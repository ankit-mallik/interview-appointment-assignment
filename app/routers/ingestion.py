import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Document
from app.schemas import ChunkingStrategy, DocumentResponse
from app.services import vector_store
from app.services.chunking import chunk_text
from app.services.embeddings import embed_texts, get_embedding_dimension
from app.services.text_extractor import SUPPORTED_EXTENSIONS, UnsupportedFileError, extract_text, get_extension

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(..., description="A .pdf or .txt file"),
    chunking_strategy: ChunkingStrategy = Form(ChunkingStrategy.SENTENCE),
    db: Session = Depends(get_db),
) -> Document:
    settings = get_settings()
    filename = file.filename or "unknown"
    extension = get_extension(filename)

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported")

    # normal `def` (not async) on purpose: embedding is CPU heavy, so FastAPI runs this in a thread pool
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File is larger than {settings.max_upload_size_mb} MB")

    # 1. extract text
    try:
        text = extract_text(filename, content)
    except UnsupportedFileError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to read file %s", filename)
        raise HTTPException(status_code=422, detail="Could not read the file. Is it a valid PDF/TXT?") from e

    if not text:
        raise HTTPException(status_code=422, detail="No text could be extracted from the file")

    # 2. chunk
    chunks = chunk_text(text, chunking_strategy)

    # 3. embed + store in Qdrant
    document_id = str(uuid.uuid4())
    try:
        vectors = embed_texts(chunks)
        vector_store.ensure_collection(get_embedding_dimension())
        vector_store.store_chunks(document_id, filename, chunks, vectors)
    except Exception as e:
        logger.exception("Failed to store embeddings")
        raise HTTPException(status_code=503, detail="Vector database is not available") from e

    # 4. save metadata in SQL
    document = Document(
        id=document_id,
        filename=filename,
        file_type=extension,
        file_size=len(content),
        chunking_strategy=chunking_strategy.value,
        num_chunks=len(chunks),
        num_characters=len(text),
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    logger.info("Ingested %s -> %d chunks (%s)", filename, len(chunks), chunking_strategy.value)
    return document


@router.get("", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return list(db.scalars(select(Document).order_by(Document.created_at.desc())))


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str, db: Session = Depends(get_db)) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> None:
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    vector_store.delete_document_chunks(document_id)
    db.delete(document)
    db.commit()
