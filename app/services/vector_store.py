import logging
import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import get_settings
from app.schemas import SourceChunk

logger = logging.getLogger(__name__)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def ensure_collection(vector_size: int) -> None:
    """Create the collection if it doesn't exist yet."""
    client = get_qdrant_client()
    collection = get_settings().qdrant_collection
    if not client.collection_exists(collection):
        logger.info("Creating Qdrant collection '%s' (size=%d)", collection, vector_size)
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def store_chunks(
    document_id: str,
    filename: str,
    chunks: list[str],
    vectors: list[list[float]],
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")

    points: list[PointStruct] = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload: dict[str, Any] = {
            "document_id": document_id,
            "filename": filename,
            "chunk_index": index,
            "text": chunk,
        }
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

    get_qdrant_client().upsert(collection_name=get_settings().qdrant_collection, points=points)


def search(query_vector: list[float], top_k: int) -> list[SourceChunk]:
    client = get_qdrant_client()
    collection = get_settings().qdrant_collection
    if not client.collection_exists(collection):
        return []  # nothing has been uploaded yet

    response = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    results: list[SourceChunk] = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            SourceChunk(
                document_id=payload.get("document_id", ""),
                filename=payload.get("filename", ""),
                chunk_index=payload.get("chunk_index", 0),
                score=point.score,
                text=payload.get("text", ""),
            )
        )
    return results


def delete_document_chunks(document_id: str) -> None:
    client = get_qdrant_client()
    collection = get_settings().qdrant_collection
    if not client.collection_exists(collection):
        return
    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        ),
    )
