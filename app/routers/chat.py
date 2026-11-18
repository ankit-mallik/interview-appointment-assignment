import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client.http.exceptions import ResponseHandlingException
from redis.exceptions import RedisError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Booking
from app.schemas import BookingResponse, ChatHistoryResponse, ChatRequest, ChatResponse
from app.services import booking as booking_service
from app.services import chat_memory
from app.services.llm import LLMError
from app.services.rag import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Conversational RAG"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    try:
        return answer_question(db, session_id, request.message)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except RedisError as e:
        logger.exception("Redis error")
        raise HTTPException(status_code=503, detail="Chat memory (Redis) is not available") from e
    except ResponseHandlingException as e:
        logger.exception("Qdrant error")
        raise HTTPException(status_code=503, detail="Vector database is not available") from e


@router.get("/chat/{session_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(session_id: str) -> ChatHistoryResponse:
    try:
        messages = chat_memory.get_history(session_id)
    except RedisError as e:
        raise HTTPException(status_code=503, detail="Chat memory (Redis) is not available") from e
    return ChatHistoryResponse(session_id=session_id, messages=messages)


@router.delete("/chat/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def clear_chat(session_id: str) -> None:
    try:
        chat_memory.clear_history(session_id)
    except RedisError as e:
        raise HTTPException(status_code=503, detail="Chat memory (Redis) is not available") from e


@router.get("/bookings", response_model=list[BookingResponse], tags=["Bookings"])
def list_bookings(db: Session = Depends(get_db)) -> list[Booking]:
    return booking_service.list_bookings(db)
