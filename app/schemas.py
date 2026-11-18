import datetime as dt
from enum import Enum

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"  # fixed size character windows with overlap
    SENTENCE = "sentence"  # groups whole sentences together


# ---------- Ingestion ----------


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    file_type: str
    file_size: int
    chunking_strategy: str
    num_chunks: int
    num_characters: int
    created_at: dt.datetime


# ---------- Chat ----------


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        max_length=64,
        description="Pass the same session_id to continue a conversation. Leave empty to start a new one.",
    )
    message: str = Field(min_length=1, max_length=4000)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    score: float
    text: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk] = []
    booking: "BookingResponse | None" = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]


# ---------- Booking ----------


class BookingCreate(BaseModel):
    """Booking details extracted by the LLM. Validated here before we save anything."""

    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    date: dt.date
    time: dt.time

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("date")
    @classmethod
    def date_not_in_past(cls, value: dt.date) -> dt.date:
        if value < dt.date.today():
            raise ValueError("interview date cannot be in the past")
        return value


class BookingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: str
    name: str
    email: str
    interview_date: dt.date
    interview_time: dt.time
    created_at: dt.datetime


ChatResponse.model_rebuild()
