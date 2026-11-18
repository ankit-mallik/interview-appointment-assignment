"""
Custom RAG pipeline (no LangChain RetrievalQAChain).

Steps for every user message:
1. Load chat history for the session from Redis.
2. If there is history, ask the LLM to rewrite the question as a standalone question
   (so follow ups like "what about the second one?" still retrieve the right chunks).
3. Embed the question and search Qdrant for the top-k chunks.
4. Build a prompt with the retrieved context + history and call the LLM.
   The LLM also has a `book_interview` tool it can call.
5. If the LLM calls the tool, validate + save the booking and send the result back to the LLM.
6. Save the user message and final answer back to Redis.
"""

import datetime as dt
import logging
from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from app.config import get_settings
from app.schemas import BookingResponse, ChatMessage, ChatResponse, SourceChunk
from app.services import booking as booking_service
from app.services import chat_memory, llm, vector_store
from app.services.booking import BookingResult
from app.services.embeddings import embed_query

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about the user's uploaded documents.
You can also book interviews.

Answering questions:
- Use the context below to answer. If the answer is not in the context, say you don't know
  instead of making something up.
- Keep answers short and clear. Mention which document the answer came from when it helps.

Booking interviews:
- To book an interview you need the user's full name, email, date and time.
- If any of these are missing, ask the user for them. Do not guess.
- Once you have all four, call the book_interview tool. Convert relative dates like "tomorrow"
  using today's date: {today} ({weekday}).
- After the tool returns, tell the user whether the booking worked.

Context from documents:
{context}"""

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the user's latest question so it can be understood without the chat history. "
    "Return ONLY the rewritten question, nothing else. If it is already standalone, return it unchanged."
)


def format_history(history: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in history)


def rewrite_question(history: list[ChatMessage], question: str) -> str:
    if not history:
        return question
    prompt = f"Chat history:\n{format_history(history)}\n\nLatest question: {question}"
    try:
        rewritten = llm.complete(REWRITE_SYSTEM_PROMPT, prompt, max_tokens=256)
    except llm.LLMError:
        logger.warning("Question rewrite failed, using original question")
        return question
    return rewritten or question


def build_context(sources: list[SourceChunk]) -> str:
    if not sources:
        return "(no relevant documents found)"
    parts = [f"[Source {i + 1}: {s.filename}, chunk {s.chunk_index}]\n{s.text}" for i, s in enumerate(sources)]
    return "\n\n".join(parts)


def build_system_prompt(sources: list[SourceChunk]) -> str:
    today = dt.date.today()
    return SYSTEM_PROMPT_TEMPLATE.format(
        today=today.isoformat(),
        weekday=today.strftime("%A"),
        context=build_context(sources),
    )


def run_tool_call(db: Session, session_id: str, name: str, arguments: Mapping[str, Any]) -> BookingResult:
    """Execute a tool the LLM asked for. Errors are returned as text so the LLM can tell the user."""
    if name != booking_service.BOOK_INTERVIEW_TOOL_NAME:
        return BookingResult(success=False, message=f"Error: unknown tool '{name}'")
    return booking_service.create_booking(db, session_id, dict(arguments))


def answer_question(db: Session, session_id: str, question: str) -> ChatResponse:
    settings = get_settings()
    history = chat_memory.get_history(session_id)

    # 1. retrieval
    search_query = rewrite_question(history, question)
    logger.info("Search query: %s", search_query)
    sources = vector_store.search(embed_query(search_query), top_k=settings.top_k)

    # 2. generation (with booking tool)
    system_prompt = build_system_prompt(sources)
    messages: list[dict[str, Any]] = [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": question})

    created_booking: BookingResponse | None = None
    tools = [booking_service.BOOK_INTERVIEW_TOOL]
    message = llm.create_message(system_prompt, messages, tools=tools)

    rounds = 0
    while message.tool_calls and rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        # the assistant's tool call has to be in the history before the tool results
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"function": {"name": call.function.name, "arguments": dict(call.function.arguments)}}
                    for call in message.tool_calls
                ],
            }
        )

        for call in message.tool_calls:
            result = run_tool_call(db, session_id, call.function.name, call.function.arguments)
            if result.booking is not None:
                created_booking = BookingResponse.model_validate(result.booking)
            messages.append({"role": "tool", "tool_name": call.function.name, "content": result.message})

        message = llm.create_message(system_prompt, messages, tools=tools)

    answer = llm.get_text(message) or "Sorry, I couldn't generate an answer. Please try again."

    # 3. save to memory (only plain text, not the tool calls)
    chat_memory.add_message(session_id, "user", question)
    chat_memory.add_message(session_id, "assistant", answer)

    return ChatResponse(session_id=session_id, answer=answer, sources=sources, booking=created_booking)
