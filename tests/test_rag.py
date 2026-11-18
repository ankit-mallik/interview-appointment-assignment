"""Tests for the RAG flow. Redis, Qdrant, embeddings and the LLM are all mocked."""

import datetime as dt
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.schemas import ChatMessage, SourceChunk
from app.services import rag


class FakeMemory:
    def __init__(self) -> None:
        self.store: dict[str, list[ChatMessage]] = {}

    def get_history(self, session_id: str) -> list[ChatMessage]:
        return list(self.store.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.store.setdefault(session_id, []).append(ChatMessage(role=role, content=content))


def text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=text, tool_calls=None)


def tool_response(tool_input: dict[str, Any], name: str = "book_interview") -> SimpleNamespace:
    call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=tool_input))
    return SimpleNamespace(content="", tool_calls=[call])


@pytest.fixture
def memory(monkeypatch: pytest.MonkeyPatch) -> FakeMemory:
    fake = FakeMemory()
    monkeypatch.setattr(rag.chat_memory, "get_history", fake.get_history)
    monkeypatch.setattr(rag.chat_memory, "add_message", fake.add_message)
    monkeypatch.setattr(rag, "embed_query", lambda q: [0.1, 0.2])
    monkeypatch.setattr(
        rag.vector_store,
        "search",
        lambda vector, top_k: [
            SourceChunk(document_id="d1", filename="policy.txt", chunk_index=0, score=0.9, text="Leave is 20 days.")
        ],
    )
    return fake


def test_answer_question_uses_context_and_saves_history(
    db: Session, memory: FakeMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_message(system: str, messages: list[dict[str, Any]], tools: Any = None, max_tokens: Any = None):
        captured["system"] = system
        return text_response("You get 20 days of leave.")

    monkeypatch.setattr(rag.llm, "create_message", fake_create_message)

    result = rag.answer_question(db, "s1", "How many leave days?")

    assert result.answer == "You get 20 days of leave."
    assert "Leave is 20 days." in captured["system"]
    assert len(result.sources) == 1
    assert [m.role for m in memory.store["s1"]] == ["user", "assistant"]


def test_follow_up_question_is_rewritten(db: Session, memory: FakeMemory, monkeypatch: pytest.MonkeyPatch) -> None:
    memory.add_message("s1", "user", "What is the leave policy?")
    memory.add_message("s1", "assistant", "20 days per year.")

    searched: list[str] = []
    monkeypatch.setattr(rag, "embed_query", lambda q: searched.append(q) or [0.1])
    monkeypatch.setattr(rag.llm, "complete", lambda system, prompt, max_tokens=0: "Can unused leave be carried over?")
    monkeypatch.setattr(rag.llm, "create_message", lambda *a, **k: text_response("Yes, up to 5 days."))

    rag.answer_question(db, "s1", "can it be carried over?")

    assert searched == ["Can unused leave be carried over?"]


def test_booking_through_tool_call(db: Session, memory: FakeMemory, monkeypatch: pytest.MonkeyPatch) -> None:
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    responses = [
        tool_response({"name": "John Doe", "email": "john@example.com", "date": tomorrow, "time": "15:00"}),
        text_response("Your interview is booked!"),
    ]
    sent_messages: list[list[dict[str, Any]]] = []

    def fake_create_message(system: str, messages: list[dict[str, Any]], tools: Any = None, max_tokens: Any = None):
        sent_messages.append(list(messages))
        return responses.pop(0)

    monkeypatch.setattr(rag.llm, "create_message", fake_create_message)

    result = rag.answer_question(db, "s1", f"Book an interview for John Doe, john@example.com on {tomorrow} at 3pm")

    assert result.booking is not None
    assert result.booking.name == "John Doe"
    assert result.answer == "Your interview is booked!"
    # second LLM call should contain the assistant tool call + the tool result
    assistant_message, tool_message = sent_messages[1][-2], sent_messages[1][-1]
    assert assistant_message["tool_calls"][0]["function"]["name"] == "book_interview"
    assert tool_message["role"] == "tool"
    assert tool_message["tool_name"] == "book_interview"
    assert "booked successfully" in tool_message["content"]


def test_invalid_booking_error_goes_back_to_llm(
    db: Session, memory: FakeMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = [
        tool_response({"name": "John Doe", "email": "bad-email", "date": "2099-01-01", "time": "10:00"}),
        text_response("That email doesn't look right, can you check it?"),
    ]
    sent_messages: list[list[dict[str, Any]]] = []

    def fake_create_message(system: str, messages: list[dict[str, Any]], tools: Any = None, max_tokens: Any = None):
        sent_messages.append(list(messages))
        return responses.pop(0)

    monkeypatch.setattr(rag.llm, "create_message", fake_create_message)

    result = rag.answer_question(db, "s1", "book it")

    assert result.booking is None
    assert sent_messages[1][-1]["content"].startswith("Error:")


def test_run_tool_call_unknown_tool(db: Session) -> None:
    result = rag.run_tool_call(db, "s1", "delete_everything", {})
    assert not result.success
