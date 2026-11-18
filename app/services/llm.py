"""
Small wrapper around Ollama so the rest of the code doesn't talk to it directly.

Ollama runs open-source models locally (default: qwen2.5:7b), so there is no API key
and no cost. Ollama is installed natively (see README), Qdrant and Redis run in Docker.
"""

import logging
from functools import lru_cache
from typing import Any

import httpx
from ollama import Client, Message, ResponseError

from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


@lru_cache
def get_llm_client() -> Client:
    settings = get_settings()
    return Client(host=settings.ollama_url, timeout=settings.llm_timeout_seconds)


def create_message(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int | None = None,
) -> Message:
    """Send a chat request and return the assistant message (may contain tool_calls)."""
    settings = get_settings()
    all_messages = [{"role": "system", "content": system}, *messages]

    try:
        response = get_llm_client().chat(
            model=settings.llm_model,
            messages=all_messages,
            tools=tools or None,
            options={
                "temperature": settings.llm_temperature,
                "num_predict": max_tokens or settings.llm_max_tokens,
                "num_ctx": settings.llm_context_window,
            },
        )
    except ResponseError as e:
        logger.exception("Ollama returned an error")
        if e.status_code == 404:
            raise LLMError(
                f"Model '{settings.llm_model}' is not downloaded. "
                f"Run: docker compose exec ollama ollama pull {settings.llm_model}"
            ) from e
        raise LLMError(f"LLM request failed: {e.error}") from e
    except (ConnectionError, httpx.ConnectError) as e:
        raise LLMError(f"Could not connect to Ollama at {settings.ollama_url}. Is it running?") from e
    except httpx.TimeoutException as e:
        raise LLMError("LLM took too long to respond") from e

    return response.message


def get_text(message: Message) -> str:
    return (message.content or "").strip()


def complete(system: str, prompt: str, max_tokens: int = 256) -> str:
    """Simple one shot call that returns just text."""
    message = create_message(system=system, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
    return get_text(message)
