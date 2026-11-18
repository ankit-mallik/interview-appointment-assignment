import json
from functools import lru_cache

import redis

from app.config import get_settings
from app.schemas import ChatMessage


@lru_cache
def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _key(session_id: str) -> str:
    return f"chat:{session_id}"


def get_history(session_id: str) -> list[ChatMessage]:
    raw_messages: list[str] = get_redis_client().lrange(_key(session_id), 0, -1)  # type: ignore[assignment]
    return [ChatMessage(**json.loads(item)) for item in raw_messages]


def add_message(session_id: str, role: str, content: str) -> None:
    settings = get_settings()
    client = get_redis_client()
    key = _key(session_id)

    message = ChatMessage(role=role, content=content)
    client.rpush(key, message.model_dump_json())
    # keep only the last N messages so the prompt doesn't grow forever
    client.ltrim(key, -settings.chat_history_limit, -1)
    client.expire(key, settings.chat_ttl_seconds)


def clear_history(session_id: str) -> None:
    get_redis_client().delete(_key(session_id))
