from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings. Values are read from environment variables / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (open-source model running locally in Ollama)
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.2
    llm_context_window: int = 8192
    llm_timeout_seconds: float = 300.0  # CPU inference can be slow

    # Vector DB
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # Redis chat memory
    redis_url: str = "redis://localhost:6379/0"
    chat_history_limit: int = 20  # how many messages we keep per session
    chat_ttl_seconds: int = 60 * 60 * 24

    # SQL database
    database_url: str = "sqlite:///./app.db"

    # Embeddings / retrieval
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 4

    # Upload limits
    max_upload_size_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
