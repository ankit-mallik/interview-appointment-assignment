import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """Load the model once and reuse it (loading takes a few seconds)."""
    model_name = get_settings().embedding_model
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def get_embedding_dimension() -> int:
    dimension = get_embedding_model().get_sentence_embedding_dimension()
    if dimension is None:
        raise RuntimeError("Could not figure out embedding dimension")
    return dimension


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_embedding_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
