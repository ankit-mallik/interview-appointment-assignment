"""
Two chunking strategies:

1. fixed    -> split text into fixed size character windows with some overlap.
               Simple and predictable, but can cut a sentence in half.
2. sentence -> split text into sentences first and then pack whole sentences
               into chunks up to a max size. Chunks are more readable for the LLM.
"""

import re

from app.schemas import ChunkingStrategy

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# split after . ! or ? when followed by whitespace (very simple sentence splitter)
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def split_into_sentences(text: str) -> list[str]:
    # treat paragraph breaks as sentence ends too
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.replace("\n", " ").strip()
        if not paragraph:
            continue
        sentences.extend(s.strip() for s in SENTENCE_SPLIT_REGEX.split(paragraph) if s.strip())
    return sentences


def sentence_chunking(
    text: str,
    max_chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_sentences: int = 1,
) -> list[str]:
    """Group sentences into chunks. The last `overlap_sentences` of a chunk are repeated in the next one."""
    sentences = split_into_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for sentence in sentences:
        # a single sentence that is too long gets split with the fixed strategy
        if len(sentence) > max_chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_length = [], 0
            chunks.extend(fixed_size_chunking(sentence, max_chunk_size, DEFAULT_CHUNK_OVERLAP))
            continue

        if current and current_length + len(sentence) + 1 > max_chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current_length = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_length += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_text(text: str, strategy: ChunkingStrategy) -> list[str]:
    if strategy == ChunkingStrategy.FIXED:
        return fixed_size_chunking(text)
    if strategy == ChunkingStrategy.SENTENCE:
        return sentence_chunking(text)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
