import pytest

from app.schemas import ChunkingStrategy
from app.services.chunking import chunk_text, fixed_size_chunking, sentence_chunking, split_into_sentences


def test_fixed_size_chunking_respects_size_and_overlap() -> None:
    text = "a" * 2000
    chunks = fixed_size_chunking(text, chunk_size=800, overlap=100)

    assert all(len(c) <= 800 for c in chunks)
    # step is 700 so we need 3 chunks to cover 2000 chars
    assert len(chunks) == 3


def test_fixed_size_chunking_short_text() -> None:
    assert fixed_size_chunking("hello world", chunk_size=800, overlap=100) == ["hello world"]


def test_fixed_size_chunking_empty_text() -> None:
    assert fixed_size_chunking("") == []


def test_fixed_size_chunking_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        fixed_size_chunking("abc", chunk_size=10, overlap=10)


def test_split_into_sentences() -> None:
    text = "First sentence. Second one! Is this the third?\n\nNew paragraph"
    assert split_into_sentences(text) == [
        "First sentence.",
        "Second one!",
        "Is this the third?",
        "New paragraph",
    ]


def test_sentence_chunking_keeps_sentences_whole() -> None:
    sentences = [f"This is sentence number {i}." for i in range(50)]
    text = " ".join(sentences)
    chunks = sentence_chunking(text, max_chunk_size=200, overlap_sentences=1)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200
        assert chunk.endswith(".")  # no sentence cut in the middle


def test_sentence_chunking_has_overlap() -> None:
    text = " ".join(f"Sentence {i} is here." for i in range(30))
    chunks = sentence_chunking(text, max_chunk_size=100, overlap_sentences=1)

    last_sentence_of_first = split_into_sentences(chunks[0])[-1]
    assert chunks[1].startswith(last_sentence_of_first)


def test_sentence_chunking_very_long_sentence() -> None:
    text = "word " * 500  # one huge "sentence" with no full stop
    chunks = sentence_chunking(text, max_chunk_size=200)
    assert all(len(c) <= 200 for c in chunks)


@pytest.mark.parametrize("strategy", list(ChunkingStrategy))
def test_chunk_text_all_strategies(strategy: ChunkingStrategy) -> None:
    chunks = chunk_text("Some text. More text.", strategy)
    assert len(chunks) == 1
