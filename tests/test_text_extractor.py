import pytest

from app.services.text_extractor import UnsupportedFileError, clean_text, extract_text, get_extension


def test_get_extension() -> None:
    assert get_extension("notes.TXT") == "txt"
    assert get_extension("my.report.pdf") == "pdf"
    assert get_extension("README") == ""


def test_extract_txt() -> None:
    assert extract_text("a.txt", b"Hello\n\n\n\nWorld") == "Hello\n\nWorld"


def test_extract_txt_non_utf8() -> None:
    assert extract_text("a.txt", "café".encode("latin-1")) == "café"


def test_unsupported_file() -> None:
    with pytest.raises(UnsupportedFileError):
        extract_text("image.png", b"123")


def test_clean_text_removes_null_chars() -> None:
    assert clean_text("abc\x00def") == "abcdef"
