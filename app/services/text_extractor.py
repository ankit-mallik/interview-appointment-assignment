import io
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {"pdf", "txt"}


class UnsupportedFileError(Exception):
    pass


def get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def extract_text_from_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            logger.warning("Page %d has no extractable text (maybe a scanned image?)", page_number + 1)
        pages.append(page_text)
    return "\n".join(pages)


def extract_text_from_txt(content: bytes) -> str:
    # try utf-8 first, fall back to latin-1 so we never crash on weird encodings
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from an uploaded file based on its extension."""
    extension = get_extension(filename)

    if extension == "pdf":
        text = extract_text_from_pdf(content)
    elif extension == "txt":
        text = extract_text_from_txt(content)
    else:
        raise UnsupportedFileError(f"Unsupported file type '.{extension}'. Only .pdf and .txt are allowed.")

    return clean_text(text)


def clean_text(text: str) -> str:
    """Basic cleanup: remove null chars and collapse extra blank lines."""
    text = text.replace("\x00", "")
    lines = [line.strip() for line in text.splitlines()]

    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if line == "":
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
        else:
            cleaned_lines.append(line)
            previous_blank = False

    return "\n".join(cleaned_lines).strip()
