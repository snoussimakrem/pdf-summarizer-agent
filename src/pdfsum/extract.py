import io
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class ExtractedPdf:
    text: str
    page_count: int


def extract_text_from_bytes(pdf_bytes: bytes) -> ExtractedPdf:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return ExtractedPdf(text="\n".join(pages), page_count=len(reader.pages))


def extract_text(pdf_path: Path) -> ExtractedPdf:
    return extract_text_from_bytes(pdf_path.read_bytes())
