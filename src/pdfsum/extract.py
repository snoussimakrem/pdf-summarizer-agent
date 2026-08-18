from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class ExtractedPdf:
    text: str
    page_count: int


def extract_text(pdf_path: Path) -> ExtractedPdf:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return ExtractedPdf(text="\n".join(pages), page_count=len(reader.pages))
