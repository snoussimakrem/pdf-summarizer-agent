from pathlib import Path

from pdfsum.extract import extract_text


def test_extract_text(sample_pdf: Path) -> None:
    result = extract_text(sample_pdf)
    assert result.page_count == 1
    assert "Hello PDF Summarizer Agent" in result.text
