from pathlib import Path

import pytest
from reportlab.pdfgen import canvas


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Hello PDF Summarizer Agent")
    c.showPage()
    c.save()
    return pdf_path
