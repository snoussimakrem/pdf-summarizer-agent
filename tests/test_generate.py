import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from pdfsum import db
from pdfsum.dataset import generate
from pdfsum.dataset.sources import SourceDocument


@pytest.fixture
def conn(tmp_path: Path):
    return db.connect(tmp_path / "registry.db")


def _make_pdf_bytes(tmp_path: Path, text: str) -> bytes:
    pdf_path = tmp_path / "doc.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, text)
    c.showPage()
    c.save()
    return pdf_path.read_bytes()


def test_short_document_uses_single_pass(conn, tmp_path, monkeypatch) -> None:
    pdf_bytes = _make_pdf_bytes(tmp_path, "A short generic document.")
    source = SourceDocument(
        domain="generic",
        source_id="test-1",
        source_url="https://example.com/doc.pdf",
        license="test",
        title="Test Doc",
        pdf_bytes=pdf_bytes,
    )

    compliant_summary = " ".join(["word"] * 140)  # within the 100-180 target range
    calls = []

    def fake_call_teacher(conn_, prompt, model=None, **kwargs):
        calls.append(prompt)
        return json.dumps(
            {"document_type": "generic", "summary": compliant_summary, "key_points": []}
        )

    monkeypatch.setattr("pdfsum.dataset.generate.teacher.call_teacher", fake_call_teacher)

    example = generate.generate_example(conn, source)

    assert example["generation_method"] == "single_pass"
    assert example["domain"] == "generic"
    assert example["length_compliant"] is True
    assert len(calls) == 1  # compliant on the first try, no retry needed
    assert "A short generic document." in example["document_text"]


def test_long_document_uses_hierarchical(conn, tmp_path, monkeypatch) -> None:
    long_text = "word " * 10_000
    pdf_bytes = _make_pdf_bytes(tmp_path, "irrelevant, we monkeypatch extraction")
    source = SourceDocument(
        domain="report",
        source_id="test-2",
        source_url="https://example.com/report.pdf",
        license="test",
        title="Test Report",
        pdf_bytes=pdf_bytes,
    )

    from pdfsum.extract import ExtractedPdf

    monkeypatch.setattr(
        "pdfsum.dataset.generate.extract_text_from_bytes",
        lambda b: ExtractedPdf(text=long_text, page_count=50),
    )

    calls = []

    def fake_call_teacher(conn_, prompt, model=None, **kwargs):
        calls.append(prompt)
        return "chunk summary" if len(calls) < 3 else '{"document_type": "report"}'

    monkeypatch.setattr("pdfsum.dataset.generate.teacher.call_teacher", fake_call_teacher)

    example = generate.generate_example(conn, source)

    assert example["generation_method"] == "hierarchical"
    assert len(calls) > 1  # per-chunk calls + final synthesis call
