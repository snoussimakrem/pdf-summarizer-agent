"""Renders arbitrary HTML/plain text into a real PDF, so non-PDF sources
(e.g. SEC EDGAR HTML filings) go through the same PDF-extraction path as
real user uploads, keeping training data and deployment input consistent.
"""
import io

from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Inline-XBRL filings (e.g. SEC EDGAR) embed a machine-readable metadata
    # block (<ix:header>, <ix:hidden>, ...) that browsers never render but a
    # plain get_text() picks up as visible text.
    for tag in soup.find_all(lambda t: t.name and t.name.lower().startswith("ix:")):
        tag.decompose()
    for tag in soup.find_all(style=True):
        style = tag["style"].replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()
    lines = [line.strip() for line in soup.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line)


def text_to_pdf_bytes(text: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = []
    for paragraph in text.split("\n"):
        if not paragraph:
            continue
        escaped = (
            paragraph.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        story.append(Paragraph(escaped, styles["Normal"]))
        story.append(Spacer(1, 6))
    doc.build(story)
    return buffer.getvalue()


def html_to_pdf_bytes(html: str) -> bytes:
    return text_to_pdf_bytes(html_to_text(html))
