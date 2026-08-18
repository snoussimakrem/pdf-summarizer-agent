"""Fetches real source PDFs for dataset generation, domain by domain.

Only HTTP GET against public APIs/static files — no LLM calls here.
"""
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from pdfsum.dataset.htmlconv import html_to_pdf_bytes, text_to_pdf_bytes

USER_AGENT = "pdf-summarizer-agent-research (snoussimakrem6@gmail.com)"
ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
EDGAR_FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"


@dataclass
class SourceDocument:
    domain: str
    source_id: str
    source_url: str
    license: str
    title: str
    pdf_bytes: bytes


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def fetch_arxiv_papers(
    query: str = "cat:cs.CL", max_results: int = 5, delay_seconds: float = 3.0
) -> list[SourceDocument]:
    """arXiv API usage guidance asks for max ~1 req/3s; this fetches metadata
    once, then paces the individual PDF downloads at the same rate."""
    url = f"{ARXIV_API}?search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    feed = _get(url)
    root = ET.fromstring(feed)

    docs: list[SourceDocument] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        arxiv_id = entry.find("atom:id", ARXIV_NS).text.rsplit("/", 1)[-1]
        title = entry.find("atom:title", ARXIV_NS).text.strip()
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

        time.sleep(delay_seconds)
        pdf_bytes = _get(pdf_url)

        docs.append(
            SourceDocument(
                domain="paper",
                source_id=arxiv_id,
                source_url=pdf_url,
                license="arXiv non-exclusive license (per-paper; see arXiv metadata)",
                title=title,
                pdf_bytes=pdf_bytes,
            )
        )
    return docs


def _edgar_search(query: str, forms: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode({"q": query, "forms": forms})
    payload = json.loads(_get(f"{EDGAR_FULL_TEXT_SEARCH}?{params}"))
    return payload["hits"]["hits"][:max_results]


def _edgar_doc_url(hit: dict) -> str:
    cik = str(int(hit["_source"]["ciks"][0]))
    accession = hit["_source"]["adsh"].replace("-", "")
    filename = hit["_id"].split(":", 1)[1]
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"


def _edgar_document_to_source(
    hit: dict, domain: str, delay_seconds: float
) -> SourceDocument:
    url = _edgar_doc_url(hit)
    time.sleep(delay_seconds)
    html = _get(url).decode("utf-8", errors="replace")
    pdf_bytes = html_to_pdf_bytes(html)
    return SourceDocument(
        domain=domain,
        source_id=hit["_id"],
        source_url=url,
        license="SEC EDGAR public filing; copyright status of underlying text not "
        "warranted by SEC or by us — store source_url/hash only, never redistribute "
        "the extracted full text",
        title=hit["_source"].get("display_names", [""])[0],
        pdf_bytes=pdf_bytes,
    )


def fetch_sec_contracts(
    query: str = "credit agreement",
    forms: str = "10-K",
    max_results: int = 5,
    delay_seconds: float = 1.0,
) -> list[SourceDocument]:
    """Material-contract exhibits (EX-10.x) filed with 10-Ks, via EDGAR full text search."""
    hits = _edgar_search(query, forms, max_results * 4)
    contract_hits = [h for h in hits if h["_source"]["file_type"].startswith("EX-10")]
    return [
        _edgar_document_to_source(h, "contract", delay_seconds)
        for h in contract_hits[:max_results]
    ]


DEFAULT_GUTENBERG_IDS = [
    "1342",  # Pride and Prejudice
    "84",  # Frankenstein
    "2701",  # Moby-Dick
    "11",  # Alice's Adventures in Wonderland
    "1661",  # The Adventures of Sherlock Holmes
]


def _strip_gutenberg_boilerplate(text: str) -> str:
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start == -1 or end == -1:
        return text
    start = text.find("\n", start) + 1
    return text[start:end].strip()


GUTENBERG_EXCERPT_CHARS = 30_000


def fetch_gutenberg_books(
    book_ids: list[str] = DEFAULT_GUTENBERG_IDS,
    max_results: int = 5,
    delay_seconds: float = 2.0,
    max_chars: int = GUTENBERG_EXCERPT_CHARS,
) -> list[SourceDocument]:
    """Public-domain books from Project Gutenberg, for the generic long-document domain.
    Truncated to an excerpt — a full novel (300+ pages) isn't representative of a
    PDF a user would realistically upload, and would blow the teacher model's
    50-req/day free-tier budget on a single example."""
    docs: list[SourceDocument] = []
    for book_id in book_ids[:max_results]:
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        time.sleep(delay_seconds)
        raw = _get(url).decode("utf-8", errors="replace")
        text = _strip_gutenberg_boilerplate(raw)[:max_chars]
        title = next(
            (line.split(":", 1)[1].strip() for line in raw.splitlines() if line.startswith("Title:")),
            f"Gutenberg #{book_id}",
        )
        docs.append(
            SourceDocument(
                domain="generic",
                source_id=book_id,
                source_url=url,
                license="Public domain (Project Gutenberg); Gutenberg's own license "
                "text applies only if redistributing their file verbatim, which we "
                "don't — we only store source_url/hash and derived summaries",
                title=title,
                pdf_bytes=text_to_pdf_bytes(text),
            )
        )
    return docs


DEFAULT_REPORT_CIKS = [
    "0000320193",  # Apple
    "0000789019",  # Microsoft
    "0001018724",  # Amazon
    "0001652044",  # Alphabet
    "0001326801",  # Meta
]


def fetch_sec_reports(
    ciks: list[str] = DEFAULT_REPORT_CIKS,
    max_results: int = 5,
    delay_seconds: float = 1.0,
) -> list[SourceDocument]:
    """Primary annual-report (10-K) filings, via EDGAR's per-company submissions API
    (full text search only reliably surfaces exhibits, not the primary document)."""
    docs: list[SourceDocument] = []
    for cik in ciks[:max_results]:
        time.sleep(delay_seconds)
        submissions = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
        recent = submissions["filings"]["recent"]
        for form, accession, primary_doc, filing_date in zip(
            recent["form"], recent["accessionNumber"], recent["primaryDocument"],
            recent["filingDate"],
        ):
            if form != "10-K":
                continue
            accession_nodash = accession.replace("-", "")
            cik_int = str(int(cik))
            url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}"
            time.sleep(delay_seconds)
            html = _get(url).decode("utf-8", errors="replace")
            docs.append(
                SourceDocument(
                    domain="report",
                    source_id=f"{cik}:{accession}",
                    source_url=url,
                    license="SEC EDGAR public filing; copyright status of underlying "
                    "text not warranted by SEC or by us — store source_url/hash only, "
                    "never redistribute the extracted full text",
                    title=f"{submissions.get('name', cik)} 10-K ({filing_date})",
                    pdf_bytes=html_to_pdf_bytes(html),
                )
            )
            break
    return docs
