"""Orchestrates dataset generation: source document -> (chunked if long) teacher
summarization -> a JSONL training example. Writes to data/dataset/ (gitignored —
these examples embed full source text, and licensing on the report/contract
sources doesn't clearly permit public redistribution of that text; see
sources.py for the per-domain license notes)."""
import json
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import Connection

from pdfsum.dataset import chunking, schemas, teacher
from pdfsum.dataset.sources import SourceDocument
from pdfsum.extract import extract_text_from_bytes

DEFAULT_OUTPUT_PATH = Path("data/dataset/examples.jsonl")


def _synthesize_from_chunks(
    conn: Connection, document_text: str, domain: str, model: str
) -> str:
    chunks = chunking.chunk_text(document_text)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = chunking.build_chunk_extraction_prompt(chunk, i, len(chunks))
        chunk_summaries.append(teacher.call_teacher(conn, prompt, model=model))

    combined = "\n\n".join(
        f"[excerpt {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    final_prompt = schemas.build_teacher_prompt(domain, combined)
    return teacher.call_teacher(conn, final_prompt, model=model)


def generate_example(
    conn: Connection,
    source: SourceDocument,
    model: str = teacher.FREE_MODELS[0],
) -> dict:
    extracted = extract_text_from_bytes(source.pdf_bytes)
    document_text = extracted.text

    if chunking.is_long_document(document_text):
        raw_output = _synthesize_from_chunks(conn, document_text, source.domain, model)
        generation_method = "hierarchical"
    else:
        prompt = schemas.build_teacher_prompt(source.domain, document_text)
        raw_output = teacher.call_teacher(conn, prompt, model=model)
        generation_method = "single_pass"

    return {
        "domain": source.domain,
        "source_id": source.source_id,
        "source_url": source.source_url,
        "license": source.license,
        "title": source.title,
        "page_count": extracted.page_count,
        "document_text": document_text,
        "teacher_model": model,
        "generation_method": generation_method,
        "teacher_output_raw": raw_output,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def append_example(example: dict, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(example) + "\n")
