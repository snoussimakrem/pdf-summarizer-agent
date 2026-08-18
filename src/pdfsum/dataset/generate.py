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


MAX_LENGTH_RETRIES = 1


def _generate_with_length_retry(
    conn: Connection, source_text: str, domain: str, model: str
) -> tuple[str, bool]:
    """Returns (raw_output, length_compliant). Retries once with explicit
    word-count feedback if the teacher missed the target length — training on
    non-compliant examples would teach the student model the wrong lesson
    about the length-control requirement. `source_text` is either the
    document itself (single-pass) or combined chunk-summaries (hierarchical)."""
    prompt = schemas.build_teacher_prompt(domain, source_text)
    raw_output = teacher.call_teacher(conn, prompt, model=model)

    for _ in range(MAX_LENGTH_RETRIES):
        if schemas.is_length_compliant(raw_output):
            return raw_output, True
        word_count = schemas.summary_word_count(raw_output)
        if word_count is None:
            break  # not recoverable via a length-retry (invalid JSON)
        retry_prompt = schemas.build_retry_prompt(domain, source_text, word_count)
        raw_output = teacher.call_teacher(conn, retry_prompt, model=model)

    return raw_output, schemas.is_length_compliant(raw_output)


def _synthesize_from_chunks(
    conn: Connection, document_text: str, domain: str, model: str
) -> tuple[str, bool]:
    chunks = chunking.chunk_text(document_text)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = chunking.build_chunk_extraction_prompt(chunk, i, len(chunks))
        chunk_summaries.append(teacher.call_teacher(conn, prompt, model=model))

    combined = "\n\n".join(
        f"[excerpt {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )
    return _generate_with_length_retry(conn, combined, domain, model)


def generate_example(
    conn: Connection,
    source: SourceDocument,
    model: str = teacher.FREE_MODELS[0],
) -> dict:
    extracted = extract_text_from_bytes(source.pdf_bytes)
    document_text = extracted.text

    if chunking.is_long_document(document_text):
        raw_output, length_compliant = _synthesize_from_chunks(
            conn, document_text, source.domain, model
        )
        generation_method = "hierarchical"
    else:
        raw_output, length_compliant = _generate_with_length_retry(
            conn, document_text, source.domain, model
        )
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
        "length_compliant": length_compliant,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def append_example(example: dict, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(example) + "\n")
