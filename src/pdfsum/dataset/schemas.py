"""Domain-adaptive structured-summary schemas.

Each schema mirrors the confirmed fine-tune target: the model must (1) detect
the document's domain, (2) emit a fixed structured JSON shape for that domain,
(3) stay within a target overall length, and (4) do this consistently even
over long/hierarchically-chunked documents.
"""
import json

SUMMARY_TARGET_WORDS = (100, 180)

DOMAIN_SCHEMAS = {
    "paper": {
        "document_type": "paper",
        "summary": "string, {}-{} words, plain-language overview",
        "methodology": "string, how the study/approach works",
        "key_findings": ["string", "..."],
        "limitations": ["string", "..."],
    },
    "contract": {
        "document_type": "contract",
        "summary": "string, {}-{} words, plain-language overview",
        "parties": ["string", "..."],
        "key_obligations": ["string", "..."],
        "risk_flags": ["string", "..."],
    },
    "report": {
        "document_type": "report",
        "summary": "string, {}-{} words, plain-language overview",
        "key_findings": ["string", "..."],
        "recommendations": ["string", "..."],
    },
    "generic": {
        "document_type": "generic",
        "summary": "string, {}-{} words, plain-language overview",
        "key_points": ["string", "..."],
    },
}


def render_schema(domain: str) -> str:
    lo, hi = SUMMARY_TARGET_WORDS
    schema = DOMAIN_SCHEMAS[domain]
    rendered = json.dumps(schema, indent=2)
    return rendered.replace("{}-{}", f"{lo}-{hi}")


def summary_word_count(teacher_output_raw: str) -> int | None:
    """Returns the word count of the "summary" field, or None if the output
    isn't valid JSON / has no summary field."""
    try:
        parsed = json.loads(teacher_output_raw)
        return len(parsed["summary"].split())
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        return None


def is_length_compliant(teacher_output_raw: str) -> bool:
    lo, hi = SUMMARY_TARGET_WORDS
    count = summary_word_count(teacher_output_raw)
    return count is not None and lo <= count <= hi


def build_retry_prompt(domain: str, document_text: str, previous_word_count: int) -> str:
    lo, hi = SUMMARY_TARGET_WORDS
    base_prompt = build_teacher_prompt(domain, document_text)
    return (
        f"Your previous attempt's \"summary\" field was {previous_word_count} words — "
        f"outside the required {lo}-{hi} word range. Try again, and this time count "
        f"the words in your summary before finishing to make sure it's within range.\n\n"
        + base_prompt
    )


def build_teacher_prompt(domain: str, document_text: str) -> str:
    lo, hi = SUMMARY_TARGET_WORDS
    schema_str = render_schema(domain)
    return f"""You are generating training data for a PDF-summarization model. \
Read the document below (a {domain} document) and produce ONLY a JSON object \
matching this exact schema — no prose before or after, no markdown fences:

{schema_str}

Rules:
- "summary" must be {lo}-{hi} words. Count words and stay in range.
- Every list field must contain only facts actually present in the document below.
- If a field doesn't apply (e.g. no explicit limitations stated), use an empty list, never invent content.
- Output must be valid JSON, nothing else.

DOCUMENT:
{document_text}
"""
