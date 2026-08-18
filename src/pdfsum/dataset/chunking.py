"""Splits long document text into word-count chunks for hierarchical
(map-reduce) teacher summarization: summarize each chunk, then synthesize
the chunk-summaries into the final structured schema output. This is what
makes "long-document handling" a property of the training data rather
than something we just hope a big context window handles well.
"""

LONG_DOCUMENT_WORD_THRESHOLD = 6_000
CHUNK_WORD_SIZE = 4_000
CHUNK_WORD_OVERLAP = 200


def is_long_document(text: str) -> bool:
    return len(text.split()) > LONG_DOCUMENT_WORD_THRESHOLD


def chunk_text(
    text: str, chunk_words: int = CHUNK_WORD_SIZE, overlap_words: int = CHUNK_WORD_OVERLAP
) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    start = 0
    step = chunk_words - overlap_words
    while start < len(words):
        chunk_words_slice = words[start : start + chunk_words]
        chunks.append(" ".join(chunk_words_slice))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


def build_chunk_extraction_prompt(chunk_text_: str, chunk_index: int, total_chunks: int) -> str:
    return f"""This is excerpt {chunk_index + 1} of {total_chunks} from a longer document. \
Extract the key facts, claims, and figures from ONLY this excerpt as a concise \
bullet list (plain text, no JSON, no preamble). Do not summarize the whole \
document — you only have this excerpt.

EXCERPT:
{chunk_text_}
"""
