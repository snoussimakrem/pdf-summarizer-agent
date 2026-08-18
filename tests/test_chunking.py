from pdfsum.dataset.chunking import chunk_text, is_long_document


def test_short_text_is_single_chunk() -> None:
    text = "word " * 100
    assert not is_long_document(text)
    assert chunk_text(text) == [text]


def test_long_text_is_split_with_overlap() -> None:
    words = [f"w{i}" for i in range(10_000)]
    text = " ".join(words)
    assert is_long_document(text)

    chunks = chunk_text(text, chunk_words=4_000, overlap_words=200)
    assert len(chunks) > 1
    # every word appears in some chunk, in order, nothing dropped
    rejoined_first_words = chunks[0].split()
    assert rejoined_first_words[0] == "w0"
    last_chunk_words = chunks[-1].split()
    assert last_chunk_words[-1] == "w9999"
