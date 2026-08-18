from pathlib import Path

from pdfsum import db


def test_insert_and_list(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "registry.db")

    doc_id = db.insert_document(
        conn, filename="doc.pdf", sha256="abc123", page_count=3, char_count=500
    )
    assert doc_id == 1

    rows = db.list_documents(conn)
    assert len(rows) == 1
    assert rows[0]["filename"] == "doc.pdf"

    assert db.find_by_hash(conn, "abc123") is not None
    assert db.find_by_hash(conn, "nope") is None
