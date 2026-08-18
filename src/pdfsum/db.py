import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path("data/registry.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    page_count INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS teacher_api_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def insert_document(
    conn: sqlite3.Connection, filename: str, sha256: str, page_count: int, char_count: int
) -> int:
    cursor = conn.execute(
        "INSERT INTO documents (filename, sha256, page_count, char_count, ingested_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (filename, sha256, page_count, char_count, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def find_by_hash(conn: sqlite3.Connection, sha256: str) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT * FROM documents WHERE sha256 = ?", (sha256,)).fetchone()


def list_documents(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute("SELECT * FROM documents ORDER BY id").fetchall()


def count_teacher_requests_today_utc(conn: sqlite3.Connection) -> int:
    """OpenRouter's free-tier quota resets at UTC midnight (calendar day),
    not on a rolling 24h window from each request — verified 2026-08-18 by
    hitting the real 429 and reading its X-RateLimit-Reset header."""
    midnight_utc = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()
    return conn.execute(
        "SELECT COUNT(*) FROM teacher_api_requests WHERE created_at > ?", (midnight_utc,)
    ).fetchone()[0]


def record_teacher_request(conn: sqlite3.Connection, model: str) -> None:
    conn.execute(
        "INSERT INTO teacher_api_requests (model, created_at) VALUES (?, ?)",
        (model, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
