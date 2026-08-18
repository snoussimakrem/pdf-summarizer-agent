import argparse
import sys
from pathlib import Path

from pdfsum import db
from pdfsum.extract import extract_text


def cmd_ingest(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"error: no such file: {pdf_path}", file=sys.stderr)
        return 1

    data = pdf_path.read_bytes()
    sha256 = db.hash_bytes(data)

    conn = db.connect(Path(args.db_path))
    existing = db.find_by_hash(conn, sha256)
    if existing is not None:
        print(f"already ingested as document #{existing['id']} ({existing['filename']})")
        return 0

    extracted = extract_text(pdf_path)
    doc_id = db.insert_document(
        conn,
        filename=pdf_path.name,
        sha256=sha256,
        page_count=extracted.page_count,
        char_count=len(extracted.text),
    )
    print(f"ingested document #{doc_id}: {pdf_path.name} "
          f"({extracted.page_count} pages, {len(extracted.text)} chars)")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    conn = db.connect(Path(args.db_path))
    rows = db.list_documents(conn)
    if not rows:
        print("no documents ingested yet")
        return 0
    for row in rows:
        print(f"#{row['id']}  {row['filename']}  "
              f"{row['page_count']} pages  {row['char_count']} chars  "
              f"{row['ingested_at']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pdfsum")
    parser.add_argument(
        "--db-path", default=str(db.DEFAULT_DB_PATH), help="path to the SQLite registry"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="extract text from a PDF and register it")
    ingest_parser.add_argument("pdf_path")
    ingest_parser.set_defaults(func=cmd_ingest)

    list_parser = subparsers.add_parser("list", help="list ingested documents")
    list_parser.set_defaults(func=cmd_list)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
