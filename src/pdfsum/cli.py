import argparse
import sys
from pathlib import Path

from pdfsum import db
from pdfsum.dataset import generate, teacher
from pdfsum.dataset.sources import fetch_arxiv_papers, fetch_gutenberg_books, fetch_sec_contracts, fetch_sec_reports
from pdfsum.extract import extract_text

DOMAIN_FETCHERS = {
    "paper": fetch_arxiv_papers,
    "contract": fetch_sec_contracts,
    "report": fetch_sec_reports,
    "generic": fetch_gutenberg_books,
}


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


def cmd_generate_dataset(args: argparse.Namespace) -> int:
    conn = db.connect(Path(args.db_path))
    fetcher = DOMAIN_FETCHERS[args.domain]
    print(f"fetching {args.count} '{args.domain}' source document(s)...")
    sources = fetcher(max_results=args.count)

    remaining_quota = teacher.DAILY_FREE_REQUEST_LIMIT - db.count_teacher_requests_today_utc(conn)
    print(f"OpenRouter free-tier requests remaining today (UTC): {remaining_quota}")

    for source in sources:
        print(f"generating example for {source.source_id} ({source.title[:60]})...")
        try:
            example = generate.generate_example(conn, source, model=args.model)
        except (teacher.QuotaExceededError, teacher.MissingApiKeyError) as e:
            print(f"stopped: {e}", file=sys.stderr)
            return 1
        generate.append_example(example)
        print(f"  -> wrote example ({example['generation_method']}, "
              f"{len(example['teacher_output_raw'])} chars teacher output)")
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

    gen_parser = subparsers.add_parser(
        "generate-dataset", help="fetch source PDFs and generate training examples via a teacher model"
    )
    gen_parser.add_argument("domain", choices=list(DOMAIN_FETCHERS))
    gen_parser.add_argument("--count", type=int, default=1)
    gen_parser.add_argument("--model", default=teacher.FREE_MODELS[0], choices=teacher.FREE_MODELS)
    gen_parser.set_defaults(func=cmd_generate_dataset)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
