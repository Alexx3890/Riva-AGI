"""Command-line interface and argument parsing for knowledge ingestion."""

import argparse
import logging
import os
import sys
from pathlib import Path

from .privacy import PrivacyGateError
from .pipeline import run_ingestion

logger = logging.getLogger("rag.ingest")


def build_arg_parser() -> argparse.ArgumentParser:
    """Configures command-line arguments for knowledge ingestion."""
    parser = argparse.ArgumentParser(description="RAG Knowledge Data Ingestion Pipeline (Qdrant)")
    parser.add_argument(
        "--source",
        type=str,
        default=os.getenv("RAG_DATA_SOURCE", "data/raw"),
        help="Path to folder or file (supports .pdf, .docx)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="auto",
        choices=["auto", "pdf", "word", "doc", "docx"],
        help="Data type format (default: auto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate ingestion and display sample documents without updating database",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to ingest",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("QDRANT_BATCH_SIZE", "50")),
        help="Bulk upsert batch size (default: 50, configurable via QDRANT_BATCH_SIZE)",
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Opt-in to redacting detected personal emails and phone numbers instead of failing (PRD S4)",
    )
    return parser


def resolve_source_path(raw_source: str) -> Path:
    """Discovers source file or directory across standard workspace candidates."""
    raw_path = Path(raw_source)
    package_dir = Path(__file__).resolve().parent.parent

    candidates = [
        raw_path,
        Path.cwd() / raw_source,
        package_dir.parent / raw_source,
        package_dir / raw_source,
    ]
    if raw_source.startswith("rag_knowledge/") or raw_source.startswith("rag_knowledge\\"):
        sub_rel = raw_source[len("rag_knowledge") + 1 :]
        candidates.append(package_dir / sub_rel)
        candidates.append(Path.cwd() / sub_rel)

    for cand in candidates:
        if cand.is_file() or cand.is_dir():
            return cand

    return raw_path


def main() -> None:
    """Main CLI entry point for knowledge ingestion."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    raw_source = (args.source or "").strip(' "\'\r\n\t')
    source_dir = resolve_source_path(raw_source)

    try:
        total = run_ingestion(
            source_dir=source_dir,
            dry_run=args.dry_run,
            limit=args.limit,
            batch_size=args.batch_size,
            data_type=args.type,
            redact=args.redact,
        )
        if not args.dry_run and total == 0:
            sys.exit(1)
    except PrivacyGateError as pge:
        print("\n[ERROR] Ingestion blocked by Privacy Gate:")
        print(f"{pge}")
        print("\nTo automatically redact detected contact details before upserting, re-run with --redact:")
        print(f"  .venv\\Scripts\\python.exe -m rag_knowledge.ingestion --source \"{raw_source}\" --redact\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
