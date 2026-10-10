"""Ingestion pipeline coordinating document loading, privacy checks, and Qdrant upserts."""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .. import load_env
from ..storage.qdrant_storage import get_global_qdrant_store
from .privacy import assert_no_privacy_leaks

logger = logging.getLogger("rag.ingest")


def _resolve_and_load_documents(
    source_path: Path,
    data_type: str,
    allow_cloud_vision: bool,
    safe_identifiers: Set[str],
    known_private_values: Set[str],
    loader_func: Callable[..., List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Path]:
    documents = loader_func(
        source_path,
        data_type=data_type,
        allow_cloud_vision=allow_cloud_vision,
        safe_identifiers=safe_identifiers,
        known_private_values=known_private_values,
    )

    if not documents and not source_path.is_file():
        candidates = [
            source_path,
            Path(__file__).resolve().parent.parent.parent / "local_data",
            Path.cwd() / "local_data",
            Path.cwd() / "rag_knowledge" / "local_data",
        ]
        for c in candidates:
            if c.exists():
                documents = loader_func(
                    c,
                    data_type=data_type,
                    allow_cloud_vision=allow_cloud_vision,
                    safe_identifiers=safe_identifiers,
                    known_private_values=known_private_values,
                )
                if documents:
                    source_path = c
                    break

    return documents, source_path


def _display_dry_run_preview(documents: List[Dict[str, Any]]) -> None:
    print("[DRY-RUN MODE] Showing sample documents that would be upserted:\n")
    sample_count = min(3, len(documents))
    for i in range(sample_count):
        d = documents[i]
        print(f"--- Document #{i+1}: [{d['id']}] ---")
        print(f"Title:    {d['title']}")
        print(f"Category: {d.get('category', 'general')}")
        print(f"Aliases:  {', '.join(d.get('aliases', [])[:6])}")
        print(f"Keywords: {', '.join(d.get('keywords', [])[:6])}")
        print(f"Summary:  {d['summary']}")
        print(f"Content Preview:\n{d.get('content', '')[:250]}...")
        print()
    print("[DRY-RUN COMPLETE] Zero changes were made to Qdrant.")


def _upsert_documents_to_qdrant(documents: List[Dict[str, Any]], batch_size: int) -> int:
    store = get_global_qdrant_store()
    store.timeout = float(os.getenv("QDRANT_INGEST_TIMEOUT", "30.0"))
    if not store.is_available():
        print("[ERROR] Qdrant storage is unreachable. Verify Qdrant configuration in .env.")
        sys.exit(1)

    print(
        f"Upserting {len(documents)} documents to Qdrant Cloud "
        f"(collection: '{store.collection_name}') in batches of {batch_size}..."
    )
    total_upserted = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        try:
            count = store.upsert_documents(batch)
            total_upserted += count
            print(
                f"  * Batch {i // batch_size + 1}: Upserted {count} documents "
                f"(Progress: {total_upserted}/{len(documents)})"
            )
        except Exception as e:
            logger.error(f"Error upserting batch {i // batch_size + 1}: {e}", exc_info=True)

    if total_upserted != len(documents):
        print(f"\n[INGESTION FAILURE] Only {total_upserted}/{len(documents)} documents were successfully upserted.")
        sys.exit(1)

    print(f"\n[INGESTION SUCCESS] Successfully upserted all {total_upserted} documents into Qdrant Cloud!")
    return total_upserted


def run_ingestion(
    source_dir: Path,
    dry_run: bool = False,
    limit: Optional[int] = None,
    batch_size: int = 50,
    data_type: str = "auto",
    redact: bool = False,
    allow_cloud_vision: bool = False,
    loader: Optional[Callable[..., List[Dict[str, Any]]]] = None,
) -> int:
    """Coordinates reading datasets, building documents, and upserting into Qdrant."""
    load_env()
    cleaned_source = str(source_dir).strip(' "\'\r\n\t') if isinstance(source_dir, (str, Path)) else source_dir
    source_path = Path(cleaned_source)

    if loader is None:
        try:
            from .loaders.dispatch import load_source_documents
        except ImportError:
            from .ingest import load_source_documents
        loader = load_source_documents

    safe_identifiers: Set[str] = set()
    known_private_values: Set[str] = set()

    documents, source_path = _resolve_and_load_documents(
        source_path,
        data_type=data_type,
        allow_cloud_vision=allow_cloud_vision,
        safe_identifiers=safe_identifiers,
        known_private_values=known_private_values,
        loader_func=loader,
    )

    if not documents:
        logger.error(f"No valid knowledge data found in {source_path}")
        return 0

    assert_no_privacy_leaks(
        documents,
        safe_identifiers=safe_identifiers,
        known_private_values=known_private_values,
        opt_in_redact=redact,
    )

    if limit and limit > 0:
        documents = documents[:limit]
        logger.info(f"Limited document ingestion to first {limit} records.")

    print("\n=======================================================")
    print("Target Backend: QDRANT VECTOR DATABASE")
    print(f"Total Unified Knowledge Documents: {len(documents)}")
    print("=======================================================\n")

    if dry_run:
        _display_dry_run_preview(documents)
        return len(documents)

    return _upsert_documents_to_qdrant(documents, batch_size=batch_size)
