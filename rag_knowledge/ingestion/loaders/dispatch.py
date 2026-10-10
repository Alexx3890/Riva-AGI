"""Dispatching loader for multi-format knowledge document ingestion."""

import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..readers.word import WordDocxReader
from .pdf import load_pdf_documents

logger = logging.getLogger("rag.ingest.loaders.dispatch")


def load_word_documents(filepath: Path) -> List[Dict[str, Any]]:
    """Loads Word .docx documents into knowledge documents using WordDocxReader."""
    reader = WordDocxReader()
    units = reader.read(filepath)
    docs: List[Dict[str, Any]] = []
    clean_stem = re.sub(r"[^a-zA-Z0-9]+", "_", filepath.stem).strip("_").lower()
    for idx, u in enumerate(units):
        title = u.get("title") or f"{filepath.stem} #{idx + 1}"
        content = u.get("content", "").strip()
        summary = content.split("\n")[0][:200] if content else title
        doc_id = f"word_{clean_stem}_{idx + 1}"
        docs.append({
            "id": doc_id,
            "title": title,
            "category": "document",
            "summary": summary,
            "content": content,
            "aliases": [title, filepath.stem],
            "keywords": ["document", "word", filepath.stem] + (u.get("section", "").lower().split()),
            "metadata": {"source": filepath.name, **u.get("provenance", {})},
            "is_active": True,
        })
    return docs


LOADER_REGISTRY: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    ".docx": load_word_documents,
    ".pdf": load_pdf_documents,
}


def _load_single_file(
    source_path: Path,
    dtype: str = "auto",
    safe_identifiers: Optional[Set[str]] = None,
    known_private_values: Optional[Set[str]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    ext = source_path.suffix.lower()
    loader = LOADER_REGISTRY.get(ext)
    if loader:
        return loader(source_path)

    logger.warning("Unsupported file type '%s' for '%s'. Skipping.", ext, source_path.name)
    return []


def _load_directory_sources(
    source_path: Path,
    dtype: str = "auto",
    safe_identifiers: Optional[Set[str]] = None,
    known_private_values: Optional[Set[str]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []

    for file in sorted(source_path.iterdir()):
        if not file.is_file():
            continue
        ext = file.suffix.lower()
        if ext in LOADER_REGISTRY:
            loader = LOADER_REGISTRY[ext]
            docs.extend(loader(file))
        else:
            logger.warning("Unsupported file type '%s' for '%s'. Skipping.", ext, file.name)

    return docs


def load_source_documents(
    source: Path,
    data_type: str = "auto",
    safe_identifiers: Optional[Set[str]] = None,
    known_private_values: Optional[Set[str]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Loads knowledge documents from a file or folder supporting Markdown, text, and JSON."""
    source_path = Path(source)
    dtype = (data_type or "auto").lower()

    if source_path.is_file():
        return _load_single_file(
            source_path,
            dtype=dtype,
            safe_identifiers=safe_identifiers,
            known_private_values=known_private_values,
            **kwargs,
        )

    if source_path.is_dir():
        return _load_directory_sources(
            source_path,
            dtype=dtype,
            safe_identifiers=safe_identifiers,
            known_private_values=known_private_values,
            **kwargs,
        )

    return []
