"""Ingestion package for RAG knowledge documents."""

from .cli import main
from .loaders.dispatch import load_source_documents, load_word_documents
from .loaders.pdf import load_pdf_documents
from .pipeline import run_ingestion
from .privacy import assert_no_privacy_leaks

__all__ = [
    "run_ingestion",
    "load_source_documents",
    "load_pdf_documents",
    "load_word_documents",
    "assert_no_privacy_leaks",
    "main",
]

