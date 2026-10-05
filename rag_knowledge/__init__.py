"""RAG Knowledge Package for Riva Voice Assistant.

Provides modular knowledge retrieval and Mistral AI synthesis.
"""

import os
from pathlib import Path


def _load_env_fallback() -> None:
    """Environment loader that reads .env from project root or package dir."""
    env_paths = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    # Optionally check cwd if explicitly enabled
    if os.getenv("RAG_LOAD_CWD_ENV", "").lower() in ("true", "1", "yes"):
        env_paths.insert(0, Path.cwd() / ".env")

    for p in env_paths:
        if p.is_file():
            try:
                # Prefer python-dotenv if installed
                try:
                    from dotenv import load_dotenv
                    load_dotenv(dotenv_path=p, override=False)
                    continue
                except ImportError:
                    pass

                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Handle export prefix
                        if line.startswith("export "):
                            line = line[7:].strip()
                        # Strip comments
                        if not line or line.startswith("#"):
                            continue
                        if "#" in line:
                            # Strip inline comment outside quotes
                            quote_idx = -1
                            comment_idx = line.find("#")
                            if '"' not in line and "'" not in line:
                                line = line[:comment_idx].strip()

                        if "=" in line:
                            k, v = line.split("=", 1)
                            key = k.strip()
                            val = v.strip().strip("'\"")
                            if key and key not in os.environ:
                                os.environ[key] = val
            except Exception:
                pass


_load_env_fallback()

from rag_knowledge.retriever import KnowledgeRetriever
from rag_knowledge.mistral_client import MistralRAGClient
from rag_knowledge.service import RAGService, query_rag, get_rag_service
from rag_knowledge.storage.mongo import MongoKnowledgeStore

__all__ = [
    "KnowledgeRetriever",
    "MistralRAGClient",
    "RAGService",
    "query_rag",
    "get_rag_service",
    "MongoKnowledgeStore",
]

