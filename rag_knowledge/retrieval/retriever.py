"""Retriever module for rag_knowledge."""

import logging
import os
from typing import Any, Dict, List, Optional

from ..storage.qdrant_storage import (
    QdrantKnowledgeStore,
    get_global_qdrant_store,
)

logger = logging.getLogger("rag.retriever")


class RetrievalError(RuntimeError):
    """Raised when knowledge retrieval fails due to a database/network error or outage."""
    pass


class KnowledgeRetriever:
    """Vector-backed knowledge retriever for RAG queries using Qdrant."""

    def __init__(self, store: Optional[QdrantKnowledgeStore] = None) -> None:
        self.store: QdrantKnowledgeStore = store or get_global_qdrant_store()

    @property
    def documents(self) -> List[Dict[str, Any]]:
        """Lists active knowledge documents from Qdrant.

        Administrative/inspection use only (e.g. CLI --list); never call on the query answer path.
        """
        return self.store.list_documents()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves top_k relevant documents for the given query using Qdrant vector search.

        Args:
            query: The user query string (e.g. 'What is the schedule for ComputeX?').
            top_k: Maximum number of relevant documents to return (defaults to RAG_TOP_K or 10).
            min_score: Minimum relevance score required (defaults to RAG_MIN_SCORE or 0.40,
                tuned for BAAI/bge-small-en-v1.5).

        Returns:
            List of matching document dicts with 'id', 'title', 'summary', 'content', and 'score'.

        Raises:
            RetrievalError: If Qdrant encounters a connection failure, timeout, or backend outage.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        # RAG_TOP_K defaults to 10 to support multi-part lists/aggregates (40 items/chunk)
        effective_top_k = top_k if top_k is not None else int(os.getenv("RAG_TOP_K", "10"))
        # 0.40 cutoff is tuned specifically for BAAI/bge-small-en-v1.5; retune if embedding model changes
        effective_min_score = min_score if min_score is not None else float(os.getenv("RAG_MIN_SCORE", "0.40"))

        try:
            return self.store.search_text(clean_query, top_k=effective_top_k, min_score=effective_min_score)
        except Exception as e:
            logger.error(f"Error retrieving knowledge from Qdrant for '{clean_query}': {e}", exc_info=True)
            raise RetrievalError(f"Qdrant retrieval error: {e}") from e

    def build_context_string(self, query: str, top_k: Optional[int] = None) -> str:
        """Retrieves and formats context with source file and page metadata into a clean text block."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""

        context_parts = []
        for r in results:
            meta = r.get("metadata") or {}
            source_file = meta.get("source") or meta.get("source_file") or meta.get("source_file_name")
            page = meta.get("page") if meta.get("page") is not None else meta.get("page_number")
            source_bits = []
            if source_file:
                source_bits.append(f"Source: {source_file}")
            if page is not None:
                source_bits.append(f"Page: {page}")
            src_str = f" [{', '.join(source_bits)}]" if source_bits else ""
            context_parts.append(f"[{r.get('title')}]{src_str}\n{r.get('content')}")
        return "\n\n".join(context_parts)
