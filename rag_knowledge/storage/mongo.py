"""MongoDB Storage & Retrieval Backend for rag_knowledge.

Provides resilient connection pooling, text search indexing, and document
upserting with graceful timeout handling and certifi TLS integration.
"""

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rag_knowledge.storage.mongo")

# Optional dependencies
try:
    import certifi
    _HAS_CERTIFI = True
except ImportError:
    _HAS_CERTIFI = False

try:
    import pymongo
    from pymongo import MongoClient
    _HAS_PYMONGO = True
except ImportError:
    _HAS_PYMONGO = False


class MongoKnowledgeStore:
    """Manages MongoDB connection, full-text search, and document storage for RAG."""

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        timeout_ms: int = 1500,
    ) -> None:
        self.uri = uri if uri is not None else os.getenv("MONGODB_URI", "").strip()
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", "riva_knowledge").strip() or "riva_knowledge"
        self.collection_name = collection_name or os.getenv("MONGODB_COLLECTION", "knowledge_documents").strip() or "knowledge_documents"
        self.timeout_ms = timeout_ms

        self._client: Optional[Any] = None
        self._db: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._is_connected: Optional[bool] = None
        self._last_fail_time: float = 0.0
        self._fail_cooldown_seconds: float = 60.0

    def connect(self) -> bool:
        """Attempts to initialize client and verify server connectivity."""
        if not _HAS_PYMONGO:
            logger.debug("pymongo is not installed; MongoDB storage is unavailable.")
            self._is_connected = False
            return False

        if not self.uri:
            logger.debug("MONGODB_URI is not set; skipping MongoDB storage initialization.")
            self._is_connected = False
            return False

        # If connection failed recently, avoid blocking in cooldown window
        if self._is_connected is False and (time.time() - self._last_fail_time < self._fail_cooldown_seconds):
            return False

        try:
            client_kwargs: Dict[str, Any] = {
                "serverSelectionTimeoutMS": self.timeout_ms,
                "connectTimeoutMS": self.timeout_ms,
                "socketTimeoutMS": self.timeout_ms,
            }
            if _HAS_CERTIFI:
                client_kwargs["tlsCAFile"] = certifi.where()

            self._client = MongoClient(self.uri, **client_kwargs)
            # Ping database to confirm connection
            self._client.admin.command("ping")
            self._db = self._client[self.db_name]
            self._collection = self._db[self.collection_name]
            self._is_connected = True
            logger.info("Connected to MongoDB successfully (db: %s, col: %s)", self.db_name, self.collection_name)
            return True
        except Exception as e:
            logger.warning("MongoDB connection failed (%s: %s). Falling back to in-memory store.", type(e).__name__, e)
            self._is_connected = False
            self._last_fail_time = time.time()
            if self._client:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None
            return False

    def is_available(self) -> bool:
        """Returns True if MongoDB is configured and responsive."""
        if self._is_connected is None:
            return self.connect()
        return self._is_connected

    def ensure_indexes(self) -> None:
        """Creates unique ID and weighted text indexes for optimal query ranking."""
        if not self.is_available() or self._collection is None:
            return

        try:
            # 1. Unique index on document ID
            self._collection.create_index([("id", pymongo.ASCENDING)], unique=True)

            # 2. Full-text search index with weighted fields
            text_weights = {
                "aliases": 10,
                "title": 8,
                "keywords": 5,
                "summary": 3,
                "content": 1,
            }
            self._collection.create_index(
                [
                    ("title", pymongo.TEXT),
                    ("aliases", pymongo.TEXT),
                    ("keywords", pymongo.TEXT),
                    ("summary", pymongo.TEXT),
                    ("content", pymongo.TEXT),
                ],
                name="knowledge_text_index",
                weights=text_weights,
                default_language="english",
            )
            logger.info("MongoDB indexes verified on '%s'", self.collection_name)
        except Exception as e:
            logger.warning("Error creating MongoDB indexes: %s", e)

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Upserts a list of knowledge documents into the collection."""
        if not self.is_available() or self._collection is None:
            raise RuntimeError("MongoDB is not available for upsert operation.")

        count = 0
        for doc in documents:
            doc_id = doc.get("id")
            if not doc_id:
                continue

            payload = dict(doc)
            # Ensure _id matches id for easy querying
            payload["_id"] = doc_id
            payload.setdefault("is_active", True)

            self._collection.update_one(
                {"_id": doc_id},
                {"$set": payload},
                upsert=True,
            )
            count += 1

        logger.info("Upserted %d document(s) into MongoDB collection '%s'", count, self.collection_name)
        return count

    def search_text(self, query: str, top_k: int = 3, min_score: float = 1.0) -> List[Dict[str, Any]]:
        """Searches MongoDB using text score ranking, falling back to regex alias/keyword match."""
        if not self.is_available() or self._collection is None:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        results: List[Dict[str, Any]] = []
        seen_ids = set()

        # 1. Primary Text Search using textScore
        try:
            cursor = self._collection.find(
                {"$text": {"$search": clean_query}, "is_active": {"$ne": False}},
                {"score": {"$meta": "textScore"}},
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k)

            for doc in cursor:
                doc_id = doc.get("id") or str(doc.get("_id"))
                score = float(doc.get("score", 0.0))
                if score >= min_score and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    results.append(self._format_doc(doc, score))

            if results:
                return results[:top_k]
        except Exception as text_err:
            logger.debug("MongoDB text search error (%s), attempting token fallback", text_err)

        # 2. Token-level Regex / Alias fallback (for exact name or token lookups)
        tokens = [re.escape(t.lower()) for t in re.findall(r"\w+", clean_query) if len(t) > 2]
        if tokens:
            try:
                regex_pattern = "|".join(tokens)
                cursor = self._collection.find(
                    {
                        "$or": [
                            {"aliases": {"$regex": regex_pattern, "$options": "i"}},
                            {"keywords": {"$regex": regex_pattern, "$options": "i"}},
                            {"title": {"$regex": regex_pattern, "$options": "i"}},
                        ],
                        "is_active": {"$ne": False},
                    }
                ).limit(top_k)

                for doc in cursor:
                    doc_id = doc.get("id") or str(doc.get("_id"))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        results.append(self._format_doc(doc, score=3.0))

            except Exception as regex_err:
                logger.warning("MongoDB token fallback search failed: %s", regex_err)

        return results[:top_k]

    def _format_doc(self, doc: Dict[str, Any], score: float) -> Dict[str, Any]:
        """Normalizes document structure to match KnowledgeRetriever standard."""
        return {
            "id": doc.get("id") or str(doc.get("_id")),
            "title": doc.get("title", ""),
            "summary": doc.get("summary", ""),
            "content": doc.get("content", ""),
            "score": round(score, 2),
        }

    def list_documents(self, filter_query: Optional[Dict[str, Any]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Lists documents from the MongoDB knowledge collection."""
        if not self.is_available() or self._collection is None:
            return []
        query = filter_query or {"is_active": {"$ne": False}}
        try:
            cursor = self._collection.find(query, {"_id": 0}).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.warning("Error listing MongoDB documents: %s", e)
            return []

    def close(self) -> None:
        """Closes the active MongoDB client connection."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._is_connected = False


_GLOBAL_STORE: Optional[MongoKnowledgeStore] = None


def get_global_mongo_store() -> MongoKnowledgeStore:
    """Returns a shared MongoKnowledgeStore instance with pooled connection."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = MongoKnowledgeStore()
    return _GLOBAL_STORE
