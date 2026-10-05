"""Unit tests for MongoDB storage backend and retriever."""

from unittest.mock import MagicMock, patch
import pytest

from rag_knowledge.storage.mongo import MongoKnowledgeStore
from rag_knowledge.retriever import KnowledgeRetriever


def test_mongo_store_not_configured():
    """MongoKnowledgeStore returns is_available() False when URI is unset."""
    store = MongoKnowledgeStore(uri="")
    assert store.is_available() is False
    assert store.search_text("chirag") == []
    assert store.list_documents() == []


def test_mongo_store_mocked_success():
    """Verifies index creation, upsert, list_documents, and text search when MongoDB is connected."""
    mock_collection = MagicMock()
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    mock_client = MagicMock()
    mock_client.__getitem__.return_value = mock_db
    mock_client.admin.command.return_value = {"ok": 1}

    # Simulate search results
    sample_doc = {
        "_id": "test_member",
        "id": "test_member",
        "title": "Test Member - AI Specialist",
        "summary": "Specialist in machine learning",
        "content": "TYPE: PERSON\nName: Test Member",
        "score": 4.5,
    }
    mock_cursor = MagicMock()
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = [sample_doc]
    mock_collection.find.return_value = mock_cursor

    with patch("rag_knowledge.storage.mongo.MongoClient", return_value=mock_client):
        store = MongoKnowledgeStore(uri="mongodb://localhost:27017")
        assert store.connect() is True
        assert store.is_available() is True

        store.ensure_indexes()
        assert mock_collection.create_index.call_count >= 2

        count = store.upsert_documents([{"id": "test_member", "title": "Test Member"}])
        assert count == 1
        assert mock_collection.update_one.called

        # Test search
        results = store.search_text("machine learning")
        assert len(results) == 1
        assert results[0]["id"] == "test_member"
        assert results[0]["score"] == 4.5

        # Test list_documents
        docs = store.list_documents()
        assert len(docs) == 1

        store.close()
        assert store.is_available() is False


def test_retriever_returns_empty_when_no_match():
    """KnowledgeRetriever returns empty list when query produces no match."""
    mock_store = MagicMock()
    mock_store.search_text.return_value = []

    retriever = KnowledgeRetriever(store=mock_store)
    results = retriever.retrieve("unknown query")
    assert results == []


def test_retriever_uses_mongo_when_available():
    """KnowledgeRetriever returns MongoDB results when available."""
    fake_mongo_result = [
        {
            "id": "custom_mongo_id",
            "title": "Mongo Custom Title",
            "summary": "From MongoDB cluster",
            "content": "Content from MongoDB",
            "score": 5.0,
        }
    ]

    mock_store = MagicMock()
    mock_store.is_available.return_value = True
    mock_store.search_text.return_value = fake_mongo_result

    retriever = KnowledgeRetriever(store=mock_store)
    results = retriever.retrieve("query")
    assert len(results) == 1
    assert results[0]["title"] == "Mongo Custom Title"
