"""Hermetic test suite for PR 1a core RAG architecture."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from rag_knowledge.clients.gemini_client import GeminiRAGClient
from rag_knowledge.ingestion.privacy import assert_no_privacy_leaks
from rag_knowledge.retrieval.retriever import KnowledgeRetriever
from rag_knowledge.service import RAGService
from rag_knowledge.storage.qdrant_storage import QdrantKnowledgeStore, string_to_point_id


def test_qdrant_storage_and_retriever():
    assert string_to_point_id("doc_1") == string_to_point_id("doc_1")

    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_hit = MagicMock(
        id="uuid_1",
        score=0.9,
        payload={"id": "doc_1", "title": "Test Title", "content": "Test Content", "metadata": {}},
    )
    mock_client.query_points.return_value = MagicMock(points=[mock_hit])
    mock_client.scroll.return_value = ([], None)

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", return_value=mock_client), \
         patch("rag_knowledge.storage.qdrant_storage.TextEmbedding") as mock_embed_cls:
        mock_model = MagicMock()
        mock_model.embed.return_value = [[0.1] * 384]
        mock_embed_cls.return_value = mock_model

        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake_key")
        assert store.connect()
        assert store.upsert_documents([{"id": "doc_1", "title": "Test"}]) == 1

        retriever = KnowledgeRetriever(store=store)
        results = retriever.retrieve("query")
        assert len(results) == 1
        assert results[0]["id"] == "doc_1"


@pytest.mark.anyio
async def test_rag_service_fallback_and_gemini():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"id": "doc_1", "title": "Doc 1", "summary": "Summary text", "content": "Content text", "score": 90.0}
    ]

    service_fallback = RAGService(retriever=mock_retriever, llm_client=GeminiRAGClient(api_key=""))
    res = await service_fallback.query("Test question")
    assert "Summary text" in res

    mock_llm = MagicMock(spec=GeminiRAGClient)
    mock_llm.is_configured = True
    mock_llm.generate_answer = AsyncMock(return_value="AI answer")
    service_llm = RAGService(retriever=mock_retriever, llm_client=mock_llm)
    assert await service_llm.query("Test question") == "AI answer"


def test_ingest_privacy_guards():
    docs = [{"id": "user_1", "title": "User One", "summary": "User One", "content": "User One is registered."}]
    assert_no_privacy_leaks(docs)
    assert "@" not in docs[0]["content"]

    leaky_doc = [{"id": "bad", "content": "user@test.local", "summary": "", "title": ""}]
    with pytest.raises(ValueError):
        assert_no_privacy_leaks(leaky_doc)


def test_pdf_and_word_dispatch_ingestion(tmp_path):
    import docx
    from rag_knowledge.ingestion import (
        load_pdf_documents,
        load_word_documents,
        load_source_documents,
    )

    doc_file = tmp_path / "handbook.docx"
    doc = docx.Document()
    doc.add_heading("Society Rules", level=1)
    doc.add_paragraph("Welcome to the handbook.")
    doc.save(str(doc_file))

    assert len(load_word_documents(doc_file)) >= 1

    docs = load_source_documents(tmp_path)
    assert len(docs) >= 1



def test_gemini_client_configuration_checks():
    """Test that is_configured requires BOTH api_key and model."""
    # Empty key, model present -> not configured
    c1 = GeminiRAGClient(api_key="", model="gemini-flash")
    assert not c1.is_configured

    # Key present, model empty -> not configured
    c2 = GeminiRAGClient(api_key="valid_key", model="")
    assert not c2.is_configured

    # Both present -> configured
    c3 = GeminiRAGClient(api_key="valid_key", model="gemini-flash")
    assert c3.is_configured


def test_build_context_string_includes_source_and_page():
    """Test build_context_string incorporates source file and page metadata."""
    mock_store = MagicMock()
    mock_store.search_text.return_value = [
        {
            "id": "doc_1",
            "title": "Club Handbook",
            "content": "Rules and guidelines.",
            "score": 0.85,
            "metadata": {"source": "handbook.md", "page": 1},
        }
    ]
    retriever = KnowledgeRetriever(store=mock_store)
    ctx = retriever.build_context_string("rules")
    assert "[Club Handbook]" in ctx
    assert "Source: handbook.md" in ctx
    assert "Rules and guidelines." in ctx


def test_retriever_raises_retrieval_error_on_storage_failure():
    """Test retriever raises RetrievalError on storage exception rather than returning empty list."""
    from rag_knowledge.retrieval.retriever import RetrievalError
    mock_store = MagicMock()
    mock_store.search_text.side_effect = RuntimeError("Qdrant cluster connection timed out")
    retriever = KnowledgeRetriever(store=mock_store)

    with pytest.raises(RetrievalError) as exc_info:
        retriever.retrieve("failing query")
    assert "Qdrant retrieval error" in str(exc_info.value)


@pytest.mark.anyio
async def test_service_query_with_sources_and_outage_handling():
    """Test query_with_sources returns sources and query handles outages."""
    from rag_knowledge.retrieval.retriever import RetrievalError

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "id": "ev_1",
            "title": "Orientation 2026",
            "summary": "Orientation is on 25 Oct.",
            "content": "Orientation 2026 will start at 10am.",
            "score": 0.9,
            "metadata": {"source": "orientation.md", "page": 1},
        }
    ]
    service = RAGService(retriever=mock_retriever, llm_client=GeminiRAGClient(api_key="", model=""))
    answer, sources = await service.query_with_sources("When is orientation?")
    assert "Orientation is on 25 Oct." in answer
    assert sources == ["orientation.md (Page 1)"]

    mock_retriever.retrieve.side_effect = RetrievalError("Qdrant offline")
    with pytest.raises(RetrievalError):
        await service.query_with_sources("test")

    outage_answer = await service.query("test")
    assert "currently unavailable due to an outage" in outage_answer


def test_max_tokens_warning_emitted():
    """Test that finish_reason == 'MAX_TOKENS' emits a warning."""
    client = GeminiRAGClient(api_key="fake", model="gemini-flash")
    fake_response_data = b'{"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "Cut off answer"}]}}]}'

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = fake_response_data
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.warns(UserWarning, match="MAX_TOKENS"):
            import asyncio
            ans = asyncio.run(client.generate_answer("query", "context"))
            assert ans == "Cut off answer"
