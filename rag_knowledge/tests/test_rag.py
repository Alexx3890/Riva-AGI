"""Concise unit & integration test suite for rag_knowledge."""

from unittest.mock import AsyncMock, MagicMock, patch
import docx
import pytest

from rag_knowledge.clients.gemini_client import GeminiRAGClient
from rag_knowledge.ingestion import (
    load_pdf_documents,
    load_word_documents,
    load_source_documents,
    assert_no_privacy_leaks,
)
from rag_knowledge.ingestion.privacy import PrivacyGateError
from rag_knowledge.retrieval.retriever import KnowledgeRetriever, RetrievalError
from rag_knowledge.service import RAGService
from rag_knowledge.storage.qdrant_storage import QdrantKnowledgeStore, string_to_point_id


def test_qdrant_store_and_retriever():
    assert string_to_point_id("doc_1") == string_to_point_id("doc_1")
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_hit = MagicMock(id="u1", score=0.9, payload={"id": "d1", "title": "T", "content": "C", "metadata": {}})
    mock_client.query_points.return_value = MagicMock(points=[mock_hit])
    mock_client.scroll.return_value = ([], None)

    with patch("rag_knowledge.storage.qdrant_storage.QdrantClient", return_value=mock_client), \
         patch("rag_knowledge.storage.qdrant_storage.TextEmbedding") as mock_emb:
        mock_emb.return_value.embed.return_value = [[0.1] * 384]
        store = QdrantKnowledgeStore(url="https://fake.qdrant.io", api_key="fake")
        assert store.connect()
        assert store.upsert_documents([{"id": "d1", "title": "T"}]) == 1

        retriever = KnowledgeRetriever(store=store)
        res = retriever.retrieve("query")
        assert len(res) == 1 and res[0]["id"] == "d1"

        mock_store = MagicMock()
        mock_store.search_text.side_effect = RuntimeError("Offline")
        with pytest.raises(RetrievalError):
            KnowledgeRetriever(store=mock_store).retrieve("failing")


def test_privacy_gate_blocking_and_redaction():
    clean = [{"id": "1", "title": "T", "content": "Clean text without contacts."}]
    assert_no_privacy_leaks(clean)

    leaky = [{"id": "2", "title": "T", "content": "Reach at user@test.local or 9876543210"}]
    with pytest.raises(PrivacyGateError):
        assert_no_privacy_leaks(leaky)

    assert_no_privacy_leaks(leaky, opt_in_redact=True)
    assert "[REDACTED_EMAIL]" in leaky[0]["content"] and "user@test.local" not in leaky[0]["content"]


def test_pdf_and_word_document_ingestion(tmp_path):
    # Word doc
    docx_file = tmp_path / "guide.docx"
    doc = docx.Document()
    doc.add_heading("Club Handbook", level=1)
    doc.add_paragraph("Orientation is on 25 October 2026.")
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Role", "Name"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "Lead", "Alex"
    doc.save(str(docx_file))

    word_docs = load_word_documents(docx_file)
    assert len(word_docs) >= 2
    assert any("Club Handbook" in d["title"] for d in word_docs)
    assert any("Table" in d["title"] for d in word_docs)

    # Dispatch directory loading
    all_docs = load_source_documents(tmp_path)
    assert len(all_docs) >= 2


@pytest.mark.anyio
async def test_gemini_client_and_rag_service():
    c = GeminiRAGClient(api_key="", model="gemini-flash")
    assert not c.is_configured
    c.api_key = "valid_key"
    assert c.is_configured

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"id": "d1", "title": "Doc", "summary": "Orientation is on 25 Oct.", "content": "Content", "score": 0.9, "metadata": {"source": "guide.pdf", "page": 2}}
    ]

    mock_llm = MagicMock(spec=GeminiRAGClient)
    mock_llm.is_configured = True
    mock_llm.generate_answer = AsyncMock(return_value="Answer from Gemini.")

    service = RAGService(retriever=mock_retriever, llm_client=mock_llm)
    ans, sources = await service.query_with_sources("When is orientation?")
    assert ans == "Answer from Gemini."
    assert sources == ["guide.pdf (Page 2)"]

    mock_retriever.retrieve.side_effect = RetrievalError("Qdrant offline")
    outage_resp = await service.query("query")
    assert "unavailable due to an outage" in outage_resp


def test_max_tokens_warning():
    client = GeminiRAGClient(api_key="k", model="m")
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b'{"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "Cut"}]}}]}'
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp), pytest.warns(UserWarning, match="MAX_TOKENS"):
        import asyncio
        assert asyncio.run(client.generate_answer("q", "ctx")) == "Cut"
