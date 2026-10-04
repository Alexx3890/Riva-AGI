# RAG Knowledge Architecture

This document describes the internal design of the `rag_knowledge` subsystem.

```
                    +-----------------------+
                    |      User Query       |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    |    RAGService         |
                    | (service.py)          |
                    +-----------+-----------+
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
  +-----------------------+             +-----------------------+
  |  KnowledgeRetriever   |             |   MistralRAGClient    |
  |  (retriever.py)       |             |  (mistral_client.py)  |
  +-----------+-----------+             +-----------+-----------+
              |                                     |
              v                                     v
  +-----------------------+             +-----------------------+
  |  MongoDB Database     |             |   Mistral Chat API    |
  | (knowledge_documents) |             | (api.mistral.ai)      |
  +-----------------------+             +-----------------------+
```

## Components

### 1. `KnowledgeRetriever` (`retriever.py`)
- Database-backed search engine querying MongoDB collection `knowledge_documents`.
- Weighted full-text search indexing on `aliases` (10x), `title` (8x), `keywords` (5x), `summary` (3x), and `content` (1x).
- Sub-millisecond token & regex alias lookup fallback.
- Returns top-k matching documents ranked by relevance score.

### 2. `MistralRAGClient` (`mistral_client.py`)
- Communicates directly with `https://api.mistral.ai/v1/chat/completions`.
- Uses standard library `urllib.request` inside an async worker executor to ensure zero event loop blocking.
- Injects a voice-tuned prompt directing Mistral to answer within 2–3 spoken sentences.
- Fail-safe: if request fails or key is missing, returns `None` so the caller gracefully falls back.

### 3. `RAGService` (`service.py`)
- Coordinates the retrieval and generation workflow.
- Falls back gracefully to raw document content if Mistral is unconfigured or encounters an error.
