# RAG Knowledge Subsystem (`rag_knowledge`)

A modular, database-backed, self-contained Retrieval-Augmented Generation (RAG) package. Designed to provide fast factual knowledge retrieval and Google Gemini synthesis for conversational agents, voice assistants, and multi-agent platforms.

---

## Highlights

- **Completely Self-Contained**: Can be run, tested, and imported independently or merged with other systems without impacting other directories.
- **Database-Backed RAG Storage**: Powered by MongoDB (`riva_knowledge.knowledge_documents`) with weighted full-text search and alias matching.
- **Dynamic Live Updates**: Add, modify, or delete knowledge documents directly in MongoDB without server restarts or redeployments.
- **Auto `.env` Discovery**: Automatically detects and loads `.env` from package or workspace roots on import.
- **Google Gemini Synthesis**: Uses Gemini Flash (`gemini-flash-latest` by default) via standard Python `urllib` with strict grounding prompts to generate concise, 2–3 sentence spoken answers.
- **Resilient Fallback**: If `GEMINI_API_KEY` is omitted, rate-limited, or unavailable, it immediately returns the factual grounded text directly so conversational pipelines never fail.

---

## Directory Structure

```text
rag_knowledge/
├── __init__.py                # Package exports (query_rag, KnowledgeRetriever, GeminiRAGClient, RAGService)
├── __main__.py                # Package entrypoint (python -m rag_knowledge)
├── cli.py                     # Command-line query tool & inspector
├── gemini_client.py           # Gemini API synthesis client with standard urllib
├── retriever.py               # Database-backed search retriever over MongoDB
├── service.py                 # RAG orchestrator coordinating retrieval & generation
├── storage/                   # Database storage layer
│   ├── __init__.py
│   └── mongo.py               # MongoDB connection pooling & full-text indexing
├── requirements.txt           # Package requirements
├── README.md                  # Main documentation
├── docs/
│   ├── architecture.md        # Technical architecture details
│   └── integration_guide.md   # Guide on integrating into any system / voice agent
└── tests/
    ├── __init__.py
    ├── test_retriever.py      # Unit tests for retriever
    ├── test_mongo_storage.py  # Unit tests for MongoDB storage layer
    ├── test_gemini_client.py  # Unit tests for Gemini API client and fallbacks
    └── test_service.py        # Unit tests for RAG service coordination
```

---

## Quick Start

### 1. Standalone CLI Usage

Query the knowledge base directly from your terminal:

```bash
# Ask a question
python -m rag_knowledge "Do you know about Ankit tomar?"

# List all stored knowledge documents
python -m rag_knowledge --list

# Query with retrieval match scores
python -m rag_knowledge "Who is Ankit tomar?" -v
```
  
### 2. Python API Usage

Import into any Python application:

```python
import asyncio
from rag_knowledge import query_rag

async def main():
    # Asynchronously query the RAG service
    answer = await query_rag("Who is Ankit tomar?")
    print("Answer:", answer)

asyncio.run(main())
```

---

## Configuration & Environment Variables

Add these to your `.env` file or environment:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MONGODB_URI` | *(None)* | MongoDB Atlas or local connection string (e.g. `mongodb+srv://...`). |
| `MONGODB_DB_NAME` | `riva_knowledge` | Target database name. |
| `MONGODB_COLLECTION` | `knowledge_documents` | Target collection name. |
| `GEMINI_API_KEY` | *(None)* | Your Google Gemini API key (from Google AI Studio). If unset, fallback mode returns raw structured facts. |
| `GEMINI_MODEL` | `gemini-flash-latest` | Gemini model identifier to use for response synthesis (e.g., `gemini-flash-latest`, `gemini-1.5-flash`). |

---

## Adding or Modifying Knowledge

Documents are stored in MongoDB collection `knowledge_documents`. Each document follows this format:

```json
{
  "_id": "person_or_topic_id",
  "id": "person_or_topic_id",
  "title": "Full Name / Title",
  "aliases": ["alias 1", "alias 2"],
  "keywords": ["keyword1", "keyword2"],
  "summary": "Short one-sentence summary.",
  "content": "Detailed facts and background information to be used as context.",
  "is_active": true
}
```

New or modified documents in MongoDB are immediately queryable without restarting applications.

---

## Running Tests

The test suite is fully isolated under `rag_knowledge/tests/`:

```bash
python -m pytest rag_knowledge/tests/ -v
```
