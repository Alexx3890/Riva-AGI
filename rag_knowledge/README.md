# RAG Knowledge Subsystem (`rag_knowledge`)

A modular Retrieval-Augmented Generation (RAG) knowledge subsystem powered by **Qdrant Vector Database** and **Google Gemini**.

---

## 1. Setup

### Installation
Install the package requirements:
```bash
pip install -r rag_knowledge/requirements.txt
```

### Configuration
Copy `.env.example` to `.env` in the project root or package directory:
```bash
cp rag_knowledge/.env.example rag_knowledge/.env
```

Configure your environment variables in `.env`:
```ini
# Google Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest

# Qdrant Vector Database
QDRANT_URL=https://your-cluster-id.us-east-1-1.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION=riva_knowledge
QDRANT_TIMEOUT=60.0
QDRANT_BATCH_SIZE=50
```

---

## 2. Ingestion (Document Processing & Indexing)

Ingest document files or entire folders into the vector database using `rag_knowledge.ingestion`:

```bash
# Ingest documents with automated privacy redaction
python -m rag_knowledge.ingestion --source "rag_knowledge/data/raw/handbook.pdf" --redact

# Dry run: parse and inspect documents without uploading to Qdrant
python -m rag_knowledge.ingestion --source "rag_knowledge/data/raw/" --dry-run
```

### CLI Ingestion Options

| Flag | Description |
| :--- | :--- |
| `--source <path>` | Path to a single file or directory of documents to ingest |
| `--type <format>` | File format (`auto`, `pdf`, `word`, `docx`) |
| `--redact` | Automatically redacts detected personal emails and phone numbers (fail-closed privacy gate) |
| `--batch-size <N>` | Points per upsert batch (default: `50`, with automatic sub-chunking on timeout) |
| `--limit <N>` | Ingest only the first `N` extracted documents |
| `--dry-run` | Extract, parse, and preview documents without modifying vector storage |

### Supported Data Formats

- **PDF Documents (`.pdf`)**: Page-aware extraction, table row parsing, and schedule/event detection.
- **Word Documents (`.docx`)**: Hierarchical section and table parsing based on headings ($H1$-$H3$).

---

## 3. Querying

### Terminal CLI
```bash
# Query with ranked retrieval matches displayed
python -m rag_knowledge "What activities are scheduled for orientation?"

# Query in quiet mode (returns answer only)
python -m rag_knowledge "What is the policy for club registration?" -q

# List indexed documents
python -m rag_knowledge --list
```

### Python API
```python
import asyncio
from rag_knowledge import query_rag

async def main():
    response = await query_rag("When is the orientation session?")
    print(response)

asyncio.run(main())
```

---

## 4. Architecture & Security Highlights

- **Privacy Gate**: Validates all documents prior to indexing against personal contact details (emails, phone numbers). Direct unredacted ingestion of PII is blocked unless `--redact` is passed.
- **Hybrid Vector Retrieval**: Combines dense ANN cosine similarity search with exact identifier and title keyword boosting for high precision.
- **Grounded Answer Synthesis**: Google Gemini answers user questions grounded in retrieved facts, returning explicit source file and page citations.
- **WAN Resilience**: Adaptive upsert retry logic with automated half-batch sub-chunking prevents socket timeouts against cloud vector instances.

---

## 5. Running Tests

Run the full test suite with pytest:

```bash
pytest rag_knowledge/tests
```
