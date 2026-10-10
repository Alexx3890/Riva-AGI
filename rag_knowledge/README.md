# RAG Knowledge Subsystem (`rag_knowledge`)

Voice-optimized Retrieval-Augmented Generation (RAG) service powered by **Qdrant Vector Database** and **Google Gemini**.

---

## 1. Quick Start (PR 1a: Text & JSON RAG)

### Installation
Install dependencies:
```bash
pip install -r rag_knowledge/requirements.txt
```

### Configuration
Copy `.env.example` to `.env` in your root or package directory:
```bash
cp rag_knowledge/.env.example rag_knowledge/.env
```

Configure your environment variables in `.env`:
```ini
# Google Gemini (for conversational answer synthesis)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest

# Qdrant Vector Database
QDRANT_URL=https://your-cluster-id.us-east-1-1.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION=riva_knowledge
```

> [!NOTE]
> **Data Protection & Privacy Notice**:
> During query synthesis, the question and the retrieved document snippets are sent to Google Gemini for answer generation. If `GEMINI_API_KEY` is omitted, the system operates hermetically, returning retrieved factual knowledge directly without contacting Google.
> Private contact data (emails, personal phone numbers) is **never** embedded or sent to LLMs; it is stripped by the fail-closed privacy gate.

---

## 2. Ingestion (Data Processing & Indexing)

Ingest supported text documents into the vector database using `rag_knowledge.ingestion`:

```bash
# Ingest markdown and JSON files with automated privacy redaction
python -m rag_knowledge.ingestion --source "path/to/documents/" --redact

# Dry-run: preview extracted chunks without modifying vector storage
python -m rag_knowledge.ingestion --source "path/to/documents/" --dry-run
```

### Supported Formats in PR 1a
- **Documents (`.txt`, `.md`, `.json`)**: Structured text sections, headings, FAQs, and lists.
- *PDF and Word (`.pdf`, `.docx`) are added in PR 1b.*
- *Spreadsheets (`.xlsx`, `.csv`, `.tsv`) are added in PR 1c.*
- *Images (`.png`, `.jpg`) with OCR are added in PR 3.*

---

## 3. Asking Questions

Ask questions against the knowledge base from the terminal:

```bash
# Ask a question (displays matches, answer, and sources cited)
python -m rag_knowledge ask "When is the orientation session scheduled?"

# Direct question invocation
python -m rag_knowledge "What is the policy for hackathon registration?"

# Quiet mode (answer and sources only)
python -m rag_knowledge "What is the policy for hackathon registration?" -q
```

---

## 4. Running Tests

Run the hermetic test suite with pytest:
```bash
pytest rag_knowledge/tests
```
