cat << 'EOF' > README.md
# AskMyDocs - Mini RAG (Retrieval-Augmented Generation) API

An end-to-end Document Question-Answering (RAG) service built from scratch using **Python (FastAPI)**, **PostgreSQL + pgvector (HNSW Cosine Index)**, and **OpenAI (`text-embedding-3-small` & `gpt-4o-mini`)**.

---

## Project Directory Structure

AskMyDocs/
├── docker-compose.yml            # pgvector/pgvector:pg16 service
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
├── sample_company_policy.txt     # Sample document for testing
└── app/
    ├── __init__.py
    ├── main.py                   # FastAPI application entrypoint & lifespan DB initialization
    ├── core/
    │   ├── __init__.py
    │   ├── config.py             # Environment variables & RAG hyperparameters
    │   └── database.py           # SQLAlchemy engine, get_db, and HNSW index setup
    ├── models/
    │   ├── __init__.py
    │   ├── domain.py             # Document & DocumentChunk SQLAlchemy models
    │   └── schemas.py            # Pydantic Request/Response schemas
    ├── services/
    │   ├── __init__.py
    │   ├── parser_service.py     # PDF (pypdf) & TXT text extraction and cleaning
    │   ├── chunker_service.py    # Recursive Character Text Splitter (500/50)
    │   ├── embedding_service.py  # OpenAI text-embedding-3-small (1536d) service
    │   └── rag_service.py        # pgvector (<=>) search, context assembly, and gpt-4o-mini
    └── api/
        ├── __init__.py
        ├── documents.py          # POST /api/documents/upload, GET & DELETE /api/documents
        └── rag.py                # POST /api/rag/ask

---

## Step-by-Step Getting Started Guide

### 1. Set Up Environment Variables
cp .env.example .env

Open .env and provide your own OPENAI_API_KEY.

### 2. Start PostgreSQL + pgvector Database
docker compose up -d

### 3. Virtual Environment Setup & Dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

### 4. Start the FastAPI Server
When the application starts up for the first time, CREATE EXTENSION IF NOT EXISTS vector;, the documents / document_chunks tables, and the HNSW (vector_cosine_ops) index are automatically initialized:

.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

- Swagger UI: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## Example curl Requests

### 1. Upload Document (POST /api/documents/upload)
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_company_policy.txt"

### 2. Query Across All Documents (POST /api/rag/ask)
curl -X POST "http://localhost:8000/api/rag/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the return period for a software license and what are the refund conditions?",
    "top_k": 3,
    "similarity_threshold": 0.30
  }'

### 3. Scoped Query Filtered by Document (document_id Filter)
curl -X POST "http://localhost:8000/api/rag/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the annual education budget?",
    "document_id": "INSERT_DOCUMENT_UUID_HERE",
    "top_k": 5
  }'

### 4. Query Not Present in Context (Hallucination Prevention Test)
curl -X POST "http://localhost:8000/api/rag/ask" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the company meal card daily allowance limit?"
  }'
EOF
