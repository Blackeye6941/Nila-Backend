# Document Embedding & Supabase Vector Ingestion Pipeline

This pipeline automatically extracts text from documents (`.pdf`, `.xlsx`, `.txt`), breaks them into semantic chunks with metadata, generates dense vector embeddings using Google Gemini (`gemini-embedding-001`), and stores them in Supabase using the `pgvector` extension.

---

## 📁 Pipeline Structure

```
pipeline/
├── __init__.py         # Package initialization
├── config.py           # Configuration & environment loader
├── loader.py           # Extracts text from PDFs (pypdf) and Excel (openpyxl)
├── chunker.py          # Splits text into overlapping chunks preserving metadata
├── embeddings.py       # Generates vector embeddings with Google Gemini
├── vector_store.py     # Supabase pgvector client (inserts, queries, clears)
├── ingest.py           # Orchestration CLI script to run the full pipeline
├── search.py           # CLI tool to test semantic similarity search
├── schema.sql          # SQL schema to run in Supabase SQL Editor
└── README.md           # Pipeline documentation
```

---

## 🚀 Setup & Execution Guide

### 1. Supabase Setup
1. Open your Supabase Dashboard project.
2. Go to **SQL Editor** -> **New query**.
3. Copy and paste the contents of `pipeline/schema.sql` and click **Run**.
   - Enables the `vector` extension.
   - Creates the `document_embeddings` table.
   - Creates the HNSW cosine similarity index.
   - Creates the `match_documents` RPC function.

### 2. Environment Configuration
Ensure your `.env` file in the project root includes:

```env
# Gemini API Key (already present)
GEMINI_API_KEY=your_gemini_api_key_here

# Supabase Credentials (from Project Settings -> API)
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your_supabase_service_role_key_or_anon_key

# Optional settings (defaults shown)
SUPABASE_TABLE=document_embeddings
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIM=768
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

> **Note:** Use your Supabase **service role key** if Row Level Security (RLS) is enabled and you need write permissions from the backend.

---

### 3. Run Ingestion

You can run the ingestion pipeline directly from the project root:

```bash
# Preview extracted chunks without calling APIs:
python -m pipeline.ingest --dry-run

# Run full ingestion (load -> chunk -> embed -> save to Supabase):
python -m pipeline.ingest

# Re-run and clear previous table data before inserting:
python -m pipeline.ingest --clear
```

---

### 4. Test Semantic Search

Test similarity retrieval against your indexed documents:

```bash
python -m pipeline.search "What are the licensing requirements for street food vendors?"
```

---

### 5. Integration in FastAPI (`app.py`)

You can easily query the vector store in your FastAPI routes:

```python
from pipeline.search import search_documents

@app.get("/search")
def search(q: str, limit: int = 5):
    results = search_documents(query=q, match_count=limit)
    return {"query": q, "results": results}
```
