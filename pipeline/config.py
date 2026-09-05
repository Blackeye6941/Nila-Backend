import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Supabase Settings
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "document_embeddings")

# Embedding Settings
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local").lower() # 'local' (open-source) or 'gemini'
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5" if EMBEDDING_PROVIDER == "local" else "gemini-embedding-001")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BATCH_DELAY = float(os.getenv("GEMINI_BATCH_DELAY", "5.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

# Chunking Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Documents Directory
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(ROOT_DIR / "documents")))
