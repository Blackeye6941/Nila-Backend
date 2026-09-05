import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm

from pipeline import config
from pipeline.loader import DocumentLoader
from pipeline.chunker import TextChunker
from pipeline.embeddings import EmbeddingGenerator
from pipeline.vector_store import SupabaseVectorStore


def run_pipeline(
    provider: str = None,
    model_name: str = None,
    clear_existing: bool = False,
    batch_size: int = None,
    delay: float = None,
    dry_run: bool = False
):
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model_name or config.EMBEDDING_MODEL
    batch_size = batch_size or (64 if provider == "local" else 25)
    batch_delay = delay if delay is not None else (0.0 if provider == "local" else config.GEMINI_BATCH_DELAY)

    print("=" * 65)
    print(" Starting Document Ingestion Pipeline")
    print("=" * 65)
    print(f"Embedding Provider  : {provider.upper()} {'(Open-Source, Local)' if provider == 'local' else '(Cloud API)'}")
    print(f"Model Name          : {model} (dim=768)")
    print(f"Batch Size          : {batch_size} chunks")
    print(f"Batch Delay         : {batch_delay}s {'(Zero delay needed)' if provider == 'local' else '(Rate-limit protection)'}")
    print(f"Supabase Table      : {config.SUPABASE_TABLE}")
    print("=" * 65)

    # 1. Load documents
    print("\n[1/3] Loading documents...")
    loader = DocumentLoader()
    raw_docs = loader.load_directory(config.DOCUMENTS_DIR)
    if not raw_docs:
        print("No documents found to process. Exiting.")
        return

    print(f"-> Extracted {len(raw_docs)} document sections/pages.")

    # 2. Chunk text
    print("\n[2/3] Chunking documents...")
    chunker = TextChunker(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunked_docs = chunker.split_documents(raw_docs)
    print(f"-> Produced {len(chunked_docs)} text chunks.")

    if dry_run:
        print("\n[DRY RUN] Skipping embedding generation and database storage.")
        return

    # Verify credentials
    if provider == "gemini" and not config.GEMINI_API_KEY:
        print("\nERROR: GEMINI_API_KEY is required when using provider='gemini'.")
        sys.exit(1)

    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        print("\nERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env to store embeddings.")
        sys.exit(1)

    # 3. Setup Supabase Vector Store
    vector_store = SupabaseVectorStore(
        url=config.SUPABASE_URL,
        key=config.SUPABASE_KEY,
        table_name=config.SUPABASE_TABLE
    )

    if clear_existing:
        print(f"\nClearing existing records in Supabase table '{config.SUPABASE_TABLE}'...")
        try:
            vector_store.clear()
            print("Table cleared.")
        except Exception as e:
            print(f"Note on table clear: {e}")

    # 4. Generate embeddings and upload to Supabase incrementally
    print(f"\n[3/3] Generating embeddings & saving incrementally to Supabase...")
    embedder = EmbeddingGenerator(
        provider=provider,
        api_key=config.GEMINI_API_KEY if provider == "gemini" else None,
        model_name=model,
        output_dim=768,
        batch_delay=batch_delay,
        max_retries=6
    )

    text_batches = [chunked_docs[i:i + batch_size] for i in range(0, len(chunked_docs), batch_size)]
    total_saved = 0

    with tqdm(total=len(chunked_docs), desc="Processing chunks") as pbar:
        for batch_num, batch in enumerate(text_batches, start=1):
            texts = [doc["content"] for doc in batch]

            # Generate embeddings for current batch
            embeddings = embedder.embed_batch(texts, batch_size=len(texts))

            records_to_insert = []
            for doc, emb in zip(batch, embeddings):
                records_to_insert.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "embedding": emb
                })

            # Save immediately to Supabase
            try:
                vector_store.insert_batch(records_to_insert)
                total_saved += len(records_to_insert)
            except Exception as e:
                print(f"\n[Warning] Supabase insert error on batch {batch_num}: {e}")

            pbar.update(len(batch))

            if batch_num < len(text_batches) and batch_delay > 0:
                time.sleep(batch_delay)

    print(f"\n-> Successfully ingested and stored {total_saved} embeddings in Supabase '{config.SUPABASE_TABLE}'!")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents, generate embeddings, and store in Supabase.")
    parser.add_argument("--provider", choices=["local", "gemini"], default=None, help="Embedding provider: 'local' (BAAI/bge-base-en-v1.5 768-dim) or 'gemini' (default: local).")
    parser.add_argument("--model", type=str, default=None, help="Model name (default: BAAI/bge-base-en-v1.5 for local).")
    parser.add_argument("--clear", action="store_true", help="Clear existing table records before inserting.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for embedding generation (default: 64 for local, 25 for gemini).")
    parser.add_argument("--delay", type=float, default=None, help="Sleep delay in seconds between batches (default: 0s for local, 5s for gemini).")
    parser.add_argument("--dry-run", action="store_true", help="Perform loading and chunking without saving.")

    args = parser.parse_args()
    run_pipeline(
        provider=args.provider,
        model_name=args.model,
        clear_existing=args.clear,
        batch_size=args.batch_size,
        delay=args.delay,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
