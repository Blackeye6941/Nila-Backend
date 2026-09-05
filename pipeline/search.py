import argparse
import sys
from pipeline import config
from pipeline.embeddings import EmbeddingGenerator
from pipeline.vector_store import SupabaseVectorStore


def search_documents(query: str, match_count: int = 5, provider: str = None, model: str = None):
    provider = (provider or config.EMBEDDING_PROVIDER).lower()
    model = model or config.EMBEDDING_MODEL

    if provider == "gemini" and not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set.")
        sys.exit(1)

    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        print("ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env.")
        sys.exit(1)

    embedder = EmbeddingGenerator(
        provider=provider,
        api_key=config.GEMINI_API_KEY if provider == "gemini" else None,
        model_name=model,
        output_dim=768
    )
    query_vector = embedder.embed_text(query)

    vector_store = SupabaseVectorStore(
        url=config.SUPABASE_URL,
        key=config.SUPABASE_KEY,
        table_name=config.SUPABASE_TABLE
    )

    results = vector_store.search(query_embedding=query_vector, match_count=match_count)
    return results


def main():
    parser = argparse.ArgumentParser(description="Query Supabase vector embeddings.")
    parser.add_argument("query", type=str, help="Search query string")
    parser.add_argument("--count", type=int, default=5, help="Number of matching results to retrieve (default: 5)")
    parser.add_argument("--provider", choices=["local", "gemini"], default=None, help="Embedding provider (default: local)")
    parser.add_argument("--model", type=str, default=None, help="Embedding model name")
    args = parser.parse_args()

    print(f"\nSearching for: '{args.query}' (top {args.count} matches)...\n")
    results = search_documents(args.query, match_count=args.count, provider=args.provider, model=args.model)

    if not results:
        print("No matching documents found.")
        return

    for idx, r in enumerate(results, start=1):
        similarity = r.get("similarity", 0)
        source = r.get("metadata", {}).get("source", "Unknown")
        page = r.get("metadata", {}).get("page", "-")
        sheet = r.get("metadata", {}).get("sheet", "-")
        content = r.get("content", "")

        print(f"[{idx}] Similarity: {similarity:.4f} | Source: {source} (Page: {page}, Sheet: {sheet})")
        print(f"     Content: {content[:250]}...")
        print("-" * 60)


if __name__ == "__main__":
    main()
