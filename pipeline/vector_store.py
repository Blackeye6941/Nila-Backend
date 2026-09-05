from typing import List, Dict, Any, Optional
from supabase import create_client, Client


class SupabaseVectorStore:
    """Manages document vector storage and similarity retrieval in Supabase."""

    def __init__(self, url: str, key: str, table_name: str = "document_embeddings"):
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be provided")
        self.client: Client = create_client(url, key)
        self.table_name = table_name

    def insert_batch(self, records: List[Dict[str, Any]], batch_size: int = 50) -> int:
        """Batch inserts document embeddings and metadata into Supabase."""
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            payload = [
                {
                    "content": r["content"],
                    "metadata": r.get("metadata", {}),
                    "embedding": r["embedding"]
                }
                for r in batch
            ]
            response = self.client.table(self.table_name).insert(payload).execute()
            if hasattr(response, "data") and response.data:
                total_inserted += len(response.data)
            else:
                total_inserted += len(batch)

        return total_inserted

    def search(self, query_embedding: List[float], match_count: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Performs vector similarity search using Supabase RPC."""
        params = {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "filter": filter_metadata or {}
        }
        try:
            res = self.client.rpc("match_documents", params).execute()
            return res.data or []
        except Exception as e:
            print(f"RPC match_documents search error: {e}")
            return []

    def get_existing_signatures(self) -> set:
        """Returns a set of existing chunk signatures (source_page_chunkIndex) to support resuming."""
        signatures = set()
        offset = 0
        limit = 1000
        try:
            while True:
                res = self.client.table(self.table_name).select("metadata").range(offset, offset + limit - 1).execute()
                if not res.data:
                    break
                for row in res.data:
                    m = row.get("metadata", {})
                    sig = f"{m.get('source')}_{m.get('page', m.get('sheet', ''))}_{m.get('chunk_index')}"
                    signatures.add(sig)
                if len(res.data) < limit:
                    break
                offset += limit
        except Exception as e:
            print(f"Could not fetch existing signatures: {e}")
        return signatures

    def clear(self, source: Optional[str] = None):
        """Removes all documents or records for a specific source document."""
        query = self.client.table(self.table_name).delete()
        if source:
            query = query.filter("metadata->>source", "eq", source)
        else:
            query = query.neq("id", 0)  # delete all
        return query.execute()
