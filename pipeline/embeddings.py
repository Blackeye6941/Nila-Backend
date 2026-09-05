import time
from typing import List, Optional
from fastembed import TextEmbedding


class EmbeddingGenerator:
    """
    Generates 768-dimensional dense vector embeddings.
    Supports:
    - 'local' (Default): Open-source BAAI/bge-base-en-v1.5 (768-dim) running locally with zero rate limits / no API cost.
    - 'gemini': Google Gemini API with rate limiting and backoff.
    """

    def __init__(
        self,
        provider: str = "local",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        output_dim: int = 768,
        batch_delay: float = 0.0,
        max_retries: int = 5
    ):
        self.provider = provider.lower()
        self.output_dim = output_dim
        self.batch_delay = batch_delay
        self.max_retries = max_retries

        if self.provider == "local":
            # Default to the top-performing 768-dim open-source model: BAAI/bge-base-en-v1.5
            self.model_name = model_name or "BAAI/bge-base-en-v1.5"
            print(f"Loading local open-source embedding model: {self.model_name} (dim=768)...")
            self.local_model = TextEmbedding(model_name=self.model_name)
        elif self.provider == "gemini":
            if not api_key:
                raise ValueError("GEMINI_API_KEY must be provided for provider='gemini'")
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.model_name = model_name or "gemini-embedding-001"
        else:
            raise ValueError(f"Unknown provider '{provider}'. Must be 'local' or 'gemini'.")

    def embed_text(self, text: str) -> List[float]:
        """Generate 768-dim embedding for a single string."""
        results = self.embed_batch([text], batch_size=1)
        return results[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate 768-dim embeddings in batches."""
        if not texts:
            return []

        # Local Open-Source Generation (FastEmbed / ONNX)
        if self.provider == "local":
            # fastembed.embed generator returns numpy arrays / iterables
            embeddings_iter = self.local_model.embed(texts, batch_size=batch_size)
            return [emb.tolist() for emb in embeddings_iter]

        # Gemini API Generation
        from google.genai import types
        all_embeddings: List[List[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(texts), batch_size), start=1):
            batch = texts[i:i + batch_size]
            backoff = 10.0

            for attempt in range(1, self.max_retries + 1):
                try:
                    config = types.EmbedContentConfig(output_dimensionality=self.output_dim) if self.output_dim else None
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                        config=config
                    )
                    batch_vectors = [list(e.values) for e in response.embeddings]
                    all_embeddings.extend(batch_vectors)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = "429" in err_str or "quota" in err_str or "exhausted" in err_str

                    if attempt < self.max_retries:
                        sleep_time = max(backoff, 25.0 if is_rate_limit else backoff)
                        print(f"\n[Rate Limit] Batch {batch_num}/{total_batches} (Attempt {attempt}/{self.max_retries}). Retrying in {sleep_time:.0f}s...")
                        time.sleep(sleep_time)
                        backoff *= 2.0
                    else:
                        raise RuntimeError(f"Failed to generate embeddings for batch {batch_num}: {e}")

            if batch_num < total_batches and self.batch_delay > 0:
                time.sleep(self.batch_delay)

        return all_embeddings
