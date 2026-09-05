from typing import List, Dict, Any


class TextChunker:
    """Splits documents into overlapping chunks with metadata."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        """Splits text into chunks of roughly chunk_size characters with overlap."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]

            # If not at the very end of the text, try to break cleanly at sentence or paragraph
            if end < len(text):
                # Search for preferred break points within the last 20% of the chunk
                search_region = chunk[int(self.chunk_size * 0.7):]
                last_para = search_region.rfind("\n\n")
                last_sentence = max(search_region.rfind(". "), search_region.rfind(".\n"))
                last_space = search_region.rfind(" ")

                if last_para != -1:
                    break_point = int(self.chunk_size * 0.7) + last_para + 2
                    chunk = chunk[:break_point]
                elif last_sentence != -1:
                    break_point = int(self.chunk_size * 0.7) + last_sentence + 2
                    chunk = chunk[:break_point]
                elif last_space != -1:
                    break_point = int(self.chunk_size * 0.7) + last_space + 1
                    chunk = chunk[:break_point]

            clean_chunk = chunk.strip()
            if clean_chunk:
                chunks.append(clean_chunk)

            # Strictly advance start forward to prevent any infinite loop
            start += step

        return chunks

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Takes loaded documents and returns chunked records with updated metadata."""
        chunked_docs = []
        for doc in documents:
            content = doc.get("content", "")
            meta = doc.get("metadata", {}).copy()
            text_chunks = self._split_text(content)

            for idx, chunk in enumerate(text_chunks):
                chunk_meta = meta.copy()
                chunk_meta["chunk_index"] = idx
                chunk_meta["total_chunks_in_section"] = len(text_chunks)
                chunk_meta["char_length"] = len(chunk)

                chunked_docs.append({
                    "content": chunk,
                    "metadata": chunk_meta
                })

        return chunked_docs
