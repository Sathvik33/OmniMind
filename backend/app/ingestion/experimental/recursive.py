from typing import List
from .base import BaseChunker


class RecursiveChunker(BaseChunker):

    def chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        size = self.config.chunk_size
        overlap = self.config.overlap

        while start < len(text):
            end = start + size
            chunk = text[start:end]
            chunks.append(chunk)
            start += size - overlap

        return chunks