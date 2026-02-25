from typing import List
from sentence_transformers import SentenceTransformer, util
import torch
from .base import BaseChunker


class SemanticChunker(BaseChunker):

    def __init__(self, config):
        super().__init__(config)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def chunk(self, text: str) -> List[str]:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if not sentences:
            return []

        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = util.cos_sim(
                embeddings[i - 1],
                embeddings[i]
            ).item()

            if similarity < self.config.similarity_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks