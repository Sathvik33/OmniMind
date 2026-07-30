import torch
from sentence_transformers import SentenceTransformer
from typing import List


class TextEmbeddingModel:
    """
    Text-only dense embedding model.
    Model : BAAI/bge-m3
    Output: 1024-dim vectors
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = 1024


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


# Backwards-compat alias so existing imports don't break
EmbeddingModel = TextEmbeddingModel