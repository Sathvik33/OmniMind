import torch
from sentence_transformers import SentenceTransformer
from typing import List


class TextEmbeddingModel:
    """
    Text-only dense embedding model.
    Model : all-MiniLM-L6-v2
    Output: 384-dim vectors
    Used  : document chunks → omnimind_text collection
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = 384

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


# Backwards-compat alias so existing imports don't break
EmbeddingModel = TextEmbeddingModel