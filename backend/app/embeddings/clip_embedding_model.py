import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
from typing import List


class CLIPEmbeddingModel:
    """
    Multi-modal embedding model using CLIP.
    Model : clip-ViT-B-32  (via sentence-transformers)
    Output: 512-dim vectors in a shared text-image space
    Used  : image descriptions + video frames → omnimind_multimodal collection

    Both images and text queries are encoded in the same CLIP space,
    enabling true cross-modal retrieval (text query → find similar images).
    """

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = 512

    # ── Image Encoding ──────────────────────────────────────────────────────────

    def embed_image(self, image_path: str) -> List[float]:
        """Encode a single image file into the CLIP space."""
        img = Image.open(image_path).convert("RGB")
        return self.model.encode(img, normalize_embeddings=True).tolist()

    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        """Batch-encode multiple image files."""
        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        return self.model.encode(
            imgs, batch_size=16, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

    # ── Text Encoding (for cross-modal queries) ────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        """Encode a text string in CLIP space for cross-modal retrieval."""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Batch-encode text strings in CLIP space."""
        return self.model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        ).tolist()
