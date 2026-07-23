import json
from typing import List, Dict, Any, Optional
import redis
import os
from backend.app.db.database import SessionLocal
from backend.app.db.models import AIModelRegistry, ModelCategory
import torch
from sentence_transformers import SentenceTransformer
import open_clip
from PIL import Image

# Setup Redis client for caching
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is not set")
redis_client = redis.from_url(REDIS_URL)

class EmbeddingService:
    def __init__(self):
        self._load_active_models()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load real text embedding model
        self.bge_model = SentenceTransformer("BAAI/bge-m3", device=self.device)
        
        # Load real vision embedding model
        self.siglip_model, _, self.siglip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16-SigLIP', 
            pretrained='webli', 
            device=self.device
        )
        self.siglip_tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')

    def _load_active_models(self):
        """Fetch active embedding models from the Model Registry."""
        db = SessionLocal()
        try:
            self.text_model = db.query(AIModelRegistry).filter_by(
                category=ModelCategory.EMBEDDING, 
                is_active=1,
                name="BAAI/bge-m3"
            ).first()
            
            self.vision_model = db.query(AIModelRegistry).filter_by(
                category=ModelCategory.EMBEDDING, 
                is_active=1,
                name="SigLIP2"
            ).first()
        finally:
            db.close()

    def _get_cache_key(self, text: str, model_name: str) -> str:
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"emb:{model_name}:{text_hash}"

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text using the registered text model.
        Checks Redis cache first.
        """
        model_name = self.text_model.name if self.text_model else "BAAI/bge-m3"
        cache_key = self._get_cache_key(text, model_name)
        
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)

        # Actual model inference using sentence_transformers (BGE-M3)
        embedding = self.bge_model.encode(text, convert_to_numpy=True).tolist()
        
        # Cache for 24 hours
        redis_client.setex(cache_key, 86400, json.dumps(embedding))
        return embedding

    def embed_image(self, image_path: str) -> List[float]:
        """
        Generate embedding for an image using the registered vision model.
        """
        model_name = self.vision_model.name if self.vision_model else "SigLIP2"
        # Actual model inference using open_clip (SigLIP)
        image = Image.open(image_path).convert('RGB')
        image_input = self.siglip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.siglip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        embedding = image_features[0].cpu().numpy().tolist()
        return embedding

    def embed_query_for_vision(self, text: str) -> List[float]:
        """
        Generate embedding for a text query using the vision model's text encoder
        for cross-modal search (Text -> Image).
        """
        text_input = self.siglip_tokenizer([text]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.siglip_model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        embedding = text_features[0].cpu().numpy().tolist()
        return embedding

embedding_service = EmbeddingService()