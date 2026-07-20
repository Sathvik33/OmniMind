import json
from typing import List, Dict, Any, Optional
import redis
import os
from backend.app.db.database import SessionLocal
from backend.app.db.models import AIModelRegistry, ModelCategory

# Setup Redis client for caching
REDIS_URL = os.getenv("REDIS_URL")
if not REDIS_URL:
    raise ValueError("REDIS_URL environment variable is not set")
redis_client = redis.from_url(REDIS_URL)

class EmbeddingService:
    def __init__(self):
        self._load_active_models()

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

        # TODO: Actual model inference using sentence_transformers (BGE-M3)
        # embedding = self._run_bge_m3(text)
        embedding = [0.0] * 1024 # Mock 1024d embedding for now
        
        # Cache for 24 hours
        redis_client.setex(cache_key, 86400, json.dumps(embedding))
        return embedding

    def embed_image(self, image_path: str) -> List[float]:
        """
        Generate embedding for an image using the registered vision model.
        """
        model_name = self.vision_model.name if self.vision_model else "SigLIP2"
        # TODO: Actual model inference using open_clip (SigLIP2)
        # embedding = self._run_siglip2(image_path)
        embedding = [0.0] * 768 # Mock 768d embedding for now
        return embedding

embedding_service = EmbeddingService()
