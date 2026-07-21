from backend.app.router.semantic_router import SemanticRouter, RouteCategory, RouteDecision
from backend.app.retrieval.pgvector_store import PgVectorStore
from backend.app.retrieval.fusion import RetrievalFusion
from backend.app.retrieval.quality_assessor import RetrievalQualityAssessor
from sqlalchemy.orm import Session
from typing import Dict, Any

class DecisionEngine:
    """
    Acts on the decision made by the Semantic Router.
    """
    def __init__(self, db: Session):
        self.db = db
        self.router = SemanticRouter()
        self.vector_store = PgVectorStore(db)
        
    def process_query(self, query: str, has_images: bool = False) -> Dict[str, Any]:
        decision = self.router.route_query(query, has_images)
        
        context = []
        if decision.category == RouteCategory.RETRIEVAL:
            # 1. Execute dense retrieval
            raw_results = self.vector_store.search(query, top_k=10)
            
            # 2. Fuse & Rerank (Placeholder for sparse search fusion)
            fused = RetrievalFusion.rrf([raw_results])
            
            # 3. Assess quality
            quality = RetrievalQualityAssessor.assess(query, fused)
            if quality["status"] == "PASS":
                context = fused
                
        return {
            "decision": decision.model_dump(),
            "context": context
        }
