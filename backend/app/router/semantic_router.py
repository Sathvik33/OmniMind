from enum import Enum
from pydantic import BaseModel

class RouteCategory(str, Enum):
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    VISION = "vision"
    FAST_QA = "fast_qa"

class RouteDecision(BaseModel):
    category: RouteCategory
    confidence: float
    reason: str

class SemanticRouter:
    """
    Determines the intent of the query and routes it to the correct capability.
    Can be powered by a fast local LLM, or via heuristics/embeddings.
    """
    
    @staticmethod
    def route_query(query: str, has_images: bool = False) -> RouteDecision:
        if has_images:
            return RouteDecision(
                category=RouteCategory.VISION,
                confidence=1.0,
                reason="Query contains images"
            )
            
        query_lower = query.lower()
        
        # Simple heuristics for routing (can be upgraded to embedding classification later)
        retrieval_keywords = ["what is", "how do i", "find", "search", "document", "file", "explain based on", "where"]
        reasoning_keywords = ["why", "compare", "analyze", "evaluate", "synthesize", "plan", "write code"]
        
        if any(kw in query_lower for kw in retrieval_keywords):
            return RouteDecision(
                category=RouteCategory.RETRIEVAL,
                confidence=0.8,
                reason="Query matches retrieval heuristics"
            )
            
        if any(kw in query_lower for kw in reasoning_keywords):
            return RouteDecision(
                category=RouteCategory.REASONING,
                confidence=0.8,
                reason="Query matches reasoning heuristics"
            )
            
        # Default fallback
        return RouteDecision(
            category=RouteCategory.FAST_QA,
            confidence=0.5,
            reason="Fallback to fast QA"
        )
