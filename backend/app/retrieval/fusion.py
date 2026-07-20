from typing import List, Dict, Any
from collections import defaultdict
# from backend.app.services.reranker_service import RerankerService # Will be created when ModelRegistry is fully loaded

class RetrievalFusion:
    @staticmethod
    def rrf(retrieval_results: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion.
        Combines multiple ranked lists (e.g., from dense search and sparse search)
        into a single unified ranked list.
        """
        rrf_scores = defaultdict(float)
        item_data = {}

        for rank_list in retrieval_results:
            for rank, item in enumerate(rank_list):
                # We use the item 'id' as the unique key
                item_id = item["id"]
                rrf_scores[item_id] += 1.0 / (k + rank + 1)
                if item_id not in item_data:
                    item_data[item_id] = item
                
        # Sort by the new RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused = []
        for item_id, score in sorted_items:
            data = item_data[item_id].copy()
            data["rrf_score"] = score
            fused.append(data)
            
        return fused

    @staticmethod
    def rerank(query: str, fused_results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks the fused results using a Cross-Encoder (BGE-Reranker-v2-m3).
        """
        if not fused_results:
            return []
            
        # Placeholder for actual cross encoder model call
        # In real scenario: scores = cross_encoder.predict([(query, item["content"]) for item in fused_results])
        # For now, we simulate by just returning top_k
        
        # Sort by simulated rerank score (here we just use existing RRF score to simulate)
        reranked = sorted(fused_results, key=lambda x: x.get("rrf_score", 0), reverse=True)
        return reranked[:top_k]
