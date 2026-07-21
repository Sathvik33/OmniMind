from typing import List, Dict, Any

class ContextBuilder:
    @staticmethod
    def build(retrieved_results: List[Dict[str, Any]]) -> str:
        """
        Deduplicates and formats the retrieved context for the LLM.
        """
        seen = set()
        unique_results = []
        
        for res in retrieved_results:
            content = res.get("content", "")
            if not content or content in seen:
                continue
            seen.add(content)
            unique_results.append(res)
            
        context_str = "\n\n---\n\n".join(
            f"Source [{res.get('artifact_id')}]: {res.get('content')}"
            for res in unique_results
        )
        return context_str
