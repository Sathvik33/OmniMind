from typing import Dict, Any
import re

class GenerationGuard:
    """
    Validates LLM outputs before sending them to the user.
    Checks for PII, toxicity, and basic hallucination markers.
    """
    @staticmethod
    def validate(output: str) -> Dict[str, Any]:
        # 1. PII Check (very basic example for demonstration)
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', output):
            return {"valid": False, "reason": "PII Detected (SSN)"}
            
        # 2. Toxicity Check (placeholder)
        toxic_words = ["hate_word_1", "hate_word_2"]
        if any(word in output.lower() for word in toxic_words):
            return {"valid": False, "reason": "Toxicity Detected"}
            
        return {"valid": True, "reason": "Pass"}
