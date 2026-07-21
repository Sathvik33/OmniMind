from typing import List, Dict

class PromptBuilder:
    @staticmethod
    def build_prompt(query: str, context: str, history: List[Dict[str, str]], max_tokens: int = 4000) -> str:
        """
        Assembles the final prompt by combining the user query, context, and history.
        Includes basic token management (truncating context if necessary).
        """
        
        system_prompt = "You are AEGIS, an advanced Multi-modal AI Assistant. Answer the user's question based on the provided context."
        
        history_str = ""
        if history:
            history_str = "Chat History:\n" + "\n".join(f"{h['role'].capitalize()}: {h['content']}" for h in history)
            
        # Basic token estimation (1 token ~= 4 chars)
        max_chars = max_tokens * 4
        
        # Build raw prompt
        raw_prompt = f"{system_prompt}\n\n"
        if history_str:
            raw_prompt += f"{history_str}\n\n"
            
        if context:
            raw_prompt += f"Context:\n{context}\n\n"
            
        raw_prompt += f"User Query: {query}\n"
        
        # Truncate context if exceeding max_chars (very rough)
        if len(raw_prompt) > max_chars:
            excess = len(raw_prompt) - max_chars
            if context and len(context) > excess:
                truncated_context = context[:-(excess + 50)] + "... [Context Truncated]"
                
                # Rebuild with truncated context
                raw_prompt = f"{system_prompt}\n\n"
                if history_str:
                    raw_prompt += f"{history_str}\n\n"
                raw_prompt += f"Context:\n{truncated_context}\n\n"
                raw_prompt += f"User Query: {query}\n"
        
        return raw_prompt
