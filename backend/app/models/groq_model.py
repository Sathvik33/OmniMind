"""
GroqModel — Groq cloud LLM wrapper for AEGIS v3.0.

Replaces local Ollama (qwen2.5:7b) with Groq-hosted llama3-70b-8192.

Why Groq?
  - 10-100x faster than local Ollama on consumer hardware
  - No GPU/VRAM required for the LLM
  - llama3-70b-8192 is significantly more capable than qwen2.5:7b

Interface is identical to OllamaModel so QueryPipeline requires zero changes.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from backend.app.core.config import GROQ_LLM_API_KEY, GROQ_GENERATION_MODEL


class GroqModel:
    """
    Wrapper around Groq's hosted LLM API.
    Drop-in replacement for OllamaModel — same .generate() / .stream() interface.
    """

    def __init__(
        self,
        model_name: str = GROQ_GENERATION_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ):
        api_key = GROQ_LLM_API_KEY or os.getenv("GROQ_LLM_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_LLM_API_KEY is not set. Add it to your .env file."
            )


        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """Blocking generation — returns full response string."""
        response = self.llm.invoke(prompt)
        return response.content

    def stream(self, prompt: str):
        """Token-by-token streaming generator."""
        for chunk in self.llm.stream(prompt):
            if chunk.content:
                yield chunk.content
