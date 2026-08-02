"""OpenRouter LLM — same generate/stream interface as GroqModel / OllamaModel."""

from __future__ import annotations

import logging

from backend.app.core.config import (
    OPENROUTER_GENERATION_MODEL,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_TEMPERATURE,
)
from backend.app.services.openrouter_client import (
    chat_completion,
    chat_completion_stream,
    openrouter_configured,
)

logger = logging.getLogger(__name__)


class OpenRouterModel:
    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = OPENROUTER_TEMPERATURE,
        max_tokens: int = OPENROUTER_MAX_TOKENS,
    ):
        if not openrouter_configured():
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.model_name = model_name or OPENROUTER_GENERATION_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info("OpenRouterModel initialized — model: %s", self.model_name)

    def generate(self, prompt: str) -> str:
        return chat_completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def stream(self, prompt: str):
        yield from chat_completion_stream(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
