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
    provider = "openrouter"
    kind_tags = ["llm", "openrouter-llm", "openrouter", "cloud", "free-tier"]

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
        from backend.app.monitoring.langsmith_logger import tracer

        with tracer.model_call(
            name=f"llm:openrouter:{self.model_name}",
            tags=self.kind_tags,
            provider=self.provider,
            model=self.model_name,
            inputs={"prompt": (prompt or "")[:6000], "prompt_chars": len(prompt or "")},
            run_type="llm",
        ) as span:
            text = chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            span["output"] = text
            return text

    def stream(self, prompt: str):
        from backend.app.monitoring.langsmith_logger import tracer

        with tracer.model_call(
            name=f"llm:openrouter:{self.model_name}:stream",
            tags=[*self.kind_tags, "stream"],
            provider=self.provider,
            model=self.model_name,
            inputs={"prompt": (prompt or "")[:6000], "prompt_chars": len(prompt or ""), "mode": "stream"},
            run_type="llm",
        ) as span:
            parts: list[str] = []
            for token in chat_completion_stream(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ):
                parts.append(token)
                yield token
            span["output"] = "".join(parts)
            span["extra"] = {"token_chunks": len(parts)}
