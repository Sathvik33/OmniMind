"""
LLM chain with cost-aware priority:

  Local Ollama (Qwen)  →  Groq  →  OpenRouter free

Local is preferred for day-to-day dev. Cloud is fallback-only.
"""

from __future__ import annotations

import logging
from typing import Any, Generator, List, Optional

from backend.app.core.config import OLLAMA_MODEL, USE_LOCAL_LLM
from backend.app.services.groq_retry import should_use_fallback
from backend.app.services.openrouter_client import openrouter_configured

logger = logging.getLogger(__name__)


def _build_chain(*, prefer_local: bool) -> List[Any]:
    chain: List[Any] = []

    if prefer_local:
        try:
            from backend.app.models.ollama_model import OllamaModel

            chain.append(OllamaModel(model_name=OLLAMA_MODEL))
        except Exception as e:
            logger.warning("Ollama primary unavailable: %s", e)

    try:
        from backend.app.models.groq_model import GroqModel

        chain.append(GroqModel())
    except Exception as e:
        logger.warning("Groq unavailable: %s", e)

    if openrouter_configured():
        try:
            from backend.app.models.openrouter_model import OpenRouterModel

            chain.append(OpenRouterModel())
        except Exception as e:
            logger.warning("OpenRouter fallback unavailable: %s", e)

    if not chain:
        raise RuntimeError(
            "No LLM backends available. Start Ollama (qwen2.5:7b) or set Groq / OpenRouter keys."
        )
    return chain


class FailoverLLM:
    """
    Drop-in for GroqModel / OllamaModel: .generate(prompt) / .stream(prompt)

    Default for local dev (USE_LOCAL_LLM=true):
      qwen2.5:7b (Ollama) → Groq → OpenRouter :free
    Cloud/demo mode (USE_LOCAL_LLM=false):
      Groq → OpenRouter :free
    """

    def __init__(
        self,
        *,
        prefer_local: Optional[bool] = None,
        chain: Optional[List[Any]] = None,
    ):
        prefer = USE_LOCAL_LLM if prefer_local is None else prefer_local
        self._chain = chain if chain is not None else _build_chain(prefer_local=prefer)
        names = [getattr(m, "model_name", type(m).__name__) for m in self._chain]
        logger.info("FailoverLLM chain: %s", " → ".join(names))

    @property
    def primary(self):
        return self._chain[0]

    @property
    def llm(self):
        """Expose first backend's LangChain chat model for structured-output bind."""
        return getattr(self.primary, "llm", None)

    @property
    def model_name(self) -> str:
        return getattr(self.primary, "model_name", "failover")

    def generate(self, prompt: str) -> str:
        last: Optional[BaseException] = None
        for i, backend in enumerate(self._chain):
            name = getattr(backend, "model_name", type(backend).__name__)
            try:
                return backend.generate(prompt)
            except Exception as e:
                last = e
                if i + 1 < len(self._chain) and should_use_fallback(e):
                    logger.warning(
                        "%s generate failed (%s); trying next backend", name, e
                    )
                    continue
                if i + 1 < len(self._chain):
                    # Non-fallbackable but still try remaining cloud only if local failed hard
                    logger.warning(
                        "%s generate failed (%s); trying next backend anyway", name, e
                    )
                    continue
                raise
        assert last is not None
        raise last

    def stream(self, prompt: str) -> Generator[str, None, None]:
        last: Optional[BaseException] = None
        for i, backend in enumerate(self._chain):
            name = getattr(backend, "model_name", type(backend).__name__)
            try:
                yielded = False
                for token in backend.stream(prompt):
                    yielded = True
                    yield token
                if yielded:
                    return
                raise RuntimeError(f"{name} stream returned no tokens")
            except Exception as e:
                last = e
                if i + 1 < len(self._chain):
                    logger.warning(
                        "%s stream failed (%s); trying next backend", name, e
                    )
                    continue
                raise
        assert last is not None
        raise last
