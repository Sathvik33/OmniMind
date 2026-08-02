"""
GroqModel — Groq cloud LLM wrapper for AEGIS v3.0.
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

    provider = "groq"
    kind_tags = ["llm", "groq-llm", "groq", "cloud"]

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
        from backend.app.monitoring.langsmith_logger import tracer

        with tracer.model_call(
            name=f"llm:groq:{self.model_name}",
            tags=self.kind_tags,
            provider=self.provider,
            model=self.model_name,
            inputs={"prompt": (prompt or "")[:6000], "prompt_chars": len(prompt or "")},
            run_type="llm",
        ) as span:
            response = self.llm.invoke(prompt)
            text = response.content
            span["output"] = text
            return text

    def stream(self, prompt: str):
        from backend.app.monitoring.langsmith_logger import tracer

        with tracer.model_call(
            name=f"llm:groq:{self.model_name}:stream",
            tags=[*self.kind_tags, "stream"],
            provider=self.provider,
            model=self.model_name,
            inputs={"prompt": (prompt or "")[:6000], "prompt_chars": len(prompt or ""), "mode": "stream"},
            run_type="llm",
        ) as span:
            parts: list[str] = []
            for chunk in self.llm.stream(prompt):
                if chunk.content:
                    parts.append(chunk.content)
                    yield chunk.content
            span["output"] = "".join(parts)
            span["extra"] = {"token_chunks": len(parts)}
