from langchain_ollama import ChatOllama
from backend.app.core.config import OLLAMA_MODEL, OLLAMA_TEMP, OLLAMA_MAX_PRED


class OllamaModel:
    """
    Local Qwen via Ollama (cost-free for day-to-day dev).

    Cloud failover (Groq → OpenRouter) is handled by FailoverLLM — this
    class only talks to Ollama and raises on failure.
    """

    provider = "ollama"
    kind_tags = ["llm", "qwen-llm", "ollama", "local"]

    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=OLLAMA_TEMP,
            num_predict=OLLAMA_MAX_PRED,
            streaming=True,
            num_thread=4,
            repeat_penalty=1.1,
        )

    def generate(self, prompt: str) -> str:
        from backend.app.monitoring.langsmith_logger import tracer

        with tracer.model_call(
            name=f"llm:ollama:{self.model_name}",
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
            name=f"llm:ollama:{self.model_name}:stream",
            tags=[*self.kind_tags, "stream"],
            provider=self.provider,
            model=self.model_name,
            inputs={"prompt": (prompt or "")[:6000], "prompt_chars": len(prompt or ""), "mode": "stream"},
            run_type="llm",
        ) as span:
            parts: list[str] = []
            for chunk in self.llm.stream(prompt):
                text = getattr(chunk, "content", None)
                if text:
                    parts.append(text)
                    yield text
            span["output"] = "".join(parts)
            span["extra"] = {"token_chunks": len(parts)}
