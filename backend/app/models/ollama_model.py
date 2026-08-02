from langchain_ollama import ChatOllama
from backend.app.core.config import OLLAMA_MODEL, OLLAMA_TEMP, OLLAMA_MAX_PRED


class OllamaModel:
    """
    Local Qwen via Ollama (cost-free for day-to-day dev).

    Cloud failover (Groq → OpenRouter) is handled by FailoverLLM — this
    class only talks to Ollama and raises on failure.
    """

    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=OLLAMA_TEMP,
            num_predict=OLLAMA_MAX_PRED,
            streaming=True,
            num_thread=4,        # CPU threads for non-GPU ops
            repeat_penalty=1.1,  # reduce repetition
        )

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

    def stream(self, prompt: str):
        for chunk in self.llm.stream(prompt):
            text = getattr(chunk, "content", None)
            if text:
                yield text