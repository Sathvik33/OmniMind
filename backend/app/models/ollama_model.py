from langchain_ollama import ChatOllama
from backend.app.core.config import OLLAMA_MODEL, OLLAMA_TEMP, OLLAMA_MAX_PRED


class OllamaModel:
    """
    Wrapper around Qwen2.5:7b via Ollama.
    Ollama serves GGUF-quantized models (Q4_K_M by default), so all
    inference is already quantized — num_gpu=99 ensures all layers
    stay on the GPU for maximum throughput.
    """

    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.llm = ChatOllama(
            model=model_name,
            temperature=OLLAMA_TEMP,
            num_predict=OLLAMA_MAX_PRED,
            streaming=True,
            num_thread=4,        # CPU threads for non-GPU ops
            repeat_penalty=1.1,  # reduce repetition
        )


    def generate(self, prompt: str) -> str:
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"⚠️ Ollama model error ({e}). Falling back to Groq Cloud LLM.")
            from backend.app.models.groq_model import GroqModel
            return GroqModel().generate(prompt)

    def stream(self, prompt: str):
        try:
            for chunk in self.llm.stream(prompt):
                yield chunk.content
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"⚠️ Ollama streaming error ({e}). Falling back to Groq Cloud LLM.")
            from backend.app.models.groq_model import GroqModel
            yield from GroqModel().stream(prompt)