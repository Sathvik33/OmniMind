from langchain_ollama import ChatOllama


class OllamaModel:

    def __init__(self, model_name: str = "llama3"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.2,
            num_predict=1500,
            streaming=True
        )

    def generate(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response.content

    def stream(self, prompt: str):
        for chunk in self.llm.stream(prompt):
            yield chunk.content