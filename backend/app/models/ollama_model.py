from langchain_community.chat_models import ChatOllama


class OllamaModel:

    def __init__(self, model_name: str = "llama3"):
        self.llm = ChatOllama(model=model_name)

    def generate(self, prompt: str):
        return self.llm.invoke(prompt)