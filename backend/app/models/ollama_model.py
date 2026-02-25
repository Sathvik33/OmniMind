from langchain_community.chat_models import ChatOllama


class OllamaModel:

    def __init__(self, model_name: str = "llama3"):
        self.llm = ChatOllama(model=model_name,
        temperature=0.2,
        num_predict=750)

    def generate(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response.content