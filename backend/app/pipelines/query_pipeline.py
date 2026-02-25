from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.models.ollama_model import OllamaModel


class QueryPipeline:

    def __init__(self):
        self.collection_manager = CollectionManager()
        self.llm = OllamaModel()

    def retrieve(self, query: str, top_k: int = 3):
        results = self.collection_manager.query(query, n_results=top_k)
        documents = results.get("documents", [])
        return documents[0] if documents else []

    def generate_answer(self, query: str, top_k: int = 3):
        context_chunks = self.retrieve(query, top_k)

        context = "\n\n".join(context_chunks)

        prompt = f"""
You are a knowledge assistant.
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.generate(prompt)

        return {
            "answer": response.content,
            "context_used": context_chunks
        }