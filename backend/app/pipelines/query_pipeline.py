from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.models.ollama_model import OllamaModel


class QueryPipeline:

    def __init__(self):
        self.collection_manager = CollectionManager()
        self.llm = OllamaModel()

    def retrieve(self, query: str, top_k: int = 3):
        results = self.collection_manager.query(query, n_results=top_k)

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if documents:
            return documents[0], metadatas[0] if metadatas else []

        return [], []

    def answer(self, query: str, top_k: int = 3):
        chunks, metadata = self.retrieve(query, top_k)

        context = "\n\n".join(chunks)

        prompt = f"""
You are a technical knowledge assistant.

Provide a detailed and well-structured explanation.
Use only the information from the context.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{query}

Answer in detailed explanation format:
"""

        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "context_used": chunks
        }