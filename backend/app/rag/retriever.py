from langsmith import traceable

class Retriever:

    def __init__(self, collection_manager):
        self.collection_manager = collection_manager

    @traceable(name="vector_retrieval")
    def retrieve(self, query: str, top_k: int = 3):
        results = self.collection_manager.query(query, n_results=top_k)
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if documents:
            return documents[0], metadatas[0] if metadatas else []

        return [], []