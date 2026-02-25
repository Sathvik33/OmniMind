from backend.app.vectorstore.collection_manager import CollectionManager


class QueryPipeline:

    def __init__(self):
        self.collection_manager = CollectionManager()

    def retrieve(self, query: str, top_k: int = 3):
        results = self.collection_manager.query(query, n_results=top_k)

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if documents:
            return {
                "chunks": documents[0],
                "metadata": metadatas[0] if metadatas else []
            }

        return {
            "chunks": [],
            "metadata": []
        }