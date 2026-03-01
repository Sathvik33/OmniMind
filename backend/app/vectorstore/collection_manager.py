from backend.app.embeddings.embedding_model import EmbeddingModel
from backend.app.vectorstore.chroma_client import get_chroma_client

class CollectionManager:

    def __init__(self, collection_name: str = "aegis_collection"):
        self.client = get_chroma_client()
        self.collection_name = collection_name
        self.embedding_model = EmbeddingModel()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_documents(self, documents, ids, metadata):
        collection = self._get_collection()

        embeddings = self.embedding_model.embed_documents(documents)

        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )

    def query(self, query_text, n_results=3):
        collection = self._get_collection()

        query_embedding = self.embedding_model.embed_query(query_text)

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return results

    def query_time_range(self, start_time, end_time):
        collection = self._get_collection()

        results = collection.get(where={"modality": "video"})

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        filtered_docs = []

        for doc, meta in zip(documents, metadatas):
            if meta.get("start_time") is not None:
                if meta["start_time"] <= end_time and meta["end_time"] >= start_time:
                    filtered_docs.append(doc)

        return filtered_docs