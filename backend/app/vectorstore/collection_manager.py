from backend.app.embeddings.embedding_model import EmbeddingModel
from backend.app.vectorstore.chroma_client import get_chroma_client


class CollectionManager:

    def __init__(self, collection_name: str = "aegis_collection"):
        self.client = get_chroma_client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        self.embedding_model = EmbeddingModel()

    def add_documents(self, documents, ids, metadata):
        embeddings = self.embedding_model.embed_documents(documents)

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )

    def query(self, query_text, n_results=3):
        query_embedding = self.embedding_model.embed_query(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results
    
    def query_time_range(self, start_time: int, end_time: int):

        results = self.collection.get(
            where={"modality": "video"}
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        filtered_docs = []

        for doc, meta in zip(documents, metadatas):
            if meta.get("start_time") is not None:
                if meta["start_time"] <= end_time and meta["end_time"] >= start_time:
                    filtered_docs.append(doc)

        return filtered_docs