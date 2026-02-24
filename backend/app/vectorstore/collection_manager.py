from sentence_transformers import SentenceTransformer
from backend.app.vectorstore.chroma_client import get_chroma_client

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

class CollectionManager:
    def __init__(self, collection_name="aegis_collection"):
        self.client=get_chroma_client()
        self.collection=self.client.get_or_create_collection(
            name=collection_name
        )

    def add_documents(self, documents, ids, metadata):
        embeddings=EMBED_MODEL.encode(documents).tolist()
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )

    def query(self, query_text, n_results=3):
        query_embedding=EMBED_MODEL.encode([query_text]).tolist()
        results=self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return results