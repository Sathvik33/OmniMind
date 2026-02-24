import chromadb
from backend.app.core.config import CHROMA_DIR

def get_chroma_client():
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )
