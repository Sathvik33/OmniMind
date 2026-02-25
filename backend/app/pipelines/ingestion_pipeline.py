from pathlib import Path
import uuid

from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.ingestion.chunking.langchain_chunker import LangchainChunker
from backend.app.ingestion.loaders.file_router import FileRouter


class IngestionPipeline:

    def __init__(self):
        self.collection_manager = CollectionManager()
        self.chunker = LangchainChunker(600, 100)
        self.router = FileRouter()

    def ingest_file(self, file_path: str):

        path = Path(file_path)

        loader = self.router.route(path)
        text = loader.load(path)

        chunks = self.chunker.chunk(text)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadata = [{"source": str(path)} for _ in chunks]

        self.collection_manager.add_documents(
            documents=chunks,
            ids=ids,
            metadata=metadata
        )

        return {
            "file": path.name,
            "chunks_added": len(chunks)
        }