import uuid
from pathlib import Path
from typing import Dict

from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.ingestion.loaders.text_loader import load_text
from backend.app.ingestion.chunking.langchain_chunker import LangchainChunker


class IngestionPipeline:
    def __init__(self):
        self.collection_manager = CollectionManager()
        self.chunker = LangchainChunker(
            chunk_size=600,
            overlap=100
        )

    def ingest_text_file(self, file_path: str) -> Dict:
        path = Path(file_path)

        text = load_text(path)

        chunks = self.chunker.chunk(text)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadata = [{"source": str(path)} for _ in chunks]

        self.collection_manager.add_documents(
            documents=chunks,
            ids=ids,
            metadata=metadata
        )

        return {
            "file": str(path.name),
            "chunks_added": len(chunks)
        }