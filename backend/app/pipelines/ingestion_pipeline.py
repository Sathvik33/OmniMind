import uuid
from pathlib import Path

from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.ingestion.chunking.semantic_chunker import SemanticChunker
from backend.app.ingestion.loaders.file_router import FileRouter


class IngestionPipeline:
    """
    Document ingestion pipeline with semantic chunking.

    Flow
    ────
    file_path → FileRouter → Loader.load() → SemanticChunker.chunk()
              → CollectionManager.add_documents() → omnimind_text collection

    NOTE: SemanticChunker runs at ingestion time (background task).
    The extra seconds do NOT affect query latency — uploads are always
    processed in the background while the user continues interacting.
    """

    def __init__(self):
        self.collection_manager = CollectionManager()
        self.chunker = SemanticChunker()   # threshold=0.75, GPU batch encoding
        self.router = FileRouter()

    def ingest_file(self, file_path: str) -> dict:
        path = Path(file_path)

        loader = self.router.route(path)
        text   = loader.load(path)

        chunks = self.chunker.chunk(text)

        if not chunks:
            return {"file": path.name, "chunks_added": 0}

        ids      = [str(uuid.uuid4()) for _ in chunks]
        metadata = [
            {"source": str(path), "modality": "document", "chunking": "semantic"}
            for _ in chunks
        ]

        self.collection_manager.add_documents(
            documents=chunks,
            ids=ids,
            metadata=metadata,
        )

        return {"file": path.name, "chunks_added": len(chunks)}