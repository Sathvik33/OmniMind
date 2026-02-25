from pathlib import Path
from backend.app.ingestion.experimental.factory import ChunkingFactory
from backend.app.ingestion.experimental.schemas import ChunkingConfig

def ingest_file(file_path: str):

    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    config = ChunkingConfig(chunk_size=600, overlap=100)
    chunker = ChunkingFactory.create("semantic", config)

    chunks = chunker.chunk(text)

    return chunks