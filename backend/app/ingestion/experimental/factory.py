from .schemas import ChunkingConfig
from .recursive import RecursiveChunker
from .semantic import SemanticChunker


class ChunkingFactory:

    @staticmethod
    def create(strategy: str, config: ChunkingConfig):
        if strategy == "recursive":
            return RecursiveChunker(config)
        elif strategy == "semantic":
            return SemanticChunker(config)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")