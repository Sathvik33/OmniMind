from abc import ABC, abstractmethod
from typing import List
from .schemas import ChunkingConfig


class BaseChunker(ABC):
    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass