from pathlib import Path
from .base_loader import BaseLoader


class TextLoader(BaseLoader):

    def load(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")