from pathlib import Path
import fitz
from .base_loader import BaseLoader


class PDFLoader(BaseLoader):

    def load(self, path: Path) -> str:
        doc = fitz.open(path)
        text_parts = []

        for page in doc:
            text_parts.append(page.get_text())

        doc.close()

        return "\n".join(text_parts)