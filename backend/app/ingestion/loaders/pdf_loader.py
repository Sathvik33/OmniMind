from pathlib import Path
from pypdf import PdfReader
from .base_loader import BaseLoader


class PDFLoader(BaseLoader):

    def load(self, path: Path) -> str:
        reader = PdfReader(str(path))
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return text 