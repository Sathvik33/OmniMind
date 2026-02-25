from pathlib import Path
from docx import Document
from .base_loader import BaseLoader


class DocxLoader(BaseLoader):

    def load(self, path: Path) -> str:
        doc = Document(str(path))
        return "\n".join([para.text for para in doc.paragraphs])