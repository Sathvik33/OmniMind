from pathlib import Path

from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .ppt_loader import PPTLoader
from .excel_loader import ExcelLoader


class FileRouter:

    def __init__(self):
        self.loaders = {
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".pdf": PDFLoader(),
            ".docx": DocxLoader(),
            ".pptx": PPTLoader(),
            ".xlsx": ExcelLoader(),
            ".xls": ExcelLoader(),
        }

    def route(self, path: Path):
        suffix = path.suffix.lower()

        if suffix not in self.loaders:
            raise ValueError(f"Unsupported file type: {suffix}")

        return self.loaders[suffix]