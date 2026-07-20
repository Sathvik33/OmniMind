from typing import Dict, Any
import os
from backend.app.db.database import SessionLocal
from backend.app.db.models import AIModelRegistry, ModelCategory

class DocumentParsingService:
    def __init__(self):
        self._load_active_parser()

    def _load_active_parser(self):
        """Fetch active parsing model from the Model Registry."""
        db = SessionLocal()
        try:
            self.parser_model = db.query(AIModelRegistry).filter_by(
                category=ModelCategory.PARSER, 
                is_active=1
            ).first()
        finally:
            db.close()

    def parse_document(self, file_path: str, mime_type: str = None) -> Dict[str, Any]:
        """
        Dynamically route to the best parser based on registry configuration and file complexity.
        """
        parser_name = self.parser_model.name if self.parser_model else "liteparse"
        
        # Simple routing logic
        if parser_name.lower() == "liteparse" or (mime_type and mime_type.startswith("text/")):
            return self._run_liteparse(file_path)
        elif parser_name.lower() == "docling":
            return self._run_docling(file_path)
        elif parser_name.lower() == "llamaparse":
            return self._run_llamaparse(file_path)
        else:
            return self._run_liteparse(file_path) # fallback

    def _run_liteparse(self, file_path: str) -> Dict[str, Any]:
        # TODO: Implement fast local parsing (PyMuPDF, pdfplumber)
        return {"text": f"Parsed content of {file_path} using LiteParse", "metadata": {}}

    def _run_docling(self, file_path: str) -> Dict[str, Any]:
        # TODO: Implement advanced structure-aware parsing
        return {"text": f"Parsed content of {file_path} using Docling", "metadata": {}}
        
    def _run_llamaparse(self, file_path: str) -> Dict[str, Any]:
        # TODO: Implement cloud API parsing
        return {"text": f"Parsed content of {file_path} using LlamaParse", "metadata": {}}

parsing_service = DocumentParsingService()
