from typing import Dict, Any
import os
from backend.app.db.database import SessionLocal
from backend.app.db.models import AIModelRegistry, ModelCategory

class DocumentParsingService:
    def __init__(self):
        self._load_active_parser()

    def _load_active_parser(self):
        """Fetch active parsing model from the Model Registry."""
        self.parser_model = None
        try:
            db = SessionLocal()
            try:
                self.parser_model = db.query(AIModelRegistry).filter_by(
                    category=ModelCategory.PARSER, 
                    is_active=1
                ).first()
            finally:
                db.close()
        except Exception:
            pass


    def parse_document(self, file_path: str, original_filename: str = "") -> Dict[str, Any]:
        """
        Dynamically route to the best parser based on registry configuration and file complexity.
        """
        parser_name = self.parser_model.name if self.parser_model else "liteparse"
        
        # Simple routing logic
        if parser_name.lower() == "liteparse":
            return self._run_liteparse(file_path, original_filename)
        elif parser_name.lower() == "docling":
            return self._run_docling(file_path)
        elif parser_name.lower() == "llamaparse":
            return self._run_llamaparse(file_path)
        else:
            return self._run_liteparse(file_path, original_filename) # fallback

    def _run_liteparse(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        import subprocess
        import tempfile
        import shutil
        
        # Create a unique output directory in the project root
        base_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(base_dir, exist_ok=True)
        
        # Use a temporary directory inside output to avoid collisions
        out_dir = tempfile.mkdtemp(dir=base_dir, prefix="liteparse_")
        
        md_path = os.path.join(out_dir, "document.md")
        json_path = os.path.join(out_dir, "document.json")
        images_dir = os.path.join(out_dir, "images")
        
        ext = os.path.splitext(original_filename)[1] or ".pdf"
        input_file = os.path.join(out_dir, f"input{ext}")
        shutil.copy(file_path, input_file)
        
        if original_filename.lower().endswith('.pdf') or ext.lower() == '.pdf':
            try:
                # 1. Pre-check document complexity using LiteParse 'is-complex'
                is_complex = True
                try:
                    res = subprocess.run(
                        ["lit", "is-complex", input_file, "--quiet"],
                        capture_output=True
                    )
                    # Exit code 0 indicates text-only document (no OCR/image extraction needed)
                    is_complex = (res.returncode != 0)
                except Exception:
                    is_complex = True

                # 2. Generate JSON document structure
                subprocess.run(
                    ["lit", "parse", input_file, "--format", "json", "-o", json_path],
                    check=True, capture_output=True
                )
                
                # 3. Adaptive Markdown Generation: fast text pass if simple, image extraction if complex
                parse_cmd = ["lit", "parse", input_file, "--format", "markdown"]
                if is_complex:
                    parse_cmd.extend(["--image-mode", "embed", "--image-output-dir", images_dir])
                else:
                    parse_cmd.append("--no-ocr")
                parse_cmd.extend(["-o", md_path])
                
                subprocess.run(parse_cmd, check=True, capture_output=True)
                
                return {
                    "markdown_path": md_path,
                    "json_path": json_path,
                    "images_dir": images_dir if (is_complex and os.path.exists(images_dir)) else None,
                    "is_complex": is_complex
                }


            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
                raise RuntimeError(f"LiteParse CLI failed: {error_msg}")
        else:
            # Fallback for non-PDFs
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
            
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(text_content)
                
            return {
                "markdown_path": md_path,
                "json_path": None,
                "images_dir": None
            }

    def _run_docling(self, file_path: str) -> Dict[str, Any]:
        # TODO: Implement advanced structure-aware parsing
        return {"text": f"Parsed content of {file_path} using Docling", "metadata": {}}
        
    def _run_llamaparse(self, file_path: str) -> Dict[str, Any]:
        # TODO: Implement cloud API parsing
        return {"text": f"Parsed content of {file_path} using LlamaParse", "metadata": {}}

parsing_service = DocumentParsingService()
