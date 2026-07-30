from typing import Dict, Any
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.app.db.database import SessionLocal
from backend.app.db.models import AIModelRegistry, ModelCategory

logger = logging.getLogger(__name__)

# Formats LiteParse can convert (Office needs LibreOffice on PATH).
_LITEPARSE_EXTS = {
    ".pdf",
    ".doc", ".docx", ".docm", ".odt", ".rtf",
    ".ppt", ".pptx", ".pptm", ".odp",
    ".xls", ".xlsx", ".xlsm", ".ods", ".csv", ".tsv",
}

# Plain text / markdown — no LiteParse needed; write markdown directly.
_TEXT_EXTS = {".txt", ".md", ".markdown", ".rst"}

# Local loaders used only when LiteParse conversion fails (e.g. LibreOffice missing).
_OFFICE_FALLBACK_EXTS = {".txt", ".docx", ".pptx", ".xlsx", ".xls"}


class DocumentParsingService:
    """
    Unified document → markdown parsing for the RAG ingest path.

    Recommended workflow for ALL document modalities:
      file → LiteParse (or text loader) → markdown → hierarchical chunk → embed

    LiteParse converts Office docs to PDF via LibreOffice, then emits structured
    markdown (headings, tables, lists) which the chunker/embedder already expect.
    """

    def __init__(self):
        self._load_active_parser()

    def _load_active_parser(self):
        self.parser_model = None
        try:
            db = SessionLocal()
            try:
                self.parser_model = db.query(AIModelRegistry).filter_by(
                    category=ModelCategory.PARSER,
                    is_active=1,
                ).first()
            finally:
                db.close()
        except Exception:
            pass

    def parse_document(self, file_path: str, original_filename: str = "") -> Dict[str, Any]:
        name = original_filename or file_path
        ext = Path(name).suffix.lower() or Path(file_path).suffix.lower()

        parser_name = (self.parser_model.name if self.parser_model else "liteparse").lower()
        if parser_name == "docling":
            return self._run_docling(file_path)
        if parser_name == "llamaparse":
            return self._run_llamaparse(file_path)

        if ext in _TEXT_EXTS:
            return self._run_text_to_markdown(file_path, original_filename=name)

        if ext in _LITEPARSE_EXTS:
            try:
                return self._run_liteparse(file_path, original_filename=name)
            except Exception as e:
                logger.warning(
                    "LiteParse failed for %s (%s). Falling back to local Office loader.",
                    name,
                    e,
                )
                if ext in _OFFICE_FALLBACK_EXTS:
                    return self._run_office_loader(file_path, original_filename=name, ext=ext)
                raise

        if ext in _OFFICE_FALLBACK_EXTS:
            return self._run_office_loader(file_path, original_filename=name, ext=ext)

        raise RuntimeError(f"Unsupported document type for parsing: {ext or '(none)'}")

    def _prepare_outdir(self, prefix: str) -> str:
        base_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(base_dir, exist_ok=True)
        return tempfile.mkdtemp(dir=base_dir, prefix=prefix)

    def _run_text_to_markdown(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        out_dir = self._prepare_outdir("textparse_")
        md_path = os.path.join(out_dir, "document.md")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        title = Path(original_filename).stem
        body = text if text.lstrip().startswith("#") else f"# {title}\n\n{text}"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(body or f"# {title}\n\n[Empty file]")
        return {
            "markdown_path": md_path,
            "json_path": None,
            "images_dir": None,
            "is_complex": False,
            "parser": "text",
        }

    def _run_office_loader(self, file_path: str, original_filename: str, ext: str) -> Dict[str, Any]:
        """Fallback extractors when LiteParse/LibreOffice is unavailable."""
        from backend.app.ingestion.loaders.file_router import FileRouter

        out_dir = self._prepare_outdir("officeparse_")
        md_path = os.path.join(out_dir, "document.md")
        text = FileRouter().route(Path(file_path)).load(Path(file_path))
        text = (text or "").strip()
        title = Path(original_filename).stem
        if not text:
            text = f"[No extractable text found in {original_filename}]"
        body = f"# {title}\n\n{text}"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(body)
        logger.info("Office fallback parse for %s (%s chars)", original_filename, len(body))
        return {
            "markdown_path": md_path,
            "json_path": None,
            "images_dir": None,
            "is_complex": False,
            "parser": "office_fallback",
        }

    def _run_liteparse(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        out_dir = self._prepare_outdir("liteparse_")
        md_path = os.path.join(out_dir, "document.md")
        json_path = os.path.join(out_dir, "document.json")
        images_dir = os.path.join(out_dir, "images")

        ext = Path(original_filename).suffix.lower() or Path(file_path).suffix.lower() or ".pdf"
        input_file = os.path.join(out_dir, f"input{ext}")
        shutil.copy(file_path, input_file)

        is_complex = True
        try:
            res = subprocess.run(
                ["lit", "is-complex", input_file, "--quiet"],
                capture_output=True,
            )
            is_complex = res.returncode != 0
        except Exception:
            is_complex = True

        # JSON structure (best-effort; don't fail the whole parse if this step fails)
        try:
            subprocess.run(
                ["lit", "parse", input_file, "--format", "json", "-o", json_path],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning(
                "LiteParse JSON pass failed for %s: %s",
                original_filename,
                (e.stderr or b"").decode("utf-8", errors="ignore")[:300],
            )
            json_path = None

        parse_cmd = ["lit", "parse", input_file, "--format", "markdown"]
        # Image extraction is most useful for PDFs / scanned-like docs
        if is_complex and ext == ".pdf":
            parse_cmd.extend(["--image-mode", "embed", "--image-output-dir", images_dir])
        elif ext == ".pdf":
            parse_cmd.append("--no-ocr")
        else:
            # Office → PDF conversion path: keep placeholders unless complex
            parse_cmd.extend(
                ["--image-mode", "embed", "--image-output-dir", images_dir]
                if is_complex
                else ["--image-mode", "placeholder"]
            )
        parse_cmd.extend(["-o", md_path])

        try:
            subprocess.run(parse_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
            raise RuntimeError(f"LiteParse CLI failed: {error_msg}") from e

        if not os.path.exists(md_path) or os.path.getsize(md_path) == 0:
            raise RuntimeError(f"LiteParse produced empty markdown for {original_filename}")

        has_images = os.path.isdir(images_dir) and any(
            n.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            for n in os.listdir(images_dir)
        )
        logger.info(
            "LiteParse complete for %s (complex=%s, images=%s)",
            original_filename,
            is_complex,
            has_images,
        )
        return {
            "markdown_path": md_path,
            "json_path": json_path,
            "images_dir": images_dir if has_images else None,
            "is_complex": is_complex,
            "parser": "liteparse",
        }

    def _run_docling(self, file_path: str) -> Dict[str, Any]:
        return {"text": f"Parsed content of {file_path} using Docling", "metadata": {}}

    def _run_llamaparse(self, file_path: str) -> Dict[str, Any]:
        return {"text": f"Parsed content of {file_path} using LlamaParse", "metadata": {}}


parsing_service = DocumentParsingService()
