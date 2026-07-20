"""
IngestionGuard — validates uploaded files before they are saved to disk.

Pure Python implementation (no external guardrails library required).

Checks:
  1. Extension whitelist
  2. File size limit (100 MB)
  3. Filename sanitization (path traversal, null bytes, unsafe chars)
"""

import re
from typing import Dict, Any

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".txt",
    ".png", ".jpg", ".jpeg",
    ".mp4", ".avi", ".mov",
}

_SAFE_FILENAME_RE = re.compile(r"^[\w\-. ()]+$")


class IngestionGuard:
    """Call validate_file() before saving any uploaded file to disk."""

    @staticmethod
    def validate_file(filename: str, size_bytes: int, content_type: str = None) -> Dict[str, Any]:
        """
        Returns {"ok": True} on success.
        Returns {"ok": False, "reason": str} on failure.
        """
        if content_type and content_type == "application/x-msdownload":
            return {"ok": False, "reason": "Executable files are not allowed."}
        # Null byte injection
        if "\x00" in filename:
            return {"ok": False, "reason": "Filename contains null byte."}

        # Path traversal
        if ".." in filename:
            return {"ok": False, "reason": "Filename contains path traversal sequence '..'."}

        # Absolute path
        if filename.startswith(("/", "\\")):
            return {"ok": False, "reason": "Filename must not be an absolute path."}

        # Safe characters only
        if not _SAFE_FILENAME_RE.match(filename):
            return {
                "ok": False,
                "reason": f"Filename '{filename}' contains disallowed characters.",
            }

        # Extension whitelist
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in ALLOWED_EXTENSIONS:
            return {
                "ok": False,
                "reason": (
                    f"File type '{suffix}' is not supported. "
                    f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                ),
            }

        # Size limit
        if size_bytes > MAX_FILE_SIZE_BYTES:
            mb = size_bytes / (1024 * 1024)
            return {
                "ok": False,
                "reason": f"File size {mb:.1f} MB exceeds the 100 MB limit.",
            }

        return {"ok": True}
