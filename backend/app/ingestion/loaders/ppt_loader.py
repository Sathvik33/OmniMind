from pathlib import Path
from pptx import Presentation
from .base_loader import BaseLoader


class PPTLoader(BaseLoader):

    def load(self, path: Path) -> str:
        prs = Presentation(str(path))
        text = []

        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)

        return "\n".join(text)