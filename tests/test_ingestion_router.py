from pathlib import Path
from backend.app.ingestion.loaders.file_router import FileRouter


def test_router_txt():
    router = FileRouter()
    loader = router.route(Path("sample.txt"))
    assert loader is not None


def test_router_pdf():
    router = FileRouter()
    loader = router.route(Path("sample.pdf"))
    assert loader is not None