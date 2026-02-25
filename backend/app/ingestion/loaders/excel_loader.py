from pathlib import Path
import pandas as pd
from .base_loader import BaseLoader


class ExcelLoader(BaseLoader):

    def load(self, path: Path) -> str:
        df = pd.read_excel(path, engine="openpyxl")
        return df.to_string()