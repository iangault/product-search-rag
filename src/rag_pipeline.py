# RAG pipeline setup

from pathlib import Path
import pandas as pd

from src.semantic import SemanticSearch
from src.prompts import build_prompt

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Processed Data
data_path = root_dir / "data" / "processed" / "processed.parquet"
# Path to the SemanticSearch index
index_path_sem = root_dir / "data" / "processed" / "semantic.index"

cols = ["product_title",
        "features",
        "description",
        "categories",
        "details",
        "review_text", # flattened aggregated review title and text
        ]

