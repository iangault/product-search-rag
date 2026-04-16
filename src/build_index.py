# Script to build BM25 and Semantic Search indices

import sys
from pathlib import Path
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.bm25 import BM25Search
from src.semantic import SemanticSearch
from src.utils import build_documents

from dotenv import load_dotenv
load_dotenv()

def main():
    """
    Execute script to build the BM25 and Semantic Search
    indices and save them to disk.
    """
    # Processed Data
    data_path = root_dir / "data" / "processed" / "processed.parquet"
    # Path to the BM25 index
    index_path_bm = root_dir / "data" / "processed" / "bm25_index.pkl"
    # Path to the SemanticSearch index
    index_path_sem = root_dir / "data" / "processed" / "semantic.index"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run src/import_process.py first.")
        return

    # load data
    print("Loading parquet...")
    df = pd.read_parquet(data_path)

    # BM25 Save
    # Columns as input into both search engines
    cols = ["product_title",
            "features",
            "description",
            "categories",
            "details",
            "review_text", # flattened aggregated review title and text
            ]

    bm25_docs = build_documents(df, cols)
    doc_ids = df["parent_asin"].tolist()
    print("Building BM25 index...")
    engine_bm = BM25Search(bm25_docs, ids=doc_ids)
    engine_bm.save(index_path_bm)

    # SemanticSearch
    print("Building semantic index...")

    engine_sem = SemanticSearch(df, cols)
    engine_sem.save(index_path_sem)


if __name__ == "__main__":
    main()
