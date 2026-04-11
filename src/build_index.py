import sys
from pathlib import Path
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.bm25 import BM25Search

def main():
    """
    Execute script to build the BM25 index and save it to disk.
    """
    data_path = root_dir / "data" / "processed" / "merged.parquet"
    index_path = root_dir / "data" / "processed" / "bm25_index.pkl"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run EDA notebook first.")
        return
    
    # load data
    df = pd.read_parquet(data_path)
    products = df["product_title"].tolist()
    engine = BM25Search(products)
    engine.save(index_path)

if __name__ == "__main__":
    main()