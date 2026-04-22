# Title: Build RAG Index
# Purpose: Build and save the vector store used by the RAG pipeline in the app.
# Date: 2026-04-22
# NOTE: codex was used to extract functions from rag_pipeline

import sys
from pathlib import Path

from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Import functions to run RAG pipeline
from src.rag_pipeline import (
    build_vectorstore,
    data_path,
    get_embeddings,
    rag_index_dir,
    save_vectorstore,
)

load_dotenv()


def main():
    """Build and save the RAG retriever artifacts for app use."""
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run src/import_process.py first.")
        return

    print("Building RAG retriever...")
    vectorstore = build_vectorstore(get_embeddings())
    save_vectorstore(vectorstore)
    print(f"Saved RAG retriever to {rag_index_dir}")


if __name__ == "__main__":
    main()
