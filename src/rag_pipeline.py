# RAG pipeline setup

from pathlib import Path
import pandas as pd

from src.semantic import SemanticSearch
from src.prompts import build_prompt
from src.utils import build_documents

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

### Generator ###


### Text Splitter ###
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap =100
)

### Functions ###
def load_chunked():
    """Load in parquet file and semantic embeddings"""
    df = pd.read_parquet(data_path).reset_index(drop=True)
    # Rebuild documents, but with chunkings, now that
    # there is an LLM in the pipeline
    sem_docs = build_documents(df, cols)

    # Change list of strings to LangChanin Document
    document = []
    for i, text in enumerate(sem_docs):
        row = df.iloc[i]
        document.append(
            Document(
                page_content=text,
                metadata={
                    "parent_asin": row["parent_asin"],
                    "product_title": row.get("product_title", "")
                }
            )
        )
    
    # Split the document for better LLM performance
    split_docs = text_splitter.split_documents(document)
    return split_docs






    retriever = SemanticSearch.load(index_path_sem)
    return by_asin, retriever