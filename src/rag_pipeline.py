# RAG pipeline setup

from pathlib import Path
import pandas as pd
import os
import sys

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils import build_documents

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

def get_llm():
    """Create the Groq client lazily so module import does not require API access."""
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.2,  # adjust higher for more creative responses,
        max_tokens=100,
    )

### Text Splitter ###
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap =100
)

### Functions ###
def load_chunked():
    """Load in parquet file and returns chunked documents"""
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
                    "parent_asin": row.get("parent_asin", "N/A"),
                    "product_title": row.get("product_title", "N/A"),
                    "derived_avg_rating": row.get("derived_avg_rating"),
                    "n_reviews": row.get("n_reviews"),
                    "helpful_vote": row.get("helpful_vote"),
                    "price": row.get("price"),
                },
            )
        )

    # Split the document for better LLM performance
    split_docs = text_splitter.split_documents(document)
    return split_docs

def build_retriever(embeddings):
    """Buils a FAISS semantic retriever over the split docs"""

    split_docs = load_chunked()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def relevant_text(retreived_docs):
    """Formats retreived docs to pass to LLM prompt"""

    blocks = []
    for doc in retreived_docs:
        blocks.append(
            f"Product Title: {doc.metadata.get('product_title', 'N/A')}\n"
            f"ASIN: {doc.metadata.get('parent_asin', 'N/A')}\n"
            f"Rating: {doc.metadata.get('derived_avg_rating', 'N/A')}\n"
            f"Review Count: {doc.metadata.get('n_reviews', 'N/A')}\n"
            f"Helpful Votes: {doc.metadata.get('helpful_vote', 'N/A')}\n"
            f"Price: {doc.metadata.get('price', 'N/A')}\n"
            f"Retrieved Text:\n{doc.page_content}"
        )

    return "\n\n".join(blocks)
