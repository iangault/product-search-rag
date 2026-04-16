# RAG pipeline setup

from pathlib import Path
import pandas as pd
import os
import sys

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils import build_documents
from src.prompts import build_prompt

# Processed Data
data_path = root_dir / "data" / "processed" / "processed.parquet"
rag_index_dir = root_dir / "data" / "processed" / "rag_faiss"

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


def get_embeddings():
    """Create the embedding model used to build the FAISS retriever."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
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
    ### Text Splitter ###
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap =100)
    
    split_docs = text_splitter.split_documents(document)
    return split_docs

def build_retriever(embeddings):
    """Buils a FAISS semantic retriever over the split docs"""

    split_docs = load_chunked()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def build_vectorstore(embeddings):
    """Build the FAISS vectorstore over chunked documents."""
    split_docs = load_chunked()
    return FAISS.from_documents(split_docs, embeddings)


def get_rag_index_version():
    """Return a version token that changes when the saved RAG index changes."""
    index_file = rag_index_dir / "index.faiss"
    pickle_file = rag_index_dir / "index.pkl"
    if not index_file.exists() or not pickle_file.exists():
        return None
    return (index_file.stat().st_mtime, pickle_file.stat().st_mtime)


def save_vectorstore(vectorstore):
    """Persist the FAISS vectorstore to disk."""
    rag_index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(rag_index_dir))


def build_and_save_retriever(embeddings):
    """Build the RAG vectorstore and save it for later app use."""
    vectorstore = build_vectorstore(embeddings)
    save_vectorstore(vectorstore)
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def load_retriever(embeddings):
    """Load the persisted FAISS retriever from disk."""
    index_file = rag_index_dir / "index.faiss"
    pickle_file = rag_index_dir / "index.pkl"

    if not index_file.exists() or not pickle_file.exists():
        raise FileNotFoundError(
            f"RAG index not found in {rag_index_dir}. "
            "Run src/build_rag.py first."
        )

    vectorstore = FAISS.load_local(
        str(rag_index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

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


def answer_query(query, retriever=None, llm=None):
    """Run retrieval, build the prompt, and return the model answer."""
    if retriever is None:
        retriever = build_retriever(get_embeddings())

    if llm is None:
        llm = get_llm()

    retrieved_docs = retriever.invoke(query)
    context = relevant_text(retrieved_docs)
    prompt = build_prompt(query, context)
    response = llm.invoke(prompt)

    return response.content, retrieved_docs
