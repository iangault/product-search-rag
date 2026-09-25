# Title: RAG Pipeline
# Purpose: Build retrieval context and generate grounded answers
# for the app's RAG modes.
# Date: 2026-04-22
# NOTE: codex was used for brainstorming, debugging,
# and making code more efficient

from pathlib import Path
import pandas as pd
import sys
import torch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.utils import build_documents
from src.prompts import build_prompt
from src.bm25 import BM25Search
from src.semantic import SemanticSearch
from src.hybrid import HybridRetriever

# Processed Data
data_path = root_dir / "data" / "processed" / "processed.parquet"
rag_index_dir = root_dir / "data" / "processed" / "rag_faiss"

# Columns to be made into a document
cols = [
    "product_title",
    "features",
    "description",
    "categories",
    "details",
    # flattened aggregated review title and text
    "review_text",
]

# Number of documents to be returned
RETRIEVER_K = 5

def get_llm():
    """Create the Groq client lazily so module import does not require API access."""
    return ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.2,  # adjust higher for more creative responses,
        max_tokens=1000,
    )

def clean_response(text):
    """Clean LLM response by removing qwen3 chain-of-thought <think> block"""
    if "<think>" in text and "</think>" in text:
        end = text.find("</think>")
        text = text[end + len("</think>"):].strip()
    return text

# Need to call model again because we are rerunning the embedding with chunking
def get_embeddings():
    """Create the embedding model used to build the FAISS retriever."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": get_torch_device()},
        show_progress=True,
    )


def get_torch_device():
    """Return the safest available device for embedding generation."""
    return "cuda" if torch.cuda.is_available() else "cpu"

# Chunk document into smaller pieces
# Documents are large based on milestone2_exploration.ipynb
# Better to feed into the LLM
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

# Store embedding for the chunks
# Embedding allows us to search for semantically similar
# Documents to queries
def build_vectorstore(embeddings):
    """Build the FAISS vectorstore over chunked documents."""
    split_docs = load_chunked()
    return FAISS.from_documents(split_docs, embeddings)

# Vectorstore already built
# Turns the vectorstore into an object callable with a quiery
# Search interface
def as_retriever(vectorstore):
    """Return the standard retriever config used across the RAG pipeline."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


def get_rag_index_version():
    """Return a version token that changes when the saved RAG index changes."""
    index_file = rag_index_dir / "index.faiss"
    pickle_file = rag_index_dir / "index.pkl"
    if not index_file.exists() or not pickle_file.exists():
        return None
    return (index_file.stat().st_mtime, pickle_file.stat().st_mtime)


def save_vectorstore(vectorstore):
    """Save the FAISS vectorstore."""
    rag_index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(rag_index_dir))


def load_retriever(embeddings):
    """Load the saved FAISS retriever from disk."""
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
        # unpickling, but know source
        allow_dangerous_deserialization=True,
    )

    return as_retriever(vectorstore)


# Context builder
def relevant_text(retrieved_docs):
    """Formats retrieved docs to pass to LLM prompt"""

    blocks = []
    for doc in retrieved_docs:
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
        raise ValueError("`retriever` is required. Load it with `load_retriever()` first.")

    if llm is None:
        llm = get_llm()

    retrieved_docs = retriever.invoke(query)
    context = relevant_text(retrieved_docs)
    prompt = build_prompt(query, context)
    response = llm.invoke(prompt)

    return clean_response(response.content), retrieved_docs

# Hybrid RAG Pipeline

# Paths for BM25 and Semantic search indices
bm25_index_path = root_dir / "data" / "processed" / "bm25_index.pkl"
semantic_index_path = root_dir / "data" / "processed" / "semantic.index"

def load_hybrid_retriever():
    """Load saved BM25 and Semantic indices from disk and return HybridRetriever instance."""
    bm25 = BM25Search.load(bm25_index_path)
    semantic = SemanticSearch.load(semantic_index_path)
    return HybridRetriever(bm25, semantic)

def relevant_text_hybrid(results, df_by_asin):
    """Format hybrid retriever results into context block for LLM prompt.

    Parameters
    ----------
    results : list of tuple
        Each tuple is (product_id, rrf_score) from HybridRetriever.
    df_by_asin : pd.DataFrame
        DataFrame indexed by parent_asin.

    Returns
    -------
    str
        Formatted context string to pass to LLM prompt.

    """

    blocks = []

    for rank, (asin, score) in enumerate(results, start=1):
        if asin not in df_by_asin.index:
            continue

        row = df_by_asin.loc[asin]

        review_text = row.get("review_text", "N/A")
        if isinstance(review_text, str) and len(review_text) > 500:
            review_text = review_text[:500] + "..."

        blocks.append(
            f"Product Title: {row.get('product_title', 'N/A')}\n"
            f"ASIN: {row.get('parent_asin', 'N/A')}\n"
            f"Rating: {row.get('derived_avg_rating', 'N/A')}\n"
            f"Review Count: {row.get('n_reviews', 'N/A')}\n"
            f"Helpful Votes: {row.get('candidate_review_helpful_vote', 'N/A')}\n"
            f"Price: {row.get('price', 'N/A')}\n"
            f"Retrieved Text:\n{review_text}"
        )

    return "\n\n".join(blocks)

def answer_query_hybrid(query, df_by_asin, hybrid_retriever=None, llm=None):
    """Run hybrid retrieval, build prompt, return model answer.

    Parameters
    ----------
    query : str
        User's search query.
    df_by_asin : pd.DataFrame
        DataFrame indexed by parent_asin.
    hybrid_retriever : HybridRetriever, optional
        Pre-loaded HybridRetriever. If None, load from disk.
    llm : ChatGroq, optional
        Pre-loaded LLm. If None, create one with get_llm().

    Returns
    -------
    tuple of (str, list)
        LLM response string and retriever results list.

    """

    if hybrid_retriever is None:
        hybrid_retriever = load_hybrid_retriever()

    if llm is None:
        llm = get_llm()

    results = hybrid_retriever.retrieve(query, top_k=RETRIEVER_K)
    context = relevant_text_hybrid(results, df_by_asin)
    prompt = build_prompt(query, context)
    response = llm.invoke(prompt)

    return clean_response(response.content), results
