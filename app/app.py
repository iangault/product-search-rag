# adapted from Streamlit "create your first app" tutorial
# https://docs.streamlit.io/library/get-started/create-an-app
# Debugged using codex

import sys
from pathlib import Path
import streamlit as st
import pandas as pd


# get proj root dir and add to path so can import from src
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir)) # tells Python to look in root dir for imports

from src.bm25 import BM25Search
from src.semantic import SemanticSearch
from src.hybrid import HybridRetriever
from src.rag_pipeline import (
    answer_query,
    answer_query_hybrid,
    get_embeddings,
    get_llm,
    get_rag_index_version,
    load_retriever,
    load_hybrid_retriever,
)

from dotenv import load_dotenv
load_dotenv()


st.title("Amazon Appliances Search 🛒", text_alignment="left")

# Processed Data
data_path = root_dir / "data" / "processed" / "processed.parquet"
# Path to the BM25 index
index_path_bm = root_dir / "data" / "processed" / "bm25_index.pkl"
# Path to the SemanticSearch index
index_path_sem = root_dir / "data" / "processed" / "semantic.index"


# Debugging step from codex to ensure caches are correct when loading
def get_data_version() -> float:
    """Invalidate cached parquet loads when the processed file changes."""
    return data_path.stat().st_mtime

def get_search_version() -> tuple[float, float, float]:
    """Invalidate cached search engines when any index artifact changes."""
    semantic_ids_path = SemanticSearch.ids_path(index_path_sem)
    return (
        index_path_bm.stat().st_mtime,
        index_path_sem.stat().st_mtime,
        semantic_ids_path.stat().st_mtime,
    )


def get_rag_version() -> tuple[float, object]:
    """Invalidate cached RAG resources when data or saved RAG index changes."""
    return (data_path.stat().st_mtime, get_rag_index_version())

# cache the data
@st.cache_data
def load_data(_version: float):
    # Keep the parquet load out of Streamlit's rerun loop so
    # searches do not repeatedly hit disk.
    return pd.read_parquet(data_path)

@st.cache_resource
def load_search(_version: tuple[float, float, float]):
    # Search indexes are heavier objects than the dataframe, so
    # cache them as resources and reuse across interactions.
    bm25 = BM25Search.load(index_path_bm)
    semantic = SemanticSearch.load(index_path_sem)
    hybrid = HybridRetriever(bm25, semantic)
    return bm25, semantic, hybrid


@st.cache_resource
def load_rag(_version: tuple[float, object]):
    # Keep loaded RAG resources alive across reruns.
    retriever = load_retriever(get_embeddings())
    llm = get_llm()
    return retriever, llm

# load data
# Changed to index by parent_asin
df = load_data(get_data_version())
df_by_asin = df.set_index("parent_asin", drop=False)
bm25_search, semantic_search, hybrid_search = load_search(get_search_version())
rag_retriever = None
rag_llm = None

if get_rag_index_version() is not None:
    rag_retriever, rag_llm = load_rag(get_rag_version())

hybrid_retriever = load_hybrid_retriever()
hybrid_llm = get_llm()

st.divider()

# search mode selector - radio buttons
search_mode = st.radio(
    "Select Search Mode:",
    ["BM25", "Semantic", "Hybrid", "RAG", "Hybrid RAG"],
    horizontal=True,
    key = "search_mode",
)

with st.form("search_form"):
    query = st.text_input("Enter search query:",
                          placeholder="eg stainless steel coffee maker",
                          key = "search_query")
    submitted = st.form_submit_button("Search")

# Debugging step to make sure old caches not held
results_placeholder = st.empty()

if submitted and query:
    with results_placeholder.container():
        st.subheader(f"Top Results for '{query}' using {search_mode} search")

        st.divider()

        # Both retrievers return `parent_asin`, so app display fields are
        # pulled by stable product id instead of fragile row position.
        # Was debugged with codex
        if search_mode == "BM25":
            results = bm25_search.retrieve(query, top_k=10)
            rag_answer = None
            retrieved_docs = []

        elif search_mode == "Semantic":
            results = semantic_search.retrieve(query, top_k=10)
            rag_answer = None
            retrieved_docs = []

        elif search_mode == "Hybrid":
            results = hybrid_search.retrieve(query, top_k=10)
            rag_answer = None
            retrieved_docs = []
        
        elif search_mode == "Hybrid RAG":
            rag_answer, results = answer_query_hybrid(
                query,
                df_by_asin=df_by_asin,
                hybrid_retriever=hybrid_retriever,
                llm=hybrid_llm,
            )

            st.markdown("### Answer")
            st.markdown(rag_answer)
            st.divider()
            st.markdown("### Supporting Sources")

        else:
            if rag_retriever is None or rag_llm is None:
                st.error("RAG index not found. Run `python src/build_rag.py` or `make build-rag` first.")
                st.stop()

            rag_answer, retrieved_docs = answer_query(
                query,
                retriever=rag_retriever,
                llm=rag_llm,
            )

            st.markdown("### Answer")
            st.markdown(rag_answer)
            st.divider()
            st.markdown("### Supporting Sources")

            seen_asins = set()
            results = []
            for doc in retrieved_docs:
                parent_asin = doc.metadata.get("parent_asin")
                if not parent_asin or parent_asin in seen_asins:
                    continue
                seen_asins.add(parent_asin)
                results.append((parent_asin, None))

        # display results
        for parent_asin, score in results:
            if parent_asin not in df_by_asin.index:
                continue

            item_data = df_by_asin.loc[parent_asin].to_dict()

            # Candidate review to be displayed based on pre-processing step
            review_title = item_data.get("candidate_review_title")
            review_text = item_data.get("candidate_review_text")

            # Rebuild the candidate review snippet from whichever title/text
            # pieces are available so missing fields do not break display.
            if pd.notna(review_title) and str(review_title).strip() and pd.notna(review_text) and str(review_text).strip():
                review_display = f"{review_title}: {review_text}"
            elif pd.notna(review_title) and str(review_title).strip():
                review_display = str(review_title)
            elif pd.notna(review_text) and str(review_text).strip():
                review_display = str(review_text)
            else:
                review_display = "No candidate review available."

            # Keep result cards compact even when the stored review text is long.
            trunc_text = review_display[:500] + "..." if len(review_display) > 500 else review_display

            try:
                avg_rating = float(item_data.get("derived_avg_rating"))
            except (ValueError, TypeError):
                avg_rating = float("nan")

            review_count = item_data.get("n_reviews", 0)
            if pd.isna(review_count):
                review_count = 0
            review_count = int(review_count)

            # Change in approach for star rating based on codex debugging
            show_stars = (not pd.isna(avg_rating)) and (review_count != 0)

            if show_stars:
                filled_stars = max(0, min(5, round(avg_rating)))
                stars = "★" * filled_stars + "☆" * (5 - filled_stars)
                rating_display = f"Average Rating: {stars} ({avg_rating:.1f}/5.0)"
            else:
                rating_display = "Average Rating: N/A"

            helpful_votes = item_data.get("candidate_review_helpful_vote")
            if review_count == 0 or pd.isna(helpful_votes):
                helpful_display = "Helpful Votes: N/A"
            else:
                helpful_display = f"Helpful Votes: {int(helpful_votes)}"

            # Debugging by codex to ensure correct display of results
            # Previously, different cards were not updating with search
            # Has now been fixed
            card_markdown = "\n\n".join(
                [line for line in [
                    f"###### {item_data.get('product_title', 'Unknown Title')}",
                    f"*{trunc_text}*",
                    f"**{rating_display}**",
                    f"**Review Count:** {review_count}",
                    f"**{helpful_display}**",
                    f"**Retrieval Score:** {score:.4f}" if score is not None else None,
                ] if line is not None]
            )
            st.markdown(card_markdown)

            if search_mode == "RAG":
                matching_doc = next(
                    (doc for doc in retrieved_docs
                     if doc.metadata.get("parent_asin") == parent_asin),
                    None,
                )
                if matching_doc is not None:
                    with st.expander("Show retrieved context"):
                        st.text(matching_doc.page_content)

            elif search_mode == "Hybrid RAG":
                with st.expander("Show retrieved context"):
                    row = df_by_asin.loc[parent_asin] if parent_asin in df_by_asin.index else None
                    if row is not None:
                        review_text = row.get("review_text", "N/A")
                        if pd.isna(review_text) or not str(review_text, str):
                            st.text("No review text available for this product.")
                        else:
                            if len(review_text) > 1000:
                                review_text = review_text[:1000] + "..."
                                st.text(review_text)
            st.divider()