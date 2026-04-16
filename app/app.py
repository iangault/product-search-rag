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

st.title("Amazon Appliances Search 🛒", text_alignment="left")

# Processed Data
data_path = root_dir / "data" / "processed" / "processed.parquet"
# Path to the BM25 index
index_path_bm = root_dir / "data" / "processed" / "bm25_index.pkl"
# Path to the SemanticSearch index
index_path_sem = root_dir / "data" / "processed" / "semantic.index"

# cache the data
@st.cache_data
def load_data():
    # Keep the parquet load out of Streamlit's rerun loop so
    # searches do not repeatedly hit disk.
    return pd.read_parquet(data_path)

@st.cache_resource
def load_search():
    # Search indexes are heavier objects than the dataframe, so
    # cache them as resources and reuse across interactions.
    bm25 = BM25Search.load(index_path_bm)
    semantic = SemanticSearch.load(index_path_sem)
    return bm25, semantic

# load data
df = load_data()
bm25_search, semantic_search = load_search()

st.divider()

# search mode selector - radio buttons
search_mode = st.radio(
    "Select Search Mode:",
    ["BM25", "Semantic"],
    horizontal=True,
    key = "search_mode",
)

with st.form("search_form"):
    query = st.text_input("Enter search query:",
                          placeholder="eg stainless steel coffee maker",
                          key = "search_query")
    submitted = st.form_submit_button("Search")

if submitted and query:
    st.subheader(f"Top 10 Results for '{query}' using {search_mode} search")

    st.divider()

    # Both retrievers return row indices into `df`, which lets the
    # app look up display fields from the same processed parquet.
    if search_mode == "BM25":
        results = bm25_search.retrieve(query, top_k=10)

    else:
        results = semantic_search.retrieve(query, top_k=10)

    # display results
    for index, score in results:

        item_data = df.iloc[index].to_dict()
        title = item_data.get("product_title", "Unknown Title")

        st.markdown(f"###### {title}")

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
        trunc_text = review_display[:200] + "..." if len(review_display) > 200 else review_display
        st.write(f"*{trunc_text}*")

        col1, col2 = st.columns(2)
        with col1:
            # Average Rating derived from the review dataset (more complete than meta)
            try:
                avg_rating = float(item_data.get("derived_avg_rating"))
            except (ValueError, TypeError):
                avg_rating = float("nan")

            # Number of Reviews
            review_count = item_data.get("n_reviews", 0)
            if pd.isna(review_count):
                review_count = 0
            review_count = int(review_count)

            if pd.isna(avg_rating) or review_count == 0:
                st.write("**Average Rating:** N/A")
            else:
                # Render a simple star summary while still showing the
                # underlying continuous average for precision.
                filled_stars = max(0, min(5, round(avg_rating)))
                stars = "★" * filled_stars + "☆" * (5 - filled_stars)
                st.write(f"**Average Rating:** {stars} ({avg_rating:.1f}/5.0)")

            st.write(f"**Review Count:** {review_count}")

        with col2:
            # Review filtered to max helpful votes, when available
            helpful_votes = item_data.get("candidate_review_helpful_vote")
            # Suppress helpful-vote display when there are no reviews or
            # when the selected candidate review has no vote count.
            if review_count == 0 or pd.isna(helpful_votes):
                st.write("**Helpful Votes:** N/A")
            else:
                st.write(f"**Helpful Votes:** {int(helpful_votes)}")
            st.write(f"**Retrieval Score:** {score:.4f}")

        st.divider()
