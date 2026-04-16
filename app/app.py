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
    return bm25, semantic

# load data
# Changed to index by parent_asin
df = load_data(get_data_version())
df_by_asin = df.set_index("parent_asin", drop=False)
bm25_search, semantic_search = load_search(get_search_version())

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

# Debugging step to make sure old caches not held
results_placeholder = st.empty()

if submitted and query:
    with results_placeholder.container():
        st.subheader(f"Top 10 Results for '{query}' using {search_mode} search")

        st.divider()

        # Both retrievers return `parent_asin`, so app display fields are
        # pulled by stable product id instead of fragile row position.
        # Was debugged with codex
        if search_mode == "BM25":
            results = bm25_search.retrieve(query, top_k=10)

        else:
            results = semantic_search.retrieve(query, top_k=10)

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
            trunc_text = review_display[:200] + "..." if len(review_display) > 200 else review_display

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
                [
                    f"###### {item_data.get('product_title', 'Unknown Title')}",
                    f"*{trunc_text}*",
                    f"**{rating_display}**",
                    f"**Review Count:** {review_count}",
                    f"**{helpful_display}**",
                    f"**Retrieval Score:** {score:.4f}",
                ]
            )
            st.markdown(card_markdown)

            st.divider()
