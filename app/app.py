# adapted from Streamlit "create your first app" tutorial
# https://docs.streamlit.io/library/get-started/create-an-app

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# get proj root dir and add to path so can import from src
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir)) # tells Python to look in root dir for imports

from src.bm25 import BM25Search
from src.semantic import SemanticSearch

st.title("Amazon Appliances Search 🛒", text_alignment="left")

# Processed Data
data_path = root_dir / "data" / "processed" / "merged.parquet"
# Path to the BM25 index
index_path_bm = root_dir / "data" / "processed" / "bm25_index.pkl"
# Path to the SemanticSearch index
index_path_sem = root_dir / "data" / "processed" / "semantic.index"

# cache the data
@st.cache_data
def load_data():
    return pd.read_parquet(data_path)

@st.cache_resource
def load_search():
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

    if search_mode == "BM25":
        results = bm25_search.retrieve(query, top_k=30) # grab larger pool of results

    else:
        results = semantic_search.retrieve(query, top_k=30) # grab larger pool of results

    seen_titles = set()
    displayed_count = 0

    # display results
    for index, score in results:

        # stop at n=10 unique titles
        if displayed_count >= 10:
            break

        item_data = df.iloc[index].to_dict()
        title = item_data.get("product_title", "Unknown Title")

        # skip if we've already displayed this title
        if title in seen_titles:
            continue

        # new title, add it to seen!
        seen_titles.add(title)
        displayed_count += 1

        st.markdown(f"###### {title}")

        review_text = str(item_data.get("text", "No review text available."))
        trunc_text = review_text[:200] + "..." if len(review_text) > 200 else review_text
        st.write(f"*{trunc_text}*")
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                rating_val = int(float(item_data.get("rating", 0)))
            except (ValueError, TypeError):
                rating_val = 0
            
            stars = "★" * rating_val + "☆" * (5 - rating_val)
            st.write(f"**Rating:** {stars}")
        
        with col2:
            st.write(f"**Retrieval Score:** {score:.4f}")

        st.divider()
