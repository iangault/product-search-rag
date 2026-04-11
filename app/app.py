# adapted from Streamlit "create your first app" tutorial
# https://docs.streamlit.io/library/get-started/create-an-app

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# get proj root dir and add to path so can import from src
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir)) # tells Python to look in root dir for imports
from src.bm25 import BM25Search
#from src.semantic import SemanticSearch 

st.title("Amazon Appliances Search 🛒", text_alignment="left")

DATA_PATH = root_dir / "data" / "processed" / "merged.parquet"
INDEX_PATH = root_dir / "data" / "processed" / "bm25_index.pkl"

# cache the data
@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

@st.cache_resource
def load_search():
    bm25 = BM25Search.load(INDEX_PATH)
    #semantic = SemanticSearch()
    return bm25#, semantic

# load data
df = load_data()
bm25_search = load_search() #bm25_search, semantic_search = load_search()

st.divider()

# search mode selector - radio buttons
search_mode = st.radio(
    "Select Search Mode:",
    ["BM25"], #["BM25", "Semantic"],
    horizontal=True
)

# query input - text box
query = st.text_input("Enter search query:", placeholder="eg stainless steel coffee maker")

if query:
    st.subheader(f"Top 3 Results for '{query}' using {search_mode} search")

    st.divider()

    if search_mode == "BM25":
        results = bm25_search.retrieve(query, top_k=3)

    #else:
        # results = semantic_search.retrieve(query, top_k=3)
    
    # display results
    for title, score in results:
        # get row in df that matches title
        matched_row = df[df["product_title"] == title]

        st.markdown(f"###### {title}")

        if not matched_row.empty:
            item_data = matched_row.iloc[0].to_dict()

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
            
        else:
            # if no match found
            st.warning("Product details not found in dataframe.")
            st.write(f"**Retrieval Score:** {score:.4f}")

        st.divider()