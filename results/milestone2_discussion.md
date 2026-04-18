
# Milestone 2

## Model Choice

The model chosen for the LLM pipeline is `qwen/qwen3-32b`, which is accessed through the Groq API. Qwen3 was chosen over the other suggested models (Phi-4, Mistral, Llama, etc) because it produced coherent answers grounded in retrieved product context. The Groq API was used because it offers a free inference tier, meaning the model runs on Groq's servers instead of locally. This is important since the 32B parameter model would require more GPU memory than a typical laptop would have available. The 32B size was the available Qwen3 option on Groq's free tier, and considering that larger models generally follow complex prompt constraints more reliably, it was a suitable choice for a product search assistant.

## Changes:

It changes import_process.py to build one row per parent_asin, retain titled products from metadata even when they have no reviews, and add aggregated fields used by the app, including derived_avg_rating, n_reviews, candidate_review_title, candidate_review_text, and the selected review’s candidate_review_helpful_vote.

It also updates app.py to read from processed.parquet instead of the old merged review-level file, removes the duplicate-result workaround, displays the candidate review snippet, shows average rating and review count, includes helpful votes for the displayed review, and handles products with no reviews without crashing.

The documents/ids alignment is consistent across the src/ pipeline. The processed dataset is built at one row per parent_asin, BM25 documents are generated from that same dataframe with one combined document per row, and the BM25 IDs come from the same dataframe’s parent_asin column. That means each BM25 document and each ID stay in a 1-to-1 positional mapping.

The semantic retrieval path follows the same pattern: it derives parent_asin IDs and searchable documents from the same dataframe, so both retrieval systems use stable product IDs aligned to one document per product row.
