
# Hybrid RAG: Changes and Qualitative Evaluation

## Model Choice

The model chosen for the LLM pipeline is `qwen/qwen3-32b`, which is accessed through the Groq API. Qwen3 was chosen over the other suggested models (Phi-4, Mistral, Llama, etc) because it produced coherent answers grounded in retrieved product context. The Groq API was used because it offers a free inference tier, meaning the model runs on Groq's servers instead of locally. This is important since the 32B parameter model would require more GPU memory than a typical laptop would have available. The 32B size was the available Qwen3 option on Groq's free tier, and considering that larger models generally follow complex prompt constraints more reliably, it was a suitable choice for a product search assistant.

## Changes

### CUDA/CPU

Implemented an automatic device fallback so semantic embedding and RAG indexing/search run on cuda when available, but otherwise fall back to CPU. This allows for both non-CUDA laptops and GPU machines. There was a lot of debugging here, trying to find the source of the error and trial and error with codex. Was able to deduce that the semantic search with one big document of metadata and available reviews was causing a crash. We needed to change the semantic search to include only metadata from the document, and keep the reviews for the RAG/LLM part that can chunk and interpret important parts. We also needed to add batching while building semantic search embeddings.

### Duplicates

We addressed duplicates in reviews by concatenating the reviews into a list of strings per product. We also chose to display the candidate title and candidate review with the highest helpful_vote count in the app. We import 20k products and 200k reviews. Around half of the products have reviews. We thought this was similar to real-world searches, where some products have many reviews while others have none. Updating the app was in 267e81a.

We changed import_process.py to build one row per parent_asin, retain titled products from metadata even when they have no reviews, and add aggregated fields used by the app, including derived_avg_rating, n_reviews, candidate_review_title, candidate_review_text, and the selected review’s candidate_review_helpful_vote.

### App

HTML tags: We updated the review rendering so HTML tags are cleanly displaced. We also added space between reviews in the hybrid-RAG's "Show product review context" section, which displays product-based reviews. For RAG, we kept it as "Show retrieved chunks."

We updated app.py to read from processed.parquet instead of the old merged review-level file, removed the duplicate-result workaround, displays the candidate review snippet, shows average rating and review count, includes helpful votes for the displayed review, and handles products with no reviews without crashing.

### parent_asin

BM25 documents are generated from that same dataframe with one combined document per row, and the BM25 IDs come from the same dataframe’s parent_asin column. That means each BM25 document and each ID stay in a 1-to-1 positional mapping.

The semantic retrieval path follows the same pattern: it derives parent_asin IDs and searchable documents from the same dataframe, so both retrieval systems use stable product IDs aligned to one document per product row.

## Qualitative Evaluation of Hybrid RAG

We ran the same 10 queries from the BM25 vs. semantic evaluation (`01_bm25_vs_semantic_eval.md`) through the hybrid RAG pipeline. Each answer is rated on the following three dimensions below:

- **Accuracy**: Is the answer factually correct based on the reviews? Yes/No
- **Completeness**: Does the answer address all aspects of the question? Yes/No
- **Fluency**: Is the answer natural, clear, and easy to read? Yes/No

### Query Results

| Query | Accuracy | Completeness | Fluency | Notes |
|-------|----------|--------------|---------|-------|
| Washing machine | No | No | Yes | Returned obscure mini washers with 1 review instead of well known washers |
| Stainless steel coffee maker | Yes | No | Yes | Found pour over drippers/filters instead of drip coffee machines |
| Replacement Parts DC61-02610A | Yes | Yes | Yes | Correctly identified and found exact product |
| Magic bullet | No | No | Yes | Found replacement parts but not the Magic Bullet blender itself |
| Whirlpool | Yes | No | Yes | Found Whirlpool replacement parts but not any Whirlpool appliances |
| Washingmachine | Yes | No | Yes | Found results despite typo (shows robustness) but returned low rated products |
| kitchen device to heat food quickly | Yes | Yes | Yes | Found relevant heating appliances with review support |
| Something to cook pizza in | Yes | Yes | Yes | Correctly identified pizza ovens |
| Best appliances for a small apartment | Yes | Yes | Yes | Found relevant compact appliances |
| Aquamarine appliance to make bread crispy | No | No | Yes | Found gas range, admits cannot confirm it colour is "Aquamarine" |

### Key Observations

Overall, the hybrid RAG pipeline performs well on semantic and medium difficulty queries but poorly on brand name lookups and general keyword queries. Fluency was consistently good across all 10 queries, the LLM produced human readable answers in every case. Accuracy and completeness dropped when the hybrid retriever returned with poor retrieved results since that was all the LLM could use to provide an answer.

Strongest results came from semantic queries such as "Something to cook pizza in" and "kitchen device to heat food quickly", where the combination of BM25 and Semantic search together returned relevant specific products with good answer quality. The weakest results came from simple keyword queries like "Washing machine" where the hybrid pipeline resulted in obscure products that have very little reviews. This is most likely due to the dataset - it contains many low reviewed niche products that matched the hybrid search criteria and was unable to distinguish them from the more popular well known products or brands.

### Limitations

1. **Retrieval does not account for product quality or popularity**: The hybrid retriever only ranks products by how well they match the query, regardless of their review count. This means that a niche product with 1 review can rank above a well-known product with hundreds of reviews just because it matches the query well.

2. **Brand name queries return accessories instead of main products**: Queries like "Magic bullet" and "Whirlpool" retrieved accessories and replacement parts instead of the brand's main products. Hybrid retriever does not distinguish between a product made by a brand and a product that is compatible with or named after the brand.

### Suggestions for Future Improvements

1. **Add minimum review count filter**: Pre-filtering out products with very few reviews before retrieval could improve result quality for general queries. This will prevent obscure niche products from dominating the results just because they match the query well.

2. **Experiment with retrieval weight tuning**: The current equal weight of 0.5/0.5 weighting was chosen as a reasonable default but was not tested for performance. Running the same 10 queries from above with different weight combinations and comparing the answer quality could reveal if a different weight combination would improve results for this dataset.