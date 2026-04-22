# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

- Number of products used

We already imported 20k products and 200k reviews in milestone 2. We did not match products with reviews; instead, we left approximately half of the products without any reviews, while other products could have a range of reviews assigned. We thought this reflected differences in popularity among Amazon products online. No changes to sampling were made.

### LLM Experiment

- Models compared (name, family, size)
- Results and discussions
  - Prompt used (copy it here)
  - Results
- Which model you chose and why

## Step 2: Additional Feature

### Quantitative Evaluation

We used a custom script-based approach rather than a RADAS workflow because the goal here was to evaluate the information retrieval component of the RAG system without introducing an LLM into the labelling pipeline itself. Codex was used to help brainstorm the purpose of this script and draft it, but was reviewed and appraised.

#### Building Retrieval Labels

The queries used to evaluate the app in Milestone 2 were exported to `retrieval_queries.json` and reused as candidate queries in our quantitative evaluation. `build_retrieval_labels.py` loads the evaluation queries from the JSON file, runs BM25, semantic, and hybrid retrieval for each query, and pools the returned candidates into a single CSV for review. "Pooling" here means taking the union of products returned by the different retrievers for the same query. Each retriever returns its top 10 products. For each `parent_asin` that appears in any of the result lists, the script creates a row if that product has not been seen yet for the query, or updates the existing row if it has. The row tracks which retrievers returned the product, stores each retriever's rank and score in separate fields such as `bm25_rank`, `semantic_rank`, `bm25_score`, and `semantic_score`, and records `best_rank` as the highest placement the product achieved across the retrievers, meaning the smallest rank number it received.

The output file, `retrieval_labels.csv`, is intended for human validation of relevance. To streamline this process, given the large number of results to be evaluated, ChatGPT-5 was first used to provide an initial screen, filling in the `relevance` column to categorize relevance to the query (`0` for not relevant, `1` for relevant). However, this process was only semi-supervised, as a spot check was manually performed using human judgment on the `relevance` column to assess quality and consistency. Relevance decisions were based on what a user would reasonably expect to retrieve for a query, not on whether a product was an exact lexical match. Some queries are broad, so relevance judgments were applied somewhat more leniently in those cases.

### Evaluation Retreival

Codex was used to brainstorm and draft `evaluate_retrieval.py`. This script evaluates the retrieval component of the system using labeled relevance judgments from `retrieval_labels.csv`. It does not compare BM25, semantic, and hybrid systems by raw scores, since those scores are not necessarily on the same scale. Instead, it evaluates the ranked lists returned by each retriever against the human relevance labels. For each labeled query, the script retrieves the top-k (5) results from BM25, semantic, and hybrid retrieval, then computes precision@k, recall@k, and reciprocal rank.

Mean precision@k and mean recall@k summarize each retriever's performance across the full query set at the chosen cutoff. MRR summarizes, on average, how early the first relevant item appears in the ranking. The output is written as a JSON file containing both summary metrics and per-query details, allowing results to be inspected directly. In the terminal, the summary across queries is also presented.

| Retriever | Queries | Mean Precision@5 | Mean Recall@5 | MRR |
|-----------|---------|------------------|---------------|-----|
| BM25 | 10 | 0.5400 | 0.3324 | 0.7250 |
| Semantic | 10 | 0.6800 | 0.3892 | 0.8500 |
| Hybrid | 10 | 0.5800 | 0.3594 | 0.8500 |

These results suggest that the semantic retriever performs best overall, compared with BM25 and even Hybrid. It achieved the highest mean precision of 0.68, meaning that about 3.4 of the top 5 retrieved products were relevant. It also achieved the highest mean recall of 0.3892, recovering a larger share of the judged relevant products within the top 5 than the other retrieval methods. For MRR, its score was tied with Hybrid search at 0.85, indicating that relevant items usually appear early in the ranking, often in the first position.

It is surprising that Semantic outperformed Hybrid, given clear weaknesses in both unimodal approaches. For example, for "Washingmachine," which contains a typo, semantic retrieval performs better than BM25, whereas for the exact serial number of a product, BM25 performs better. This may depend on the choice of search queries, as a few were deliberately designed to target semantic interpretation, for which exact matching would perform poorly. Also, for some items such as "Stainless Steel Coffee Maker," there were many matching keywords in product titles, but correctly identifying the best result likely still required semantic interpretation.

Overall, the quantitative assessment shows modest performance

An improvement to our analysis would be to add a filter to our searches so that either the top 10 results are returned or only results that are at least 80% of the top result's score are returned. Through the quantitative analysis, it became apparent that `Magic Bullet` has only one truly relevant product, yet nine other products are returned even though their scores are much worse. This may add noise to the interpretation of the quantitative measurements across retrieval modes.

## Step 3: Improve Documentation and Code Quality

### Documentation Update

Summary of `README` improvements:

- Updated the mermaid file structure based on the additional feature
- Updated the makefile to include controls for the new feature
- Added document in the custom Quantitative Evaluation feature

### Code Quality Changes

- Summary of cleanups

## Step 4: Cloud Deployment Plan

(See Step 4 above for required subsections)
