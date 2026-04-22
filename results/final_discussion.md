# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

- Number of products used

We already imported 20k products and 200k reviews in milestone 2. We did not match products with reviews; instead, we let approximately half of the products not have any reviews, while other products can have a range in the number of reviews assigned. We thought this reflected differences in popularity among amazon products online. No changes in sampling was taken.

### LLM Experiment

- Models compared (name, family, size)
- Results and discussions
  - Prompt used (copy it here)
  - Results
- Which model you chose and why

## Step 2: Additional Feature (state which option you chose)

### Quantitative Evaluation

We used a custom script-based approach rather than a RADAS workflow because the goal here was to evaluate the information retrieval component of the RAG system without introducing an LLM into the labeling pipeline itself. Codex was used to help brainstorm the purpose of this script and draft it, but was reviewed and appraised.

#### Building Retrieval Labels

The queries used to evaluate the app in milestone 2 were exported into `retrieval_queries.json` to continue as candidate queries in our quantitative evaluation.`build_retrieval_labels.py` loads the evaluation queries from the JSON file, runs BM25, semantic, and hybrid retrieval for each query, and pools the returned candidates into a single CSV for review. "Pooling" here means taking the union of products returned by the different retrievers for the same query. Each retreiver returns its top 10 products. For each parent_asin that appears in any of the result lists, it creates a row if that product has not been seen yet for the query, or updates the existing row if it has. The row keeps track of which retrievers returned the product, stores each retriever’s rank and score in separate fields such as bm25_rank, semantic_rank, bm25_score, and semantic_score, and records best_rank as the highest placement the product achieved across the retrievers, meaning the smallest rank number it received.

The output file, `retrieval_labels.csv`, is intended for human validation of relevance. To streamline this process, given the large number of results to be evaluated, ChatGPT5 was first used on the file as an initial screen, filling in the column `relevance` to categorize relevance to the query (0 for not relevance, 1 for relevant). However, this only semi-supervised, as a spot check was manually done using human judgement on the `relevance` column to assess quality and consistency. Relevance decisions were based on what a user would reasonably expect to retrieve for a query, not on whether a product was an exact lexical match. Some queries are broad, so relevance judgments were applied somewhat more leniently in those cases.

### Evaluation Retreival

Codex was used to help brainstorm and draft `evaluation_retreival.py`. This script evaluates the retrieval component of the system using labeled relevance judgments collected in `retrieval_labels.csv`. This script does not compare BM25, semantic, and hybrid systems by raw score values, since those scores are not necessarily on the same scale. Instead, it evaluates the ranked lists returned by each retriever against the human labels of relevance.For each labeled query, the script retrieves the top-k (5) results from BM25, semantic, and hybrid retrieval, then computes precision@k, recall@k, and reciprocal rank.

Mean precision@k and mean recall@k summarize how well each retriever performs across the full query set at the chosen cutoff. MRR summarizes how early the first relevant item appears in the ranking on average. The output is written as a JSON file with both summary metrics and per-query details, so results can be inspected directly. In the terminal, the summary across queries is presented. 

| Retriever | Queries | Mean Precision@5 | Mean Recall@5 | MRR |
|-----------|---------|------------------|---------------|-----|
| BM25 | 10 | 0.5400 | 0.3324 | 0.7250 |
| Semantic | 10 | 0.6800 | 0.3892 | 0.8500 |
| Hybrid | 10 | 0.5800 | 0.3594 | 0.8500 |



- Description of the feature
- Key results or examples
  
## Step 3: Improve Documentation and Code Quality

### Documentation Update

- Summary of `README` improvements

### Code Quality Changes

- Summary of cleanups

## Step 4: Cloud Deployment Plan

(See Step 4 above for required subsections)
