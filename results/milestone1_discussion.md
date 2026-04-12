# Milestone 1: Qualitative Evaluation

Our dataset is a join of `metadata` and `product reviews`, based on the unique amazon ID. This means that there are unique reviews, but the product meta data is repeated. For the moment, we focused our qualitative evaluation on product meta data; however, there is room for improvement sort out some of the current bugs. 

We conducted a qualitive evaluation to compare BM25 and semantic search across queries of varying difficulty. BM25 relies on keywords from the `product_title`, while the semantic model combined `product_title`, `category`, `details`, `features` into a document to establish a larger corpus for contextual interpretation. 

Current issues:

- We have duplicates in our search results. We are currently screening for 100 unique product titles and presenting the top 10 scores, but this may be contributing to errors coming up for some products within a webapp search.
- The different in inputs between BM25 and semantic search may be a confounding factor. There may be some 'junk' terms in the merged document that is not in natural language and reducing the effectiveness of the embedding index.
- Performance for semantic search is poor, further inquiry into the data pipeline is needed in a future milestone. Question why irrelevant results are returned sometimes.
- More edge cases will be needed, such as return score 0.0.

## Query Inputs

| Query | Type | Difficulty |
|------|------|------------|
| Washing machine | Keyword | Easy |
| Stainless steel coffee maker | Keyword + Attribute | Easy |
| Replacement Parts DC61-02610A | Keyword | Easy |
| Magic bullet | Specific Keyword | Medium |
| Whirlpool | Keyword + Semantic | Medium |
| Washingmachine | Keyword (Robustness) | Medium |
| kitchen device to heat food quickly | Semantic | Medium |
| Something to cook pizza in | Semantic | Medium |
| Best appliances for a small apartment | Semantic | Hard |
| Aquamarine appliance to make bread crispy | Semantic + Compositional | Hard |


## Retrieval Results

### Query 1: Stainless steel coffee maker

### Query 2: Best appliances for a small apartment

### Query 3:

### Query 4:

### Query 5:

## Summary
