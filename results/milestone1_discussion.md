# Milestone 1: Qualitative Evaluation

Our dataset is a join of `metadata` and `product reviews` on the unique Amazon ID. This means that reviews are unique, but product metadata is repeated. For now, we have focused our qualitative evaluation on product metadata; however, there is room to address some current bugs. 

We conducted a qualitative evaluation comparing BM25 and semantic search across queries of varying difficulty. BM25 relies on keywords from the `product_title`, while the semantic model combines `product_title`, `category`, `details`, and `features` into a document to create a larger corpus for contextual interpretation. 

**Current issues**:

- We have duplicates in our search results. We are screening for 100 unique product titles and presenting the top 10 scores, but this may be contributing to errors for some products in a webapp search.
- The difference in inputs between BM25 and semantic search may be a confounding factor. There may be 'junk' terms in the merged document that are not in natural language and reduce the effectiveness of the embedding index.
- Performance for semantic search is poor. Further inquiry into the data pipeline is needed in a future milestone. We also need to understand why irrelevant results are sometimes returned.
- More edge cases will be needed, such as returning a score of 0.0.

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

### Query 1: Replacement Parts DC61-02610A

**BM25**: Succeeds at finding the exact match of the serial number as the top candidate

**SemanticSearch**: Fails to find the exact match even within the top 10 results

**Comment**: BM25 is effective at finding the exact match, while the SemanticSearch does not. It seems as though the SemanticSearch is finding a similar pattern of product name paired with series-type numbers, though it's surprising it doesn't return the exact match as one of the semantically similar products.

### Query 2: Stainless steel coffee makers

**BM25**: Effective at finding variations of the exact keywords. The top candidates include all four keywords, but the lower-scoring products lose semantic similarity and begin matching only on individual words or attributes, such as a "Stainless steel" product, even if that product is an ice maker.

**SemanticSearch**: It fails to find the exact match found by the BM25 approach; however, of the 10 results, all products are semantically related to coffee, such as coffee pots, pods, filters, and milk frothers.

**Comment**: Both approaches have their advantages and disadvantages. This query example shows that both are somewhat effective, but a hybrid approach would be better. Overall, a user would want the exact match returned by the BM25 approach.

### Query 3: kitchen device to heat food quickly

**BM25**: This fails to capture the query's semantic meaning. It's trying to find keyword matches, but the top result is a kitchen faucet.

**SemanticSearch**: This is somewhat successful in finding appliances that heat food, such as "rapid corn & potato cooker | Microwave" or an electric cooktop. However, the categorization is fairly broad.

**Comment**: The SemanticSearch is a better approach here, which makes sense given the semantic query and the lack of direct keywords. Further refinement of the SemanticSearch is needed to improve the results.

### Query 4: Aquamarine appliance to make bread crispy

**BM25**: It fails to find matching keywords

**SemanticSearch**: It picks up on "crisper" or "crisper pan" or "AQUA", but the query is too nonsensical for the current semantic embedding to decipher it as a blue toaster.

**Comment**: neither is successful in finding a complex interpretation of the query.

### Query 5: Washingmachine vs Washing Machine

**BM25**: It is successful in finding an exact match for "Washing machine" but fails to interpret the typo "Washingmachine."

**SemanticSearch**: It is successful for both "Washing Machine" and the typo "Washingmachine"

**Comment**: It's likely that queries will be misspelled; the semantic search can still find semantically similar results when the exact match fails.

## Summary

BM25 and SemanticSearch use different scoring systems, making quantitative performance comparisons difficult. Results can vary widely depending on the type and quality of the query. When queries contain direct, straightforward keyword matches, BM25 performs much better than SemanticSearch. However, when queries are abstract, semantically related, or contain typos, SemanticSearch is at least able to direct the user to the right area of the dataset, albeit one that may be a little broad and imprecise.

Both perform poorly on complex queries that require careful consideration of context and relational data between words in the query. RAG will be helpful for incorporating specific knowledge we are looking for to supplement the current corpus.
