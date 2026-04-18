from src.bm25 import BM25Search
from src.semantic import SemanticSearch

class HybridRetriever:
    """
    A hybrid retriever that combines BM25 and semantic search results into a single ranked list.

    Parameters
    ----------
    bm25 : BM25Search
        An instance of BM25Search class, loaded with the same product documents as the semantic search engine.
    semantic : SemanticSearch
        An instance of SemanticSearch class, loaded with the same product documents as the BM25 search engine.
    bm25_weight : float, optional
        Weighting factor for BM25 scores in final ranking. Default is 0.5.
    semantic_weight : float, optional
        Weighting factor for semantic search scores in final ranking. Default is 0.5.
    rrf_k : int, optional
        RRF parameter k for score normalization. Default is 60. Large k reduces impact of rank position, small k emphasizes top ranks.
    
    """

    def __init__(self, bm25, semantic, bm25_weight=0.5, semantic_weight=0.5, rrf_k=60):
        self.bm25 = bm25
        self.semantic = semantic
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.rrf_k = rrf_k

    def retrieve(self, query, top_k=5):
        """
        Run both retrievers on a query and merge results using RRF
        
        Parameters
        ----------
        query : str
            User's search query
        top_k : int, optional
            Number of top results to return after merging. Default is 5.

        Returns
        -------
        list of tuple
            List of tuples (product_id, rrf_score) tuples, sorted best-first
        
        """

        # fetch extra results to allow for better merging
        fetch_k = max(top_k * 3, 20)

        # results from each retriever as lists of (product_id, score)
        bm25_results = self.bm25.retrieve(query, top_k=fetch_k)
        semantic_results = self.semantic.retrieve(query, top_k=fetch_k)

        # track combined RRF score for each product_id
        rrf_scores = {}

        # process BM25 results with weighting and RRF scoring
        for rank, (product_id, score) in enumerate(bm25_results):
            rrf_weight = self.bm25_weight * (1.0 / (self.rrf_k + rank)) # 575_lec05_hybrid retrieval_RRF
            rrf_scores[product_id] = rrf_scores.get(product_id, 0.0) + rrf_weight

        # process semantic results with weighting and RRF scoring
        for rank, (product_id, score) in enumerate(semantic_results):
            rrf_weight = self.semantic_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[product_id] = rrf_scores.get(product_id, 0.0) + rrf_weight

        # sort by combined RRF score, highest first
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return ranked[:top_k]