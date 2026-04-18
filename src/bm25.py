# adapted from 575_lec05 "comparison between BM25 and embedding-based search"
# Note: debugged by codex to incorporate parent_asin at index id

from rank_bm25 import BM25Okapi
from src.utils import text_preprocessor
import pickle

class BM25Search:
    """
    Search engine class that performs information retrieval using BM25 algorithm.

    Attributes
    ----------
    products : list of str
        Original list of products.
    tokenized_corpus : list of list of str
        Corpus after being processed and tokenized.
    bm25 : BM25Okapi
        Indexed BM25 model from library rank_bm25.

    Examples
    --------
    >>> appliances = ["Coffee Maker", "Toaster Oven"]
    >>> appliance_ids = ["A1", "A2"]
    >>> engine = BM25Search(appliances, appliance_ids)
    >>> engine.retrieve("coffee", top_k=1)
    [('A1', 0.0)]
    """

    def __init__(self, products, ids):
        """
        Initialize the BM25 search engine by indexing input documents.

        Parameters
        ----------
        products : list of str
            Documents to index for BM25 retrieval.
        ids : list
            Product identifiers aligned with `products`, such as
            `parent_asin`.

        """

        if isinstance(products, str):
            products = [products]

        if len(products) != len(ids):
            raise ValueError("`products` and `ids` must have the same length.")

        self.products = products
        self.ids = ids
        self.tokenized_corpus = [text_preprocessor(doc) for doc in products]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def retrieve(self, query, top_k=5):
        """
        Calculates BM25 scores and returns the top k results with their scores.

        Parameters
        ----------
        query : str
            The search string.
        top_k : int, optional
            Number of results to be returned. Default is 5.

        Returns
        -------
        list of tuple
            List of tuples where each tuple contains (product_id, score).
        
        """

        tokenized_query = text_preprocessor(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_idx = sorted(range(len(scores)), 
                             key=lambda i: scores[i], 
                             reverse=True)[:top_k]
        
        results = [(self.ids[i], float(scores[i])) for i in top_idx]

        return results
    
    # adapted from 524 coursebook 21.6.3 Model objects
    def save(self, filepath="data/processed/bm25_index.pkl"):
        """
        Save the tokenized corpus and BM25 index to pickle file.
        
        Parameters
        ----------
        filepath : str
            Path where pickle file will be saved.

        """
 
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath="data/processed/bm25_index.pkl"):
        """
        Load a saved BM25 search engine from disk.

        Parameters
        ----------
        filepath : str
            Path to the pickle file containing the saved BM25 engine.

        Returns
        -------
        BM25Search
            Loaded BM25 search engine with products, ids, and BM25 index.
        """

        with open(filepath, "rb") as f:
            return pickle.load(f)
