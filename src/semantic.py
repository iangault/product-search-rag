from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import build_documents

# NOTE: Brainstorming with ChatGBT5 was done to develop the class
# and learn about new packages
# Debugged with codex to index based on `parent_asin`

class SemanticSearch:
    """Search engine class that performs semantic retrieval
    using sentence transformer embeddings and a FAISS index.
    """

    def __init__(self, df, columns):
        """Initialize the semantic search engine.

        Parameters
        ----------
        df : pandas.DataFrame
            Processed product dataframe containing one row per product,
            including `parent_asin`.
        columns : str or list of str
            Column name(s) to combine into one searchable document per
            row.
        """

        self.df = df.reset_index(drop=True)
        self.ids = self.df["parent_asin"].tolist()

        # If only one columns as a string is passed
        self.columns = [columns] if isinstance(columns, str) else columns

        # Build documents, which internally normalizes column types
        self.documents = build_documents(self.df, self.columns)

        # Set model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create embeddings from documents
        self.embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True
        ).astype("float32")

        # Rescale each doc vector to length 1
        faiss.normalize_L2(self.embeddings)
        # Create FAISS index that scores normalized vectors
        # via cosine similarity
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        # Adds document vector to that index
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k):
        """Retrieve the top semantic matches for a query.

        Parameters
        ----------
        query : str
            Search query from the user.
        top_k : int
            Number of top results to return.

        Returns
        -------
        list of tuple
            List of `(product_id, score)` tuples ordered by decreasing
            similarity.
        """

        # Create query embeddings
        query_embedding = (self.model.encode([query], convert_to_numpy=True)
                           .astype("float32"))

        # Normalize the query embedding
        faiss.normalize_L2(query_embedding)
        # Compares query embedding to all indexed document embedding,
        # returning the score and the row position for most similar doc

        # Safer depending on returned results
        top_k = min(top_k, self.index.ntotal) 

        scores, indices = self.index.search(query_embedding, top_k)

        # Create a list for results presentation
        results = [(self.ids[int(i)], float(scores[0][rank]))
                   for rank, i in enumerate(indices[0])
                   if i != -1]

        return results

    @staticmethod
    def ids_path(filepath):
        """Return the sidecar file that stores stable document ids."""
        return filepath.with_suffix(".ids.npy")

    def save(self, filepath="data/processed/semantic.index"):
        """Save the semantic index and aligned product ids to disk.

        Parameters
        ----------
        filepath : str or Path, optional
            Path where the FAISS index will be saved. A sidecar
            `.ids.npy` file containing aligned product ids is written
            alongside it.
        """

        filepath = Path(filepath)

        # Save the searchable vector index itself, then persist a sidecar
        # array that maps FAISS row positions back to stable product ids.
        faiss.write_index(self.index, str(filepath))
        np.save(self.ids_path(filepath), np.asarray(self.ids, dtype=str))

    @staticmethod
    def load(filepath="data/processed/semantic.index"):
        """Load a saved semantic index and aligned product ids.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to the saved FAISS index.

        Returns
        -------
        SemanticSearch
            Semantic search engine with the saved FAISS index, loaded
            product ids, and query encoder.
        """

        filepath = Path(filepath)

        # Create empty object to add out engine to
        engine = SemanticSearch.__new__(SemanticSearch)

        # SentenceTransformer initialized for new queries
        engine.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Read in searchable vector index
        engine.index = faiss.read_index(str(filepath))

        ids_path = SemanticSearch.ids_path(filepath)
        if not ids_path.exists():
            raise FileNotFoundError(
                f"Semantic id mapping not found at {ids_path}. "
                "Rebuild the semantic index to regenerate both files."
            )

        engine.ids = np.load(ids_path, allow_pickle=False).tolist()

        return engine
