from pathlib import Path
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
from src.utils import build_documents

# NOTE: Brainstorming with ChatGBT5 was done to develop the class
# and learn about new packages
# Debugged with codex to index based on `parent_asin` and CUDA

# No longer using review data
# With metadata still need to run in batches
# Now that we can explicitly choosing the safest device
# to run the code
MAX_SEMANTIC_DOC_CHARS = 8000
SEMANTIC_BATCH_SIZE = 8
SEMANTIC_ENCODE_CHUNK_SIZE = 1000

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
        self.documents = [truncate_document(doc) for doc in self.documents]

        # Set model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=get_torch_device(),
        )

        # Encode in chunks to avoid native crashes on large corpora.
        self.embeddings = encode_documents(self.model, self.documents)

        # Rescale each doc vector to length 1
        self.embeddings = normalize_embeddings(self.embeddings)
        # Create FAISS index that scores normalized vectors
        # via cosine similarity
        faiss = get_faiss()
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
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")
        query_embedding = np.ascontiguousarray(query_embedding)

        # Normalize the query embedding
        query_embedding = normalize_embeddings(query_embedding)
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
        faiss = get_faiss()
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
        engine.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=get_torch_device(),
        )

        # Read in searchable vector index
        faiss = get_faiss()
        engine.index = faiss.read_index(str(filepath))

        ids_path = SemanticSearch.ids_path(filepath)
        if not ids_path.exists():
            raise FileNotFoundError(
                f"Semantic id mapping not found at {ids_path}. "
                "Rebuild the semantic index to regenerate both files."
            )

        engine.ids = np.load(ids_path, allow_pickle=False).tolist()

        if len(engine.ids) != engine.index.ntotal:
            raise ValueError(
                "Semantic index/id mapping mismatch: "
                f"loaded {len(engine.ids)} ids for {engine.index.ntotal} vectors. "
                "Rebuild the semantic index so both artifacts are aligned."
            )

        return engine

# NOTE: codex created function below during the debugging process

def get_torch_device():
    """Return the safest available device for embedding generation."""
    if os.environ.get("FORCE_SEMANTIC_CPU", "").lower() in {"1", "true", "yes"}:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_faiss():
    """Import FAISS lazily to avoid native import-order crashes."""
    import faiss

    return faiss

def truncate_document(doc, max_chars=MAX_SEMANTIC_DOC_CHARS):
    """Trim oversized semantic documents to keep embedding stable."""
    if len(doc) <= max_chars:
        return doc
    return doc[:max_chars].rsplit(" ", 1)[0].strip()

def encode_documents(model, documents):
    """Encode the corpus in smaller chunks to keep embedding stable."""
    chunks = []
    total = len(documents)

    for start in range(0, total, SEMANTIC_ENCODE_CHUNK_SIZE):
        end = min(start + SEMANTIC_ENCODE_CHUNK_SIZE, total)
        chunk_embeddings = model.encode(
            documents[start:end],
            batch_size=SEMANTIC_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        chunk_embeddings = np.asarray(chunk_embeddings, dtype="float32")
        chunk_embeddings = np.ascontiguousarray(chunk_embeddings)
        chunks.append(chunk_embeddings)

    return np.vstack(chunks)

def normalize_embeddings(embeddings):
    """Normalize embedding rows with NumPy to avoid FAISS L2 crashes."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return np.ascontiguousarray(normalized.astype("float32"))
