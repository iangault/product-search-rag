import faiss
import pickle
from sentence_transformers import SentenceTransformer
from src.normalize import normalize
from src.build_documents import build_documents

# NOTE: Brainstorming with ChatGBT5 was done to develop the class
# and learn about new packages

class SemanticSearch:
    """Search engine class that performs semantic retrieval
    using sentence transformer embeddings and a FAISS index.
    """

    def __init__(self, df, columns):
        """Initializes the semantic search engine.
        - builds document strings
        - generates embeddings
        - creates a FAISS index

        Args:
            df (DataFrame): 
                Input with source text fields
            columns (str or list of str or dict):
                columnnames to combine into one searchable 
                document per row
        """

        self.df = df.reset_index(drop=True)

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

    def embedding_search(self, query, top_k=5):
        """Retrieve top k rows whose document embeddings are most
        similar to the query embedding

        Args:
            query (str): Search query from user
            top_k (int, optional): number of top results to return.
                Defaults to 5.

        Returns:
            DataFrame: DF of top matching rows with similarity scores
            and a combined document of text.
        """

        # Create query embeddings
        query_embedding = (self.model.encode([query], convert_to_numpy=True)
                           .astype("float32"))

        # Normalize the query embedding
        faiss.normalize_L2(query_embedding)
        # Compares query embedding to all indexed document embedding, returning the score and the row position for most similar doc
        scores, indices = self.index.search(x = query_embedding, k = top_k)

        # Create a list for results presentation
        results = []

        for rank, i in enumerate(indices[0]):
            results.append({
                "index": int(i),
                "score": float(scores[0][rank]),
                "product_title": self.df.iloc[i]["product_title"],
                "price": self.df.iloc[i]["price"],
                "categories": self.df.iloc[i]["categories"],
                "document": self.documents[i]
            })

        return results

    def save(self, index_path="data/processed/semantic.index",
             docs_path = "data/processed/semantic_docs.pkl"):
        """Save the FAISS structure of ducment embeddings of corpus,
        so that it can later be compared to query embeddings.

        Args:
            index_path (str, optional): location of index. Defaults to
            "data/processed/semantic.index".
            docs_path (str, optional): location of docs. Defaults to
            "data/processed/semantic_docs.pkl".
        """

        # Saves the searchable vector index (vectors and positions)
        faiss.write_index(self.index, index_path)

        # Saves the original built document text for readable results
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    @staticmethod
    def load(index_path = "data/processed/semantic.index",
             docs_path = "data/processed/semantic_docs.pkl"):
        """Load a previously saved FAISS index and document string.

        Args:
            index_path (str, optional): location of index.
                Defaults to "data/processed/semantic.index".
            docs_path (str, optional): location of docs.
                Defaults to "data/processed/semantic_docs.pkl".

        Returns:
            SemanticSearch: A SemanticSearch object with
            saved index and documents
        """

        # Create empty object to add out engine to
        engine = SemanticSearch.__new__(SemanticSearch)
        # Create columns attribute
        engine.df = None
        engine.columns = None
        # SentenceTransformer initialized for new queries
        engine.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Read in searchable vector index
        engine.index = faiss.read_index(index_path)

        # Read in document text
        with open(docs_path, "rb") as f:
            engine.documents = pickle.load(f)

        # Don't need because we have engine.index for retreival
        engine.embeddings = None
        return engine


