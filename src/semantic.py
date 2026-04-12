import faiss
from sentence_transformers import SentenceTransformer
from src.utils import normalize
from src.utils import build_documents

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

    def retrieve(self, query, top_k):
        """Retrieve top k rows whose document embeddings are most
        similar to the query embedding

        Args:
            query (str): Search query from user
            top_k (int, optional): number of top results to return.

        Returns:
            DataFrame: DF of top matching rows with similarity scores
            and a combined document of text.
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
        results = [(int(i), float(scores[0][rank]))
                   for rank, i in enumerate(indices[0])
                   if i != -1]

        return results

    def save(self, filepath="data/processed/semantic.index"):
        """Save the FAISS structure of ducment embeddings of corpus,
        so that it can later be compared to query embeddings.

        Args:
            index_path (str, optional): location of index. Defaults to
            "data/processed/semantic.index".
            docs_path (str, optional): location of docs. Defaults to
            "data/processed/semantic_docs.pkl".
        """

        #### Separate option, will keep here#
        # index_path="data/processed/semantic.index",
        # docs_path = "data/processed/semantic_docs.pkl"
        # # Saves the searchable vector index (vectors and positions)
        # faiss.write_index(self.index, str(index_path))
        # # Saves the original built document text for readable results
        # with open(docs_path, "wb") as f:
        #     pickle.dump(self.documents, f)

        # Saves the searchable vector index (vectors and positions)
        faiss.write_index(self.index, str(filepath))

    @staticmethod
    def load(filepath="data/processed/semantic.index"):
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

        # Option of keeping separate
        # index_path = "data/processed/semantic.index",
        # docs_path = "data/processed/semantic_docs.pkl"
        # # Create empty object to add out engine to
        # engine = SemanticSearch.__new__(SemanticSearch)
        # # Create columns attribute
        # engine.df = None
        # engine.columns = None
        # # SentenceTransformer initialized for new queries
        # engine.model = SentenceTransformer("all-MiniLM-L6-v2")
        # Read in searchable vector index
        # engine.index = faiss.read_index(str(index_path))

        # Create empty object to add out engine to
        engine = SemanticSearch.__new__(SemanticSearch)

        # SentenceTransformer initialized for new queries
        engine.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Read in searchable vector index
        engine.index = faiss.read_index(str(filepath))

        return engine
