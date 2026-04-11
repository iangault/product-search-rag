import faiss
import pickle
from sentence_transformers import SentenceTransformer
from src.normalize import normalize
from src.build_documents import build_documents

class SemanticSearch:

    def __init__(self, df, columns):

        self.df = df.reset_index(drop=True)
        self.columns = [columns] if isinstance(columns, str) else columns
        self.documents = build_documents(self.df, self.columns)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.embeddings = self.model.encode(
            self.documents,
            convert_to_numpy=True
        ).astype("float32")

        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def embedding_search(self, query, top_k=5):
        query_embedding = self.model.encode([query],
                                       convert_to_numpy=True).astype("float32")

        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

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

        faiss.write_index(self.index, index_path)

        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    @staticmethod
    def load(index_path = "data/processed/semantic.index",
             docs_path = "data/processed/semantic_docs.pkl"):

        engine = SemanticSearch.__new__(SemanticSearch)
        engine.columns = None
        engine.model_name = "all-MiniLM-L6-v2"
        engine.index = faiss.read_index(index_path)

        with open(docs_path, "rb") as f:
            engine.documents = pickle.load(f)

        engine.embeddings = None
        return engine


import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.semantic import SemanticSearch

print("\nreading in parquet...\n")
df = pd.read_parquet("data/processed/merged.parquet")

cols = ["product_title", "features", "description", "categories", "details"]

print("\nSemantic Search...\n")
engine = SemanticSearch(df, cols)

query = "Blender for fruit smoothies"
results = engine.embedding_search(query)

results_df = pd.DataFrame(results)

print(f"Query: {query}\n")

print("Semantic search top results:")

print(results_df[["product_title", "score", "price"]])
