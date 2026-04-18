# Current Retrieval Pipeline Flow

## End-to-end flow

```mermaid
flowchart TD
    A[Amazon Reviews JSONL.gz<br/>Amazon Metadata JSONL.gz] --> B[src/import_process.py]
    B --> C[reviews_raw.parquet<br/>meta_raw.parquet]
    C --> D[LEFT JOIN on parent_asin]
    D --> E[Aggregate to one row per product]
    E --> F[processed.parquet]

    F --> G[src/build_index.py]

    G --> H[BM25 branch]
    H --> H1[build_documents on selected product fields]
    H1 --> H2[text_preprocessor]
    H2 --> H3[BM25Search with parent_asin ids]
    H3 --> H4[bm25_index.pkl]

    G --> I[Semantic branch]
    I --> I1[build_documents on selected product fields]
    I1 --> I2[SentenceTransformer embeddings]
    I2 --> I3[FAISS IndexFlatIP]
    I3 --> I4[semantic.index<br/>semantic.ids.npy]

    F --> J[app/app.py]
    H4 --> J
    I4 --> J
    J --> K[User selects BM25, Semantic,<br/>Hybrid, RAG, or Hybrid RAG]
    K --> L[Selected retriever returns parent_asin and score<br/>or supporting products for RAG modes]
    L --> M[App looks up product by parent_asin]
    M --> N[Display title, candidate review, rating,<br/>helpful votes, and retrieval score]
```

## Summary

The updated pipeline is product-level rather than review-row-level. `src/import_process.py` joins raw review and metadata files, aggregates them to one row per `parent_asin`, and saves `processed.parquet`. `src/build_index.py` then builds both BM25 and semantic retrieval artifacts from that same processed product table, keeping `parent_asin` aligned to each searchable document.

In the app, the retrieval modes return stable product IDs instead of row positions. `app.py` uses `parent_asin` to look up the product in `processed.parquet` and display the product title, candidate review snippet, rating summary, helpful votes, and retrieval score. The current app also includes Hybrid, RAG, and Hybrid RAG modes, though this diagram remains focused on the shared processed-data and retrieval-index flow.
