"""
Title: Build Retrieval Labels
Purpose: Generate a pooled manual-labeling file for retrieval evaluation by
combining BM25, semantic, and hybrid candidates.
Date: 2026-04-22

Notes
- Codex was used to help brainstorm the purpose of this script and draft it.
- We used a custom script-based approach rather than a RADAS workflow because
  the goal here was to evaluate the information retrieval component of the RAG
  system without introducing an LLM into the labeling pipeline itself.
- This script loads evaluation queries from a JSON file, runs BM25, semantic,
  and hybrid retrieval for each query, and pools the returned candidates into a
  single CSV for review.
- "Pooling" here means taking the union of products returned by the different
  retrievers for the same query. Raw BM25 and semantic scores are not combined
  directly, because they are on different scales. Instead, the script records
  each retriever's rank and score in separate columns.
- The output file, `retrieval_labels.csv`, is intended for human validation of
  relevance.
- ChatGPT (a much larger model) was used to add in relevance (0 for not relevant,
  1 for relevant) into the csv file. A spot check was done for human judgement
  to assess quality and consistency. Relevance decisions were based on
  what a user would reasonably expect to retrieve for a query, not on whether a
  product was an exact lexical match.
- Some queries are broad, so relevance judgments were applied somewhat more
  leniently in those cases.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.bm25 import BM25Search
from src.hybrid import HybridRetriever
from src.semantic import SemanticSearch


DEFAULT_BM25_PATH = root_dir / "data" / "processed" / "bm25_index.pkl"
DEFAULT_SEMANTIC_PATH = root_dir / "data" / "processed" / "semantic.index"
DEFAULT_DATA_PATH = root_dir / "data" / "processed" / "processed.parquet"
DEFAULT_QUERIES_PATH = root_dir / "data" / "eval" / "retrieval_queries.json"
DEFAULT_OUTPUT_PATH = root_dir / "data" / "eval" / "retrieval_labels.csv"


def parse_args():
    """Parse command-line arguments for label-set generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Pool BM25, semantic, and hybrid retrieval candidates into a "
            "CSV file for manual relevance labeling."
        )
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to JSON file containing evaluation queries.",
    )
    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=DEFAULT_BM25_PATH,
        help="Path to the saved BM25 index pickle.",
    )
    parser.add_argument(
        "--semantic-index",
        type=Path,
        default=DEFAULT_SEMANTIC_PATH,
        help="Path to the saved semantic FAISS index.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to processed.parquet used to enrich label rows.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to pull from each retriever before pooling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the manual labeling CSV should be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def load_queries(queries_path):
    """Load evaluation queries from JSON."""
    if not queries_path.exists():
        raise FileNotFoundError(
            f"Query file not found at {queries_path}. Create it first or pass --queries."
        )

    with open(queries_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list) or not payload:
        raise ValueError("Query file must contain a non-empty list.")

    queries = []
    for item in payload:
        if isinstance(item, str) and item.strip():
            queries.append(item.strip())
            continue

        if isinstance(item, dict):
            query = str(item.get("query", "")).strip()
            if query:
                queries.append(query)

    if not queries:
        raise ValueError("No valid queries were found in the query file.")

    return queries


def load_product_metadata(data_path, asins):
    """Load product metadata for a set of ASINs from processed.parquet."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {data_path}. Run src/import_process.py first."
        )

    asin_list = sorted(set(asins))
    if not asin_list:
        return {}

    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT
            parent_asin,
            product_title,
            derived_avg_rating,
            n_reviews,
            candidate_review_title,
            candidate_review_text
        FROM read_parquet('{data_path}')
        WHERE parent_asin IN ({",".join(["?"] * len(asin_list))})
        """,
        asin_list,
    ).fetchall()

    con.close()

    metadata = {}
    for row in rows:
        metadata[row[0]] = {
            "product_title": row[1] or "",
            "derived_avg_rating": row[2],
            "n_reviews": row[3],
            "candidate_review_title": row[4] or "",
            "candidate_review_text": row[5] or "",
        }

    return metadata


def truncate_text(text, max_chars=280):
    """Shorten long review snippets for CSV readability."""
    text = str(text or "").strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_pooled_candidates(queries, bm25, semantic, hybrid, top_k):
    """Pool candidates from BM25, semantic, and hybrid retrieval."""
    pooled = defaultdict(dict)

    for query in queries:
        retriever_outputs = {
            "bm25": bm25.retrieve(query, top_k=top_k),
            "semantic": semantic.retrieve(query, top_k=top_k),
            "hybrid": hybrid.retrieve(query, top_k=top_k),
        }

        for retriever_name, results in retriever_outputs.items():
            for rank, (parent_asin, score) in enumerate(results, start=1):
                row = pooled[query].setdefault(
                    parent_asin,
                    {
                        "query": query,
                        "parent_asin": parent_asin,
                        "retrievers": set(),
                        "best_rank": rank,
                        "bm25_rank": "",
                        "semantic_rank": "",
                        "hybrid_rank": "",
                        "bm25_score": "",
                        "semantic_score": "",
                        "hybrid_score": "",
                    },
                )
                row["retrievers"].add(retriever_name)
                row["best_rank"] = min(row["best_rank"], rank)
                row[f"{retriever_name}_rank"] = rank
                row[f"{retriever_name}_score"] = round(float(score), 4)

    return pooled


def write_labels_csv(output_path, pooled, metadata):
    """Write pooled candidates to CSV for manual relevance labeling."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query",
        "parent_asin",
        "product_title",
        "derived_avg_rating",
        "n_reviews",
        "candidate_review_title",
        "candidate_review_text",
        "retrievers",
        "best_rank",
        "bm25_rank",
        "bm25_score",
        "semantic_rank",
        "semantic_score",
        "hybrid_rank",
        "hybrid_score",
        "relevant",
        "notes",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for query in pooled:
            rows = sorted(
                pooled[query].values(),
                key=lambda row: (row["best_rank"], row["parent_asin"]),
            )
            for row in rows:
                product_meta = metadata.get(row["parent_asin"], {})
                writer.writerow(
                    {
                        "query": row["query"],
                        "parent_asin": row["parent_asin"],
                        "product_title": product_meta.get("product_title", ""),
                        "derived_avg_rating": product_meta.get("derived_avg_rating", ""),
                        "n_reviews": product_meta.get("n_reviews", ""),
                        "candidate_review_title": product_meta.get(
                            "candidate_review_title", ""
                        ),
                        "candidate_review_text": truncate_text(
                            product_meta.get("candidate_review_text", "")
                        ),
                        "retrievers": ",".join(sorted(row["retrievers"])),
                        "best_rank": row["best_rank"],
                        "bm25_rank": row["bm25_rank"],
                        "bm25_score": row["bm25_score"],
                        "semantic_rank": row["semantic_rank"],
                        "semantic_score": row["semantic_score"],
                        "hybrid_rank": row["hybrid_rank"],
                        "hybrid_score": row["hybrid_score"],
                        "relevant": "",
                        "notes": "",
                    }
                )


def main():
    """Generate a pooled candidate CSV for manual labeling."""
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("`--top-k` must be a positive integer.")
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Output file already exists at {args.output}. "
            "Use `--force` to overwrite it."
        )

    queries = load_queries(args.queries)

    if not args.bm25_index.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {args.bm25_index}. Run src/build_index.py first."
        )

    if not args.semantic_index.exists():
        raise FileNotFoundError(
            f"Semantic index not found at {args.semantic_index}. "
            "Run src/build_index.py first."
        )

    bm25 = BM25Search.load(args.bm25_index)
    semantic = SemanticSearch.load(args.semantic_index)
    hybrid = HybridRetriever(bm25, semantic)

    pooled = build_pooled_candidates(queries, bm25, semantic, hybrid, args.top_k)
    all_asins = {
        parent_asin
        for query_rows in pooled.values()
        for parent_asin in query_rows
    }
    metadata = load_product_metadata(args.data, all_asins)
    write_labels_csv(args.output, pooled, metadata)

    print(f"Wrote manual labeling file to {args.output}")
    print("Fill the `relevant` column with yes/no (or true/false, 1/0).")


if __name__ == "__main__":
    main()
