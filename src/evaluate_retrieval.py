"""
Title: Evaluate Retrieval
Purpose: Score BM25, semantic, and hybrid retrieval against labeled relevance
judgments using ranking metrics.
Date: 2026-04-22

Notes
- Codex was used to help brainstorm and draft this script.
- This script evaluates the retrieval component of the system using manually
  labeled relevance judgments collected in `retrieval_labels.csv`.
- The labels file is expected to come from the pooled candidate-generation
  workflow in `build_retrieval_labels.py`, followed by human review of the
  `relevant` column.
- This script does not compare BM25, semantic, and hybrid systems by raw score
  values, since those scores are not necessarily on the same scale. Instead, it
  evaluates the ranked lists returned by each retriever against the human
  labels.
- For each labeled query, the script retrieves the top-k results from BM25,
  semantic, and hybrid retrieval, then computes precision@k, recall@k, and
  reciprocal rank.
- Mean precision@k and mean recall@k summarize how well each retriever performs
  across the full query set at the chosen cutoff. MRR summarizes how early the
  first relevant item appears in the ranking on average.
- The script requires every pooled candidate for a query to be labeled before
  evaluation, and it requires at least one positive relevant item per query.
  This is intended to reduce ambiguity in the evaluation set.
- The output is written as a JSON file with both summary metrics and per-query
  details, so results can be inspected directly and reused in the report.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.bm25 import BM25Search
from src.hybrid import HybridRetriever
from src.semantic import SemanticSearch


DEFAULT_BM25_PATH = root_dir / "data" / "processed" / "bm25_index.pkl"
DEFAULT_SEMANTIC_PATH = root_dir / "data" / "processed" / "semantic.index"
DEFAULT_LABELS_PATH = root_dir / "data" / "eval" / "retrieval_labels.csv"
DEFAULT_OUTPUT_PATH = root_dir / "results" / "retrieval_eval_results.json"


def parse_args():
    """Parse command-line arguments for retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BM25, semantic, and hybrid retrieval using "
            "precision@k, recall@k, and MRR."
        )
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help="Path to CSV file containing manually labeled retrieval candidates.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Cutoff k for precision@k and recall@k.",
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where evaluation results JSON will be written.",
    )
    return parser.parse_args()


def parse_relevant_label(value):
    """Parse a human-entered relevance label from CSV."""
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    if normalized == "":
        return None
    raise ValueError(
        "Unsupported relevance label "
        f"`{value}`. Use yes/no, true/false, or 1/0."
    )


def load_labels(labels_path):
    """Load and validate manually labeled retrieval candidates from CSV."""
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found at {labels_path}. "
            "Create it first or pass --labels."
        )

    grouped = {}

    with open(labels_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"query", "parent_asin", "relevant"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Labels file is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            query = str(row.get("query", "")).strip()
            parent_asin = str(row.get("parent_asin", "")).strip()
            relevant = parse_relevant_label(row.get("relevant", ""))

            if not query or not parent_asin:
                raise ValueError(
                    "Each row in the labels file must include `query` and `parent_asin`."
                )

            query_group = grouped.setdefault(
                query,
                {
                    "query": query,
                    "relevant_asins": set(),
                    "judged_asins": set(),
                    "unlabeled_asins": set(),
                },
            )

            if relevant is None:
                query_group["unlabeled_asins"].add(parent_asin)
                continue

            query_group["judged_asins"].add(parent_asin)
            if relevant:
                query_group["relevant_asins"].add(parent_asin)

    if not grouped:
        raise ValueError("Labels file must contain at least one row.")

    validated = []
    for query, entry in grouped.items():
        if entry["unlabeled_asins"]:
            raise ValueError(
                f"Query `{query}` still has unlabeled candidates. "
                "Finish labeling all pooled rows before evaluation."
            )

        if not entry["relevant_asins"]:
            raise ValueError(
                f"Query `{query}` has no positive labels. "
                "Mark at least one relevant product for each query."
            )

        validated.append(
            {
                "query": query,
                "relevant_asins": sorted(entry["relevant_asins"]),
                "num_judged": len(entry["judged_asins"]),
            }
        )

    return validated


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Compute precision at cutoff k."""
    top_k = retrieved_ids[:k]
    if k <= 0:
        return 0.0
    hits = sum(doc_id in relevant_ids for doc_id in top_k)
    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Compute recall at cutoff k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(doc_id in relevant_ids for doc_id in top_k)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Compute reciprocal rank for a single query."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(name, retriever, labels, k):
    """Evaluate a single retriever over a set of labeled queries."""
    per_query = []

    for entry in labels:
        query = entry["query"]
        relevant_ids = set(entry["relevant_asins"])

        results = retriever.retrieve(query, top_k=k)
        retrieved_ids = [product_id for product_id, _ in results]

        precision = precision_at_k(retrieved_ids, relevant_ids, k)
        recall = recall_at_k(retrieved_ids, relevant_ids, k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        per_query.append(
            {
                "query": query,
                "relevant_asins": sorted(relevant_ids),
                "num_judged": entry["num_judged"],
                "retrieved_asins": retrieved_ids,
                f"precision@{k}": round(precision, 4),
                f"recall@{k}": round(recall, 4),
                "reciprocal_rank": round(rr, 4),
            }
        )

    n_queries = len(per_query)
    summary = {
        "retriever": name,
        "num_queries": n_queries,
        f"mean_precision@{k}": round(
            sum(row[f"precision@{k}"] for row in per_query) / n_queries, 4
        ),
        f"mean_recall@{k}": round(
            sum(row[f"recall@{k}"] for row in per_query) / n_queries, 4
        ),
        "mrr": round(
            sum(row["reciprocal_rank"] for row in per_query) / n_queries, 4
        ),
    }

    return {
        "summary": summary,
        "per_query": per_query,
    }


def print_summary(result, k):
    """Print a concise metrics summary for one retriever."""
    summary = result["summary"]
    print(f"\n{summary['retriever'].upper()}")
    print(f"  queries: {summary['num_queries']}")
    print(f"  mean_precision@{k}: {summary[f'mean_precision@{k}']:.4f}")
    print(f"  mean_recall@{k}: {summary[f'mean_recall@{k}']:.4f}")
    print(f"  mrr: {summary['mrr']:.4f}")


def main():
    """Run retrieval evaluation for BM25, semantic, and hybrid retrievers."""
    args = parse_args()

    if args.k <= 0:
        raise ValueError("`--k` must be a positive integer.")

    labels = load_labels(args.labels)

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

    all_results = {
        "metadata": {
            "k": args.k,
            "labels_path": str(args.labels),
            "bm25_index": str(args.bm25_index),
            "semantic_index": str(args.semantic_index),
        },
        "bm25": evaluate_retriever("bm25", bm25, labels, args.k),
        "semantic": evaluate_retriever("semantic", semantic, labels, args.k),
        "hybrid": evaluate_retriever("hybrid", hybrid, labels, args.k),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print(f"Wrote retrieval evaluation results to {args.output}")
    print_summary(all_results["bm25"], args.k)
    print_summary(all_results["semantic"], args.k)
    print_summary(all_results["hybrid"], args.k)


if __name__ == "__main__":
    main()
