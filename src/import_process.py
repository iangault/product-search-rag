from pathlib import Path
import duckdb
import requests
from tqdm import tqdm
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CATEGORY = "Appliances"
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
REVIEWS_URL = f"{BASE_URL}/review_categories/{CATEGORY}.jsonl.gz"
META_URL = f"{BASE_URL}/meta_categories/meta_{CATEGORY}.jsonl.gz"
DATA_DIR = ROOT/ "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

c2 = duckdb.connect()

c2.execute(
    f"""
      COPY (SELECT * FROM read_json_auto('{REVIEWS_URL}')  LIMIT 60000)
      TO '{RAW_DIR}/reviews_raw.parquet'
      (FORMAT PARQUET, COMPRESSION ZSTD)
  """
)

c2.execute(
    f"""
      COPY (SELECT * FROM read_json_auto('{META_URL}') LIMIT 20000)
      TO '{RAW_DIR}/meta_raw.parquet'
      (FORMAT PARQUET, COMPRESSION ZSTD)
  """
)

# Brainstormed with ChatGBT to get a complex
# SQL function to process the data
# 1) Imports raw review and metadata files
# 2) joins them by `parent_asin`
# 3) aggregates review title + text pairs into product-level review fields
# 4) derives summary features like average rating and candidate review +
# title to present in the app
# 5) saves the processed parquet for retrieval and app use.

c2.execute(
    f"""
    COPY (
        WITH joined AS (
            SELECT
                r.parent_asin,
                r.rating,
                r.helpful_vote,
                r.title AS review_title,
                r.text,
                m.title AS product_title,
                m.features,
                m.description,
                m.categories,
                m.details,
                m.price,
                CASE
                    WHEN r.title IS NOT NULL AND TRIM(r.title) <> ''
                         AND r.text IS NOT NULL AND TRIM(r.text) <> ''
                        THEN r.title || ': ' || r.text
                    WHEN r.title IS NOT NULL AND TRIM(r.title) <> ''
                        THEN r.title
                    WHEN r.text IS NOT NULL AND TRIM(r.text) <> ''
                        THEN r.text
                    ELSE NULL
                END AS review_doc
            FROM read_parquet('{RAW_DIR}/reviews_raw.parquet') r
            LEFT JOIN read_parquet('{RAW_DIR}/meta_raw.parquet') m
                USING (parent_asin)
        ),
        ranked_reviews AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY parent_asin
                    ORDER BY
                        CASE
                            WHEN (review_title IS NOT NULL AND TRIM(review_title) <> '')
                              OR (text IS NOT NULL AND TRIM(text) <> '')
                            THEN 1
                            ELSE 0
                        END DESC,
                        COALESCE(helpful_vote, -1) DESC,
                        LENGTH(COALESCE(text, '')) DESC,
                        LENGTH(COALESCE(review_title, '')) DESC
                ) AS rn
            FROM joined
        ),
        aggregated AS (
            SELECT
                parent_asin,
                product_title,
                features,
                description,
                categories,
                details,
                price,
                AVG(rating) AS derived_avg_rating,
                MAX(helpful_vote) AS max_helpful_vote,
                COUNT(*) AS n_reviews,
                LIST(review_doc) FILTER (WHERE review_doc IS NOT NULL) AS review_docs,
                STRING_AGG(review_doc, ' ') FILTER (WHERE review_doc IS NOT NULL) AS review_text
            FROM joined
            GROUP BY
                parent_asin,
                product_title,
                features,
                description,
                categories,
                details,
                price
        ),
        candidate_reviews AS (
            SELECT
                parent_asin,
                review_title AS candidate_review_title,
                text AS candidate_review_text,
                helpful_vote AS candidate_review_helpful_vote
            FROM ranked_reviews
            WHERE rn = 1
        )
        SELECT
            a.*,
            c.candidate_review_title,
            c.candidate_review_text,
            c.candidate_review_helpful_vote
        FROM aggregated a
        LEFT JOIN candidate_reviews c USING (parent_asin)
    )
    TO '{DATA_DIR}/processed.parquet'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """
)
