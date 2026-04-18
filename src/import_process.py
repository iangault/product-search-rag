# Script to import meta and review datasets and process them to be inputs

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

def main():
    """
    Imports and pre-processes data
    """
    c2 = duckdb.connect()

    # Review Data
    c2.execute(
        f"""
        COPY (SELECT *
                FROM read_json_auto('{REVIEWS_URL}') 
                LIMIT 200000)
        TO '{RAW_DIR}/reviews_raw.parquet'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    )

    # Meta data
    # Filter out NaNs for product title
    c2.execute(
        f"""
        COPY (SELECT *
                FROM read_json_auto('{META_URL}')
                WHERE title IS NOT NULL
                AND TRIM(title) <> ''
        LIMIT 20000)
        TO '{RAW_DIR}/meta_raw.parquet'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    )

    ##### PROCESSING ######

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
                -- Join product metadata to raw review rows and
                -- normalize review text.
                SELECT
                    m.parent_asin,
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
                    -- `review_doc`: combine title/text into
                    -- one displayable review string.
                    -- Prefer "title: text" when both exist, otherwise
                    -- fall back to whichever field is populated.
                    CASE
                        WHEN r.title IS NOT NULL AND TRIM(r.title) <> ''
                            AND r.text IS NOT NULL AND TRIM(r.text) <> ''
                            THEN r.title || ': ' || r.text
                        WHEN r.title IS NOT NULL AND TRIM(r.title) <> ''
                            THEN r.title
                        WHEN r.text IS NOT NULL AND TRIM(r.text) <> ''
                            THEN r.text
                        ELSE NULL
                    END AS review_doc,
                    -- `has_review`: mark rows with any review
                    -- signal as real reviews.
                    CASE
                        WHEN r.rating IS NOT NULL
                        OR r.helpful_vote IS NOT NULL
                        OR (r.title IS NOT NULL AND TRIM(r.title) <> '')
                        OR (r.text IS NOT NULL AND TRIM(r.text) <> '')
                        THEN 1
                        ELSE 0
                    END AS has_review
                FROM read_parquet('{RAW_DIR}/meta_raw.parquet') m
                LEFT JOIN read_parquet('{RAW_DIR}/reviews_raw.parquet') r
                    USING (parent_asin)
                WHERE m.title IS NOT NULL
                AND TRIM(m.title) <> ''
            ),
            ranked_reviews AS (
                -- Prefer reviews with visible text/title,
                -- then break ties by helpfulness and length.
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY parent_asin
                        ORDER BY
                            -- First prioritize rows with any readable
                            -- title/text, then use helpful votes and
                            -- longer content as tie-breakers.
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
                -- Collapse review rows to one product row
                -- with product-level aggregates.
                SELECT
                    parent_asin,
                    product_title,
                    features,
                    description,
                    categories,
                    details,
                    price,
                    -- Average rating across real review rows only.
                    AVG(rating) FILTER (WHERE has_review = 1 AND rating IS NOT NULL) AS derived_avg_rating,
                    -- Count only rows that passed the `has_review`
                    -- screen above.
                    COUNT(*) FILTER (WHERE has_review = 1) AS n_reviews,
                    -- Flatten review snippets into one searchable field
                    -- for downstream retrieval and UI use. Separate
                    -- reviews with blank lines so the app can render
                    -- them as distinct blocks.
                    STRING_AGG(review_doc, '\n\n') FILTER (WHERE review_doc IS NOT NULL) AS review_text
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
                -- Keep one representative review per product
                -- for app display.
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

if __name__ == "__main__":
    main()
