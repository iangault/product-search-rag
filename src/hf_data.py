# Title: Hugging Face Data Sync
# Purpose: Download the prebuilt processed data and indexes from a public
# Hugging Face dataset repo so the deployed app does not need them in git.
# Run this file directly to upload local artifacts after a rebuild.
# Date: 2026-09-24

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

root_dir = Path(__file__).resolve().parent.parent
processed_dir = root_dir / "data" / "processed"

HF_REPO_ID = "gaultian/product-search-rag-data"

# Files the app needs at startup, relative to data/processed
REQUIRED_FILES = [
    "processed.parquet",
    "bm25_index.pkl",
    "semantic.index",
    "semantic.ids.npy",
    "rag_faiss/index.faiss",
    "rag_faiss/index.pkl",
]


def missing_files():
    """Return the required files that are not present locally."""
    return [f for f in REQUIRED_FILES if not (processed_dir / f).exists()]


def ensure_processed_data():
    """Download any missing artifacts from the Hugging Face dataset repo."""
    missing = missing_files()
    if not missing:
        return
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=processed_dir,
        allow_patterns=missing,
    )


def upload_processed_data():
    """Upload local artifacts to the Hugging Face dataset repo.

    Needs a write token, e.g. from `huggingface-cli login`.
    """
    missing = missing_files()
    if missing:
        raise FileNotFoundError(
            f"Missing local artifacts: {missing}. Run `make all` first."
        )
    api = HfApi()
    api.create_repo(HF_REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        folder_path=processed_dir,
        allow_patterns=REQUIRED_FILES,
        commit_message="Update processed data and indexes",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{HF_REPO_ID}")


if __name__ == "__main__":
    upload_processed_data()
