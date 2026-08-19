"""Shared constants for the TMDB loader: dataset identifiers, local data
paths, report/figures output paths, and dataset provenance dates.

Dataset provenance — cite these dates verbatim in the dissertation's data
provenance section. Update them if a dataset is (re-)downloaded on a
different date than assumed here.
"""

from __future__ import annotations

from pathlib import Path

TMDB_KAGGLE_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
TMDB_DOWNLOAD_DATE = "2026-07-11"  # date this dataset was first downloaded via the Kaggle API, as the author of the dataset updates it daily and the Kaggle API always returns the latest version of the dataset, which may change over time

# Sentence-embedding model used for text features (see src/eda/helper.py's
# token_length_stats and _get_tokenizer). Multilingual since TMDB overview
# text spans many original_language values.
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SBERT_MAX_SEQ_LENGTH = 128

# The one reproducibility seed for every sample/shuffle/split in this
# project -- every notebook and .py file must import this rather than
# hardcoding its own 42, so a single value is auditable/changeable in one
# place instead of trusting that every local literal agrees with it.
RANDOM_SEED = 42

# Anchored to this file's location (not the process cwd), so these resolve
# correctly regardless of where a notebook kernel or script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
TMDB_RAW_DIR = DATA_RAW_DIR / "tmdb"

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FINAL_DIR = REPO_ROOT / "data" / "final"

REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"