"""Shared constants for the TMDB and Airbnb loaders: dataset identifiers,
local data paths, and dataset provenance dates.

Dataset provenance — cite these dates verbatim in the dissertation's data
provenance section. Update them if a dataset is (re-)downloaded on a
different date than assumed here.
"""

from __future__ import annotations

from pathlib import Path

TMDB_KAGGLE_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
TMDB_DOWNLOAD_DATE = "2026-07-11"  # date this dataset was first downloaded via the Kaggle API, as the author of the dataset updates it daily and the Kaggle API always returns the latest version of the dataset, which may change over time

AIRBNB_SOURCE_URL = "https://insideairbnb.com/get-the-data/"
AIRBNB_SCRAPE_DATE = "2026-06-19"  #  Airbnb's landing page has the date of possible last scraping of data
AIRBNB_DOWNLOAD_DATE = "2026-07-11"  # date the file was manually downloaded and placed in data/raw/airbnb/

# Sentence-embedding model used for text features (see src/eda/helper.py's
# token_length_stats and _get_tokenizer). Multilingual since TMDB overview
# text spans many original_language values.
SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SBERT_MAX_SEQ_LENGTH = 128

# Anchored to this file's location (not the process cwd), so these resolve
# correctly regardless of where a notebook kernel or script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = REPO_ROOT / "data" / "raw"
TMDB_RAW_DIR = DATA_RAW_DIR / "tmdb"
AIRBNB_RAW_DIR = DATA_RAW_DIR / "airbnb"

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FINAL_DIR = REPO_ROOT / "data" / "final"