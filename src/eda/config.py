"""Shared constants for the TMDB and Airbnb loaders: dataset identifiers,
local data paths, and dataset provenance dates.

Dataset provenance — cite these dates verbatim in the dissertation's data
provenance section. Update them if a dataset is (re-)downloaded on a
different date than assumed here.
"""

from __future__ import annotations

from pathlib import Path

TMDB_KAGGLE_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
TMDB_DOWNLOAD_DATE = "2026-07-10"  # date this dataset was first downloaded via the Kaggle API

AIRBNB_SOURCE_URL = "https://insideairbnb.com/get-the-data/"
AIRBNB_SCRAPE_DATE = "2026-07-10"  # Inside Airbnb's landing page has no dated URL; this is the download date, used as the best available reference
AIRBNB_DOWNLOAD_DATE = "2026-07-10"  # date the file was manually downloaded and placed in data/raw/airbnb/

DATA_RAW_DIR = Path("data/raw")
TMDB_RAW_DIR = DATA_RAW_DIR / "tmdb"
AIRBNB_RAW_DIR = DATA_RAW_DIR / "airbnb"