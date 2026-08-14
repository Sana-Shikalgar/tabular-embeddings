"""Loaders for the two candidate datasets used in this dissertation's EDA:
TMDB (movies, downloaded via the Kaggle API) and Inside Airbnb London
(listings, downloaded manually — Inside Airbnb has no stable scriptable URL).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd

from src.config import (
    AIRBNB_RAW_DIR,
    AIRBNB_SOURCE_URL,
    TMDB_KAGGLE_DATASET,
    TMDB_RAW_DIR,
)

# dtype hints for the TMDB CSV (930k-movies dump). This intentionally covers
# 22 of the dataset's columns; "adult" is left to pandas' default bool
# inference and "release_date" is handled via parse_dates in load_tmdb(), so
# neither appears in this dict — it is not the full column list.
TMDB_DTYPES = {
    "id": "Int64",
    "title": "string",
    "vote_average": "float64",
    "vote_count": "Int64",
    "status": "category",
    "revenue": "float64",
    "runtime": "float64",
    "budget": "float64",
    "original_language": "category",
    "original_title": "string",
    "overview": "string",
    "popularity": "float64",
    "tagline": "string",
    "genres": "string",
    "production_companies": "string",
    "production_countries": "string",
    "spoken_languages": "string",
    "keywords": "string",
    "imdb_id": "string",
    "homepage": "string",
    "backdrop_path": "string",
    "poster_path": "string",
}

_TMDB_MANUAL_INSTRUCTIONS = """\
No Kaggle credentials found (expected a KAGGLE_API_TOKEN environment
variable, or a token file at ~/.kaggle/access_token), and no CSV file was
found in {raw_dir}.

To fetch the Full TMDB Movies Dataset 2024 manually:
  1. Go to https://www.kaggle.com/datasets/{dataset}
  2. Click "Download" (requires a free Kaggle account).
  3. Unzip the archive and place the CSV file (e.g. TMDB_movie_dataset_v11.csv)
     into: {raw_dir}

Alternatively, set up Kaggle API credentials and re-run this function to
download automatically: generate a token at
https://www.kaggle.com/settings/api and save it to ~/.kaggle/access_token
(or set the KAGGLE_API_TOKEN environment variable).
"""

_AIRBNB_MANUAL_INSTRUCTIONS = """\
No Airbnb listings file found in {raw_dir}.

This dataset must be downloaded manually from Inside Airbnb (there is no
stable scriptable URL):
  1. Go to {url}
     (or browse http://insideairbnb.com/get-the-data/ for a later snapshot)
  2. Download the "listings.csv.gz" file for London, England.
  3. Place it, unmodified, at: {raw_dir}/listings.csv.gz

Then re-run this loader.
"""


def _has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    access_token_path = config_dir / "access_token"
    return access_token_path.exists() and access_token_path.stat().st_size > 0


def _download_tmdb_via_kaggle(raw_dir: Path) -> None:
    print(f"Downloading '{TMDB_KAGGLE_DATASET}' from Kaggle into {raw_dir}...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            TMDB_KAGGLE_DATASET,
            "-p",
            str(raw_dir),
            "--unzip",
        ],
        check=True,
    )


def load_tmdb(raw_dir: Path = TMDB_RAW_DIR) -> pd.DataFrame:
    """Load the Full TMDB Movies Dataset 2024, downloading it via the Kaggle
    API into ``raw_dir`` first if it isn't already present there.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        if not _has_kaggle_credentials():
            raise FileNotFoundError(
                _TMDB_MANUAL_INSTRUCTIONS.format(raw_dir=raw_dir, dataset=TMDB_KAGGLE_DATASET)
            )
        try:
            _download_tmdb_via_kaggle(raw_dir)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise FileNotFoundError(
                _TMDB_MANUAL_INSTRUCTIONS.format(raw_dir=raw_dir, dataset=TMDB_KAGGLE_DATASET)
            ) from exc
        csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            _TMDB_MANUAL_INSTRUCTIONS.format(raw_dir=raw_dir, dataset=TMDB_KAGGLE_DATASET)
        )

    df = pd.read_csv(csv_files[0], dtype=TMDB_DTYPES, parse_dates=["release_date"])
    return df


def load_airbnb(raw_dir: Path = AIRBNB_RAW_DIR) -> pd.DataFrame:
    """Load Inside Airbnb London listings from ``raw_dir``. Raises a helpful
    error if no file is present, since this dataset must be fetched manually.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(raw_dir.glob("listings.csv*"))
    if not candidates:
        raise FileNotFoundError(
            _AIRBNB_MANUAL_INSTRUCTIONS.format(raw_dir=raw_dir, url=AIRBNB_SOURCE_URL)
        )

    df = pd.read_csv(candidates[0], low_memory=False)

    if "price" in df.columns:
        df["price"] = (
            df["price"].astype("string").str.replace(r"[\$,]", "", regex=True).astype("float64")
        )

    return df
