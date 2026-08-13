# smart-tabular-embeddings
Creating Smart Embeddings for Complex Multi-Type Tabular Datasets


## Environment Setup

This project uses a conda environment (Python 3.11).

```bash
conda env create -f environment.yml
conda activate smart-tabular-embeddings
```

To use this environment as a Jupyter kernel in VS Code / Jupyter, run once
from within the activated environment:

```bash
python -m ipykernel install --user --name smart-tabular-embeddings --display-name "Python (smart-tabular-embeddings)"
```

then select "Python (smart-tabular-embeddings)" as the kernel for notebooks
in `notebooks/`.

## Data Sources & Licensing

This branch works with a single dataset:

- **Full TMDB Movies Dataset 2024 (1M Movies)** — sourced via Kaggle.
  Licensed under the [Open Data Commons Attribution License (ODC-By) v1.0](https://opendatacommons.org/licenses/by/1-0/index.html).

Raw data files are not included in this repository. See `src/eda/loaders.py`
for how to obtain the dataset locally.

## What's in This Branch

A TMDB-only data pipeline, in four notebooks: characterizing the raw
dataset, cleaning it, constructing and comparing candidate feature sets, and
producing two final train/val/test splits. See "Notebook Run Order" below
for how the notebooks depend on each other, and "Dataset Shapes" for what
each stage produces.

## Dataset Shapes

| Stage | Rows | Columns |
|---|---:|---:|
| Raw (Kaggle download) | 1,456,829 | 24 |
| Cleaned (`01_overview_cleaning`) | 863,858 | 19 |
| Chosen candidate A -- `tmdb_br_gt0` | 10,504 | 20 |
| Chosen candidate B -- `tmdb_nbr_gt5` | 107,219 | 18 |

Two candidates come out of `02_preprocessing` and carry through
`03_feature_split`, differing in how they trade off row count against the
`budget`/`revenue` features:

- **`tmdb_br_gt0`** (`tmdb_budget_revenue_gt0`): rows with non-null
  `budget` **and** `revenue`, filtered to `vote_count > 0`. Keeps `budget`
  and `revenue` as features (20 columns), at the cost of a much smaller row
  count. Split 80/10/10 -> train 8,403 / val 1,050 / test 1,051.
- **`tmdb_nbr_gt5`** (`tmdb_no_budget_revenue_gt5`): `budget` and `revenue`
  dropped entirely as columns (too sparse to keep -- see `02_preprocessing`
  Section 1), filtered to `vote_count > 5`. 18 columns, roughly 10x the
  rows of the other candidate. Split 70/15/15 -> train 75,053 / val 16,083
  / test 16,083.

Both splits are stratified on `original_language`. All six resulting
train/val/test files are saved to `data/final/`.

## Features

**Raw dataset** (24 columns): `id`, `title`, `vote_average`, `vote_count`,
`status`, `release_date`, `revenue`, `runtime`, `adult`, `backdrop_path`,
`budget`, `homepage`, `imdb_id`, `original_language`, `original_title`,
`overview`, `popularity`, `poster_path`, `tagline`, `genres`,
`production_companies`, `production_countries`, `spoken_languages`,
`keywords`.

**`tmdb_br_gt0`** (20 columns): `id`, `title`, `vote_average`, `revenue`,
`runtime`, `budget`, `original_language`, `original_title`, `overview`,
`popularity`, `genres`, `production_companies`, `production_countries`,
`spoken_languages`, `keywords`, `has_production_companies`,
`has_keywords`, `release_year`, `release_month_sin`, `release_month_cos`.

**`tmdb_nbr_gt5`** (18 columns): same as `tmdb_br_gt0` above, minus
`revenue` and `budget`.

## Notebook Run Order

Each notebook (except `00`) reads a file the previous one writes, so they
must be run in order:

1. **`00_tmdb_eda.ipynb`** -- characterizes the raw dataset (missingness,
   cardinality, text/token-length, language detection). Produces no saved
   file; its findings inform the cleaning decisions made in `01`.
2. **`01_overview_cleaning.ipynb`** -- cleans the raw data (recodes,
   dedup, drops, caps and drops) ->
   `data/processed/tmdb_clean.parquet`.
3. **`02_preprocessing.ipynb`** -- builds and compares 7 candidate
   variants, refines and selects the two carried forward ->
   `data/processed/tmdb_br_gt0.parquet`, `data/processed/tmdb_nbr_gt5.parquet`.
4. **`03_feature_split.ipynb`** -- stratified train/val/test split, rare-
   language and vocabulary bucketing -> the six files in `data/final/`.

Each notebook also logs a timestamped summary to `reports/eda_log.md`.

## Shared Code (`src/eda/`)

Every notebook's Setup cell imports from `src/eda/` rather than
duplicating logic: `loaders.py` (dataset loading), `helper.py` (EDA/
cleaning helper functions), `report.py` (figure saving and the
`reports/eda_log.md` logger), and `config.py` (shared constants).

## Development Notes

### Notebook outputs and `nbstripout`

This repo has [`nbstripout`](https://github.com/kynan/nbstripout) installed
as a git filter (`nbstripout --install`), which strips notebook cell
outputs from `.ipynb` files at commit time. This means:

- Rendered outputs stay in your local working copy — nothing is deleted on
  disk, and re-running a notebook restores them.
- `git diff` / commits only show code and markdown changes, not embedded
  image/output blobs, so notebook diffs stay readable.
- If a GitHub view of a notebook looks like it's missing output cells,
  that's expected — it's the last *committed* version, which never has
  outputs baked in.

If you want a specific notebook's outputs to actually go into a commit
(e.g. so it renders with output on GitHub), you have two options:

- Temporarily disable the filter for that commit: `git commit --no-verify`
  won't help here since this is a content filter, not a hook — instead run
  `nbstripout --uninstall` first, commit, then optionally re-run
  `nbstripout --install` afterwards to resume stripping.
- Or force-add the notebook bypassing the filter: `git add --renormalize`
  won't restore stripped output either, since stripping happens on the
  `git add`/commit path itself — uninstalling before committing is the
  reliable option.
