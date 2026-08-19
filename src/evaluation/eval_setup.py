"""Shared evaluation setup used throughout 06_evaluation_protocol.ipynb: loads
representations and candidate splits, checks row alignment, and provides the
shared XGBoost config and label-preparation helpers every evaluation step
builds on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

REPR_NAMES = ("raw", "classical", "ft_transformer", "subtab", "scarf")
FIT_SPLIT = "train"
EVAL_SPLIT = "test"
SENSITIVE_COL = "original_language"
TARGET_COL = "vote_average"
GENRE_COL = "genres"


@dataclass(frozen=True)
class SharedXGBConfig:
    """Fixed, un-tuned XGBoost config applied identically to every
    representation, so a downstream score difference reflects the
    representation, not a per-representation tuning advantage.
    """

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = config.RANDOM_SEED
    n_jobs: int = -1

    def regressor_kwargs(self) -> dict:
        """Shared fields plus the regression objective/eval_metric, ready for xgboost.XGBRegressor(**kwargs)."""
        return {**asdict(self), "objective": "reg:squarederror", "eval_metric": "rmse"}

    def classifier_kwargs(self) -> dict:
        """Shared fields plus the classification objective/eval_metric, ready for xgboost.XGBClassifier(**kwargs)."""
        return {**asdict(self), "objective": "multi:softprob", "eval_metric": "mlogloss"}


def load_representations(
    models_dir: Path,
    repr_names: tuple[str, ...] = REPR_NAMES,
    splits: tuple[str, ...] = config.SPLIT_NAMES,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Loads every representation's saved (N, d) feature table from
    models_dir / {representation} / f"{split}.parquet", one row per input
    row in the source split's own order. Returns
    representations[repr_name][split] -> pd.DataFrame. Raises
    FileNotFoundError naming the exact missing path if any file is absent.
    """
    representations: dict[str, dict[str, pd.DataFrame]] = {}
    for repr_name in repr_names:
        representations[repr_name] = {}
        for split in splits:
            path = models_dir / repr_name / f"{split}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing representation file for representation={repr_name!r}, "
                    f"split={split!r}: {path} -- Step 3.3 (raw/classical) or Step 3.4 "
                    "(ft_transformer/subtab/scarf) must produce it first"
                )
            representations[repr_name][split] = pd.read_parquet(path)
    return representations


def load_candidate_splits(
    final_dir: Path,
    candidate: str = config.ACTIVE_CANDIDATE,
    splits: tuple[str, ...] = config.SPLIT_NAMES,
) -> dict[str, pd.DataFrame]:
    """Loads the candidate's own final split parquet files from
    final_dir / f"tmdb_{candidate}_{split}.parquet". These frames carry
    `id`, `genres`, `original_language`, `vote_average` -- label columns
    none of the representation frames (load_representations) carry
    directly.
    """
    candidate_splits: dict[str, pd.DataFrame] = {}
    for split in splits:
        path = final_dir / f"tmdb_{candidate}_{split}.parquet"
        assert path.exists(), f"Missing candidate split file: {path}"
        candidate_splits[split] = pd.read_parquet(path)
    return candidate_splits


def assert_row_alignment(
    representations: dict[str, dict[str, pd.DataFrame]],
    candidate_splits: dict[str, pd.DataFrame],
    splits: tuple[str, ...] = config.SPLIT_NAMES,
) -> None:
    """Asserts every representation frame is in the same row order as
    candidate_splits[split]: first checks row counts match, then compares
    raw's "runtime" column elementwise against candidate_splits' own
    "runtime" (np.allclose, since it's an unstandardised float passthrough)
    as a genuine row-order proof, not just a length check. Raises
    AssertionError naming the split and representation on mismatch.
    """
    for split in splits:
        expected_len = len(candidate_splits[split])
        for repr_name, split_frames in representations.items():
            actual_len = len(split_frames[split])
            assert actual_len == expected_len, (
                f"row count mismatch for representation={repr_name!r}, split={split!r}: "
                f"{actual_len} rows vs candidate_splits[{split!r}]'s {expected_len} rows"
            )

    for split in splits:
        raw_runtime = representations["raw"][split]["runtime"].to_numpy()
        candidate_runtime = candidate_splits[split]["runtime"].to_numpy()
        assert np.allclose(raw_runtime, candidate_runtime), (
            f"row-order mismatch for representation='raw', split={split!r}: "
            "raw's 'runtime' column does not elementwise match candidate_splits' "
            "'runtime' column -- the two frames are not in the same row order"
        )


def derive_genre_label(
    candidate_splits: dict[str, pd.DataFrame], fit_split: str = FIT_SPLIT
) -> dict[str, pd.Series]:
    """Derives each row's genre label as genres[0]. Empty-list rows are
    filled with the most frequent first-listed genre in fit_split alone
    (never derived from val/test), so the fallback value is a fixed
    constant reused across every split. Prints how many rows in each split
    needed the fallback.
    """
    first_genre_by_split: dict[str, pd.Series] = {
        split: df[GENRE_COL].apply(lambda genres: genres[0] if len(genres) > 0 else None)
        for split, df in candidate_splits.items()
    }

    non_null_fit = first_genre_by_split[fit_split].dropna()
    assert not non_null_fit.empty, (
        f"fit_split={fit_split!r} has no rows with a non-empty genres list -- "
        "cannot derive a fallback genre from it"
    )
    fallback_genre = non_null_fit.value_counts().idxmax()

    labels: dict[str, pd.Series] = {}
    for split, first_genre in first_genre_by_split.items():
        n_fallback = int(first_genre.isna().sum())
        print(
            f"derive_genre_label: {split}: {n_fallback} row(s) used the "
            f"fallback genre {fallback_genre!r}"
        )
        labels[split] = first_genre.fillna(fallback_genre)

    return labels


def drop_sensitive_attribute_columns(df: pd.DataFrame, representation: str) -> pd.DataFrame:
    """Drops the sensitive-attribute column(s) from a raw/classical
    representation, so a recoverability probe can't trivially recover
    original_language from a column that IS original_language. "raw" drops
    the single SENSITIVE_COL column; "classical" drops SENSITIVE_COL and
    every one-hot column starting with SENSITIVE_COL + "_". Any other
    representation is returned unchanged. Prints the columns actually
    dropped, or "none".
    """
    if representation == "raw":
        cols_to_drop = [c for c in df.columns if c == SENSITIVE_COL]
    elif representation == "classical":
        prefix = SENSITIVE_COL + "_"
        cols_to_drop = [c for c in df.columns if c == SENSITIVE_COL or c.startswith(prefix)]
    else:
        cols_to_drop = []

    if cols_to_drop:
        print(f"drop_sensitive_attribute_columns[{representation}]: dropped {cols_to_drop}")
    else:
        print(f"drop_sensitive_attribute_columns[{representation}]: dropped none")

    return df.drop(columns=cols_to_drop)
