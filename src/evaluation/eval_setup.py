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
from src.helper import load_representation_with_id

REPR_NAMES = ("raw", "classical", "ft_transformer", "subtab", "scarf")
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
    row in the source split's own order. Every file carries `id` as its
    first column (the save_representation_with_id convention); this
    function strips it back out via load_representation_with_id so
    representations[repr_name][split] is features only, never leaking
    `id` into a downstream .to_numpy() call. Raises FileNotFoundError
    naming the exact missing path if any file is absent.
    """
    representations: dict[str, dict[str, pd.DataFrame]] = {}
    for repr_name in repr_names:
        representations[repr_name] = {}
        for split in splits:
            path = models_dir / repr_name / f"{split}.parquet"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing representation file for representation={repr_name!r}, "
                    f"split={split!r}: {path} -- 04_feature_encoding.ipynb (raw/classical) "
                    "or 05_embedding_paradigms.ipynb (ft_transformer/subtab/scarf) must "
                    "produce it first"
                )
            features_df, _id = load_representation_with_id(path)
            representations[repr_name][split] = features_df
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
    candidate_splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Derives each row's genre label as genres[0]. A row with an empty
    genres list has no first genre to derive -- rather than fabricating
    one, this returns two parallel dicts: labels[split] (genres[0] where
    available, NaN elsewhere) and valid_mask[split] (True where
    labels[split] is real). A caller using labels[split] as a clustering
    label must first filter both the label and its matching feature rows
    to valid_mask[split] -- e.g. X[mask], labels[split][mask] -- since
    NaN is not a usable label. Prints how many rows in each split have no
    first genre and are excluded.
    """
    labels: dict[str, pd.Series] = {}
    valid_mask: dict[str, pd.Series] = {}

    for split, df in candidate_splits.items():
        first_genre = df[GENRE_COL].apply(lambda genres: genres[0] if len(genres) > 0 else None)
        mask = first_genre.notna()
        n_excluded = int((~mask).sum())
        print(
            f"derive_genre_label: {split}: {n_excluded} row(s) have no first "
            "genre and are excluded from clustering labels"
        )
        labels[split] = first_genre
        valid_mask[split] = mask

    return labels, valid_mask


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
