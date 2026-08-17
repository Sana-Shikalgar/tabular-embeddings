"""Section 3.5's shared evaluation setup: the one loading/alignment/labelling
layer every downstream Chapter 4 comparison (predictive performance,
[D46]'s recoverability probe, etc.) builds on, so those steps load
already-checked data instead of each re-deriving it.

load_representations loads all five representations' saved feature tables
(Steps 3.3/3.4's own "Save ..." cells in 04_feature_encoding.ipynb /
05_embedding_paradigms.ipynb); load_candidate_splits loads the candidate's
own final split files, which carry the label/id columns none of the five
representation frames carry directly. assert_row_alignment is the one
check every later step relies on without re-checking: that a representation
frame's row i is candidate_splits' row i, for every representation and
every split. derive_genre_label and drop_sensitive_attribute_columns both
prepare inputs to specific downstream probes -- a genre clustering label,
and a leakage guard for the sensitive-attribute recoverability probe --
rather than being generic utilities.

SharedXGBConfig is the one model configuration every representation is
scored with, deliberately fixed and un-tuned (see its own docstring).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVE_CANDIDATE = "br_gt0"
REPR_NAMES = ("raw", "classical", "ft_transformer", "subtab", "scarf")
SPLITS = ("train", "val", "test")
FIT_SPLIT = "train"
EVAL_SPLIT = "test"
SENSITIVE_COL = "original_language"
TARGET_COL = "vote_average"
GENRE_COL = "genres"
RANDOM_SEED = 42


@dataclass(frozen=True)
class SharedXGBConfig:
    """Section 3.5's shared XGBoost config, applied IDENTICALLY to every
    representation (raw, classical, ft_transformer, subtab, scarf) --
    deliberately un-tuned, fixed defaults, matching the D12
    no-per-method-tuning precedent already set in Sections 3.1/3.4
    (SubTabConfig.svd_variance_threshold, SharedTrainingConfig.patience
    overriding each paper's own reported number): comparing
    representations is the point of Chapter 4, not finding the best
    XGBoost hyperparameters for any one of them. Not benchmarked, not
    optimised per representation -- one config, five representations, so
    any downstream score difference reflects the representation, not a
    tuning advantage one of them happened to get.

    n_estimators/max_depth/learning_rate/subsample/colsample_bytree/
    reg_lambda: common, reasonable general-purpose XGBoost defaults, not
    sourced from a specific paper -- unlike FT-Transformer/SubTab/SCARF,
    Section 3.5 has no single source paper to match. Chosen once here and
    reused for every fit.

    regressor_kwargs()/classifier_kwargs() each merge these shared fields
    with the one task-specific objective/eval_metric pair into one dict a
    caller passes straight to xgboost.XGBRegressor(**kwargs) /
    xgboost.XGBClassifier(**kwargs) -- the shared fields are never
    duplicated or restated at the call site.
    """

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = RANDOM_SEED
    n_jobs: int = -1

    def regressor_kwargs(self) -> dict:
        return {**asdict(self), "objective": "reg:squarederror", "eval_metric": "rmse"}

    def classifier_kwargs(self) -> dict:
        return {**asdict(self), "objective": "multi:softprob", "eval_metric": "mlogloss"}


def load_representations(
    artifacts_dir: Path,
    repr_names: tuple[str, ...] = REPR_NAMES,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Loads every representation's saved (N, d) feature table --
    ft_transformer/subtab/scarf from Step 3.4's "Save ... embeddings"
    cells (05_embedding_paradigms.ipynb), raw/classical from Step 3.3's
    "Save raw and classical baselines" cell (04_feature_encoding.ipynb).
    All five were written to the identical
    artifacts_dir / "models" / {representation} / f"{split}.parquet"
    path convention, one row per input row, in the source split's own row
    order -- no id/target column saved alongside any of them.

    Returns representations[repr_name][split] -> pd.DataFrame.

    Raises FileNotFoundError naming the exact missing path if any file is
    absent: ft_transformer/subtab/scarf require Step 3.4 to have already
    produced them; raw/classical require Step 3.3.
    """
    representations: dict[str, dict[str, pd.DataFrame]] = {}
    for repr_name in repr_names:
        representations[repr_name] = {}
        for split in splits:
            path = artifacts_dir / "models" / repr_name / f"{split}.parquet"
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
    candidate: str = ACTIVE_CANDIDATE,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, pd.DataFrame]:
    """Loads the candidate's own final split parquet files -- the same
    final_dir / f"tmdb_{candidate}_{split}.parquet" paths the CANDIDATES
    dict in both 04_feature_encoding.ipynb and 05_embedding_paradigms.ipynb
    reads. These frames carry `id`, `genres`, `original_language`,
    `vote_average` -- the label source for every downstream evaluation
    step, since none of the five representation frames
    (load_representations) carry these columns directly: each
    representation frame is features only, in row-position alignment with
    these splits, not a superset that includes them.
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
    splits: tuple[str, ...] = SPLITS,
) -> None:
    """Every representation's frame must be in the exact same row order as
    candidate_splits[split] -- every downstream step (label derivation,
    sensitive-attribute dropping, XGBoost fitting) assumes this without
    re-checking it per call, so it is checked here, once, up front.

    Row COUNT alone doesn't prove row ORDER: two frames can have identical
    length while being silently permuted relative to each other. "raw" is
    the one representation that keeps a column verbatim from the source
    frame (build_raw_baseline's numeric_cols pass through unstandardised
    -- baselines.py), so comparing raw[split]["runtime"] against
    candidate_splits[split]["runtime"] elementwise is a genuine row-order
    proof for "raw" specifically, not just another length check.
    np.allclose, not exact equality, since runtime is an unstandardised
    float passthrough surviving a parquet round-trip, not an
    integer/string id column guaranteed bit-for-bit stable.

    Raises AssertionError naming the split and representation responsible
    on the first mismatch found.
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
    """First-listed genre; empty-list fallback = most frequent first-listed
    genre in the training split, applied identically to val/test so the
    fallback constant is never derived from data outside train.

    Each row's label is genres[0] -- except when genres is an empty list
    (NOT NaN; 02_preprocessing.ipynb already normalises missing list
    fields to [], confirmed on the real br_gt0 data: 0 NaNs, 55 empty-list
    rows in train alone), which has no [0] to take. Those rows are filled
    with whichever first-listed genre is most common across fit_split's
    own non-empty rows -- computed once, on fit_split only, then reused as
    a fixed constant for every split (including fit_split's own empty
    rows), so val/test's fallback value is never influenced by val/test's
    own data.

    Prints how many rows in each split needed the fallback.
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
    """[D46] Without this, the Step 6 recoverability probe on raw/classical
    would be predicting original_language from a column that IS
    original_language (raw's own ordinal-encoded SENSITIVE_COL column, or
    classical's one-hot SENSITIVE_COL_* block) -- producing a meaningless
    ~100% score rather than a genuine test of whether the REST of the
    representation leaks the sensitive attribute.

    "raw": drops SENSITIVE_COL itself if present -- build_raw_baseline
    keeps it as one ordinal-encoded column, named exactly SENSITIVE_COL
    (baselines.py).

    "classical": drops every column named SENSITIVE_COL or starting with
    SENSITIVE_COL + "_" -- build_classical_baseline one-hot expands it
    into f"{lang_col}_{token}" / f"{lang_col}_oov" columns (baselines.py),
    so no single column name matches; the prefix catches the whole block.
    Confirmed against the real classical baseline's actual columns:
    original_language_oov/_en/_es/_fr/_hi/_other/_ru.

    Any other representation (ft_transformer/subtab/scarf): returned
    unchanged -- these are opaque learned embeddings, not one-hot/ordinal
    features, so there is no column that IS original_language to begin
    with, nothing to drop.

    Prints the columns actually dropped (or "none" for non-raw/classical,
    or if the column(s) were already absent) so the caller can see the
    guard fired.
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
