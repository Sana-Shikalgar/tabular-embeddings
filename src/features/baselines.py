"""Builds the "raw" and "classical" baseline feature encodings that the
learned embedding paradigms are benchmarked against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.categorical import CategoricalLookup
from src.features.list_pooling import ListFieldPooler
from src.features.standardize import Standardizer
from src.features.target_split import FeatureTargetSplit

# Free text: no classical (one-hot/multi-hot) encoding applies, and no
# naive numeric encoding either -- dropped from both the raw and the
# classical condition. 
EXCLUDED_TEXT_COLS = {"overview", "original_title"}


def build_raw_baseline(
    df: pd.DataFrame,
    split: FeatureTargetSplit,
    numeric_cols: list[str],
    categorical_lookup: CategoricalLookup,
    list_valued_cols: list[str],
) -> pd.DataFrame:
    """Encodes df with minimal preprocessing: numeric columns as-is,
    original_language as a single integer code, list fields as item counts."""
    lang_col = categorical_lookup.col

    missing_numeric = [c for c in numeric_cols if c not in split.feature_cols]
    assert not missing_numeric, f"numeric_cols not in split.feature_cols: {missing_numeric}"
    missing_list_cols = [c for c in list_valued_cols if c not in split.feature_cols]
    assert not missing_list_cols, f"list_valued_cols not in split.feature_cols: {missing_list_cols}"
    assert lang_col in split.feature_cols, f"{lang_col!r} not in split.feature_cols"

    # numeric_cols: original scale, no standardisation.
    numeric_part = df[numeric_cols].copy()

    # original_language: a single train-fit integer per row (ordinal, not
    # one-hot) -- CategoricalLookup.transform() already is exactly this.
    lang_part = pd.DataFrame({lang_col: categorical_lookup.transform(df)}, index=df.index)

    # list-valued fields: raw item count per row, not a vocab-based encoding.
    def list_len(items):
        return len(items) if pd.api.types.is_list_like(items) else 0

    list_part = pd.DataFrame(
        {col: df[col].apply(list_len) for col in list_valued_cols},
        index=df.index,
    )

    # Everything else in feature_cols that isn't numeric, original_language,
    # a list field, or excluded free text: already-numeric/boolean columns
    # pass through unchanged -- asserted, not assumed.
    handled_cols = set(numeric_cols) | {lang_col} | set(list_valued_cols) | EXCLUDED_TEXT_COLS
    passthrough_cols = [c for c in split.feature_cols if c not in handled_cols]
    non_numeric_passthrough = [
        c for c in passthrough_cols
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]
    assert not non_numeric_passthrough, (
        f"unhandled non-numeric passthrough columns: {non_numeric_passthrough}"
    )
    passthrough_part = df[passthrough_cols].copy()

    return pd.concat([numeric_part, lang_part, list_part, passthrough_part], axis=1)


def build_classical_baseline(
    df: pd.DataFrame,
    split: FeatureTargetSplit,
    numeric_cols: list[str],
    standardizer: Standardizer,
    categorical_lookup: CategoricalLookup,
    list_vocabs: dict[str, ListFieldPooler],
) -> pd.DataFrame:
    """Encodes df with standard preprocessing: numeric columns standardized,
    original_language one-hot, list fields multi-hot."""
    assert set(numeric_cols) == set(standardizer.cols), (
        "numeric_cols must match the already-fitted standardizer's own cols: "
        f"numeric_cols={numeric_cols}, standardizer.cols={standardizer.cols}"
    )
    numeric_part = standardizer.transform(df)[numeric_cols]

    # original_language: Map original language to fixed integer codes, 
    # then use get_dummies with all possible codes to ensure consistent columns across train/val/test.
    lang_col = categorical_lookup.col
    lang_index_to_token = {idx: token for token, idx in categorical_lookup.token_to_index.items()}
    n_lang_tokens = len(categorical_lookup.token_to_index)

    lang_codes = df[lang_col].map(lambda tok: categorical_lookup.token_to_index.get(tok, 0))
    lang_codes = pd.Categorical(lang_codes, categories=range(n_lang_tokens + 1))
    lang_dummies = pd.get_dummies(lang_codes).astype(int)
    lang_dummies.columns = [
        f"{lang_col}_oov" if idx == 0 else f"{lang_col}_{lang_index_to_token[idx]}"
        for idx in lang_dummies.columns
    ]
    lang_dummies.index = df.index

    # Each list field: multi-hot against its own fitted vocabulary, via the
    # same per-row index lists ListFieldPooler.transform() already produces
    # (already reusing the train-derived vocab and OOV convention).
    list_parts = []
    for field, pooler in list_vocabs.items():
        idx_series = pooler.transform(df)
        n_tokens = len(pooler.token_to_index)
        mat = np.zeros((len(df), n_tokens + 1), dtype=int)
        for row_i, indices in enumerate(idx_series):
            if indices:
                mat[row_i, indices] = 1

        field_index_to_token = {idx: token for token, idx in pooler.token_to_index.items()}
        columns = [
            f"{field}_oov" if idx == 0 else f"{field}_{field_index_to_token[idx]}"
            for idx in range(n_tokens + 1)
        ]
        list_parts.append(pd.DataFrame(mat, columns=columns, index=df.index))

    # Everything else in feature_cols that isn't numeric, original_language,
    # a list field, or excluded free text: already-numeric/boolean columns
    # (e.g. has_keywords, release_month_sin/cos) pass through unchanged.
    handled_cols = set(numeric_cols) | {lang_col} | set(list_vocabs.keys()) | EXCLUDED_TEXT_COLS
    passthrough_cols = [c for c in split.feature_cols if c not in handled_cols]
    non_numeric_passthrough = [
        c for c in passthrough_cols
        if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]
    assert not non_numeric_passthrough, (
        f"unhandled non-numeric passthrough columns: {non_numeric_passthrough}"
    )
    passthrough_part = df[passthrough_cols].copy()

    return pd.concat([numeric_part, lang_dummies, *list_parts, passthrough_part], axis=1)
