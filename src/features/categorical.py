"""Fits a string-to-integer vocabulary for a categorical column, reserving
index 0 for values unseen at inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CategoricalLookup:
    """Vocabulary-based integer encoder for a single categorical column."""

    def __init__(self, col: str):
        """Stores which column to encode; the vocabulary is empty until fit()."""
        self.col = col
        self.token_to_index: dict[str, int] | None = None

    def fit(self, train_df: pd.DataFrame) -> "CategoricalLookup":
        """Builds the vocabulary from the training split's distinct values."""
        # Reuses whatever cardinality bucket_rare_languages() already
        # produced in the saved parquet -- no recomputation here.
        tokens = sorted(train_df[self.col].dropna().unique().tolist())
        self.token_to_index = {token: i + 1 for i, token in enumerate(tokens)}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Maps the column's values to their fitted indices, 0 for unseen values."""
        return (
            df[self.col]
            .map(lambda token: self.token_to_index.get(token, 0))
            .to_numpy(dtype=np.int64)
        )

    def to_dict(self) -> dict:
        """Serializes the fitted vocabulary to a plain dict."""
        return {
            "col": self.col,
            "token_to_index": self.token_to_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CategoricalLookup":
        """Reconstructs a fitted CategoricalLookup from to_dict()'s output."""
        obj = cls(d["col"])
        obj.token_to_index = dict(d["token_to_index"])
        return obj
