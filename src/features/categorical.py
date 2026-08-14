"""Categorical embedding lookup per Eq. 3.6: maps a fitted, bucketed
categorical column to integer indices for an `nn.Embedding` table. Index
0 is reserved for an explicit unseen-at-inference fallback -- val/test
should never actually need it, since `03_feature_split.ipynb`'s
`bucket_rare_languages()` already folds anything outside the training
vocabulary into `"other"` before this class ever sees the column, but a
silent crash on a genuinely unseen token would be worse than a defined
fallback.

`embedding_dim` should be set from the post-binning cardinality already
computed in `03_feature_split.ipynb`'s Section 4 (`entity_embedding_dim`),
not the raw 179-language-code figure -- that recomputation already
accounts for bucketing having collapsed the vocabulary down to the kept
languages plus `"other"`.

The `nn.Embedding` table itself belongs in the paradigm-specific model
code (Section 3.4), not here -- this class's only job is the fitted
vocabulary and the int lookup, dataset- and framework-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CategoricalLookup:
    def __init__(self, col: str, embedding_dim: int):
        self.col = col
        self.embedding_dim = embedding_dim
        self.token_to_index: dict[str, int] | None = None

    def fit(self, train_df: pd.DataFrame) -> "CategoricalLookup":
        # Reuses whatever cardinality bucket_rare_languages() already
        # produced in the saved parquet -- no recomputation here.
        tokens = sorted(train_df[self.col].dropna().unique().tolist())
        self.token_to_index = {token: i + 1 for i, token in enumerate(tokens)}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return (
            df[self.col]
            .map(lambda token: self.token_to_index.get(token, 0))
            .to_numpy(dtype=np.int64)
        )

    def to_dict(self) -> dict:
        return {
            "col": self.col,
            "embedding_dim": self.embedding_dim,
            "token_to_index": self.token_to_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CategoricalLookup":
        obj = cls(d["col"], d["embedding_dim"])
        obj.token_to_index = dict(d["token_to_index"])
        return obj
