"""Builds a per-item vocabulary for a list-valued column and pools its
item embeddings (mean or sum) into one fixed-size vector per row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ListFieldPooler:
    """Vocabulary-based integer encoder for a list-valued column, plus a
    static pooling helper for its item embeddings."""

    def __init__(self, col: str, embedding_dim: int, pooling: str = "mean"):
        """Stores the column, embedding dimension, and pooling mode."""
        self.col = col
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.token_to_index: dict[str, int] | None = None

    def fit(self, train_df: pd.DataFrame) -> "ListFieldPooler":
        """Builds the vocabulary from every item seen in the training split."""
        # One code path for every list-valued field: explode to one row per
        # item, drop the NaN an empty list produces, take the unique tokens.
        vocab = sorted(train_df[self.col].explode().dropna().unique().tolist())
        self.token_to_index = {token: i + 1 for i, token in enumerate(vocab)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        """Maps each row's list to a list of fitted indices, 0 for unseen items."""
        def to_indices(items):
            if not pd.api.types.is_list_like(items):
                return []
            return [self.token_to_index.get(item, 0) for item in items]

        return df[self.col].apply(to_indices)

    @staticmethod
    def pool(item_embeddings: np.ndarray, mode: str = "mean") -> np.ndarray:
        """Mean- or sum-pools a row's item embeddings; an empty set pools to zeros."""
        if len(item_embeddings) == 0:
            # (1/|S|) * sum(e_i) is 0/0 at |S|=0; a zero vector rather than
            # NaN, so a row with no items doesn't poison whatever this feeds
            # into downstream.
            embedding_dim = item_embeddings.shape[1] if item_embeddings.ndim == 2 else 0
            return np.zeros(embedding_dim, dtype=np.float32)

        if mode == "mean":
            return item_embeddings.mean(axis=0)
        elif mode == "sum":
            return item_embeddings.sum(axis=0)
        else:
            raise ValueError(f"Unknown pooling mode: {mode!r}")

    def to_dict(self) -> dict:
        """Serializes the fitted vocabulary and settings to a plain dict."""
        return {
            "col": self.col,
            "embedding_dim": self.embedding_dim,
            "pooling": self.pooling,
            "token_to_index": self.token_to_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ListFieldPooler":
        """Reconstructs a fitted ListFieldPooler from to_dict()'s output."""
        obj = cls(d["col"], d["embedding_dim"], pooling=d["pooling"])
        obj.token_to_index = dict(d["token_to_index"])
        return obj
