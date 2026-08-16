"""Pooled embeddings for list-valued fields, Eq. 3.5:
pool(S) = (1/|S|) * sum_{i in S} e_i for mean, sum_{i in S} e_i for sum.
Zaheer et al. (2017, Thm 2) establish that sum/mean pooling over a set's
element embeddings is permutation-invariant -- the pooled vector doesn't
depend on the order items happen to appear in the list, which is exactly
the property needed here since `genres`/`keywords`/etc. have no
meaningful order.

ListFieldPooler itself only builds the vocabulary and the per-row token
index lists; the actual per-item nn.Embedding table and the call to
`pool()` on its output belong in the paradigm-specific model code, same
division of responsibility as CategoricalLookup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ListFieldPooler:
    def __init__(self, col: str, embedding_dim: int, pooling: str = "mean"):
        self.col = col
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.token_to_index: dict[str, int] | None = None

    def fit(self, train_df: pd.DataFrame) -> "ListFieldPooler":
        # One code path for every list-valued field: explode to one row per
        # item, drop the NaN an empty list produces, take the unique tokens.
        vocab = sorted(train_df[self.col].explode().dropna().unique().tolist())
        self.token_to_index = {token: i + 1 for i, token in enumerate(vocab)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        def to_indices(items):
            if not pd.api.types.is_list_like(items):
                return []
            return [self.token_to_index.get(item, 0) for item in items]

        return df[self.col].apply(to_indices)

    @staticmethod
    def pool(item_embeddings, mode: str = "mean"):
        if len(item_embeddings) == 0:
            # (1/|S|) * sum(e_i) is 0/0 at |S|=0; a zero vector rather than
            # NaN, so a row with no items doesn't poison whatever this feeds
            # into downstream. Matches item_embeddings' own type: pool() is
            # called both on plain np.ndarray and, inside a torch
            # nn.Module's forward(), on torch.Tensor -- a numpy return in
            # the latter case would silently break autograd and device
            # placement when stacked alongside real tensor tokens.
            embedding_dim = item_embeddings.shape[1] if item_embeddings.ndim == 2 else 0
            if isinstance(item_embeddings, np.ndarray):
                return np.zeros(embedding_dim, dtype=np.float32)
            # Not np.ndarray -- assume torch.Tensor (NumPy 2.x arrays also
            # carry a .device attribute now, so that alone can't
            # distinguish the two; isinstance against np.ndarray can).
            import torch

            return torch.zeros(
                embedding_dim, dtype=item_embeddings.dtype, device=item_embeddings.device
            )

        if mode == "mean":
            return item_embeddings.mean(axis=0)
        elif mode == "sum":
            return item_embeddings.sum(axis=0)
        else:
            raise ValueError(f"Unknown pooling mode: {mode!r}")

    def to_dict(self) -> dict:
        return {
            "col": self.col,
            "embedding_dim": self.embedding_dim,
            "pooling": self.pooling,
            "token_to_index": self.token_to_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ListFieldPooler":
        obj = cls(d["col"], d["embedding_dim"], pooling=d["pooling"])
        obj.token_to_index = dict(d["token_to_index"])
        return obj
