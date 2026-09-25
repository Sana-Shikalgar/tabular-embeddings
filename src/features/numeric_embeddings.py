"""Piecewise-linear encoding: bins a numeric column by training-quantile
edges and encodes each value by its fractional position within its bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PiecewiseLinearEncoder:
    """Quantile-binned piecewise-linear encoder for one or more numeric columns."""

    def __init__(self, cols: list[str], n_bins: int = 10):
        """Stores the columns to encode and the target number of bins."""
        self.cols = list(cols)
        self.n_bins = n_bins
        self.edges_: dict[str, np.ndarray] | None = None
        self.n_bins_: dict[str, int] | None = None

    def fit(self, train_df: pd.DataFrame) -> "PiecewiseLinearEncoder":
        """Computes quantile bin edges per column from the training split."""
        edges_ = {}
        n_bins_ = {}
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        for col in self.cols:
            edges = train_df[col].quantile(quantiles).to_numpy(dtype=float)
            edges = np.unique(edges)  # drop trivial (zero-width) bins
            edges_[col] = edges
            n_bins_[col] = len(edges) - 1

        self.edges_ = edges_
        self.n_bins_ = n_bins_
        return self

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Encodes each column's values by their position within their fitted bin."""
        result = {}
        for col in self.cols:
            edges = self.edges_[col]
            x = df[col].to_numpy(dtype=float)[:, None]
            lower = edges[:-1][None, :]
            upper = edges[1:][None, :]
            fractional = (x - lower) / (upper - lower)

            below_bin = x < lower
            below_bin[:, 0] = False  # first bin: no zero-override, formula handles x < b_0
            above_bin = x >= upper
            above_bin[:, -1] = False  # last bin: no one-override, formula handles x >= b_T

            encoded = np.where(below_bin, 0.0, fractional)
            encoded = np.where(above_bin, 1.0, encoded)
            result[col] = encoded
        return result

    def fit_transform(self, train_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Fits on train_df, then transforms it."""
        self.fit(train_df)
        return self.transform(train_df)

    def to_dict(self) -> dict:
        """Serializes the fitted bin edges and settings to a plain dict."""
        return {
            "cols": self.cols,
            "n_bins": self.n_bins,
            "edges_": {col: edges.tolist() for col, edges in self.edges_.items()},
            "n_bins_": self.n_bins_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PiecewiseLinearEncoder":
        """Reconstructs a fitted PiecewiseLinearEncoder from to_dict()'s output."""
        obj = cls(d["cols"], n_bins=d["n_bins"])
        obj.edges_ = {col: np.array(edges, dtype=float) for col, edges in d["edges_"].items()}
        obj.n_bins_ = dict(d["n_bins_"])
        return obj
