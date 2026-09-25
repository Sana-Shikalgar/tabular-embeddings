"""Z-score standardization: mean/std fit on the training split only,
then applied unchanged to any split.
"""

from __future__ import annotations

import pandas as pd


class Standardizer:
    """Fits per-column mean/std from a training split and standardizes any split with them."""

    def __init__(self, cols: list[str]):
        """Stores the columns to standardize."""
        self.cols = list(cols)
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> "Standardizer":
        """Computes per-column mean and population std from the training split."""
        mean_ = train_df[self.cols].mean()
        std_ = train_df[self.cols].std(ddof=0)

        nan_cols = std_[std_.isna()].index.tolist()
        assert not nan_cols, (
            f"NaN std for columns {nan_cols} -- all-NaN column, should not "
            "happen given 03's missingness check"
        )
        zero_cols = std_[std_ == 0].index.tolist()
        assert not zero_cols, f"Zero std (constant column) for {zero_cols}"

        self.mean_ = mean_
        self.std_ = std_
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes the fitted columns using the fitted mean/std."""
        df = df.copy()
        df[self.cols] = (df[self.cols] - self.mean_) / self.std_
        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fits on train_df, then transforms it."""
        self.fit(train_df)
        return self.transform(train_df)

    def to_dict(self) -> dict:
        """Serializes the fitted mean/std to a plain dict."""
        return {
            "cols": self.cols,
            "mean_": self.mean_.to_dict(),
            "std_": self.std_.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        """Reconstructs a fitted Standardizer from to_dict()'s output."""
        obj = cls(d["cols"])
        obj.mean_ = pd.Series(d["mean_"])[obj.cols]
        obj.std_ = pd.Series(d["std_"])[obj.cols]
        return obj
