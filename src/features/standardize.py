"""Standardization per Eq. 3.1: z = (x - mu) / sigma. mu and sigma are fit
on the training split only, then applied unchanged to val/test, so no
information about the val/test distribution leaks into scaling.
"""

from __future__ import annotations

import pandas as pd


class Standardizer:
    def __init__(self, cols: list[str]):
        self.cols = list(cols)
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> "Standardizer":
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
        df = df.copy()
        df[self.cols] = (df[self.cols] - self.mean_) / self.std_
        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)

    def to_dict(self) -> dict:
        return {
            "cols": self.cols,
            "mean_": self.mean_.to_dict(),
            "std_": self.std_.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        obj = cls(d["cols"])
        obj.mean_ = pd.Series(d["mean_"])[obj.cols]
        obj.std_ = pd.Series(d["std_"])[obj.cols]
        return obj
