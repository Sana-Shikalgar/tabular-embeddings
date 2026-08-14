"""Enforces the target/feature/identifier split defined in Section 3.2:
`vote_average` is the supervised target only. Every non-supervised path
must build its feature frame via `.drop_target()`, so the target column
isn't just "unused" by convention -- it is physically not present in the
frame handed to that code.

`id` is the only column excluded as non-generalisable. `title` and
`original_title` are deliberately treated as text features, not
identifiers -- both carry distinct, fairness-relevant content once broken
down by `original_language`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureTargetSplit:
    target_col: str
    id_cols: list[str]
    feature_cols: list[str]

    def select(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df[self.feature_cols], df[self.target_col]

    def drop_target(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.feature_cols + self.id_cols]


def build_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "vote_average",
    id_cols: tuple[str, ...] = ("id", "title",),
) -> FeatureTargetSplit:
    assert target_col in df.columns, f"target_col {target_col!r} not in df.columns"
    for col in id_cols:
        assert col in df.columns, f"id_col {col!r} not in df.columns"

    excluded = {target_col, *id_cols}
    feature_cols = [c for c in df.columns if c not in excluded]

    return FeatureTargetSplit(
        target_col=target_col,
        id_cols=list(id_cols),
        feature_cols=feature_cols,
    )
