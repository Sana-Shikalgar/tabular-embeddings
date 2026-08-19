"""Splits a candidate DataFrame's columns into target/id/feature groups,
and further partitions feature columns by type.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureTargetSplit:
    """Fixed target/id/feature column split for a candidate DataFrame."""

    target_col: str
    id_cols: list[str]
    feature_cols: list[str]

    def select(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Returns (feature columns, target column) as (DataFrame, Series)."""
        return df[self.feature_cols], df[self.target_col]

    def drop_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns df restricted to feature and id columns, excluding the target."""
        return df[self.feature_cols + self.id_cols]


def build_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "vote_average",
    id_cols: tuple[str, ...] = ("id", "title",),
) -> FeatureTargetSplit:
    """Builds a FeatureTargetSplit from df's columns, given the target and id column names."""
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


@dataclass(frozen=True)
class ColumnGroups:
    """Feature columns partitioned by type: numeric, bypass, categorical, text, list-valued."""

    numeric_cols: list[str]
    bypass_cols: list[str]
    categorical_cols: list[str]
    text_cols: list[str]
    list_cols: list[str]


def _is_list_valued(s: pd.Series) -> bool:
    """True if s holds list-like values in every non-null cell."""
    if s.dtype != object:
        return False
    non_null = s.dropna()
    if non_null.empty:
        return False
    # pd.read_parquet round-trips list cells as numpy.ndarray, not Python
    # list -- is_list_like (not `type(x) is list`) is the check that
    # actually survives that round trip.
    return non_null.apply(pd.api.types.is_list_like).all()


def _is_bypass(s: pd.Series, name: str) -> bool:
    """True if s is boolean, or a sin/cos-named float column, both passed through unchanged."""
    if pd.api.types.is_bool_dtype(s):
        return True
    return pd.api.types.is_float_dtype(s) and (name.endswith("_sin") or name.endswith("_cos"))


def build_column_groups(df: pd.DataFrame, split: FeatureTargetSplit) -> ColumnGroups:
    """Partitions split.feature_cols into five type-based groups by
    inspecting df's actual dtypes, asserting every column lands in
    exactly one group."""
    feature_cols = split.feature_cols

    list_cols = [c for c in feature_cols if _is_list_valued(df[c])]
    bypass_cols = [c for c in feature_cols if _is_bypass(df[c], c)]
    categorical_cols = [c for c in feature_cols if isinstance(df[c].dtype, pd.CategoricalDtype)]
    text_cols = [c for c in feature_cols if isinstance(df[c].dtype, pd.StringDtype)]

    already_claimed = set(list_cols) | set(bypass_cols) | set(categorical_cols) | set(text_cols)
    numeric_cols = [
        c for c in feature_cols
        if c not in already_claimed and pd.api.types.is_numeric_dtype(df[c])
    ]

    groups = ColumnGroups(
        numeric_cols=numeric_cols,
        bypass_cols=bypass_cols,
        categorical_cols=categorical_cols,
        text_cols=text_cols,
        list_cols=list_cols,
    )

    assignment_counts = Counter(
        c
        for group in (numeric_cols, bypass_cols, categorical_cols, text_cols, list_cols)
        for c in group
    )
    unassigned = [c for c in feature_cols if assignment_counts[c] == 0]
    double_assigned = sorted(c for c, n in assignment_counts.items() if n > 1)
    assert not unassigned and not double_assigned, (
        "build_column_groups did not produce a clean partition of feature_cols: "
        f"unassigned={unassigned}, double_assigned={double_assigned}"
    )

    return groups
