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

from collections import Counter
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


@dataclass(frozen=True)
class ColumnGroups:
    numeric_cols: list[str]
    bypass_cols: list[str]
    categorical_cols: list[str]
    text_cols: list[str]
    list_cols: list[str]


def _is_list_valued(s: pd.Series) -> bool:
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
    if pd.api.types.is_bool_dtype(s):
        return True
    return pd.api.types.is_float_dtype(s) and (name.endswith("_sin") or name.endswith("_cos"))


def build_column_groups(df: pd.DataFrame, split: FeatureTargetSplit) -> ColumnGroups:
    """Partitions split.feature_cols into five groups by inspecting the
    actual dtypes/values in df -- not by a hardcoded column-name list, so
    a future schema change is caught by the exhaustiveness check below
    rather than silently mis-routed."""
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
