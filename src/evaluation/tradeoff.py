"""Section 3.5's RQ4/DO5 trade-off table -- the basis for Table 3.3, combining
one metric from each of the three axes already computed elsewhere in this
notebook (clustering quality, interpretability agreement, fairness
amplification) into a single per-representation comparison.

This module deliberately does NOT include the Step 4 host regressor's
r2/mae anywhere. That score exists solely so Step 5's LIME/SHAP have
something to explain (host_model.py's own module docstring) -- it is a DO3
sanity check, never a DO2 representation-quality axis, and
Chapter3_Methodology_Structure.md is explicit that folding it into Table 3.3
would be "smuggling" a fourth axis in under cover of the other three. If a
future edit adds an r2/mae column here, that edit is wrong on its face --
this docstring is the guard against it happening by accident.

assemble_raw_table pulls exactly one column per source table (silhouette,
calinski_harabasz, davies_bouldin from clustering_table; agreement from
interpretability_table's attribution_agreement; delta_accuracy from
fairness_table) into one frame, asserting all three inputs share the same
representation index/order first -- concatenating misaligned tables would
silently mix one representation's clustering score with another's fairness
score. normalize_metrics then min-max scales that frame to [0, 1] per
column, in a fixed higher-is-better direction, for Table 3.3's combined
view only; see its own docstring for what is lost in that combination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assemble_raw_table(
    clustering_table: pd.DataFrame,
    interpretability_table: pd.DataFrame,
    fairness_table: pd.DataFrame,
) -> pd.DataFrame:
    """One row per representation, exactly these columns: silhouette,
    calinski_harabasz, davies_bouldin (clustering_table, Eq. 3.16-3.18),
    agreement (interpretability_table's attribution_agreement, [D49]),
    delta_accuracy (fairness_table, Eq. 3.20, signed).

    Asserts clustering_table, interpretability_table, and fairness_table
    all share the same row index in the same order before pulling any
    column out of them -- concatenating by column position rather than by
    a checked, matching index would silently attribute one
    representation's clustering score to a different representation's row.

    fairness_table's delta_accuracy is NaN for the "raw" representation
    itself (amplification is only defined relative to raw, per
    fairness.compute_amplification's own docstring) -- that NaN is carried
    through into this table's "raw" row unchanged, not filled or dropped.
    normalize_metrics documents what happens to it downstream.
    """
    assert clustering_table.index.equals(interpretability_table.index), (
        "clustering_table and interpretability_table do not share the same "
        f"representation index/order: {clustering_table.index.tolist()} vs "
        f"{interpretability_table.index.tolist()}"
    )
    assert clustering_table.index.equals(fairness_table.index), (
        "clustering_table and fairness_table do not share the same "
        f"representation index/order: {clustering_table.index.tolist()} vs "
        f"{fairness_table.index.tolist()}"
    )

    return pd.DataFrame(
        {
            "silhouette": clustering_table["silhouette"],
            "calinski_harabasz": clustering_table["calinski_harabasz"],
            "davies_bouldin": clustering_table["davies_bouldin"],
            "agreement": interpretability_table["attribution_agreement"],
            "delta_accuracy": fairness_table["delta_accuracy"],
        },
        index=clustering_table.index,
    )


def normalize_metrics(raw_table: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalises raw_table to [0, 1] per column, oriented so 1 is
    always the more favourable value, per this fixed direction convention:

        silhouette:         higher_better
        calinski_harabasz:  higher_better
        davies_bouldin:     lower_better  (normalise, then 1 - x)
        agreement:          higher_better
        delta_accuracy:     abs(delta_accuracy), then lower_better

    delta_accuracy is normalised on its ABSOLUTE VALUE, not its signed
    value: this treats a representation that amplifies the sensitive
    attribute by +0.10 and one that obscures it by -0.10 as equally
    unfavourable for this single combined score. That collapses the
    amplified-vs-obscured distinction Step 6's own fairness_table
    preserves in its signed delta_accuracy column -- Table 3.3's caption
    must point back to that signed column for readers who need to know
    WHICH direction a representation moved, not just how far. This
    function's output is a trade-off-view summary only, the same caveat
    interpretability.attribution_agreement's own docstring makes about its
    single agreement number.

    A column with zero variance across representations (every value
    identical) returns 0.5 for every row with a non-NaN input, rather than
    dividing by zero -- 0.5 signals "no representation differs on this
    metric," not an arbitrary min or max.

    raw_table's "raw" row carries a NaN delta_accuracy (assemble_raw_table's
    own docstring) -- that NaN is excluded from the min/max computed for
    the delta_accuracy column, and the "raw" row's own normalised
    delta_accuracy stays NaN in the output; it is never imputed to 0.5 or
    any other value, even when the column is otherwise zero-variance.
    """
    direction = {
        "silhouette": "higher_better",
        "calinski_harabasz": "higher_better",
        "davies_bouldin": "lower_better",
        "agreement": "higher_better",
        "delta_accuracy": "lower_better",
    }

    working = raw_table.copy()
    working["delta_accuracy"] = working["delta_accuracy"].abs()

    normalized = pd.DataFrame(index=raw_table.index)
    for col, col_direction in direction.items():
        values = working[col]
        col_min = values.min()
        col_max = values.max()

        if col_max == col_min:
            scaled = pd.Series(np.where(values.isna(), np.nan, 0.5), index=values.index)
        else:
            scaled = (values - col_min) / (col_max - col_min)
            if col_direction == "lower_better":
                scaled = 1 - scaled

        normalized[col] = scaled

    return normalized
