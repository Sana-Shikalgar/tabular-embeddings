"""Assembles and normalises 06_evaluation_protocol.ipynb's Table 3.3
trade-off table from clustering, interpretability, and fairness results
computed earlier in that notebook. Deliberately excludes the host
regressor's r2/mae -- that's a sanity check, not a quality axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assemble_raw_table(
    clustering_table: pd.DataFrame,
    interpretability_table: pd.DataFrame,
    fairness_table: pd.DataFrame,
) -> pd.DataFrame:
    """Combines one column from each source table into a single per-
    representation frame: silhouette, calinski_harabasz, davies_bouldin
    from clustering_table; agreement from interpretability_table's
    attribution_agreement; delta_accuracy from fairness_table. Asserts all
    three inputs share the same row index/order first, so a column is
    never attributed to the wrong representation. fairness_table's NaN
    delta_accuracy for "raw" is carried through unchanged.
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
    """Min-max scales raw_table to [0, 1] per column, oriented so 1 is
    always more favourable: silhouette/calinski_harabasz/agreement are
    higher_better as-is; davies_bouldin and delta_accuracy (on its
    absolute value) are flipped to lower_better. A zero-variance column
    returns 0.5 for every non-NaN row instead of dividing by zero.
    "raw"'s NaN delta_accuracy is excluded from the column's min/max and
    stays NaN in the output, never imputed.
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
