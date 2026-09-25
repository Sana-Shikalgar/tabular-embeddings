"""LIME and SHAP explanations of 06_evaluation_protocol.ipynb's host models
(src/evaluation/host_model.py), plus a SHAP/LIME agreement score for the
Table 3.3 trade-off view. num_samples for LIME is hardcoded low (150) to
avoid a known segfault in this environment on wide representations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr


def shap_explain(model: xgboost.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Returns per-sample, per-feature SHAP values, shape (n_samples, n_features), via shap.TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


LIME_NUM_SAMPLES = 150


def lime_explain_sample(
    model: xgboost.XGBRegressor,
    X_train: np.ndarray,
    X_sample: np.ndarray,
    feature_names: list[str],
    random_state: int,
    num_features: int = 10,
) -> list:
    """Returns one LIME explanation object per row of X_sample, from a
    single LimeTabularExplainer fit on X_train (mode="regression",
    discretize_continuous=True) and reused for every row. num_samples is
    hardcoded to LIME_NUM_SAMPLES (150, well below LIME's own default of
    5000) to avoid a known crash in this environment on wide
    representations. LIME is the slow path in this module; prints progress
    every 20 rows.
    """
    explainer = LimeTabularExplainer(
        training_data=X_train,
        mode="regression",
        feature_names=feature_names,
        discretize_continuous=True,
        random_state=random_state,
    )

    explanations = []
    for i, row in enumerate(X_sample):
        explanations.append(
            explainer.explain_instance(
                row, model.predict, num_features=num_features, num_samples=LIME_NUM_SAMPLES
            )
        )
        if (i + 1) % 20 == 0:
            print(f"lime_explain_sample: explained {i + 1}/{len(X_sample)} rows")
    return explanations


def summarize_shap(shap_values: np.ndarray, feature_names: list[str], top_k: int = 10) -> pd.Series:
    """Returns the top_k features by mean absolute SHAP value, sorted descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(top_k)


def summarize_lime(lime_explanations: list, top_k: int = 10) -> pd.Series:
    """Returns the top_k features by mean absolute LIME weight, averaged
    across all explained instances. LIME only returns each instance's own
    top-num_features features, so an instance where a feature is absent
    from its own explanation contributes 0 for that feature, rather than
    being skipped.
    """
    n = len(lime_explanations)
    totals: dict[int, float] = {}
    for exp in lime_explanations:
        for feature_idx, weight in exp.as_map()[1]:
            totals[feature_idx] = totals.get(feature_idx, 0.0) + abs(weight)

    feature_names = lime_explanations[0].domain_mapper.feature_names
    means = pd.Series(
        {feature_names[idx]: total / n for idx, total in totals.items()}
    )
    return means.sort_values(ascending=False).head(top_k)


def attribution_agreement(shap_summary: pd.Series, lime_summary: pd.Series) -> dict[str, float]:
    """Returns {"spearman": ..., "overlap_at_10": ...} comparing the full
    (untruncated) shap_summary/lime_summary Series -- callers should pass
    everything summarize_shap/summarize_lime can produce, not a
    pre-truncated top_k, so real values are compared rather than an
    artificial rank structure.

    "spearman" is scipy.stats.spearmanr's correlation computed directly
    on the two summaries' own values, reindexed to the union of both
    summaries' features; a feature absent from one summary is filled
    with 0.0 there (genuinely missing, not a rank placeholder).
    spearmanr handles tied values internally, so no manual tiebreak is
    needed. "overlap_at_10" is the size of the intersection of the two
    summaries' own top-10 feature sets (0-10) -- a plain feature-set
    agreement, independent of value magnitude or rank.

    A single per-representation agreement score for the Table 3.3
    trade-off view -- it does not surface WHERE the two methods disagree.
    """
    union = shap_summary.index.union(lime_summary.index)
    shap_aligned = shap_summary.reindex(union).fillna(0.0)
    lime_aligned = lime_summary.reindex(union).fillna(0.0)
    spearman = spearmanr(shap_aligned, lime_aligned).correlation

    shap_top10 = set(shap_summary.head(10).index)
    lime_top10 = set(lime_summary.head(10).index)
    overlap_at_10 = len(shap_top10 & lime_top10)

    return {"spearman": float(spearman), "overlap_at_10": overlap_at_10}
