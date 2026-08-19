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


def attribution_agreement(shap_summary: pd.Series, lime_summary: pd.Series, top_k: int = 10) -> float:
    """Returns the Spearman correlation between shap_summary's and
    lime_summary's feature rankings over the union of their top_k
    features (a feature missing from one summary is treated as 0 there;
    ties broken alphabetically). A single per-representation agreement
    score for the Table 3.3 trade-off view -- it does not surface WHERE
    the two methods disagree.
    """
    shap_top = shap_summary.head(top_k)
    lime_top = lime_summary.head(top_k)
    union = sorted(set(shap_top.index) | set(lime_top.index))

    def _ranks(summary: pd.Series) -> list[int]:
        values = {f: summary.get(f, 0.0) for f in union}
        order = sorted(union, key=lambda f: (-values[f], f))
        rank_by_feature = {f: rank + 1 for rank, f in enumerate(order)}
        return [rank_by_feature[f] for f in union]

    shap_ranks = _ranks(shap_top)
    lime_ranks = _ranks(lime_top)
    return spearmanr(shap_ranks, lime_ranks).correlation
