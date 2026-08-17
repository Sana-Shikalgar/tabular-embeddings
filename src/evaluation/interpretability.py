"""Section 3.5's LIME and SHAP explanations of Step 4's host models
(src/evaluation/host_model.py) -- Ribeiro, M.T., Singh, S. & Guestrin, C.
(2016), "'Why Should I Trust You?': Explaining the Predictions of Any
Classifier," KDD 2016 (the KDD conference paper version, per [D29], not the
earlier arXiv preprint); Lundberg, S.M. & Lee, S.-I. (2017), "A Unified
Approach to Interpreting Model Predictions," NeurIPS 2017.

shap_explain uses shap.TreeExplainer, not the model-agnostic Kernel/sampling
SHAP explainer -- see its own docstring for why that's a legitimate
implementation-level choice, not a change of what's being measured.

lime_explain_sample and summarize_lime work through LIME's per-instance
top-num_features explanations, not a dense per-instance attribution matrix
like SHAP's -- summarize_lime's own docstring states the zero-fill
convention this implies.

attribution_agreement ([D49]) reduces a whole representation's SHAP/LIME
comparison to one Spearman correlation for Table 3.3's trade-off view; see
its own docstring for what that number does and does not stand in for.

KNOWN ENVIRONMENT CRASH -- LimeTabularExplainer.explain_instance segfaults
the Python process outright (same class of Windows BLAS/LAPACK bug already
documented in subtab.py, clustering_metrics.py, and tsne_viz.py), this time
in the local Ridge regression LIME fits on its default num_samples=5000
perturbation matrix. Confirmed on real data: crashes on classical (4282
columns) and subtab (1024 columns) at the literal default, both as a bare
segfault in a standalone script and as a real kernel death mid-cell in
06_evaluation_protocol.ipynb; raw (16), ft_transformer (192), and scarf
(256) never crashed, even at the default. num_samples=300 was confirmed
safe for classical and subtab in isolation -- a fresh kernel loading only
what lime_explain_sample itself needs -- but STILL crashed the kernel when
run as part of the real notebook's full cell sequence (host-model fitting,
SHAP, clustering, t-SNE all having already accumulated state in the same
kernel first). num_samples=150 was then confirmed safe end-to-end: a full
`jupyter nbconvert --execute` run of the real notebook, top to bottom,
completed cleanly for all five representations (classical's 100-row LIME
loop took 95s, the slowest, well under this module's own 180s warning
threshold). lime_explain_sample hardcodes num_samples=150 for every
representation, not just the two that needed it -- the small fidelity cost
on raw/ft_transformer/scarf's local explanations buys one code path with no
per-representation branching, matching the "hardcode the minimum necessary
fix" precedent already set in tsne_viz.py. If this crash resurfaces on a
future environment or larger candidate split, re-verify against a full
notebook run, not just an isolated script or a fresh-kernel probe -- both
under-reproduced the real crash here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr


def shap_explain(model: xgboost.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Returns per-sample, per-feature SHAP values, shape (n_samples, n_features).

    TreeExplainer exploits XGBoost's tree structure directly, giving exact
    Shapley values faster than the model-agnostic Kernel/sampling SHAP
    explainer. This is a legitimate implementation-level optimisation -- it
    does not change what is being measured (the same Shapley value
    definition), only how it is computed.
    """
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
    """Returns one LIME explanation object per row of X_sample.

    Builds a single LimeTabularExplainer fit on X_train (mode="regression",
    discretize_continuous=True) and calls .explain_instance(row,
    model.predict, num_features=num_features, num_samples=LIME_NUM_SAMPLES)
    for each row of X_sample, reusing the same explainer rather than
    rebuilding it per row. num_samples=150 (LIME's own default is 5000) is
    hardcoded, not a parameter -- see this module's own docstring, KNOWN
    ENVIRONMENT CRASH, for why: the literal default segfaults this
    environment on wide representations (classical, subtab), and 150 was
    confirmed safe, uniformly, for every representation this project uses,
    in a full end-to-end run of the real notebook. LIME is the slow path in
    this module (one local surrogate model fit per row); prints progress
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
    """Top_k features by mean absolute SHAP value, index = feature name,
    values = mean |shap|, sorted descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(top_k)


def summarize_lime(lime_explanations: list, top_k: int = 10) -> pd.Series:
    """Top_k features by mean absolute LIME weight across all explained
    instances, index = feature name, values = mean |weight|, sorted
    descending.

    LIME's explain_instance only returns each instance's own
    top-num_features features (as_map()), not a dense per-instance vector
    over every feature. This function's mean is over ALL explained
    instances for every feature that appears in ANY instance's
    explanation: an instance where a given feature is absent from its
    own top-num_features contributes 0 for that feature, not a skipped
    value -- so a feature that only ever shows up as a strong outlier in
    a few instances is not overstated relative to one that appears
    consistently but more mildly.
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
    """[D49] This single number is a derived summary for the Step 8 /
    Table 3.3 trade-off view only. It does not replace the qualitative
    per-representation LIME-vs-SHAP disagreement discussion Section 3.5's
    prose requires -- a high correlation can still hide a specific,
    informative disagreement on one important feature, and this function
    is not designed to surface that.

    Takes the union of shap_summary's and lime_summary's own top_k
    features, ranks each summary's features within that union (a feature
    present in one summary's top_k but absent from the other's values is
    treated as 0 for the other, then ranked accordingly; ties broken by
    alphabetical feature name for determinism), and returns
    scipy.stats.spearmanr's correlation between the two rank sequences.
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
