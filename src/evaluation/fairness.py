"""Section 3.5's recoverability probe and amplification metric (Eq. 3.20) --
Song, C. & Raghunathan, A. (2020), "Information Leakage in Embedding
Models," ACM CCS 2020, for the recoverability-probe mechanism: train a
classifier to predict a sensitive attribute FROM a representation, and read
its score as how much of that attribute the representation leaks, whether or
not the representation was ever meant to encode it. Zemel, R., Wu, Y.,
Swersky, K., Pitassi, T. & Dwork, C. (2013), "Learning Fair Representations,"
ICML 2013, for the converse case compute_amplification must also report
correctly: a representation can OBSCURE the sensitive attribute relative to
raw features, not just amplify it, and Zemel et al. (2013) is the source for
treating that obscuring outcome as a real, reportable result in its own
right, not a degenerate case of "no leakage."

fit_recoverability_probe/evaluate_probe operate on ONE representation at a
time -- Section 3.5's actual Step 6 cell calls them once per representation,
on eval_setup.drop_sensitive_attribute_columns'd raw/classical features and
the untouched ft_transformer/subtab/scarf embeddings ([D46]'s leakage
guard), then feeds each representation's own accuracy/macro_f1 and raw's
accuracy/macro_f1 into compute_amplification to get that representation's
amplification score.
"""

from __future__ import annotations

import numpy as np
import xgboost
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.eval_setup import SharedXGBConfig


def fit_recoverability_probe(
    X_train: np.ndarray, y_train_encoded: np.ndarray, xgb_config: SharedXGBConfig
) -> xgboost.XGBClassifier:
    """Described in the dissertation text as "a classifier trained to
    recover the sensitive attribute", per [D44] -- not as "the probing
    classifier", since that term does not originate in Song & Raghunathan
    (2020). Same classifier type and same fixed hyperparameters as every
    other representation's probe, per [D45] -- no per-representation
    tuning: xgb_config is eval_setup.SharedXGBConfig, the identical config
    every other XGBoost fit in this project uses, fit here with
    xgb_config.classifier_kwargs().

    y_train_encoded is the sensitive attribute (original_language) already
    label-encoded to integer class ids -- this function does not encode it
    itself, so the same encoding must be used for fit and evaluate_probe.
    """
    model = xgboost.XGBClassifier(**xgb_config.classifier_kwargs())
    model.fit(X_train, y_train_encoded)
    return model


def evaluate_probe(model: xgboost.XGBClassifier, X_test: np.ndarray, y_test_encoded: np.ndarray) -> dict:
    """Returns {"accuracy": ..., "macro_f1": ...} -- both, not accuracy
    alone. The sensitive attribute's classes are almost certainly
    imbalanced after the training-split rare-language bucketing in
    03_feature_split.ipynb (some kept languages will have far more rows
    than others, plus the "other" bucket), so a single majority-class-
    dominated accuracy figure could understate a representation's
    recoverability of the less common classes. macro_f1 (sklearn's
    f1_score, average="macro") weights every class equally regardless of
    its support, surfacing exactly the leakage accuracy alone would hide.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test_encoded, y_pred)),
        "macro_f1": float(f1_score(y_test_encoded, y_pred, average="macro")),
    }


def compute_amplification(metric_embedding: float, metric_raw: float) -> float:
    """Eq. 3.20: metric_embedding - metric_raw, for a single metric
    (accuracy or macro_f1) computed identically for both.

    Positive = amplified relative to raw features; negative = obscured
    (the Zemel et al. 2013 converse case is real and should be reported as
    such, not treated as an uninteresting null result). This function
    returns the signed difference exactly as computed -- it does not clamp,
    take an absolute value, or otherwise collapse the two cases into one
    number that reads the same either way.
    """
    return metric_embedding - metric_raw
