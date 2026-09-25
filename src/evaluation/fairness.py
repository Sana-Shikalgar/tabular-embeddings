"""Recoverability probe and amplification metric used in
06_evaluation_protocol.ipynb's fairness check: trains a classifier to
predict the sensitive attribute (original_language) from a representation,
then compares its score against raw features to see whether the
representation amplifies or obscures that attribute.
"""

from __future__ import annotations

import numpy as np
import xgboost
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.eval_setup import SharedXGBConfig


def fit_recoverability_probe(
    X_train: np.ndarray, y_train_encoded: np.ndarray, xgb_config: SharedXGBConfig
) -> xgboost.XGBClassifier:
    """Fits an XGBoost classifier to predict the label-encoded sensitive
    attribute (y_train_encoded) from X_train, using the shared XGBoost
    config's classifier_kwargs(). y_train_encoded must already be
    label-encoded to integer class ids -- the same encoding must be used
    for evaluate_probe.
    """
    model = xgboost.XGBClassifier(**xgb_config.classifier_kwargs())
    model.fit(X_train, y_train_encoded)
    return model


def evaluate_probe(model: xgboost.XGBClassifier, X_test: np.ndarray, y_test_encoded: np.ndarray) -> dict:
    """Returns {"accuracy": ..., "macro_f1": ...} for the probe on
    X_test/y_test_encoded. macro_f1 is included because the sensitive
    attribute's classes are imbalanced, so accuracy alone could understate
    recoverability of the less common classes.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test_encoded, y_pred)),
        "macro_f1": float(f1_score(y_test_encoded, y_pred, average="macro")),
    }


def compute_amplification(metric_embedding: float, metric_raw: float) -> float:
    """Returns metric_embedding - metric_raw for a single metric (accuracy
    or macro_f1): positive means amplified relative to raw features,
    negative means obscured. Returned as the signed difference, never
    clamped or absolute-valued.
    """
    return metric_embedding - metric_raw
