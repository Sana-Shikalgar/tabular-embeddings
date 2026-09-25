"""Host regressor used in 06_evaluation_protocol.ipynb's interpretability
step, fit only so LIME/SHAP have something to explain -- not a
representation-quality measure. Note: the FT-Transformer representation was
itself trained to predict this model's same target, giving it a structural
head start that shouldn't be read as a quality ranking across
representations.
"""

from __future__ import annotations

import numpy as np
import xgboost
from sklearn.metrics import mean_absolute_error, r2_score

from src.evaluation.eval_setup import SharedXGBConfig


def fit_host_regressor(
    X_train: np.ndarray, y_train: np.ndarray, xgb_config: SharedXGBConfig
) -> xgboost.XGBRegressor:
    """Fits an XGBoost regressor on X_train/y_train using the shared XGBoost config."""
    model = xgboost.XGBRegressor(**xgb_config.regressor_kwargs())
    model.fit(X_train, y_train)
    return model


def evaluate_host(model: xgboost.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Returns {"r2": ..., "mae": ...} for the fitted model on X_test/y_test."""
    y_pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
