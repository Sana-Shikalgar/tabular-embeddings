"""This model exists solely to give Step 5's LIME/SHAP something to explain
(RQ2/DO3) -- it is explicitly NOT a DO2 quality axis. Matches the "DO3 sanity
check, not a DO2 quality result" framing already in
Chapter3_Methodology_Structure.md's equation table for Eq. 3.19: the host
regressor's own predictive score is not this project's measure of
representation quality (that's DO2's own metrics elsewhere in Chapter 3/4),
it is only the thing LIME/SHAP need fit and predicting well enough to explain
meaningfully.

fit_host_regressor uses xgb_config.regressor_kwargs() -- the SAME
eval_setup.SharedXGBConfig every other XGBoost fit in this project uses, no
early stopping, no per-representation hyperparameter search. [D12]: this
project's own precedent (already set for SubTab's svd_variance_threshold,
SharedTrainingConfig's patience overriding each paradigm's own paper-reported
number) against per-method tuning -- one fixed config, applied identically,
so a representation's host score reflects the representation, not a tuning
advantage it happened to get.

The FT-Transformer representation was itself trained to predict
vote_average (the same TARGET_COL this host model predicts) -- see [D3]. Its
host R2/MAE therefore has a structural head start the reconstruction
(SubTab) and similarity (SCARF) paradigms do not share. This asymmetry is
disclosed, not concealed: the host's score is read per-representation as an
above-chance sanity check only (is this a real predictor at all?), never
compared across representations as a quality ranking. That comparison would
measure "which embedding was trained on this target", not "which
representation is better".
"""

from __future__ import annotations

import numpy as np
import xgboost
from sklearn.metrics import mean_absolute_error, r2_score

from src.evaluation.eval_setup import SharedXGBConfig


def fit_host_regressor(
    X_train: np.ndarray, y_train: np.ndarray, xgb_config: SharedXGBConfig
) -> xgboost.XGBRegressor:
    model = xgboost.XGBRegressor(**xgb_config.regressor_kwargs())
    model.fit(X_train, y_train)
    return model


def evaluate_host(model: xgboost.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
