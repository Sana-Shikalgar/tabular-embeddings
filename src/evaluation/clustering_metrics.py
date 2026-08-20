"""Cluster-validity metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)
used in 06_evaluation_protocol.ipynb's representation-quality comparison,
computed identically for every representation against the derived genre
label (eval_setup.py). Thin wrappers around sklearn.metrics.

silhouette_score is subsampled (silhouette_sample_size, default
DEFAULT_SILHOUETTE_SAMPLE_SIZE=200): its full-size pairwise-distance
computation segfaults this environment on wide, real-size representation
arrays when BLAS runs single-threaded -- compute_clustering_metrics wraps
just that call in threadpoolctl.threadpool_limits(limits=1) rather than
pinning threads for the whole process, so the crash workaround doesn't
leave every other BLAS-backed call (XGBoost fits, the rest of this
notebook) running single-threaded too. calinski_harabasz_score/
davies_bouldin_score are unaffected and always run on the full array,
unpinned. run_silhouette_variance discloses how much a single 200-row
draw can vary, by repeating it over several seeded draws and reporting
the mean/std across them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import threadpoolctl
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.evaluation.eval_setup import REPR_NAMES


DEFAULT_SILHOUETTE_SAMPLE_SIZE = 200


def compute_clustering_metrics(
    X: np.ndarray,
    labels: pd.Series | np.ndarray,
    random_state: int,
    silhouette_sample_size: int | None = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
) -> dict[str, float]:
    """Computes silhouette, Calinski-Harabasz, and Davies-Bouldin scores
    for one representation against labels. X is cast to float64 first.
    silhouette_sample_size (default 200) is passed to sklearn's
    silhouette_score(..., sample_size=...) to avoid a known crash in this
    environment on full-size input; pass None to score every row instead.
    calinski_harabasz_score/davies_bouldin_score are never subsampled.
    random_state seeds silhouette_score's subsampling and internal ties;
    the other two metrics are deterministic and take no random_state.
    """
    assert X.shape[0] == len(labels), (
        f"X and labels must have the same number of rows: "
        f"X.shape[0]={X.shape[0]}, len(labels)={len(labels)}"
    )
    X = np.asarray(X, dtype=np.float64)
    with threadpoolctl.threadpool_limits(limits=1):
        silhouette = float(
            silhouette_score(
                X, labels, sample_size=silhouette_sample_size, random_state=random_state
            )
        )
    return {
        "silhouette": silhouette,
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def run_silhouette_variance(
    X: np.ndarray,
    labels: pd.Series | np.ndarray,
    random_state: int,
    n_draws: int = 20,
    sample_size: int = 200,
) -> dict[str, float]:
    """Repeats silhouette_score over n_draws distinct sample_size-row
    subsamples of one representation (draw i seeded with random_state +
    i, so every draw samples different rows), and returns
    {"mean": ..., "std": ...} across those n_draws scores -- discloses
    how much compute_clustering_metrics's single 200-row silhouette draw
    could have varied under a different seed.
    """
    X = np.asarray(X, dtype=np.float64)
    scores = [
        float(
            silhouette_score(
                X, labels, sample_size=sample_size, random_state=random_state + i
            )
        )
        for i in range(n_draws)
    ]
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


def run_all(
    representations: dict[str, np.ndarray],
    labels: pd.Series | np.ndarray,
    random_state: int,
    silhouette_sample_size: int | None = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Runs compute_clustering_metrics for every representation in
    eval_setup.REPR_NAMES against the same labels, returning one row per
    representation (index = representation name). Raises KeyError naming
    any REPR_NAMES entry missing from representations.
    """
    rows: dict[str, dict[str, float]] = {}
    for repr_name in REPR_NAMES:
        if repr_name not in representations:
            raise KeyError(
                f"run_all: representations is missing an entry for {repr_name!r} "
                f"(expected one for every eval_setup.REPR_NAMES entry: {REPR_NAMES})"
            )
        rows[repr_name] = compute_clustering_metrics(
            representations[repr_name], labels, random_state, silhouette_sample_size
        )
    return pd.DataFrame.from_dict(rows, orient="index")
