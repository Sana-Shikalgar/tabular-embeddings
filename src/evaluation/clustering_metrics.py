"""Eq. 3.16 (silhouette), Eq. 3.17 (Calinski-Harabasz / variance ratio
criterion), Eq. 3.18 (Davies-Bouldin) -- Chapter3_Methodology_Structure.md's
own cluster-validity metrics, computed identically for every representation
so Chapter 4 can compare how well each representation's own geometry
separates derive_genre_label's clustering label (eval_setup.py), not
whether any one representation was tuned to look good on it.

Thin wrappers around sklearn.metrics' own implementations, not
reimplementations:
- sklearn.metrics.silhouette_score -- Rousseeuw, P.J. (1987), "Silhouettes:
  a graphical aid to the interpretation and validation of cluster
  analysis," Journal of Computational and Applied Mathematics. Higher is
  better; range [-1, 1].
- sklearn.metrics.calinski_harabasz_score -- Calinski, T. & Harabasz, J.
  (1974), "A dendrite method for cluster analysis," Communications in
  Statistics-theory and Methods. Higher is better; unbounded above.
- sklearn.metrics.davies_bouldin_score -- Davies, D.L. & Bouldin, D.W.
  (1979), "A Cluster Separation Measure," IEEE Transactions on Pattern
  Analysis and Machine Intelligence. LOWER is better, the one metric of
  the three where that direction is reversed; unbounded above, 0 is the
  best-possible (fully separated) score.

These are the same sources sklearn's own documentation cites for each
function, matching how src/models/paradigm_config.py cites its own
sources (Gorishniy et al., Ucar et al., Bahri et al.) rather than
paraphrasing them from memory.

KNOWN ENVIRONMENT CRASH -- silhouette_score segfaults the Python process
outright (not a Python exception, so no traceback) in this project's
Windows conda environment. calinski_harabasz_score/davies_bouldin_score
are NOT affected -- confirmed fine on the full 1051-row br_gt0 test
split, every representation, no workaround needed. The crash is isolated
entirely to silhouette_score's own O(n^2) pairwise-distance computation,
and needs ALL THREE of the following true at once to reproduce (any one
being false is enough to avoid it):

  (a) X is float32 -- ft_transformer/subtab/scarf's saved arrays' native
      dtype (raw/classical's are dtype=object, a parquet round-trip
      artifact, which also crashes uncast). Fixed inside this module:
      compute_clustering_metrics() always casts X to float64
      (np.asarray(X, dtype=np.float64)) before calling any metric.

  (b) BLAS is running its default multi-threaded configuration. Fixed by
      the CALLER, not this module -- numpy's BLAS backend is already
      loaded by the time this module's own code runs, so
      threadpoolctl.threadpool_limits(1) around the call does NOT help;
      OMP_NUM_THREADS=1 / OPENBLAS_NUM_THREADS=1 / MKL_NUM_THREADS=1 must
      be set as environment variables before numpy is first imported
      ANYWHERE in the process (same placement discipline as
      06_evaluation_protocol.ipynb's own KMP_DUPLICATE_LIB_OK line;
      06_evaluation_protocol.ipynb's setup cell already does this).

  (c) The array is large enough -- confirmed on real data, single
      threaded, float64: as a plain script (no Jupyter), scarf (256-wide)
      crashes between N=500 (fine) and N=600 (crashes), and the real test
      split (N=1051) crashes for every representation except raw
      (16-wide) even with (a) and (b) already fixed. THE THRESHOLD IS
      LOWER STILL INSIDE AN ACTUAL JUPYTER KERNEL (ipykernel) THAN IN A
      PLAIN SCRIPT -- confirmed via nbclient driving a real kernel, not
      inferred: sample_size=500 (safe as a script) still crashes the
      kernel; sample_size=300 crashes it on scarf; sample_size=200 was
      confirmed safe inside a real kernel on BOTH classical (4282-wide,
      the widest representation) and scarf (256-wide). No further lower
      bound was established -- 200 is what was verified, not a
      mathematically tight threshold. Unlike (a)/(b), this one CANNOT be
      worked around by this module always doing the "safe" thing,
      because silhouette_score's whole point is to score the full split
      -- so compute_clustering_metrics()/run_all() below expose
      silhouette_sample_size (default DEFAULT_SILHOUETTE_SAMPLE_SIZE=200,
      sklearn's own `sample_size` argument, applied to silhouette_score
      ONLY) rather than silently picking one. This is a real
      methodological trade-off, disclosed here rather than hidden inside
      a passing test: 200 is NOT derived from any accuracy/stability
      analysis of how much a 200-row subsample's silhouette estimate can
      vary from the true full-set value, only from what this specific
      environment's crash threshold was verified to tolerate.
      calinski_harabasz_score/davies_bouldin_score are unaffected by (c)
      and always run on the FULL X, every row -- only silhouette is
      subsampled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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
    """One representation's three cluster-validity scores against `labels`
    (e.g. derive_genre_label's output for one split).

    X is cast to float64 before any metric is computed -- see this
    module's own docstring, crash cause (a): saved representation arrays
    are float32 or dtype=object, and silhouette_score segfaults on either
    in this environment.

    silhouette_sample_size (default DEFAULT_SILHOUETTE_SAMPLE_SIZE=200) is
    passed straight through to sklearn's own silhouette_score(...,
    sample_size=...) -- crash cause (c) in this module's own docstring:
    silhouette_score's pairwise-distance computation segfaults on the
    real, full-size test split regardless of (a)/(b) already being fixed,
    and unlike those two this one cannot be fixed by this module always
    doing the "safe" thing, since scoring the full split is the whole
    point. Pass None to disable subsampling and score every row (only
    safe at whatever scale this environment's crash threshold allows --
    NOT the real N=1051 test split, per this module's own docstring).
    calinski_harabasz_score/davies_bouldin_score are never subsampled --
    both are unaffected by crash cause (c) and always run on the FULL X.

    random_state is passed to silhouette_score for BOTH the subsampling
    above (reproducible which rows get sampled) and internal ties;
    calinski_harabasz_score/davies_bouldin_score take no random_state,
    both fully deterministic given X and labels.
    """
    assert X.shape[0] == len(labels), (
        f"X and labels must have the same number of rows: "
        f"X.shape[0]={X.shape[0]}, len(labels)={len(labels)}"
    )
    X = np.asarray(X, dtype=np.float64)
    return {
        "silhouette": float(
            silhouette_score(
                X, labels, sample_size=silhouette_sample_size, random_state=random_state
            )
        ),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def run_all(
    representations: dict[str, np.ndarray],
    labels: pd.Series | np.ndarray,
    random_state: int,
    silhouette_sample_size: int | None = DEFAULT_SILHOUETTE_SAMPLE_SIZE,
) -> pd.DataFrame:
    """compute_clustering_metrics for every representation in
    eval_setup.REPR_NAMES, against the SAME labels each time -- one row
    per representation (index = representation name, REPR_NAMES order),
    one column per metric. silhouette_sample_size is forwarded unchanged
    to every compute_clustering_metrics() call -- see that function's own
    docstring for what it controls and why it defaults to 200, not None.

    representations must have an entry for every REPR_NAMES key; a
    missing one raises KeyError naming it rather than silently producing
    a DataFrame with fewer rows than representations actually exist.
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
