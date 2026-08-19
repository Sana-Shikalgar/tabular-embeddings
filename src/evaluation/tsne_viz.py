"""t-SNE visualisation used in 06_evaluation_protocol.ipynb's representation-
quality plots. method="exact" and n_jobs=1 are hardcoded: sklearn's default
TSNE settings segfault this environment on wide representation matrices, and
this combination is the smallest fix that keeps init="pca" usable at all.
"""

from __future__ import annotations

import numpy as np
from sklearn.manifold import TSNE


def kobak_berens_tsne_params(n_samples: int) -> dict:
    """Scales perplexity and learning_rate to n_samples instead of using
    sklearn's fixed defaults: perplexity = max(30, n_samples / 100),
    learning_rate = max(200, n_samples / 12), both cast to int.
    """
    perplexity = max(30, n_samples / 100)
    learning_rate = max(200, n_samples / 12)
    return {"perplexity": int(perplexity), "learning_rate": int(learning_rate)}


def fit_tsne(
    X: np.ndarray,
    random_state: int,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits a 2D TSNE embedding on X (or a reproducible subsample of it,
    if max_samples is given and len(X) exceeds it), using perplexity/
    learning_rate scaled to the row count actually fit via
    kobak_berens_tsne_params. X is cast to float64 first, and
    method="exact"/n_jobs=1 are hardcoded (see this module's docstring).
    Prints the row count and parameters used before fitting.

    Returns (embedding, sample_indices): embedding has shape (n, 2);
    sample_indices are the row positions into X that embedding's rows
    correspond to (np.arange(len(X)) when no subsampling occurs), so a
    caller can always align parallel per-row data via
    `parallel_array[sample_indices]` without a None-check.
    """
    n_samples = X.shape[0]
    if max_samples is not None and n_samples > max_samples:
        rng = np.random.default_rng(random_state)
        sample_indices = np.sort(rng.choice(n_samples, size=max_samples, replace=False))
        X_fit = X[sample_indices]
    else:
        sample_indices = np.arange(n_samples)
        X_fit = X

    X_fit = np.asarray(X_fit, dtype=np.float64)
    params = kobak_berens_tsne_params(X_fit.shape[0])
    print(
        f"fit_tsne: fitting {X_fit.shape[0]}/{n_samples} rows, "
        f"perplexity={params['perplexity']}, learning_rate={params['learning_rate']}"
    )
    tsne = TSNE(
        n_components=2,
        perplexity=params["perplexity"],
        learning_rate=params["learning_rate"],
        init="pca",
        method="exact",
        n_jobs=1,
        random_state=random_state,
    )
    embedding = tsne.fit_transform(X_fit)
    return embedding, sample_indices
