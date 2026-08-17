"""Section 3.5's t-SNE visualisation -- Van der Maaten, L.J.P. & Hinton, G.E.
(2008), "Visualizing Data using t-SNE," Journal of Machine Learning Research.

t-SNE, not UMAP: McInnes, L., Healy, J. & Melville, J. (2018), "UMAP: Uniform
Manifold Approximation and Projection for Dimension Reduction," is the
alternative this deliberately doesn't use -- UMAP is not implemented anywhere
in this module; cited here only to name the choice being made, not as a
source this module's code follows.

Perplexity/learning-rate scaling rule: Kobak, D. & Berens, P. (2019), "The
art of using t-SNE for single-cell transcriptomics," Nature Communications.
See kobak_berens_tsne_params's own docstring for the important caveat on
this citation's exact form.

KNOWN ENVIRONMENT CRASH -- sklearn.manifold.TSNE segfaults the Python
process outright (same class of Windows BLAS/LAPACK bug already
documented in subtab.py's TruncatedSVD note and clustering_metrics.py's
silhouette_score note, hitting yet another sklearn code path here), and
this one is worse than either of those: with the literal originally
specified call (init="pca", sklearn's own default method="barnes_hut",
no n_jobs override), it crashes on real 256-wide scarf test-split data
even at N=50 -- there was no usable N for the spec as originally
written. Tried and failed to fully fix it: n_jobs=1 alone (still crashes
at N=1051); method="exact" with n_jobs=1 and init="random" (raises the
threshold to N=400 OK / N=700 crash, still short of real split sizes);
various PCA svd_solver choices for the init="pca" step specifically
(full/arpack/randomized all crash the same way PCA does standalone,
confirmed crashing on its own between N=50 (fine) and N=200 (crashes)).

The combination that DOES work, empirically, on real data: init="pca" +
method="exact" + n_jobs=1, safe at N=100, crashes at N=200. fit_tsne()
hardcodes method="exact" and n_jobs=1 (not exposed as parameters) as the
minimum fix needed to make init="pca" -- the literally specified init --
usable AT ALL in this environment; this is a stated, cited deviation
from an unspecified default (method was never named in the original
spec), not a silent one. Given no tested combination clears the real
~1000-row split sizes this project's splits actually have, fit_tsne()
also exposes an opt-in max_samples parameter (default None, i.e. off --
matching the function's original single-argument behaviour exactly for
any X small enough not to need it) rather than silently capping N
itself. ~100 is the largest value confirmed safe on real data in this
environment; a caller choosing a larger max_samples should re-verify it
against this same crash before trusting it.
"""

from __future__ import annotations

import numpy as np
from sklearn.manifold import TSNE


def kobak_berens_tsne_params(n_samples: int) -> dict:
    """perplexity = max(30, n_samples / 100), learning_rate = max(200,
    n_samples / 12), both cast to int.

    sklearn's TSNE defaults (perplexity=30 fixed, learning_rate='auto')
    are NOT scaled to dataset size; using them unscaled here would
    undermine the validated-settings defence this dissertation's
    methodology claims. The exact constants (n/100, n/12) are the
    commonly-cited form of Kobak & Berens (2019)'s guidance; the source
    PDF was not available when this function was written -- verify the
    precise recommended form against the actual paper before quoting
    these numbers in the dissertation text, not just in code.
    """
    perplexity = max(30, n_samples / 100)
    learning_rate = max(200, n_samples / 12)
    return {"perplexity": int(perplexity), "learning_rate": int(learning_rate)}


def fit_tsne(
    X: np.ndarray,
    random_state: int,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits sklearn.manifold.TSNE(n_components=2, init="pca") on X (or a
    subsample of it, see max_samples below), with perplexity/
    learning_rate from kobak_berens_tsne_params(n) -- see that function's
    own docstring for the exact formula and its citation caveat. n is the
    row count actually fit (len(X) normally, or max_samples when
    subsampling triggers) -- Kobak & Berens' scaling rule is about the
    size of what t-SNE is actually embedding, not some notional original
    count. method="exact" and n_jobs=1 are hardcoded, not parameters --
    see this module's own docstring, KNOWN ENVIRONMENT CRASH, for why
    both are required just to get init="pca" (the literally specified
    init) working at all in this environment. Prints the row count
    actually fit and the perplexity/learning_rate used before fitting,
    so a run's console output alone says what was run without
    re-deriving it.

    X is cast to float64 before fitting -- ft_transformer/subtab/scarf's
    saved arrays are float32 and raw/classical's are dtype=object (a
    parquet round-trip artifact); clustering_metrics.py's own
    silhouette_score investigation found both crash sklearn routines in
    this same environment, so this function casts defensively too rather
    than waiting to find out TSNE has the identical sensitivity.

    max_samples (default None = no subsampling, this function's original
    single-argument behaviour exactly for any X small enough not to need
    it): when given and len(X) > max_samples, draws max_samples row
    indices via np.random.default_rng(random_state).choice(...,
    replace=False) -- reproducible with the same random_state -- fits
    TSNE on just those rows, and returns their positions sorted
    ascending (a stable, deterministic row order, not shuffled).

    Returns (embedding, sample_indices):
      - embedding: shape (n, 2), n = len(X) or max_samples per above.
      - sample_indices: shape (n,) int array, the row positions into X
        that embedding's rows correspond to, in order. ALWAYS returned,
        even when max_samples is None or no subsampling actually
        triggered (sample_indices is then np.arange(len(X)), unchanged
        order) -- so a caller never needs a None-check branch: align any
        parallel per-row data (e.g. genre labels) to embedding via
        `parallel_array[sample_indices]`, unconditionally, every time.
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
