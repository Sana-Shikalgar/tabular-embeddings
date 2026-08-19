"""Implements SubTab (Ucar et al. 2021): reduces high-cardinality list
fields via SVD, splits the flat feature vector into overlapping column
subsets, and trains a shared autoencoder to reconstruct the full row."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.config import RANDOM_SEED
from src.features.baselines import build_classical_baseline
from src.features.pipeline import FittedEncoders
from src.features.target_split import build_feature_target_split
from src.paradigms.ft_transformer import TEXT_FIELDS
from src.paradigms.paradigm_config import SubTabConfig

# Fixed, documented concatenation order for build_flat_vector /
# flat_vector_segment_bounds -- multi-hot list fields grouped first, then
# the two SVD-reduced fields.
FLAT_VECTOR_SEGMENT_ORDER = (
    "numeric",
    "original_language",
    "genres",
    "production_countries",
    "spoken_languages",
    "keywords",
    "production_companies",
    "overview",
    "original_title",
)


class SubtabListReducer:
    """Reduces one list field's sparse multi-hot indicator matrix to
    n_components dimensions via TruncatedSVD."""

    def __init__(self):
        """Initializes an unfit reducer."""
        self.field: str | None = None
        self.n_components: int | None = None
        self.svd: TruncatedSVD | None = None

    def fit(
        self, train_df: pd.DataFrame, field: str, fitted_encoders: FittedEncoders, n_components: int
    ) -> "SubtabListReducer":
        """Fits a TruncatedSVD to field's multi-hot indicator matrix built
        from train_df."""
        self.field = field
        self.n_components = n_components

        field_matrix = self._build_field_matrix(train_df, fitted_encoders)
        # algorithm="arpack", not the default "randomized" -- randomized
        # segfaults on this sparse matrix shape once n_components grows
        # past ~150 in this environment; arpack is exact for this problem
        # size and does not hit that crash.
        self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED, algorithm="arpack")
        self.svd.fit(field_matrix)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        fitted_encoders: FittedEncoders,
        classical_baseline: pd.DataFrame | None = None,
    ):
        """Projects df's field column into the fitted SVD's
        n_components-dimensional space."""
        assert self.svd is not None, "SubtabListReducer.transform() called before fit()"
        field_matrix = self._build_field_matrix(df, fitted_encoders, classical_baseline)
        return self.svd.transform(field_matrix)

    def _build_field_matrix(
        self,
        df: pd.DataFrame,
        fitted_encoders: FittedEncoders,
        classical_baseline: pd.DataFrame | None = None,
    ) -> csr_matrix:
        """Builds field's sparse multi-hot indicator matrix from df's
        classical baseline, computing the baseline if not already given."""
        # classical_baseline is rebuilt here only when the caller doesn't
        # already have one -- build_flat_vector computes it once per df and
        # passes it to both fields' transform() calls, since it's identical
        # for both and otherwise gets rebuilt redundantly per field.
        if classical_baseline is None:
            split = build_feature_target_split(df)
            classical_baseline = build_classical_baseline(
                df,
                split,
                fitted_encoders.column_groups.numeric_cols,
                fitted_encoders.standardizer,
                fitted_encoders.language_lookup,
                fitted_encoders.list_poolers,
            )
        field_cols = [c for c in classical_baseline.columns if c.startswith(f"{self.field}_")]
        assert field_cols, f"no columns found for field={self.field!r} in the classical baseline"
        return csr_matrix(classical_baseline[field_cols].to_numpy())


def _segment_width(
    name: str, fitted_encoders: FittedEncoders, list_reducers: dict, include_text: bool = True
) -> int:
    """Returns one FLAT_VECTOR_SEGMENT_ORDER segment's column width,
    derived from fitted_encoders/list_reducers alone -- no df/batch
    needed. list_reducers is checked before column_groups.list_cols:
    keywords/production_companies are list fields too, but take the SVD
    path when present in list_reducers rather than the multi-hot path.
    include_text=False rejects overview/original_title -- callers that
    filter FLAT_VECTOR_SEGMENT_ORDER down before iterating should never
    reach this branch, but it's guarded here too rather than silently
    returning a width for an excluded field."""
    if name == "numeric":
        return fitted_encoders.dims["numeric"]
    if name == "original_language":
        return len(fitted_encoders.language_lookup.token_to_index) + 1
    if name in list_reducers:
        return list_reducers[name].n_components
    if name in fitted_encoders.column_groups.list_cols:
        return len(fitted_encoders.list_poolers[name].token_to_index) + 1
    if name in TEXT_FIELDS:
        if not include_text:
            raise ValueError(f"_segment_width: {name!r} is a text field but include_text=False")
        return fitted_encoders.dims[name]
    raise ValueError(f"unknown segment name: {name!r}")


def flat_vector_segment_bounds(
    fitted_encoders: FittedEncoders, list_reducers: dict, include_text: bool = True
) -> dict[str, tuple[int, int]]:
    """Returns each FLAT_VECTOR_SEGMENT_ORDER segment's (start, end)
    column range in build_flat_vector's output, without changing what
    build_flat_vector itself returns. include_text=False excludes
    overview/original_title from the iteration (and the returned dict)
    -- FLAT_VECTOR_SEGMENT_ORDER itself is never mutated, only filtered
    per-call."""
    segment_order = (
        FLAT_VECTOR_SEGMENT_ORDER
        if include_text
        else tuple(name for name in FLAT_VECTOR_SEGMENT_ORDER if name not in TEXT_FIELDS)
    )
    bounds: dict[str, tuple[int, int]] = {}
    offset = 0
    for name in segment_order:
        width = _segment_width(name, fitted_encoders, list_reducers, include_text=include_text)
        bounds[name] = (offset, offset + width)
        offset += width
    return bounds


def _multihot(index_lists, width: int) -> np.ndarray:
    """Expands index_lists into a dense multi-hot matrix of the given
    width."""
    mat = np.zeros((len(index_lists), width), dtype=np.float64)
    for row_i, indices in enumerate(index_lists):
        if len(indices):
            mat[row_i, indices] = 1.0
    return mat


def build_flat_vector(
    df: pd.DataFrame, fitted_encoders: FittedEncoders, list_reducers: dict, include_text: bool = True
) -> np.ndarray:
    """Concatenates every fitted_encoders.transform(df) key into one flat,
    fixed-order array (FLAT_VECTOR_SEGMENT_ORDER) -- the reconstruction
    target for SubTab's autoencoder. Numeric, one-hot language, and the
    three small multi-hot list fields pass through as-is; keywords/
    production_companies go through list_reducers' SVD instead of full
    multi-hot; the two text fields pass through as their 768-d SBERT
    vectors. include_text=False excludes overview/original_title from
    the iteration (and from fitted_encoders.transform()'s own lookup)
    -- FLAT_VECTOR_SEGMENT_ORDER itself is never mutated, only filtered
    per-call."""
    batch = fitted_encoders.transform(df, include_text=include_text)
    segments: list[np.ndarray] = []

    segment_order = (
        FLAT_VECTOR_SEGMENT_ORDER
        if include_text
        else tuple(name for name in FLAT_VECTOR_SEGMENT_ORDER if name not in TEXT_FIELDS)
    )

    # Built once here rather than once per list_reducers field -- both
    # keywords and production_companies otherwise each independently rebuild
    # the identical classical_baseline for this same df inside
    # SubtabListReducer.transform().
    classical_baseline = None
    if list_reducers:
        split = build_feature_target_split(df)
        classical_baseline = build_classical_baseline(
            df,
            split,
            fitted_encoders.column_groups.numeric_cols,
            fitted_encoders.standardizer,
            fitted_encoders.language_lookup,
            fitted_encoders.list_poolers,
        )

    for name in segment_order:
        if name == "numeric":
            segments.append(batch["numeric"])
        elif name == "original_language":
            width = _segment_width(name, fitted_encoders, list_reducers, include_text=include_text)
            segments.append(np.eye(width, dtype=np.float64)[batch["original_language"]])
        elif name in list_reducers:
            segments.append(
                list_reducers[name].transform(df, fitted_encoders, classical_baseline)
            )
        elif name in fitted_encoders.column_groups.list_cols:
            width = _segment_width(name, fitted_encoders, list_reducers, include_text=include_text)
            segments.append(_multihot(batch[name], width))
        elif name in TEXT_FIELDS:
            segments.append(batch[name])
        else:
            raise ValueError(f"unknown segment name: {name!r}")

    flat = np.concatenate(segments, axis=1)

    bounds = flat_vector_segment_bounds(fitted_encoders, list_reducers, include_text=include_text)
    expected_width = bounds[segment_order[-1]][1]
    assert flat.shape[1] == expected_width, (
        f"build_flat_vector produced width {flat.shape[1]}, but "
        f"flat_vector_segment_bounds computed {expected_width} -- the two must agree"
    )

    print(f"build_flat_vector: total width = {flat.shape[1]}")
    return flat


def make_column_subsets(width: int, k: int, overlap: float) -> list[np.ndarray]:
    """Splits width columns into k overlapping, contiguous column-index
    subsets, following SubTab's reference implementation: each subset i>0
    is columns[i*n_column_subset - n_overlap : (i+1)*n_column_subset],
    subset 0 instead extends n_overlap forward since it has no predecessor
    to borrow from. Column order and membership are fixed, not randomized."""
    n_column_subset = int(width / k)
    n_overlap = int(overlap * n_column_subset)
    column_idx = list(range(width))

    subsets: list[np.ndarray] = []
    for i in range(k):
        if i == 0:
            start_idx = 0
            stop_idx = n_column_subset + n_overlap
        else:
            start_idx = i * n_column_subset - n_overlap
            stop_idx = (i + 1) * n_column_subset
        subsets.append(np.array(column_idx[start_idx:stop_idx], dtype=np.int64))

    return subsets


class SubTabAutoencoder(nn.Module):
    """One shared encoder + one shared decoder, applied identically to
    every subset make_column_subsets() produces, reconstructing the full
    row from each subset's latent representation."""

    def __init__(self, subset_width: int, full_width: int, config: SubTabConfig):
        """Builds the encoder/decoder MLPs (Linear-LeakyReLU-Linear each)
        from subset_width, full_width, and config."""
        super().__init__()
        self.config = config

        hidden_dim = config.encoder_dims[0]
        latent_dim = config.latent_dim
        assert config.encoder_dims[-1] == latent_dim, (
            "SubTabConfig.encoder_dims[-1] and SubTabConfig.latent_dim must "
            "agree -- in Ucar et al.'s own Encoder, the second layer's width "
            "IS the latent width, not two independently-chosen numbers"
        )

        self.encoder = nn.Sequential(
            nn.Linear(subset_width, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, full_width),
        )

    def forward(self, x_subset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes x_subset to a latent vector and decodes it back to a
        full-row reconstruction. Returns (latent, x_recon)."""
        latent = self.encoder(x_subset)
        x_recon = self.decoder(latent)
        return latent, x_recon


def reconstruction_loss(
    x_full: torch.Tensor, reconstructions: list[torch.Tensor]
) -> torch.Tensor:
    """Computes L_r: the mean reconstruction MSE across all subset
    reconstructions against the shared target x_full."""
    losses = [F.mse_loss(x_full, x_recon_k) for x_recon_k in reconstructions]
    return torch.stack(losses).mean()


def per_segment_reconstruction_mse(
    x_full: torch.Tensor,
    reconstructions: list[torch.Tensor],
    segment_bounds: dict[str, tuple[int, int]],
) -> dict[str, float]:
    """Diagnostic only -- does not feed backprop or change the training
    objective. Splits build_flat_vector's columns via segment_bounds into
    each FLAT_VECTOR_SEGMENT_ORDER segment present in segment_bounds
    (numeric, original_language, each multi-hot list field, keywords/
    production_companies, and the two text fields when segment_bounds
    includes them), and returns one mean reconstruction MSE per segment,
    averaged the same way as reconstruction_loss."""

    def _avg_mse(col_idx: torch.Tensor) -> float:
        losses = [
            F.mse_loss(x_full[:, col_idx], x_recon_k[:, col_idx])
            for x_recon_k in reconstructions
        ]
        return torch.stack(losses).mean().item()

    with torch.no_grad():
        return {
            name: _avg_mse(torch.arange(*segment_bounds[name]))
            for name in FLAT_VECTOR_SEGMENT_ORDER
            if name in segment_bounds
        }
