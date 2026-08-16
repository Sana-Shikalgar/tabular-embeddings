"""SubtabListReducer compresses one list field's sparse multi-hot indicator
matrix down to n_components via TruncatedSVD -- SVD applied to a sparse
term/item-indicator matrix for dimensionality reduction is an established
technique (Deerwester et al. 1990, Latent Semantic Analysis), and
sklearn.decomposition.TruncatedSVD itself runs Halko, Martinsson & Tropp
(2011)'s randomized SVD algorithm under the hood.

TruncatedSVD does NOT center the data before decomposing it (unlike PCA)
-- by design, so a sparse input matrix stays sparse rather than becoming
dense once the (nonzero) mean is subtracted. This means the first
component tends to absorb the matrix's overall uncentered magnitude
rather than its sharpest axis of *variation*, which is why Prompt 3.1's
cumulative-variance curves look front-loaded on the first component --
expected behavior of uncentered SVD on indicator data, not a bug.

Each field is fit at the exact k Prompt 3.1's diagnostic settled on for
that field specifically (keywords=206, production_companies=92 -- NOT a
shared value; a fresh TruncatedSVD(n_components=k) fit at that final k,
not a slice of the 300-component exploratory fit, so what's saved is
exactly the transform actually used, with no leftover unused components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.features.baselines import build_classical_baseline
from src.features.pipeline import FittedEncoders
from src.features.target_split import build_feature_target_split
from src.models.ft_transformer import TEXT_FIELDS
from src.models.paradigm_config import SubTabConfig

# Fixed, documented concatenation order for build_flat_vector /
# flat_vector_segment_bounds -- multi-hot list fields grouped first, then
# the two SVD-reduced fields, matching how Prompt 3.6b needs to slice the
# two SVD segments out as a contiguous block.
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
    def __init__(self):
        self.field: str | None = None
        self.n_components: int | None = None
        self.svd: TruncatedSVD | None = None

    def fit(
        self, train_df: pd.DataFrame, field: str, fitted_encoders: FittedEncoders, n_components: int
    ) -> "SubtabListReducer":
        self.field = field
        self.n_components = n_components

        field_matrix = self._build_field_matrix(train_df, fitted_encoders)
        # algorithm="arpack", not the default "randomized" -- randomized
        # segfaults on this sparse matrix shape once n_components grows
        # past ~150 in this environment (Prompt 3.1); arpack is exact for
        # this problem size and does not hit that crash.
        self.svd = TruncatedSVD(n_components=n_components, random_state=0, algorithm="arpack")
        self.svd.fit(field_matrix)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        fitted_encoders: FittedEncoders,
        classical_baseline: pd.DataFrame | None = None,
    ):
        assert self.svd is not None, "SubtabListReducer.transform() called before fit()"
        field_matrix = self._build_field_matrix(df, fitted_encoders, classical_baseline)
        return self.svd.transform(field_matrix)

    def _build_field_matrix(
        self,
        df: pd.DataFrame,
        fitted_encoders: FittedEncoders,
        classical_baseline: pd.DataFrame | None = None,
    ) -> csr_matrix:
        # classical_baseline is rebuilt here only when the caller doesn't
        # already have one (e.g. Prompt 3.1's standalone diagnostic) --
        # build_flat_vector computes it once per df and passes it to both
        # fields' transform() calls, since it's identical for both and
        # otherwise gets rebuilt redundantly per field.
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


def _segment_width(name: str, fitted_encoders: FittedEncoders, list_reducers: dict) -> int:
    """Width of one FLAT_VECTOR_SEGMENT_ORDER segment, derived from
    fitted_encoders/list_reducers alone -- no df/batch needed. list_reducers
    is checked before column_groups.list_cols: keywords/production_companies
    are list fields too, but take the SVD path when present in
    list_reducers rather than the multi-hot path."""
    if name == "numeric":
        return fitted_encoders.dims["numeric"]
    if name == "original_language":
        return len(fitted_encoders.language_lookup.token_to_index) + 1
    if name in list_reducers:
        return list_reducers[name].n_components
    if name in fitted_encoders.column_groups.list_cols:
        return len(fitted_encoders.list_poolers[name].token_to_index) + 1
    if name in TEXT_FIELDS:
        return fitted_encoders.dims[name]
    raise ValueError(f"unknown segment name: {name!r}")


def flat_vector_segment_bounds(
    fitted_encoders: FittedEncoders, list_reducers: dict
) -> dict[str, tuple[int, int]]:
    """Each FLAT_VECTOR_SEGMENT_ORDER segment's (start, end) column range in
    build_flat_vector's output -- a companion lookup so later code (Prompt
    3.6b) can isolate the two SVD segments (or any other segment) without
    re-deriving the concatenation order or widths by hand. Does not itself
    change what build_flat_vector returns."""
    bounds: dict[str, tuple[int, int]] = {}
    offset = 0
    for name in FLAT_VECTOR_SEGMENT_ORDER:
        width = _segment_width(name, fitted_encoders, list_reducers)
        bounds[name] = (offset, offset + width)
        offset += width
    return bounds


def _multihot(index_lists, width: int) -> np.ndarray:
    mat = np.zeros((len(index_lists), width), dtype=np.float64)
    for row_i, indices in enumerate(index_lists):
        if len(indices):
            mat[row_i, indices] = 1.0
    return mat


def build_flat_vector(
    df: pd.DataFrame, fitted_encoders: FittedEncoders, list_reducers: dict
) -> np.ndarray:
    """Concatenates every fitted_encoders.transform(df) key into one flat,
    fixed-order array (FLAT_VECTOR_SEGMENT_ORDER) -- the reconstruction
    target for SubTab's autoencoder:

    - "numeric": passes through as-is (already real-valued, PLE-encoded).
    - "original_language": one-hot expanded from its int64 index array,
      width len(language_lookup.token_to_index) + 1 -- fixed,
      non-trainable, unchanged from the original decision.
    - "genres"/"production_countries"/"spoken_languages": expanded to full
      multi-hot at their existing Section 3.3 vocab width -- unaffected by
      any of this, never the problem.
    - "keywords"/"production_companies": list_reducers[field].transform()
      instead of multi-hot -- a fixed, non-trainable, but now
      dense-and-small representation of what used to be a ~2,001-wide
      sparse block. This is the reconstruction target's whole point.
    - "overview"/"original_title": pass through as-is (768-d each).
    """
    batch = fitted_encoders.transform(df)
    segments: list[np.ndarray] = []

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

    for name in FLAT_VECTOR_SEGMENT_ORDER:
        if name == "numeric":
            segments.append(batch["numeric"])
        elif name == "original_language":
            width = _segment_width(name, fitted_encoders, list_reducers)
            segments.append(np.eye(width, dtype=np.float64)[batch["original_language"]])
        elif name in list_reducers:
            segments.append(
                list_reducers[name].transform(df, fitted_encoders, classical_baseline)
            )
        elif name in fitted_encoders.column_groups.list_cols:
            width = _segment_width(name, fitted_encoders, list_reducers)
            segments.append(_multihot(batch[name], width))
        elif name in TEXT_FIELDS:
            segments.append(batch[name])
        else:
            raise ValueError(f"unknown segment name: {name!r}")

    flat = np.concatenate(segments, axis=1)

    bounds = flat_vector_segment_bounds(fitted_encoders, list_reducers)
    expected_width = bounds[FLAT_VECTOR_SEGMENT_ORDER[-1]][1]
    assert flat.shape[1] == expected_width, (
        f"build_flat_vector produced width {flat.shape[1]}, but "
        f"flat_vector_segment_bounds computed {expected_width} -- the two must agree"
    )

    print(f"build_flat_vector: total width = {flat.shape[1]}")
    return flat


def make_column_subsets(width: int, k: int, overlap: float) -> list[np.ndarray]:
    """Splits build_flat_vector's width columns into k overlapping, contiguous
    column-index subsets -- SubTab's Eq.-less "dividing tabular data to
    multiple subsets" step (Ucar, Hajiramezanali & Edwards 2021).

    Uses SubTabConfig.n_subsets=5, overlap=0.25 (Prompt 1.1): Ucar et al.
    (2021) Table A1/S3.3 report 5 subsets, 25% overlap as their best
    configuration for Income, the closest structural analogue to TMDB among
    their five datasets (both wide, mixed-type, non-image tabular data).

    The paper's own SPEC IS UNDERSPECIFIED for the exact column arithmetic.
    Section 2's Method preamble states only: "we divide tabular data to
    multiple subsets. Neighbouring subsets can have overlapping regions,
    defined as a percentage of a dimension of the subset" -- and, on why
    subsets stay contiguous column blocks rather than randomly resampled:
    "we don't change the relative order of features in a subset." Section
    2.1 itself is "Strategies for adding noise" (Gaussian/swap/zero-out),
    NOT subset generation -- there's no subsection, equation, or appendix
    that turns "a percentage of a dimension" into an exact index formula
    anywhere in the paper.

    The formula below is instead sourced directly from the paper authors'
    own reference implementation, AstraZeneca/SubTab, src/model.py's
    `subset_generator` (test-time / fixed-order branch -- SubTab only
    randomizes subset ORDER at train time, never which columns are IN a
    subset, matching the "relative order... unchanged" quote above):

        n_column_subset = int(width / n_subsets)
        n_overlap = int(overlap * n_column_subset)
        subset 0:      columns[0 : n_column_subset + n_overlap]
        subset i (>0): columns[i*n_column_subset - n_overlap : (i+1)*n_column_subset]

    Every subset ends up the same width, n_column_subset + n_overlap: subset
    0 has no predecessor to borrow overlap from, so it instead extends that
    same n_overlap forward into what would otherwise be subset 1's territory.
    `int()` truncates twice (subset width, then overlap width), so if width
    doesn't divide evenly by k, the trailing width % n_column_subset columns
    belong to no subset at all -- this is the reference implementation's own
    behavior, not a bug introduced here.

    One consequence worth knowing before reading these subsets' overlaps
    pairwise: subset 0 and subset 1 share 2*n_overlap columns, not
    n_overlap -- subset 0's own forward extension (no left neighbour to
    balance against) stacks with subset 1 independently pulling back by
    n_overlap on its own left edge. Every OTHER adjacent pair (1&2, 2&3,
    ...) shares exactly n_overlap, as expected. Verified against real
    n_subsets=5/overlap=0.25/width=2266 data: subsets 0-4 are all width
    566, subset-0/1 overlap is 226, every later adjacent pair is 113.
    """
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
    """One shared encoder + one shared decoder, applied identically to every
    subset make_column_subsets() produces -- "shared" as in Ucar et al.
    (2021)'s own design (Figure 1, SS3.2 "Training"): the same two networks
    process every subset, there is no per-subset encoder/decoder.

    Architecture is Ucar et al.'s own Income/Obesity configuration, not a
    generic choice: SS3.3 "Results" states Income's architecture is "same as
    in Obesity," and the Obesity paragraph specifies "a two-layer encoder
    with [1024, 1024] dimensions. Second layer is a linear layer." Appendix
    Table A1 confirms Income's row directly: Encoder=[1024, 1024],
    Decoder=[1024, 1024], Projection=No. Cross-checked against the paper
    authors' own reference implementation (AstraZeneca/SubTab,
    utils/model_utils.py's Encoder/Decoder/HiddenLayers classes) to resolve
    what the paper's prose alone leaves implicit:
      - hidden layer: Linear(in, encoder_dims[0]) then LeakyReLU (the
        reference's HiddenLayers class; negative_slope defaults to 0.01,
        not stated in the paper, so left at nn.LeakyReLU()'s own default
        rather than invented).
      - bottleneck/output layer: plain nn.Linear, no activation -- this is
        the paper's own "second layer is a linear layer" line, confirmed by
        the reference code building Encoder.latent and Decoder.logits as
        bare nn.Linear layers outside HiddenLayers' activation-adding loop.
      - encoder_dims[-1] (the second layer's width) IS the latent width in
        the reference implementation -- Encoder.latent is
        Linear(dims[-2], dims[-1]), so SubTabConfig.encoder_dims[-1] and
        SubTabConfig.latent_dim are the same number by construction, not
        two independent choices; asserted below rather than silently
        trusted, so a future config edit that breaks this is caught, not
        silently wrong.

    encoder_dims=(1024, 1024) is Income/Obesity-scale, sized for those
    papers' own (much narrower) feature counts. This dataset's flat-vector
    width (Prompt 3.3, e.g. 2266 for br_gt0) will generally differ from
    Income/Obesity's, and the paper gives no general rule for scaling
    encoder_dims to input width -- so subset_width/full_width are read from
    the actual data at construction time, while encoder_dims/latent_dim
    keep Ucar et al.'s literal reported numbers and shrinkage pattern
    (hidden width == latent width, not a funnel) rather than inventing new
    widths to "fit" this dataset better.

    No projection network: Appendix Table A1 lists Projection=No for
    Income, consistent with D28 (this project's own decision) disabling
    the contrastive/distance losses that a projection head would feed.
    AEWrapper's linear_layer1/linear_layer2/z-normalization block in the
    reference code is therefore not ported here.

    Reconstructs the FULL row from a subset's latent, not just that
    subset's own input columns -- Ucar et al.'s Figure 1 / SS2 "we chose the
    latter [reconstructing the complete feature space] in our experiments
    since it is more effective in learning good representations." So
    decoder's output width is full_width (build_flat_vector's total width),
    independent of which/how-wide a subset was fed to the encoder.
    """

    def __init__(self, subset_width: int, full_width: int, config: SubTabConfig):
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
        latent = self.encoder(x_subset)
        x_recon = self.decoder(latent)
        return latent, x_recon


def reconstruction_loss(
    x_full: torch.Tensor, reconstructions: list[torch.Tensor]
) -> torch.Tensor:
    """Eq. 3.12: L_r = (1/K) sum_k MSE(X, X~_k) -- Ucar et al. (2021) Eq. (3)
    (also Appendix Algorithm 1): L_r = (1/K) sum_{k=1}^{K} s_k, where
    s_k = (1/N) sum_i (X^(i) - X~_k^(i))^2, X the true full feature vector,
    X~_k subset k's full-row reconstruction (SubTabAutoencoder.forward()'s
    x_recon), K the number of subsets, N the batch size. s_k is exactly
    what F.mse_loss(x_full, x_recon_k) computes (mean squared error over
    the batch), so this is that, averaged over reconstructions' K entries.

    reconstructions is one x_recon tensor per subset (e.g. K=SubTabConfig.n_subsets
    calls to SubTabAutoencoder.forward(), one per make_column_subsets() subset,
    collected into a list by the training loop) -- not itself responsible for
    running the K forward passes, only for combining their already-computed
    reconstructions against the one shared x_full target.

    D28 (this project's own decision): only L_r is implemented here, not
    L_c (contrastive) or L_d (distance) -- L_t = L_r + L_c + L_d is Ucar et
    al.'s Eq. (2), but L_c/L_d are both optional per-dataset additions, not
    part of L_r itself. This isn't an override of the paper's own choice for
    the closest analogue dataset: SS3.3 "Results" states, verbatim, both "For
    the base model, we only used reconstruction loss" for Income AND for
    Blog -- i.e. Income's own base model (already the architecture/subset
    numbers this file follows) is reconstruction-loss-only to begin with, so
    dropping L_c/L_d here is direct precedent-following, not a deviation.
    Table 2's MNIST ablation (RL only: 97.13 vs RL+CL: 97.26) is cited only
    as separate, general supporting evidence that the contrastive term is
    cheap to drop -- MNIST is not the dataset whose base-model definition
    this decision follows; Income/Blog's own "base model" statement above is.
    """
    losses = [F.mse_loss(x_full, x_recon_k) for x_recon_k in reconstructions]
    return torch.stack(losses).mean()


# Pre-registered Chapter 4 watch item: SubTab's reconstruction target
# compresses keywords/production_companies through a 50%-variance-retention
# SVD (Prompt 3.1b), a lossier target than every other segment. If SubTab's
# downstream embeddings underperform specifically on company/franchise-
# driven signal in Chapter 4, that is a disclosed, anticipated limitation
# of this design choice -- not a surprise to explain away post hoc.
# per_segment_reconstruction_mse() (Prompt 3.6c) gives an early, in-training
# signal for this, well before Chapter 4's downstream results would.


def per_segment_reconstruction_mse(
    x_full: torch.Tensor,
    reconstructions: list[torch.Tensor],
    segment_bounds: dict[str, tuple[int, int]],
) -> dict[str, float]:
    """Diagnostic only -- does NOT feed backprop or change the training
    objective; reconstruction_loss (Prompt 3.6) is still what Prompt 3.7's
    optimizer actually minimises. This exists because whether 50% variance
    retention (Prompt 3.1b) was adequate or too aggressive for keywords/
    production_companies is otherwise only answerable much later and
    indirectly, via Chapter 4's downstream results -- this gives a cheap,
    direct, early signal during training instead, and is what the
    pre-registered watch item above (SubTab underperforming on company/
    franchise structure) is checked against: if "svd_segments"' MSE is
    visibly worse than "rest"'s, that confirms the disclosed limitation
    early rather than surprising Chapter 4.

    Splits build_flat_vector's columns into two groups via segment_bounds
    (flat_vector_segment_bounds's output, Prompt 3.3):
      - "svd_segments": the "keywords" + "production_companies" column
        ranges combined -- the two fields going through lossy SVD
        compression instead of full multi-hot.
      - "rest": every other segment's columns combined (numeric,
        original_language, genres, production_countries, spoken_languages,
        overview, original_title) -- reconstructed from the full,
        uncompressed target, so any MSE gap between the two groups isolates
        the SVD segments' own reconstruction difficulty, not noise from an
        unrelated field.

    Averaged over reconstructions' K entries the exact same way as
    reconstruction_loss: per group, per-subset MSE (restricted to that
    group's columns) is computed for each of the K reconstructions, then
    averaged -- so "svd_segments" and "rest" here are directly comparable
    to each other AND to reconstruction_loss's own overall number.
    """
    svd_fields = ("keywords", "production_companies")
    svd_idx = torch.cat([torch.arange(*segment_bounds[field]) for field in svd_fields])
    all_idx = torch.arange(x_full.shape[1])
    svd_mask = torch.zeros(x_full.shape[1], dtype=torch.bool)
    svd_mask[svd_idx] = True
    rest_idx = all_idx[~svd_mask]
    assert len(svd_idx) + len(rest_idx) == x_full.shape[1], (
        "svd_segments/rest column groups must partition x_full's columns exactly"
    )

    def _avg_mse(col_idx: torch.Tensor) -> float:
        losses = [
            F.mse_loss(x_full[:, col_idx], x_recon_k[:, col_idx])
            for x_recon_k in reconstructions
        ]
        return torch.stack(losses).mean().item()

    with torch.no_grad():
        return {"svd_segments": _avg_mse(svd_idx), "rest": _avg_mse(rest_idx)}
