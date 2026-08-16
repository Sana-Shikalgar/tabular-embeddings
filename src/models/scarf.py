"""SCARF (Bahri et al. 2022) self-supervised contrastive pretraining: corrupt
a random subset of each row's raw feature values, encode both the clean row
and its corrupted twin, and pull their representations together while
pushing every other row in the batch apart (Eq. 3.13-3.15).

fitted_encoders.corruption_eligible_cols already defaults to every
transform() key except "overview"/"original_title" (pipeline.py's own
docstring/FittedEncoders.fit()) -- reused as-is below, not redefined here.

Corruption granularity resolution: corrupt_dataframe() below corrupts the
raw DataFrame's own columns, not fitted_encoders.transform()'s output
groups directly. Bahri et al. (2022)'s Algorithm 1 draws an independent
Bernoulli(p) corruption mask per FEATURE (not per transform()-output
group, which for a list field or "numeric" would bundle many raw columns
under one mask draw), and this matches the existing precedent in
04_feature_encoding.ipynb's own timing-test cell, which corrupts
"revenue", "runtime", "original_language" individually rather than as one
bundled "numeric" swap. Concretely, corrupt_dataframe() expands each
corruption_eligible_cols group key back to its underlying raw column(s)
and draws a separate mask per raw column.

corruption_rate=0.6 / temperature=1.0 (SCARFConfig's own defaults) are
Bahri et al. (2022) SS4 "Experiments"' own recommendations, quoted
verbatim: "we thus recommend a default setting of 60%" (corruption rate)
and "We recommend using a default temperature of 1."

Corruption happens before any encoding at all -- one-hot for
original_language, embedding+pooling for the five list fields (Prompt
4.2), SBERT lookup for the two text fields (untouched by corruption to
begin with). corrupt_dataframe() itself never calls fitted_encoders.transform()
or looks at any encoding scheme; it only ever reads and replaces raw
DataFrame values. Encoding of the corrupted copy happens downstream, in
build_scarf_views().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.list_pooling import ListFieldPooler
from src.features.pipeline import FittedEncoders
from src.models.ft_transformer import TEXT_FIELDS, FTTransformerConfig
from src.models.paradigm_config import SCARFConfig

# ScarfEmbeddingTables' shared e_dim: FTTransformerConfig.d_token (192).
# FT-Transformer's tokenizer embeds every field -- numeric, categorical,
# list, text -- to one shared d_token width; it is not truly "per-field",
# but it is the actual width FT-Transformer trains its own list-field
# embeddings at, unlike ListFieldPooler.embedding_dim (fixed at 8 in
# 04_feature_encoding.ipynb), which IS per-field-fitted but is never read
# by FT-Transformer's FeatureTokenizer (it hardcodes d_token instead).
# Reused here as a reasonable default; SCARF shares no weights with
# FT-Transformer, so this is a starting point, not a hard dependency.
DEFAULT_E_DIM = FTTransformerConfig().d_token


def corrupt_dataframe(
    df: pd.DataFrame,
    fitted_encoders: FittedEncoders,
    p: float,
    rng: np.random.Generator,
    eligible_group_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Bahri et al. (2022) Algorithm 1's corruption step: for each eligible
    raw column independently, draw a per-row Bernoulli(p) mask, and for
    masked rows, replace that row's value with a value drawn (with
    replacement) from that same column's own values in df -- the empirical
    marginal distribution of that feature, per Bahri et al.'s definition
    ("we sample a random value for each feature independently from the
    empirical marginal distribution"). Every column is corrupted
    independently: a row masked-in on "revenue" is not necessarily also
    masked-in on "runtime".

    eligible_group_keys defaults to fitted_encoders.corruption_eligible_cols
    (every existing call site is unaffected), but can be overridden to a
    different subset of those same keys -- which fields SCARF is allowed to
    corrupt is itself a fairness-relevant knob Bahri et al.'s paper doesn't
    examine (D2), and this project's original Section 3.4 scaffold kept open
    reporting >=2 corruption schemes as an ablation, e.g.
    eligible_group_keys=[k for k in fitted_encoders.corruption_eligible_cols
    if k != "original_language"] to compare with/without original_language
    eligible for corruption. Not run here -- this is just the one-argument
    change needed to run it later, from a caller, without editing this
    function again.

    Eligible raw columns are eligible_group_keys' group keys, expanded back
    to underlying df columns via fitted_encoders.column_groups (see module
    docstring for why raw columns, not transform()-output groups, is the
    right granularity):
      - "numeric" -> column_groups.numeric_cols + column_groups.bypass_cols
      - "original_language" -> fitted_encoders.language_lookup.col
      - each list-field key -> fitted_encoders.list_poolers[key].col

    `id` and the two text columns are untouched: neither is ever a valid
    group key to begin with (see FittedEncoders.fit()'s own docstring), so
    they're never added to the raw-column list below regardless of which
    keys eligible_group_keys selects.

    Returns a corrupted COPY of df -- df itself is never mutated.
    """
    if eligible_group_keys is None:
        eligible_group_keys = fitted_encoders.corruption_eligible_cols

    raw_cols: list[str] = []
    for key in eligible_group_keys:
        if key == "numeric":
            raw_cols.extend(fitted_encoders.column_groups.numeric_cols)
            raw_cols.extend(fitted_encoders.column_groups.bypass_cols)
        elif key == "original_language":
            raw_cols.append(fitted_encoders.language_lookup.col)
        elif key in fitted_encoders.list_poolers:
            raw_cols.append(fitted_encoders.list_poolers[key].col)
        else:
            raise ValueError(
                f"corrupt_dataframe: don't know how to expand "
                f"eligible_group_keys entry {key!r} to a raw df column"
            )

    corrupted = df.copy()
    n_rows = len(df)
    for col in raw_cols:
        mask = rng.random(n_rows) < p
        if not mask.any():
            continue
        source_values = df[col].to_numpy()
        sampled = rng.choice(source_values, size=int(mask.sum()), replace=True)
        col_values = corrupted[col].to_numpy(copy=True)
        col_values[mask] = sampled
        corrupted[col] = col_values

    return corrupted


class ScarfEmbeddingTables(nn.Module):
    """SCARF's own trainable per-item embedding tables for the five
    list-valued fields (genres, keywords, production_companies,
    production_countries, spoken_languages) -- separate weights from
    FT-Transformer's (ft_transformer.py's FeatureTokenizer.list_embeddings),
    even though both are keyed to the same Section-3.3 vocabulary
    (fitted_encoders.list_poolers[field].token_to_index): the two paradigms
    never share weights.

    One nn.Embedding(fitted_encoders.dims[field] + 1, e_dim) per field --
    the "+1" reserves index 0 for the OOV fallback, matching
    CategoricalLookup/ListFieldPooler's own convention (index 0 = unseen
    token) and FeatureTokenizer's identical "+1" for its own list
    embeddings.

    e_dim is a single shared width across all five fields (see
    DEFAULT_E_DIM above for why 192, FT-Transformer's own d_token, is the
    default). original_language stays a fixed, non-trainable one-hot --
    handled directly in build_scarf_flat_vector(), not given a table here.

    Also stores each field's pooling mode (from
    fitted_encoders.list_poolers[field].pooling), so pool_list_fields()
    below doesn't need its own fitted_encoders argument -- mirrors
    FeatureTokenizer's own self._list_pooling_modes cache.
    """

    def __init__(self, fitted_encoders: FittedEncoders, e_dim: int = DEFAULT_E_DIM):
        super().__init__()
        self.e_dim = e_dim
        self.fields = list(fitted_encoders.column_groups.list_cols)
        self.tables = nn.ModuleDict(
            {
                field: nn.Embedding(fitted_encoders.dims[field] + 1, e_dim)
                for field in self.fields
            }
        )
        self.pooling_modes: dict[str, str] = {
            field: fitted_encoders.list_poolers[field].pooling for field in self.fields
        }


def pool_list_fields(
    transformed: dict[str, np.ndarray], tables: ScarfEmbeddingTables
) -> dict[str, torch.Tensor]:
    """For each list field in tables.fields: embed every item in each row
    via tables.tables[field], then pool the row's item embeddings via
    ListFieldPooler.pool(mode=tables.pooling_modes[field]) (Eq. 3.5) --
    exactly mirroring FeatureTokenizer.forward()'s own per-field embed
    -> pool loop in ft_transformer.py, just against SCARF's own tables
    instead of FT-Transformer's.

    transformed is one fitted_encoders.transform(df) call's output (or the
    corrupted-copy equivalent) -- transformed[field] is a dtype=object
    array of one variable-length list of int indices per row.
    """
    device = next(tables.parameters()).device
    pooled: dict[str, torch.Tensor] = {}
    for field in tables.fields:
        embedding_table = tables.tables[field]
        pooling_mode = tables.pooling_modes[field]
        pooled_rows = []
        for row_indices in transformed[field]:
            idx_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
            item_embeddings = embedding_table(idx_tensor)
            pooled_rows.append(ListFieldPooler.pool(item_embeddings, mode=pooling_mode))
        pooled[field] = torch.stack(pooled_rows)
    return pooled


def build_scarf_flat_vector(
    transformed: dict[str, np.ndarray],
    tables: ScarfEmbeddingTables,
    fitted_encoders: FittedEncoders,
) -> torch.Tensor:
    """Concatenates one fitted_encoders.transform(df) batch (clean or
    corrupted -- this function doesn't care which) into one flat vector,
    fixed order:

      transformed["numeric"] (as-is) ++ original_language one-hot (fixed,
      non-trainable, width len(language_lookup.token_to_index) + 1) ++
      pool_list_fields()'s five pooled list-field vectors, in
      fitted_encoders.column_groups.list_cols order ++ overview (768-d,
      as-is) ++ original_title (768-d, as-is).

    This is SCARF's own flat vector, not SubTab's (subtab.py's
    FLAT_VECTOR_SEGMENT_ORDER): list fields are embed+pool here, not
    multi-hot or SVD-reduced, since SCARF's encoder f takes this vector as
    input rather than reconstructing it as a target.
    """
    device = next(tables.parameters()).device

    numeric = torch.from_numpy(transformed["numeric"]).float().to(device)

    # .copy(): CategoricalLookup.transform()'s pandas .map()/.to_numpy() output
    # is a read-only view under pandas' copy-on-write -- torch.from_numpy would
    # otherwise wrap that same read-only buffer and warn.
    lang_idx = torch.from_numpy(transformed["original_language"].copy()).long().to(device)
    lang_width = len(fitted_encoders.language_lookup.token_to_index) + 1
    language_onehot = F.one_hot(lang_idx, num_classes=lang_width).float()

    pooled = pool_list_fields(transformed, tables)
    list_vectors = [pooled[field] for field in fitted_encoders.column_groups.list_cols]

    text_vectors = [
        torch.from_numpy(transformed[field]).float().to(device) for field in TEXT_FIELDS
    ]

    return torch.cat([numeric, language_onehot, *list_vectors, *text_vectors], dim=1)


def build_scarf_views(
    df: pd.DataFrame,
    fitted_encoders: FittedEncoders,
    tables: ScarfEmbeddingTables,
    p: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds SCARF's (clean, corrupted) view pair for one batch: corrupts
    a copy of df (corrupt_dataframe(), Prompt 4.1), calls
    fitted_encoders.transform() on both the original df and the corrupted
    copy, then build_scarf_flat_vector() (Prompt 4.3) on each using the
    SAME shared-weight tables -- this is the "calls transform() twice per
    batch" behaviour pipeline.py's own module docstring already commits to
    (and the reason text embeddings are cached rather than re-encoded
    live: two transform() calls per batch, every batch, would otherwise
    mean two live SBERT passes per batch too).
    """
    corrupted_df = corrupt_dataframe(df, fitted_encoders, p, rng)

    clean_transformed = fitted_encoders.transform(df)
    corrupted_transformed = fitted_encoders.transform(corrupted_df)

    clean_vector = build_scarf_flat_vector(clean_transformed, tables, fitted_encoders)
    corrupted_vector = build_scarf_flat_vector(corrupted_transformed, tables, fitted_encoders)

    return clean_vector, corrupted_vector


class ScarfEncoder(nn.Module):
    """f: Bahri et al. (2022) SS4's own encoder -- "f consists of 4 layers"
    at "hidden dimension 256". config.encoder_layers=4 Linear layers total:
    (encoder_layers - 1) hidden Linear+ReLU pairs at encoder_hidden_dim,
    then one final Linear to output_dim -- 4 Linear layers, 3 ReLUs,
    matching a standard MLP reading of "4 layers" (the last layer is the
    output projection, not itself followed by an activation).

    output_dim defaults to 256, matching f's own hidden width -- Bahri et
    al. state the hidden dimension throughout f/g/h but do not fix f's
    OUTPUT width anywhere in the paper; 256 is a stated, explicit default
    here (reusing the one width the paper does give), not left implicit.

    f's output IS the representation Chapter 4 uses downstream (Eq.
    3.13) -- unlike g (ScarfProjectionHead), f is kept at inference time.
    """

    def __init__(self, input_dim: int, config: SCARFConfig, output_dim: int = 256):
        super().__init__()
        hidden_dim = config.encoder_hidden_dim
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(config.encoder_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScarfProjectionHead(nn.Module):
    """g: Bahri et al. (2022) SS4's own pretraining head -- "both g and h
    have 2 layers" at "hidden dimension 256", and SS3: "the pre-train head
    network l2-normalizes the outputs so that they lie on the unit
    hypersphere." config.head_layers=2 Linear layers total: one hidden
    Linear+ReLU pair at head_hidden_dim, then one final Linear to
    output_dim, L2-normalised (Eq. 3.14's cosine similarity is exactly a
    dot product once both operands already lie on the unit hypersphere).

    output_dim defaults to 256, matching ScarfEncoder's own output_dim
    default for the same reason: Bahri et al. give the hidden width, not
    g's own output width, anywhere in the paper.

    g is discarded after pretraining -- Bahri et al. SS3: "after
    pre-training, g is discarded and a classification head h is applied
    on top of the learned f." Chapter 4's representation is f's output
    only (Eq. 3.13); g exists solely to shape the contrastive loss
    (Eq. 3.14-3.15) during pretraining.
    """

    def __init__(self, input_dim: int, config: SCARFConfig, output_dim: int = 256):
        super().__init__()
        hidden_dim = config.head_hidden_dim
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(config.head_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


def nt_xent_loss(z: torch.Tensor, z_tilde: torch.Tensor, temperature: float) -> torch.Tensor:
    """Eq. 3.14 (cosine similarity) + Eq. 3.15 (InfoNCE), Bahri et al.
    (2022)'s OWN asymmetric formula -- confirmed directly against the
    paper's Section 3 (not the task's own paraphrase, which described a
    symmetric SimCLR-style negative set; resolved in favour of the paper's
    actual equation, per user choice):

        s_i,j = z^(i)^T z~^(j) / (||z^(i)||_2 ||z~^(j)||_2),  i, j in [N]
        L_cont = (1/N) sum_i -log( exp(s_i,i / tau) / sum_k exp(s_i,k / tau) )

    For row i, the anchor is ONLY z_i (the clean view) -- there is no
    reverse direction anchored from z_tilde_i. The negative set for anchor
    i is the OTHER rows' corrupted views z_tilde_j (j != i) only; other
    rows' clean views z_j (j != i) are never used as negatives. z, z_tilde
    are z_i^(i=1..N) / z_tilde^(i=1..N) stacked into (N, dim) batches, one
    row per example, positive pair (z_i, z_tilde_i) at matching row index.

    z/z_tilde are expected already L2-normalised (ScarfProjectionHead's
    own output) -- F.normalize is applied here too regardless, so this
    function implements Eq. 3.14 itself rather than assuming the caller
    already has, and is a no-op on already-unit vectors.

    Implemented via F.cross_entropy(similarity, arange(N)): PyTorch's
    cross_entropy computes exactly (1/N) sum_i -log(softmax(logits_i)[target_i]),
    softmax(logits_i)[k] = exp(s_i,k/tau) / sum_k exp(s_i,k/tau) -- Eq.
    3.15's own formula, up to Eq. 3.15's own "(1/N) sum_k" INSIDE the
    denominator: exp(s_i,i/tau) / [(1/N) sum_k exp(s_i,k/tau)] =
    N * softmax(logits_i)[i], so Eq. 3.15's loss is this function's return
    value plus a constant -log(N) offset -- independent of z/z_tilde, so
    it does not affect gradients, only the raw logged loss value. A
    stated, cited simplification, not a silent deviation.
    """
    z = F.normalize(z, p=2, dim=-1)
    z_tilde = F.normalize(z_tilde, p=2, dim=-1)

    similarity = z @ z_tilde.T / temperature
    targets = torch.arange(z.shape[0], device=z.device)
    return F.cross_entropy(similarity, targets)
