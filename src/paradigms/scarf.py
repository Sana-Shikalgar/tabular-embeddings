"""Implements SCARF (Bahri et al. 2022): corrupts a random subset of each
row's features, encodes both the clean and corrupted views, and trains
them together via a contrastive loss."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.list_pooling import ListFieldPooler
from src.features.pipeline import FittedEncoders
from src.paradigms.ft_transformer import TEXT_FIELDS, FTTransformerConfig
from src.paradigms.paradigm_config import SCARFConfig

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
    """Corrupts df by replacing a Bernoulli(p)-selected subset of each
    eligible column's values with values resampled from that column's own
    empirical distribution, independently per column. Returns a corrupted
    copy; df is never mutated."""
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
    list-valued fields, plus each field's pooling mode."""

    def __init__(self, fitted_encoders: FittedEncoders, e_dim: int = DEFAULT_E_DIM):
        """Builds one embedding table per list field, sized from
        fitted_encoders, plus each field's pooling mode."""
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
    """Embeds and pools each list field's items into one vector per row,
    for every field in tables."""
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
    """Concatenates one fitted_encoders.transform() batch (clean or
    corrupted) into SCARF's flat input vector: numeric, one-hot language,
    pooled list fields, then the two text embeddings, in that fixed
    order."""
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
    a copy of df, transforms both the original and corrupted copies, and
    flattens each into SCARF's input vector via the same shared-weight
    tables."""
    corrupted_df = corrupt_dataframe(df, fitted_encoders, p, rng)

    clean_transformed = fitted_encoders.transform(df)
    corrupted_transformed = fitted_encoders.transform(corrupted_df)

    clean_vector = build_scarf_flat_vector(clean_transformed, tables, fitted_encoders)
    corrupted_vector = build_scarf_flat_vector(corrupted_transformed, tables, fitted_encoders)

    return clean_vector, corrupted_vector


class ScarfEncoder(nn.Module):
    """f: SCARF's encoder MLP, mapping the flat input vector to the
    representation used downstream."""

    def __init__(self, input_dim: int, config: SCARFConfig, output_dim: int = 256):
        """Builds an MLP of encoder_layers Linear+ReLU layers at
        encoder_hidden_dim, ending in a Linear to output_dim."""
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
        """Encodes x through the MLP."""
        return self.net(x)


class ScarfProjectionHead(nn.Module):
    """g: SCARF's pretraining projection head, mapping the encoder's
    output to an L2-normalized vector for the contrastive loss."""

    def __init__(self, input_dim: int, config: SCARFConfig, output_dim: int = 256):
        """Builds an MLP of head_layers Linear+ReLU layers at
        head_hidden_dim, ending in a Linear to output_dim."""
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
        """Projects x through the MLP and L2-normalizes the result."""
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


def nt_xent_loss(z: torch.Tensor, z_tilde: torch.Tensor, temperature: float) -> torch.Tensor:
    """Computes the NT-Xent contrastive loss between clean embeddings z
    and corrupted embeddings z_tilde: for each row, its clean view is the
    anchor and every other row's corrupted view is a negative, with its own
    corrupted view as the positive."""
    z = F.normalize(z, p=2, dim=-1)
    z_tilde = F.normalize(z_tilde, p=2, dim=-1)

    similarity = z @ z_tilde.T / temperature
    targets = torch.arange(z.shape[0], device=z.device)
    return F.cross_entropy(similarity, targets)
