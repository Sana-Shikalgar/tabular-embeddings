"""Hyperparameter configs for FT-Transformer, SubTab, and SCARF, plus one
shared training/early-stopping config used across all three paradigms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FTTransformerConfig:
    """FT-Transformer hyperparameters (Gorishniy et al. 2021, Table 12):
    architecture, dropout, and optimizer settings."""

    n_layers: int = 3
    d_token: int = 192
    n_heads: int = 8
    ffn_factor: float = 4 / 3
    activation: str = "relu"
    attention_dropout: float = 0.2
    ffn_dropout: float = 0.1
    residual_dropout: float = 0.0
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 256


@dataclass(frozen=True)
class SubTabConfig:
    """SubTab hyperparameters (Ucar et al. 2021): subset/overlap, encoder/
    decoder architecture, optimizer settings, and list-field SVD reduction
    settings."""

    n_subsets: int = 5
    overlap: float = 0.25
    encoder_dims: tuple[int, int] = (1024, 1024)
    latent_dim: int = 1024
    aggregation: str = "mean"
    optimizer: str = "adamw"
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-7
    lr: float = 0.001
    batch_size: int = 256
    svd_variance_threshold: float = 0.50
    svd_n_components_ceiling: int = 300
    svd_fields: tuple[str, str] = ("keywords", "production_companies")


@dataclass(frozen=True)
class SCARFConfig:
    """SCARF hyperparameters (Bahri et al. 2022): encoder/head architecture,
    corruption rate, temperature, and optimizer settings."""

    encoder_layers: int = 4
    encoder_hidden_dim: int = 256
    head_layers: int = 2
    head_hidden_dim: int = 256
    corruption_rate: float = 0.6
    temperature: float = 1.0
    optimizer: str = "adam"
    lr: float = 0.001
    batch_size: int = 128


@dataclass(frozen=True)
class SharedTrainingConfig:
    """Shared max-epochs/patience/early-stopping settings used across all
    three paradigms; each paradigm's own val_loss quantity differs, only the
    stopping mechanics are shared."""

    max_epochs: int = 100
    patience: int = 16
    early_stop_metric: str = "val_loss"
