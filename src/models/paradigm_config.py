from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FTTransformerConfig:
    """Gorishniy et al. (2021), Table 12 (FT-Transformer defaults).

    activation="relu": Gorishniy et al. (2021) use ReGLU by default, but
    Appendix E.1 states they "did not observe strong difference between
    ReGLU and ReLU in preliminary experiments." Plain PyTorch has no
    built-in ReGLU, so nn.ReLU is used here -- a stated, cited
    simplification, not a silent deviation.

    weight_decay=1e-5 is Table 12's default; the paper documents it as
    0.0 specifically for the feature tokenizer, LayerNorm, and biases,
    not applied uniformly.

    batch_size=256: Gorishniy et al. (2021) use a smaller batch size for
    all but their two largest datasets (where 1024 is used); 256 matches
    that regime and this dataset's scale, same uncited-but-consistent
    treatment as SubTabConfig's svd_variance_threshold.
    """

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
    """Uçar et al. (2021).

    n_subsets/overlap: §3.3 "Results", UCI Adult Income & BlogFeedback --
    "the best performance is obtained using 5 subsets with 25% overlap."

    encoder_dims/latent_dim: §3.3 "Results" says Income's architecture is
    "same as in Obesity," and the Obesity paragraph specifies "a
    two-layer encoder with [1024, 1024] dimensions. Second layer is a
    linear layer."

    batch_size: Appendix Table A1, Income row (256).

    optimizer/optimizer_betas/optimizer_eps/lr: §C.4 "Implementation and
    resources" -- AdamW, betas (0.9, 0.999), eps 1e-07; "Learning rate
    of 0.001 is used for all experiments."

    svd_variance_threshold/svd_n_components_ceiling/svd_fields: the rule
    from Prompt 3.1/3.2, not paper-sourced -- kept here alongside the
    rest of SubTab's config for one place to look.
    """

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
    svd_variance_threshold: float = 0.80
    svd_n_components_ceiling: int = 300
    svd_fields: tuple[str, str] = ("keywords", "production_companies")


@dataclass(frozen=True)
class SCARFConfig:
    """Bahri et al. (2022), §4 "Model architecture and training."

    List-valued fields use trainable embed-then-pool (Eq. 3.5), matching
    FT-Transformer, not Bahri et al.'s own one-hot treatment -- see the
    module docstring (D12) for why. The embedding-table code lives in
    Prompt 4.2; this file only holds the encoder/head/training numbers.
    """

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
    """Shared stopping rule across all three paradigms -- NOT a shared
    loss, since each paradigm's val_loss is a different quantity (MSE
    for FT-Transformer, reconstruction L_r for SubTab, InfoNCE for
    SCARF). Only the early-stopping mechanics are shared.

    patience=16: Gorishniy et al. (2021)'s own general training
    protocol -- "patience = 16 for all algorithms."

    max_epochs=100 is explicitly PROVISIONAL: to be confirmed against
    actual compute budget (Table 3.4 risk register).

    train_val_split is unchanged from 03_feature_split.ipynb's existing
    splits -- not re-derived here.
    """

    max_epochs: int = 100
    patience: int = 16
    early_stop_metric: str = "val_loss"
