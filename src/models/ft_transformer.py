"""FeatureTokenizer implements Eq. 3.7: turns one fitted_encoders.transform()
batch into a (batch, n_tokens, d_token) sequence for FT-Transformer's
self-attention, which requires every token to share the same width.

Per-key tokenization:
- numeric: re-sliced back into one PLE sub-vector per column_groups.numeric_cols
  entry (offsets follow pipeline.py's own documented concatenation order --
  numeric_cols first, summing ple.n_bins_[col], then one width-1 slice per
  bypass_cols entry), then one nn.Linear(ple.n_bins_[col], d_token) per
  numeric column, with NO shared weights across features -- Gorishniy,
  Rubachev & Babenko (2022) §3.2: "one linear layer after PLE, without
  sharing weights between features."
- bypass columns: FT-Transformer's own original numeric tokenizer,
  T_j = b_j + x_j * W_j, W_j in R^d_token (Gorishniy et al. 2021, §3,
  Feature Tokenizer) -- implemented as nn.Linear(1, d_token), which computes
  exactly that affine form.
- original_language: nn.Embedding(dims["original_language"] + 1, d_token);
  the +1 covers the reserved OOV index 0 (categorical.py's own convention).
- each list field: nn.Embedding(dims[field] + 1, d_token) applied per item,
  pooled into one d_token-wide token via ListFieldPooler.pool(mode=...)
  (Eq. 3.5/3.6). Full Section 3.3 vocabulary, no cap or reduction --
  embeddings compress to d_token regardless of vocab size, so this
  paradigm never had a cardinality problem the way one-hot/multi-hot did.
- overview / original_title: nn.Linear(768, d_token) each, per Prompt 2.1's
  decision (self-attention's shared-width requirement rules out using the
  frozen 768-d SBERT vectors as-is without abandoning d_token=192 globally).

Token order (fixed, and required by anything reading the output positionally):
column_groups.numeric_cols, then column_groups.bypass_cols, then
original_language, then column_groups.list_cols, then overview, then
original_title.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.features.list_pooling import ListFieldPooler
from src.features.pipeline import FittedEncoders
from src.models.paradigm_config import FTTransformerConfig

TEXT_FIELDS = ("overview", "original_title")


class FeatureTokenizer(nn.Module):
    def __init__(self, fitted_encoders: FittedEncoders, d_token: int):
        super().__init__()
        self.column_groups = fitted_encoders.column_groups
        self.d_token = d_token

        # Slice offsets into the "numeric" array, matching pipeline.py's
        # transform() concatenation order exactly: numeric_cols first
        # (each ple.n_bins_[col] wide), then bypass_cols (each width 1).
        self._numeric_slices: list[tuple[str, int, int]] = []
        self._bypass_slices: list[tuple[str, int, int]] = []
        offset = 0
        for col in self.column_groups.numeric_cols:
            width = fitted_encoders.ple.n_bins_[col]
            self._numeric_slices.append((col, offset, offset + width))
            offset += width
        for col in self.column_groups.bypass_cols:
            self._bypass_slices.append((col, offset, offset + 1))
            offset += 1

        self.numeric_tokenizers = nn.ModuleDict(
            {col: nn.Linear(end - start, d_token) for col, start, end in self._numeric_slices}
        )
        self.bypass_tokenizers = nn.ModuleDict(
            {col: nn.Linear(1, d_token) for col, _, _ in self._bypass_slices}
        )

        self.language_embedding = nn.Embedding(
            fitted_encoders.dims["original_language"] + 1, d_token
        )

        self._list_pooling_modes: dict[str, str] = {
            field: pooler.pooling for field, pooler in fitted_encoders.list_poolers.items()
        }
        self.list_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(fitted_encoders.dims[field] + 1, d_token)
                for field in self.column_groups.list_cols
            }
        )

        self.text_projections = nn.ModuleDict(
            {field: nn.Linear(fitted_encoders.dims[field], d_token) for field in TEXT_FIELDS}
        )

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        device = self.language_embedding.weight.device
        tokens: list[torch.Tensor] = []

        numeric = torch.from_numpy(batch["numeric"]).float().to(device)
        for col, start, end in self._numeric_slices:
            tokens.append(self.numeric_tokenizers[col](numeric[:, start:end]))
        for col, start, end in self._bypass_slices:
            tokens.append(self.bypass_tokenizers[col](numeric[:, start:end]))

        lang_idx = torch.from_numpy(batch["original_language"]).long().to(device)
        tokens.append(self.language_embedding(lang_idx))

        for field in self.column_groups.list_cols:
            embedding_table = self.list_embeddings[field]
            pooling_mode = self._list_pooling_modes[field]
            pooled_rows = []
            for row_indices in batch[field]:
                idx_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
                item_embeddings = embedding_table(idx_tensor)
                pooled_rows.append(ListFieldPooler.pool(item_embeddings, mode=pooling_mode))
            tokens.append(torch.stack(pooled_rows))

        for field in TEXT_FIELDS:
            vec = torch.from_numpy(batch[field]).float().to(device)
            tokens.append(self.text_projections[field](vec))

        return torch.stack(tokens, dim=1)


class FTTransformer(nn.Module):
    """Eq. 3.8: prepends a learnable [CLS] token to FeatureTokenizer's
    output, then applies config.n_layers pre-norm Transformer encoder
    blocks (norm_first=True, matching Gorishniy et al. 2021 Appendix
    E.1's PreNorm choice). Returns the full encoded sequence
    (batch, n_tokens + 1, d_token), [CLS] at position 0 -- extracting
    and predicting from [CLS] is a later step, not done here.

    nn.TransformerEncoderLayer exposes one dropout value, not the three
    FTTransformerConfig has (attention_dropout, ffn_dropout,
    residual_dropout). config.ffn_dropout is used for that slot -- the
    structurally closest match, since PyTorch applies it inside the FFN
    and after each sublayer. attention_dropout/residual_dropout are not
    applied anywhere in this implementation: a stated, cited
    simplification (the same treatment as FTTransformerConfig's own
    ReLU/ReGLU substitution note), not a silent deviation.

    self.tokenizer (a FeatureTokenizer) and self.encoder (an
    nn.TransformerEncoder) are both directly accessible attributes, so
    either can be unit-tested independently of forward().
    """

    def __init__(self, fitted_encoders: FittedEncoders, config: FTTransformerConfig):
        super().__init__()
        self.config = config
        self.tokenizer = FeatureTokenizer(fitted_encoders, d_token=config.d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_token,
            nhead=config.n_heads,
            dim_feedforward=round(config.d_token * config.ffn_factor),
            dropout=config.ffn_dropout,
            activation=config.activation,
            norm_first=True,
            batch_first=True,
        )
        # enable_nested_tensor's fast path never applies here anyway --
        # norm_first=True already disables it -- so this just silences
        # PyTorch's UserWarning about the two settings disagreeing.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers, enable_nested_tensor=False
        )

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        tokens = self.tokenizer(batch)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return self.encoder(tokens)


class FTTransformerModel(nn.Module):
    """Wraps FTTransformer's tokenizer + encoder stack (self.ft_transformer)
    with a prediction head for training.

    .embed(batch) returns the final-layer [CLS] token, shape
    (batch, d_token) -- Eq. 3.9, the representation Chapter 4 actually
    uses downstream. This is the one call Section 3.4 points to.

    .forward(batch) = .embed(batch) through one more prediction head,
    squeezed to shape (batch,) -- for training only, not itself the
    representation. The head is Gorishniy et al. (2021, §3)'s own
    "Prediction" formula, y_hat = Linear(ReLU(LayerNorm(T_CLS))) -- a
    LayerNorm and ReLU before the final linear, not a bare nn.Linear.
    .embed(batch) is unaffected by this: Eq. 3.9's representation is the
    raw final-layer [CLS] token, defined independently of how it's used
    for supervised training.

    Loss: MSE on vote_average (Eq. 3.10). vote_average is continuous
    under the D3 target decision, so no CE branch is implemented here --
    a deliberate omission, not a silent one: a classification head would
    need a per-class softmax output, which this single-scalar regression
    head structurally isn't built for.
    """

    def __init__(self, fitted_encoders: FittedEncoders, config: FTTransformerConfig):
        super().__init__()
        self.ft_transformer = FTTransformer(fitted_encoders, config)
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(config.d_token), nn.ReLU(), nn.Linear(config.d_token, 1)
        )

    def embed(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        encoded = self.ft_transformer(batch)
        return encoded[:, 0, :]

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        return self.prediction_head(self.embed(batch)).squeeze(-1)
