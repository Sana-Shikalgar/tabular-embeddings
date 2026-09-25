"""
Tokenizes each row's features (numeric, categorical, list, text) into a
shared-width sequence, encodes it with a Transformer, and predicts
vote_average from the resulting [CLS] embedding.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.features.list_pooling import ListFieldPooler
from src.features.pipeline import FittedEncoders
from src.paradigms.paradigm_config import FTTransformerConfig

TEXT_FIELDS = ("overview", "original_title")


class FeatureTokenizer(nn.Module):
    """Embeds/projects every feature type to a common d_token width and
    returns one token per feature, in a fixed column order."""

    def __init__(self, fitted_encoders: FittedEncoders, d_token: int, include_text: bool = True):
        """Builds one submodule per feature (numeric/bypass linear layers,
        categorical and list embedding tables, text projections), sized
        from fitted_encoders. text_projections is only built when
        include_text is True."""
        super().__init__()
        self.column_groups = fitted_encoders.column_groups
        self.d_token = d_token
        self.include_text = include_text

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

        if self.include_text:
            self.text_projections = nn.ModuleDict(
                {field: nn.Linear(fitted_encoders.dims[field], d_token) for field in TEXT_FIELDS}
            )

    def forward(self, batch: dict[str, np.ndarray], include_text: bool = True) -> torch.Tensor:
        """Tokenizes one fitted_encoders.transform() batch into a
        (batch, n_tokens, d_token) tensor, one token per feature in the
        fixed column order. Text tokens are appended only when both this
        call's include_text and the tokenizer's own (set at construction)
        are True -- skipped otherwise, since text_projections doesn't
        exist unless the tokenizer itself was built with include_text."""
        device = self.language_embedding.weight.device
        tokens: list[torch.Tensor] = []

        numeric = torch.from_numpy(batch["numeric"]).float().to(device)
        for col, start, end in self._numeric_slices:
            tokens.append(self.numeric_tokenizers[col](numeric[:, start:end]))
        for col, start, end in self._bypass_slices:
            tokens.append(self.bypass_tokenizers[col](numeric[:, start:end]))

        # .copy(): CategoricalLookup.transform()'s pandas .map()/.to_numpy() output
        # is a read-only view under pandas' copy-on-write -- torch.from_numpy would
        # otherwise wrap that same read-only buffer and warn.
        lang_idx = torch.from_numpy(batch["original_language"].copy()).long().to(device)
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

        if self.include_text and include_text:
            for field in TEXT_FIELDS:
                vec = torch.from_numpy(batch[field]).float().to(device)
                tokens.append(self.text_projections[field](vec))

        return torch.stack(tokens, dim=1)


class FTTransformer(nn.Module):
    """Prepends a learnable [CLS] token to FeatureTokenizer's output and
    encodes the sequence with a stack of pre-norm Transformer encoder
    layers."""

    def __init__(
        self, fitted_encoders: FittedEncoders, config: FTTransformerConfig, include_text: bool = True
    ):
        """Builds the feature tokenizer, [CLS] token, and Transformer
        encoder stack from fitted_encoders and config. include_text=False
        builds a tokenizer with no text_projections, for a genuine
        text-free retrain rather than an inference-time skip."""
        super().__init__()
        self.config = config
        self.tokenizer = FeatureTokenizer(fitted_encoders, d_token=config.d_token, include_text=include_text)
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
        """Tokenizes batch, prepends [CLS], and returns the full encoded
        sequence (batch, n_tokens + 1, d_token)."""
        tokens = self.tokenizer(batch)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        return self.encoder(tokens)


class FTTransformerModel(nn.Module):
    """Wraps FTTransformer with a prediction head; .embed() returns the
    [CLS] representation used downstream, .forward() adds the regression
    head for training."""

    def __init__(
        self, fitted_encoders: FittedEncoders, config: FTTransformerConfig, include_text: bool = True
    ):
        """Builds the FTTransformer backbone and a LayerNorm/ReLU/Linear
        prediction head. include_text=False builds a text-free backbone
        (see FTTransformer)."""
        super().__init__()
        self.ft_transformer = FTTransformer(fitted_encoders, config, include_text=include_text)
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(config.d_token), nn.ReLU(), nn.Linear(config.d_token, 1)
        )

    def embed(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        """Returns the final-layer [CLS] token embedding, shape
        (batch, d_token)."""
        encoded = self.ft_transformer(batch)
        return encoded[:, 0, :]

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        """Predicts vote_average from embed()'s [CLS] token, shape
        (batch,)."""
        return self.prediction_head(self.embed(batch)).squeeze(-1)
