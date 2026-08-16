"""Trains FTTransformerModel to minimize MSE against vote_average (Eq. 3.10) --
FTTransformerModel.forward() already returns the single scalar prediction this
loss compares against the target.

Weight decay follows Table 12: zeroed for the feature tokenizer, LayerNorm, and
biases rather than applied uniformly. Implemented as two AdamW param groups --
no_decay covers any parameter belonging to an nn.LayerNorm module (norm1/norm2
inside each nn.TransformerEncoderLayer) or whose full name contains "bias" or
"tokenizer" (FeatureTokenizer's per-column/per-field weights all live under
self.ft_transformer.tokenizer, so this one substring check exempts the whole
feature tokenizer, matching Table 12's "not applied uniformly" note).

train_df/val_df are transformed exactly once each, not per-batch/epoch --
fitted_encoders.transform() re-runs the standardizer/PLE/list-pooler pipeline
and a dict-based SBERT cache lookup on every call, so encoding once and
slicing the resulting arrays per minibatch avoids repeating that work every
epoch for no benefit.

shared_config.patience only decides when training STOPS, not which epoch's
weights get kept -- the loop can run 16 epochs past the best val MSE before
early stopping fires. So the best-val-MSE state_dict is checkpointed in
memory every time a new best is found, and reloaded into model once the
loop ends (whether by early stopping or by exhausting max_epochs), so the
model returned is always the best-validation-epoch one, never whatever
epoch training happened to stop at.

Returns (model, train_losses, val_losses) as plain lists. Prompt 5.2 will
retrofit this into the shared logger -- that wiring is not done here.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features.pipeline import FittedEncoders
from src.models.ft_transformer import FTTransformerModel
from src.models.paradigm_config import FTTransformerConfig, SharedTrainingConfig

TARGET_COL = "vote_average"


def _build_optimizer(model: nn.Module, config: FTTransformerConfig) -> torch.optim.AdamW:
    no_decay_ids = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters(recurse=False):
                no_decay_ids.add(id(param))

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in no_decay_ids or "bias" in name or "tokenizer" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.lr,
    )


def _slice_batch(batch: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[idx] for key, value in batch.items()}


def _run_epoch(
    model: FTTransformerModel,
    batch: dict[str, np.ndarray],
    target: np.ndarray,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """One pass over batch/target in batch_size chunks. Shuffles row order
    and updates weights when optimizer is given (training); otherwise runs
    in a fixed order under torch.no_grad() (validation). Returns the
    row-count-weighted mean MSE across chunks."""
    n_rows = len(target)
    device = next(model.parameters()).device
    row_order = np.random.permutation(n_rows) if optimizer is not None else np.arange(n_rows)
    model.train(optimizer is not None)

    total_loss, total_rows = 0.0, 0
    for start in range(0, n_rows, batch_size):
        idx = row_order[start : start + batch_size]
        chunk = _slice_batch(batch, idx)
        target_chunk = torch.from_numpy(target[idx]).float().to(device)

        if optimizer is not None:
            optimizer.zero_grad()
            preds = model(chunk)
            loss = F.mse_loss(preds, target_chunk)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(chunk)
                loss = F.mse_loss(preds, target_chunk)

        total_loss += loss.item() * len(idx)
        total_rows += len(idx)

    return total_loss / total_rows


def train_ft_transformer(
    model: FTTransformerModel,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fitted_encoders: FittedEncoders,
    ft_config: FTTransformerConfig,
    shared_config: SharedTrainingConfig,
) -> tuple[FTTransformerModel, list[float], list[float]]:
    train_batch = fitted_encoders.transform(train_df)
    train_target = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    val_batch = fitted_encoders.transform(val_df)
    val_target = val_df[TARGET_COL].to_numpy(dtype=np.float32)

    optimizer = _build_optimizer(model, ft_config)

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, shared_config.max_epochs + 1):
        train_loss = _run_epoch(
            model, train_batch, train_target, ft_config.batch_size, optimizer
        )
        val_loss = _run_epoch(model, val_batch, val_target, ft_config.batch_size, None)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= shared_config.patience:
                break

    model.load_state_dict(best_state_dict)
    print(f"Restored epoch {best_epoch}'s weights (val MSE: {best_val_loss:.4f})")

    return model, train_losses, val_losses
