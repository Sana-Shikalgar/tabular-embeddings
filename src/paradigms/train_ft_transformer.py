"""Trains FTTransformerModel to minimize MSE against vote_average, with
zero weight decay for the tokenizer/LayerNorm/bias parameters and
best-validation-epoch checkpointing."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.features.pipeline import FittedEncoders
from src.paradigms.ft_transformer import FTTransformerModel
from src.paradigms.paradigm_config import FTTransformerConfig, SharedTrainingConfig
from src.paradigms.training_diagnostics import TrainingLogger

TARGET_COL = "vote_average"


def _build_optimizer(model: nn.Module, config: FTTransformerConfig) -> torch.optim.AdamW:
    """Builds an AdamW optimizer with two parameter groups: zero weight
    decay for LayerNorm/bias/tokenizer parameters, config.weight_decay for
    everything else."""
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
    """Slices every array in batch to rows idx."""
    return {key: value[idx] for key, value in batch.items()}


def _run_epoch(
    model: FTTransformerModel,
    batch: dict[str, np.ndarray],
    target: np.ndarray,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator,
) -> float:
    """One pass over batch/target in batch_size chunks. Shuffles row order
    (via rng) and updates weights when optimizer is given (training);
    otherwise runs in a fixed order under torch.no_grad() (validation).
    Returns the row-count-weighted mean MSE across chunks."""
    n_rows = len(target)
    device = next(model.parameters()).device
    row_order = rng.permutation(n_rows) if optimizer is not None else np.arange(n_rows)
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
    models_dir: Path,
    seed: int = config.RANDOM_SEED,
) -> tuple[FTTransformerModel, TrainingLogger]:
    """Trains model against train_df/val_df, early-stopping on validation
    MSE, restores the best-validation-epoch weights, and saves the
    training curve via TrainingLogger. Returns (model, logger)."""
    rng = np.random.default_rng(seed)
    train_batch = fitted_encoders.transform(train_df)
    train_target = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    val_batch = fitted_encoders.transform(val_df)
    val_target = val_df[TARGET_COL].to_numpy(dtype=np.float32)

    optimizer = _build_optimizer(model, ft_config)

    logger = TrainingLogger()
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, shared_config.max_epochs + 1):
        train_loss = _run_epoch(
            model, train_batch, train_target, ft_config.batch_size, optimizer, rng
        )
        val_loss = _run_epoch(model, val_batch, val_target, ft_config.batch_size, None, rng)
        logger.log(epoch, train_loss, val_loss)

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
    logger.save(models_dir, "ft_transformer")

    return model, logger
