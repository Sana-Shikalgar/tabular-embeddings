"""Trains SubTabAutoencoder to minimize the reconstruction loss L_r across
overlapping column subsets, with best-validation-epoch checkpointing and
per-segment MSE diagnostics."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config
from src.features.pipeline import FittedEncoders
from src.paradigms.paradigm_config import SharedTrainingConfig, SubTabConfig
from src.paradigms.subtab import (
    SubTabAutoencoder,
    SubtabListReducer,
    build_flat_vector,
    flat_vector_segment_bounds,
    make_column_subsets,
    per_segment_reconstruction_mse,
    reconstruction_loss,
)
from src.paradigms.training_diagnostics import TrainingLogger


def _run_train_epoch(
    model: SubTabAutoencoder,
    train_full: torch.Tensor,
    subsets: list[torch.Tensor],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator = np.random.default_rng(config.RANDOM_SEED),
) -> float:
    """One shuffled pass (via rng) over train_full, updating model on each
    subset's reconstruction loss. Returns the row-count-weighted mean L_r
    across chunks."""
    n_rows = train_full.shape[0]
    row_order = rng.permutation(n_rows)
    model.train()

    total_loss, total_rows = 0.0, 0
    for start in range(0, n_rows, batch_size):
        idx = row_order[start : start + batch_size]
        chunk = train_full[idx]

        optimizer.zero_grad()
        reconstructions = [model(chunk[:, subset_idx])[1] for subset_idx in subsets]
        loss = reconstruction_loss(chunk, reconstructions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)
        total_rows += len(idx)

    return total_loss / total_rows


def _run_validation(
    model: SubTabAutoencoder,
    val_full: torch.Tensor,
    subsets: list[torch.Tensor],
    batch_size: int,
    segment_bounds: dict[str, tuple[int, int]],
) -> tuple[float, dict[str, float]]:
    """Fixed-order pass over val_full, computing L_r and per-segment
    reconstruction MSE from the same reconstructions. Returns
    (val_loss, val_segment_mse)."""
    n_rows = val_full.shape[0]
    model.eval()

    total_loss = 0.0
    total_svd_segments = 0.0
    total_rest = 0.0
    total_rows = 0
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            chunk = val_full[start : start + batch_size]
            reconstructions = [model(chunk[:, subset_idx])[1] for subset_idx in subsets]

            chunk_loss = reconstruction_loss(chunk, reconstructions)
            chunk_segment_mse = per_segment_reconstruction_mse(
                chunk, reconstructions, segment_bounds
            )

            n = chunk.shape[0]
            total_loss += chunk_loss.item() * n
            total_svd_segments += chunk_segment_mse["svd_segments"] * n
            total_rest += chunk_segment_mse["rest"] * n
            total_rows += n

    val_loss = total_loss / total_rows
    val_segment_mse = {
        "keywords_and_production_companies": total_svd_segments / total_rows,
        "rest": total_rest / total_rows,
    }
    return val_loss, val_segment_mse


def train_subtab(
    model: SubTabAutoencoder,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fitted_encoders: FittedEncoders,
    list_reducers: dict[str, SubtabListReducer],
    subtab_config: SubTabConfig,
    shared_config: SharedTrainingConfig,
    models_dir: Path,
    rng: np.random.Generator = np.random.default_rng(config.RANDOM_SEED),
) -> tuple[SubTabAutoencoder, TrainingLogger, list[dict[str, float]]]:
    """Trains model against train_df/val_df, early-stopping on validation
    L_r, restores the best-validation-epoch weights, and saves the
    training curve via TrainingLogger. Returns
    (model, logger, val_segment_mse)."""
    train_flat = build_flat_vector(train_df, fitted_encoders, list_reducers)
    val_flat = build_flat_vector(val_df, fitted_encoders, list_reducers)
    full_width = train_flat.shape[1]

    subset_arrays = make_column_subsets(full_width, subtab_config.n_subsets, subtab_config.overlap)
    subsets = [torch.from_numpy(s).long() for s in subset_arrays]
    segment_bounds = flat_vector_segment_bounds(fitted_encoders, list_reducers)

    train_full = torch.from_numpy(train_flat).float()
    val_full = torch.from_numpy(val_flat).float()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=subtab_config.lr,
        betas=subtab_config.optimizer_betas,
        eps=subtab_config.optimizer_eps,
        weight_decay=0.0,
    )

    logger = TrainingLogger()
    val_segment_mse: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, shared_config.max_epochs + 1):
        train_loss = _run_train_epoch(
            model, train_full, subsets, subtab_config.batch_size, optimizer, rng
        )
        val_loss, val_seg = _run_validation(
            model, val_full, subsets, subtab_config.batch_size, segment_bounds
        )
        logger.log(epoch, train_loss, val_loss)
        val_segment_mse.append(val_seg)

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
    restored_segment_mse = val_segment_mse[best_epoch - 1]
    print(f"Restored epoch {best_epoch}'s weights (val L_r: {best_val_loss:.4f})")
    print(
        f"  at restored epoch: keywords_and_production_companies MSE = "
        f"{restored_segment_mse['keywords_and_production_companies']:.4f}, "
        f"rest MSE = {restored_segment_mse['rest']:.4f}"
    )
    logger.save(models_dir, "subtab")

    return model, logger, val_segment_mse
