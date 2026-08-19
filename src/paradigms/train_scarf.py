"""Trains SCARF's encoder, projection head, and embedding tables jointly
to minimize the NT-Xent contrastive loss between each row's clean and
corrupted views."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import config
from src.features.pipeline import FittedEncoders
from src.paradigms.paradigm_config import SCARFConfig, SharedTrainingConfig
from src.paradigms.scarf import (
    ScarfEmbeddingTables,
    ScarfEncoder,
    ScarfProjectionHead,
    build_scarf_views,
    expand_group_keys_to_raw_cols,
    nt_xent_loss,
)
from src.paradigms.training_diagnostics import TrainingLogger


def _run_batch(
    tables: ScarfEmbeddingTables,
    encoder: ScarfEncoder,
    head: ScarfProjectionHead,
    batch_df: pd.DataFrame,
    fitted_encoders: FittedEncoders,
    scarf_config: SCARFConfig,
    rng: np.random.Generator,
    marginals: dict[str, np.ndarray] | None = None,
) -> torch.Tensor:
    """Builds one batch's clean/corrupted views (resampling corrupted
    values from marginals when given, otherwise from batch_df itself) and
    returns their NT-Xent loss."""
    clean_vec, corrupted_vec = build_scarf_views(
        batch_df, fitted_encoders, tables, scarf_config.corruption_rate, rng, marginals=marginals
    )
    z_clean = head(encoder(clean_vec))
    z_corrupted = head(encoder(corrupted_vec))
    return nt_xent_loss(z_clean, z_corrupted, scarf_config.temperature)


def _run_train_epoch(
    tables: ScarfEmbeddingTables,
    encoder: ScarfEncoder,
    head: ScarfProjectionHead,
    train_df: pd.DataFrame,
    fitted_encoders: FittedEncoders,
    scarf_config: SCARFConfig,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer,
    marginals: dict[str, np.ndarray] | None = None,
) -> float:
    """One shuffled pass (via rng) over train_df, updating tables/encoder/
    head on each batch's loss. Returns the row-count-weighted mean
    NT-Xent loss across batches."""
    n_rows = len(train_df)
    row_order = rng.permutation(n_rows)
    tables.train()
    encoder.train()
    head.train()

    total_loss, total_rows = 0.0, 0
    for start in range(0, n_rows, scarf_config.batch_size):
        idx = row_order[start : start + scarf_config.batch_size]
        batch_df = train_df.iloc[idx]

        optimizer.zero_grad()
        loss = _run_batch(
            tables, encoder, head, batch_df, fitted_encoders, scarf_config, rng, marginals=marginals
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)
        total_rows += len(idx)

    return total_loss / total_rows


def _run_validation(
    tables: ScarfEmbeddingTables,
    encoder: ScarfEncoder,
    head: ScarfProjectionHead,
    val_df: pd.DataFrame,
    fitted_encoders: FittedEncoders,
    scarf_config: SCARFConfig,
    rng: np.random.Generator,
    marginals: dict[str, np.ndarray] | None = None,
) -> float:
    """Fixed-order pass over val_df, computing NT-Xent loss without
    gradient updates. Returns the row-count-weighted mean loss across
    batches."""
    n_rows = len(val_df)
    tables.eval()
    encoder.eval()
    head.eval()

    total_loss, total_rows = 0.0, 0
    with torch.no_grad():
        for start in range(0, n_rows, scarf_config.batch_size):
            batch_df = val_df.iloc[start : start + scarf_config.batch_size]
            loss = _run_batch(
                tables, encoder, head, batch_df, fitted_encoders, scarf_config, rng, marginals=marginals
            )

            total_loss += loss.item() * len(batch_df)
            total_rows += len(batch_df)

    return total_loss / total_rows


def train_scarf(
    tables: ScarfEmbeddingTables,
    encoder: ScarfEncoder,
    head: ScarfProjectionHead,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fitted_encoders: FittedEncoders,
    scarf_config: SCARFConfig,
    shared_config: SharedTrainingConfig,
    models_dir: Path,
    seed: int = config.RANDOM_SEED,
) -> tuple[ScarfEmbeddingTables, ScarfEncoder, ScarfProjectionHead, TrainingLogger]:
    """Trains tables/encoder/head against train_df/val_df, early-stopping
    on validation loss, restores the best-validation-epoch weights, and
    saves the training curve via TrainingLogger. Returns
    (tables, encoder, head, logger)."""
    rng = np.random.default_rng(seed)

    # Fixed once from train_df, before the epoch loop -- every batch, every
    # epoch, every corrupted value is resampled from this same train-set
    # marginal distribution, never rebuilt from whatever batch is being
    # corrupted (which would leak that batch's own values back into itself).
    raw_cols = expand_group_keys_to_raw_cols(fitted_encoders, fitted_encoders.corruption_eligible_cols)
    marginals = {col: train_df[col].to_numpy() for col in raw_cols}

    optimizer = torch.optim.Adam(
        itertools.chain(tables.parameters(), encoder.parameters(), head.parameters()),
        lr=scarf_config.lr,
    )

    logger = TrainingLogger()
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, dict] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, shared_config.max_epochs + 1):
        train_loss = _run_train_epoch(
            tables, encoder, head, train_df, fitted_encoders, scarf_config, rng, optimizer,
            marginals=marginals,
        )
        val_loss = _run_validation(
            tables, encoder, head, val_df, fitted_encoders, scarf_config, rng,
            marginals=marginals,
        )
        logger.log(epoch, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                "tables": copy.deepcopy(tables.state_dict()),
                "encoder": copy.deepcopy(encoder.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= shared_config.patience:
                break

    tables.load_state_dict(best_state["tables"])
    encoder.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])
    print(f"Restored epoch {best_epoch}'s weights (val NT-Xent: {best_val_loss:.4f})")
    logger.save(models_dir, "scarf")

    return tables, encoder, head, logger
