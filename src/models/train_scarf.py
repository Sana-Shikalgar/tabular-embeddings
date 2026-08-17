"""Trains SCARF's encoder f, projection head g, and ScarfEmbeddingTables
jointly to minimize nt_xent_loss (Eq. 3.14-3.15, scarf.py) -- the
contrastive alignment between each row's clean and corrupted views
(scarf.py's build_scarf_views).

Per batch, NOT once up front: unlike train_ft_transformer.py/
train_subtab.py, which each call fitted_encoders.transform() on the whole
train_df/val_df exactly once before the epoch loop (a deterministic
function of the row alone, cheap to reuse across epochs),
corrupt_dataframe() draws a fresh Bernoulli(p) mask every call -- Bahri et
al. (2022)'s own procedure is a new random corruption every batch, every
epoch, not one fixed augmentation reused throughout training. So
build_scarf_views() is called once per batch here, for both training and
validation chunks alike -- this is also the "calls transform() twice per
batch" behaviour pipeline.py's own module docstring already commits to,
and build_scarf_views' own docstring names.

tables.parameters() is included in the optimizer alongside
encoder.parameters()/head.parameters(): ScarfEmbeddingTables' five
nn.Embedding tables are SCARF's own trainable weights (scarf.py's
ScarfEmbeddingTables docstring), unlike FT-Transformer's fixed one-hot
original_language treatment. Leaving tables out of the optimizer would
silently freeze every list-field embedding at its random initialisation --
no error, just weights that never move.

Optimizer: Adam (SCARFConfig.optimizer="adam") at scarf_config.lr. Bahri
et al. (2022) SS4 don't specify betas or weight decay beyond whatever
their own implementation defaults to, and SCARFConfig itself carries no
such fields (unlike FTTransformerConfig.weight_decay or
SubTabConfig.optimizer_betas/eps) -- so none are set explicitly here;
torch.optim.Adam's own defaults apply, not a value the paper/config never
specified.

shared_config.patience (16) governs early stopping here, NOT Bahri et al.
(2022)'s own reported patience: SS4 "Model architecture and training"
states, verbatim, "Unsupervised pre-training methods all use early
stopping with patience 3 on the validation loss, unless otherwise noted."
This project's own D12 decision (see SubTabConfig/train_subtab.py's
identical treatment of Ucar et al.'s own 20-epoch Income run) is to use
one shared, patience-based stopping rule across all three paradigms
rather than each paradigm's own paper-specific number;
SharedTrainingConfig's own docstring covers the general "why".

Returns (tables, encoder, head, logger) -- Prompt 5.2's TrainingLogger
retrofit (train_ft_transformer.py's identical change): the train_losses/
val_losses lists this function used to return directly are gone; logger is
a TrainingLogger already .save()'d to artifacts_dir / "models" / "scarf" /
"diagnostics" / "loss_curve.csv" before this function returns. Read the
per-epoch values back via logger.to_dataframe().
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.pipeline import FittedEncoders
from src.models.paradigm_config import SCARFConfig, SharedTrainingConfig
from src.models.scarf import (
    ScarfEmbeddingTables,
    ScarfEncoder,
    ScarfProjectionHead,
    build_scarf_views,
    nt_xent_loss,
)
from src.models.training_diagnostics import TrainingLogger


def _run_batch(
    tables: ScarfEmbeddingTables,
    encoder: ScarfEncoder,
    head: ScarfProjectionHead,
    batch_df: pd.DataFrame,
    fitted_encoders: FittedEncoders,
    scarf_config: SCARFConfig,
    rng: np.random.Generator,
) -> torch.Tensor:
    clean_vec, corrupted_vec = build_scarf_views(
        batch_df, fitted_encoders, tables, scarf_config.corruption_rate, rng
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
) -> float:
    """One shuffled pass over train_df, row-count-weighted mean NT-Xent
    loss across chunks -- mirrors train_subtab.py's _run_train_epoch."""
    n_rows = len(train_df)
    row_order = np.random.permutation(n_rows)
    tables.train()
    encoder.train()
    head.train()

    total_loss, total_rows = 0.0, 0
    for start in range(0, n_rows, scarf_config.batch_size):
        idx = row_order[start : start + scarf_config.batch_size]
        batch_df = train_df.iloc[idx]

        optimizer.zero_grad()
        loss = _run_batch(tables, encoder, head, batch_df, fitted_encoders, scarf_config, rng)
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
) -> float:
    """Fixed-order pass over val_df, row-count-weighted mean NT-Xent loss
    across chunks -- same corruption procedure as training (a fresh
    Bernoulli(p) draw per chunk), just without gradient updates."""
    n_rows = len(val_df)
    tables.eval()
    encoder.eval()
    head.eval()

    total_loss, total_rows = 0.0, 0
    with torch.no_grad():
        for start in range(0, n_rows, scarf_config.batch_size):
            batch_df = val_df.iloc[start : start + scarf_config.batch_size]
            loss = _run_batch(tables, encoder, head, batch_df, fitted_encoders, scarf_config, rng)

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
    artifacts_dir: Path,
) -> tuple[ScarfEmbeddingTables, ScarfEncoder, ScarfProjectionHead, TrainingLogger]:
    rng = np.random.default_rng()

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
            tables, encoder, head, train_df, fitted_encoders, scarf_config, rng, optimizer
        )
        val_loss = _run_validation(
            tables, encoder, head, val_df, fitted_encoders, scarf_config, rng
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
    logger.save(artifacts_dir, "scarf")

    return tables, encoder, head, logger
