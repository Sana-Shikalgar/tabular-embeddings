"""Trains SubTabAutoencoder to minimize L_r, the reconstruction loss (Prompt
3.6/Eq. 3.12) -- L_c/L_d are not implemented at all (D28), so L_r is the
whole objective here, not one term among several being summed.

Flat vectors (Prompt 3.3, build_flat_vector) and column subsets (Prompt 3.4,
make_column_subsets) are built ONCE up front for train_df/val_df, not
per-epoch/per-batch -- same "encode once, slice per minibatch" reasoning as
train_ft_transformer.py: fitted_encoders.transform() and the per-field SVD
transforms inside build_flat_vector aren't cheap, and column subsets are a
fixed, deterministic function of full_width/n_subsets/overlap that never
changes across epochs.

Optimizer is AdamW at subtab_config.lr, betas=subtab_config.optimizer_betas,
eps=subtab_config.optimizer_eps -- Ucar et al. (2021) Appendix SC.4
"Implementation and resources" + Table A1's Income row (see SubTabConfig's
own docstring for the exact quotes). weight_decay is explicitly 0.0: neither
that quote nor SubTabConfig itself specifies a weight-decay value, and
torch.optim.AdamW's own default (0.01) is not something the paper stated --
left explicit rather than silently inheriting a PyTorch default the paper
never mentioned.

shared_config.max_epochs/patience govern stopping here, NOT Ucar et al.'s
own reported 20-epoch Income run (Appendix Table A1) -- this project's own
decision (D12) to use one shared, patience-based stopping rule across all
three paradigms rather than each paradigm's paper-specific epoch count;
SharedTrainingConfig's own docstring covers the general "why" for this.

Each epoch's validation pass computes per_segment_reconstruction_mse
(Prompt 3.6b) on the SAME reconstructions already computed for that epoch's
val L_r -- an extra read of already-computed tensors, not an extra forward
pass, per_segment_reconstruction_mse's own docstring). Best-epoch
checkpointing follows train_ft_transformer.py's exact discipline: L_r (val)
is checkpointed on improvement, restored after the loop ends, and the
segment-MSE values printed at the end are read from val_segment_mse[the
restored epoch], not val_segment_mse[-1] -- the same "restored, not final"
bug class train_ft_transformer.py's own val-MSE print was once caught on.

Returns (model, train_losses, val_losses, val_segment_mse) as plain lists/
dicts. val_segment_mse is one {"keywords_and_production_companies": ...,
"rest": ...} dict per epoch, same length and index alignment as val_losses
-- per_segment_reconstruction_mse's own return keys ("svd_segments"/"rest")
are remapped to "keywords_and_production_companies"/"rest" here, since that
is this function's own return-contract naming, not a second/competing
definition of the segments themselves.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from src.features.pipeline import FittedEncoders
from src.models.paradigm_config import SharedTrainingConfig, SubTabConfig
from src.models.subtab import (
    SubTabAutoencoder,
    SubtabListReducer,
    build_flat_vector,
    flat_vector_segment_bounds,
    make_column_subsets,
    per_segment_reconstruction_mse,
    reconstruction_loss,
)


def _run_train_epoch(
    model: SubTabAutoencoder,
    train_full: torch.Tensor,
    subsets: list[torch.Tensor],
    batch_size: int,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One shuffled pass over train_full, row-count-weighted mean L_r
    across chunks -- mirrors train_ft_transformer.py's _run_epoch."""
    n_rows = train_full.shape[0]
    row_order = np.random.permutation(n_rows)
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
    """Fixed-order pass over val_full. Computes L_r AND
    per_segment_reconstruction_mse per chunk off the SAME reconstructions,
    then row-count-weights both across chunks the same way L_r itself is."""
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
) -> tuple[SubTabAutoencoder, list[float], list[float], list[dict[str, float]]]:
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

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_segment_mse: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, shared_config.max_epochs + 1):
        train_loss = _run_train_epoch(
            model, train_full, subsets, subtab_config.batch_size, optimizer
        )
        val_loss, val_seg = _run_validation(
            model, val_full, subsets, subtab_config.batch_size, segment_bounds
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
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

    return model, train_losses, val_losses, val_segment_mse
