"""Prompt 5.5's shared diagnostics logger -- 05_embedding_paradigms.ipynb's own
FT-Transformer training-curve cell already names this: "Ad hoc sanity check
only -- not the final asset. Prompt 5.5's shared diagnostics logging will
later replace this plot with the identical-across-paradigms version (same
axes/styling for FT-Transformer, SubTab, and SCARF alike)." TrainingLogger is
that replacement.

TrainingLogger itself carries no paradigm identity (no paradigm_name in
__init__) -- it only ever appears as an explicit argument to .save(), never
stored on the instance. This is deliberate, not an oversight: the whole
point (per the task this class implements) is that FT-Transformer's,
SubTab's, and SCARF's curves are visually comparable side by side even
though their loss units differ (MSE vs L_r vs NT-Xent) -- so .plot()
produces the SAME axis labels ("epoch"/"loss"), gridlines, scale, and
line colors no matter which paradigm's losses it's given, and never prints
a paradigm-specific unit onto the shared axes. Any paradigm-specific
title/annotation is the CALLER's job, applied to the returned Axes after
.plot() returns -- not this class's.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class TrainingLogger:
    def __init__(self):
        self.records: list[dict[str, float]] = []

    def log(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records, columns=["epoch", "train_loss", "val_loss"])

    def save(self, artifacts_dir: Path, paradigm_name: str) -> Path:
        diagnostics_dir = Path(artifacts_dir) / "models" / paradigm_name / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        path = diagnostics_dir / "loss_curve.csv"
        self.to_dataframe().to_csv(path, index=False)
        print(f"Saved: {path}")
        return path

    def plot(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Identical figure style regardless of paradigm -- see module
        docstring. Colors are set explicitly (not left to matplotlib's
        default cycle) so train/val are the same two colors on every
        paradigm's plot, even if a caller reuses an ax that already has
        other lines drawn on it."""
        if ax is None:
            _, ax = plt.subplots()

        df = self.to_dataframe()
        ax.plot(df["epoch"], df["train_loss"], label="train", color="C0")
        ax.plot(df["epoch"], df["val_loss"], label="val", color="C1")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_yscale("linear")
        ax.grid(True)
        ax.legend()
        return ax
