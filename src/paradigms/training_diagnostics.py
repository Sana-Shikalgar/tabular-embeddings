"""TrainingLogger: a shared per-epoch (train_loss, val_loss) recorder and
plotter used identically by all three paradigms' training loops."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class TrainingLogger:
    """Records, saves, and plots per-epoch train/val loss, with the same
    axes/styling regardless of which paradigm's losses it's given."""

    def __init__(self):
        """Initializes an empty record list."""
        self.records: list[dict[str, float]] = []

    def log(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Appends one epoch's (train_loss, val_loss) record."""
        self.records.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    def to_dataframe(self) -> pd.DataFrame:
        """Returns all recorded epochs as a DataFrame."""
        return pd.DataFrame(self.records, columns=["epoch", "train_loss", "val_loss"])

    def save(self, models_dir: Path, paradigm_name: str) -> Path:
        """Writes the recorded epochs to
        models_dir / paradigm_name / "diagnostics" / "loss_curve.csv"."""
        diagnostics_dir = Path(models_dir) / paradigm_name / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        path = diagnostics_dir / "loss_curve.csv"
        self.to_dataframe().to_csv(path, index=False)
        print(f"Saved: {path}")
        return path

    def plot(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plots train/val loss vs. epoch on ax (or a new Axes), with
        fixed colors/labels/gridlines so every paradigm's plot looks the
        same."""
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
