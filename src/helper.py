"""JSON save/load helpers for any fitted encoder that exposes
to_dict()/from_dict().
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_json(obj, filename: str, dir_path: Path) -> Path:
    """Writes obj.to_dict() as JSON to <dir_path>/<filename>,
    creating the directory if needed. Returns the path saved to."""
    encoders_dir = Path(dir_path)
    encoders_dir.mkdir(parents=True, exist_ok=True)
    path = encoders_dir / filename
    with open(path, "w") as f:
        json.dump(obj.to_dict(), f)
    print(f"Saved: {path}")
    return path


def load_json(cls, path: Path):
    """Reads a JSON file written by save_json back into a fitted
    instance of cls, via cls.from_dict()."""
    with open(path) as f:
        return cls.from_dict(json.load(f))


def save_representation_with_id(df: pd.DataFrame, id_series: pd.Series, path: Path) -> None:
    """The single shared implementation of this project's id-as-first-column
    convention for saved representation parquet files -- exists so the
    convention and its integrity checks live in exactly one place, not
    reimplemented per representation per notebook."""
    assert len(id_series) == len(df), (
        f"id_series length {len(id_series)} does not match df row count {len(df)}"
    )
    assert id_series.is_unique, "id_series contains duplicate values"
    assert "id" not in df.columns, "df already has an 'id' column -- refusing to silently overwrite it"

    df = df.reset_index(drop=True)
    id_series = id_series.reset_index(drop=True).rename("id")
    out = pd.concat([id_series, df], axis=1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"Saved: {path} (rows={len(out)}, cols={out.shape[1]}, id first)")


def load_representation_with_id(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """The inverse of save_representation_with_id, kept in the same module
    so the two stay in sync if the id-first-column convention ever changes."""
    path = Path(path)
    df = pd.read_parquet(path)
    assert df.columns[0] == "id", (
        f"expected 'id' as the first column of {path}, got {df.columns[0]!r}"
    )
    id_series = df["id"]
    features_df = df.drop(columns=["id"])
    return features_df, id_series
