"""Shared I/O helpers for src/features/ artifacts -- fitted objects that
each expose a .to_dict() (Standardizer, PiecewiseLinearEncoder,
CategoricalLookup, ...).
"""

from __future__ import annotations

import json
from pathlib import Path


def save_artifact(obj, filename: str, artifacts_dir: Path) -> Path:
    """Serializes obj.to_dict() to <artifacts_dir>/<filename> as JSON,
    creating the directory if needed. Returns the path saved to."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / filename
    with open(path, "w") as f:
        json.dump(obj.to_dict(), f)
    print(f"Saved: {path}")
    return path


def load_artifact(cls, path: Path):
    """Reloads a fitted object of type `cls` (any class exposing a
    from_dict classmethod) from a JSON file written by save_artifact."""
    with open(path) as f:
        return cls.from_dict(json.load(f))
