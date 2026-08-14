"""Shared I/O helpers for fitted objects that expose a .to_dict()
(Standardizer, PiecewiseLinearEncoder, CategoricalLookup, ...). Callers
pass the candidate's artifact directory; save_json/load_json own the
encoders/ subfolder themselves, the same way src/eda/report.py's
savefig owns the figures/ subfolder.
"""

from __future__ import annotations

import json
from pathlib import Path


def save_json(obj, filename: str, dir_path: Path) -> Path:
    """Serializes obj.to_dict() to <dir_path>/encoders/<filename> as
    JSON, creating the directory if needed. Returns the path saved to."""
    encoders_dir = Path(dir_path) / "encoders"
    encoders_dir.mkdir(parents=True, exist_ok=True)
    path = encoders_dir / filename
    with open(path, "w") as f:
        json.dump(obj.to_dict(), f)
    print(f"Saved: {path}")
    return path


def load_json(cls, path: Path):
    """Reloads a fitted object of type `cls` (any class exposing a
    from_dict classmethod) from a JSON file written by save_json."""
    with open(path) as f:
        return cls.from_dict(json.load(f))
