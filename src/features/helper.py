"""JSON save/load helpers for any fitted encoder that exposes
to_dict()/from_dict().
"""

from __future__ import annotations

import json
from pathlib import Path


def save_json(obj, filename: str, dir_path: Path) -> Path:
    """Writes obj.to_dict() as JSON to <dir_path>/encoders/<filename>,
    creating the directory if needed. Returns the path saved to."""
    encoders_dir = Path(dir_path) / "encoders"
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
