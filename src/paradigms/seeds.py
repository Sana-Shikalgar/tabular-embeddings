"""Seeds Python's random, numpy, and torch (CPU + all CUDA devices) from
one call, so a notebook run starts every paradigm's training from the
same global state."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Seeds random, np.random, torch, and torch.cuda (all devices) with
    seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
