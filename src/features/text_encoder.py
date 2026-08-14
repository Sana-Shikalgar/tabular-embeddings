"""Frozen SBERT sentence encoding, Eq. 3.7: a one-time, offline forward
pass through a frozen sentence-transformers model, mean-pooled to a
fixed-size vector per text. No fine-tuning anywhere -- the encoder's
weights never update, so the same TextEncoder instance is applied
identically to train, val, and test (no fit/transform split, unlike the
other src/features/ classes, since there's no train-only state to fit).
Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings using
Siamese BERT-Networks."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from src.config import SBERT_MODEL_NAME


class TextEncoder:
    def __init__(self, model_name: str = SBERT_MODEL_NAME, batch_size: int = 64):
        # Deferred import, same reasoning and DLL workaround as
        # src/eda/helper.py's _get_tokenizer(): importing sentence_transformers
        # pulls in torch, which some Windows Jupyter kernel processes fail to
        # load unless torch's own lib directory is registered first.
        if hasattr(os, "add_dll_directory"):
            torch_lib_dir = Path(sys.executable).parent / "Lib" / "site-packages" / "torch" / "lib"
            if torch_lib_dir.exists():
                os.add_dll_directory(str(torch_lib_dir))

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Frozen, no fine-tuning anywhere: verify rather than assume.
        assert not self.model.training, "model must be in eval mode (frozen, no fine-tuning)"
        assert all(not p.requires_grad for p in self.model.parameters()), (
            "model parameters must not require grad (frozen, no fine-tuning)"
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(embeddings, dtype=np.float32)
