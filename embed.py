"""Embedding (all-MiniLM-L6-v2, ou modele fine-tune si artifacts/finetuned_minilm existe)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME

_model = None
_FINETUNED = Path(__file__).resolve().parent / "artifacts" / "finetuned_minilm"


def get_model():
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        name = str(_FINETUNED) if _FINETUNED.exists() else EMBEDDING_MODEL_NAME
        _model = SentenceTransformer(name, device=device)
    return _model


def embed_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    """Renvoie des embeddings L2-normalises, shape (n, dim)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_queries(queries: List[str], *, batch_size: int = 64) -> np.ndarray:
    return embed_texts(queries, batch_size=batch_size)

