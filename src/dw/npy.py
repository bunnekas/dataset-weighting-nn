from __future__ import annotations
from pathlib import Path
import numpy as np


def save_bf16(path: Path, a: np.ndarray) -> None:
    """
    Save embeddings in a compact dtype. Prefer bfloat16 if NumPy supports it,
    otherwise fall back to float16 for portability.
    """
    path = Path(path)
    a = np.asarray(a)

    try:
        bf16 = np.dtype("bfloat16")
        out = a.astype(bf16)
    except TypeError:
        # NumPy without bf16 support (common on clusters)
        out = a.astype(np.float16)

    np.save(path, out)


def load_for_faiss(path: Path) -> np.ndarray:
    """
    Load embeddings and cast to float32 for FAISS IndexFlatIP.
    """
    x = np.load(path)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x
