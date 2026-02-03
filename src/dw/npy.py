from __future__ import annotations
from pathlib import Path
import numpy as np


def save_fp16(path: Path, array: np.ndarray) -> None:
    """Save array as float16."""
    if array.dtype != np.float16:
        array = array.astype(np.float16)
    np.save(path, array)


def load_for_faiss(path: Path) -> np.ndarray:
    """
    Load embeddings and cast to float32 for FAISS IndexFlatIP.
    """
    x = np.load(path)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x
