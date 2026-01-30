from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_max_edge(img: Image.Image, max_edge: int) -> Image.Image:
    """Resize while preserving aspect ratio so that max(H, W) == max_edge (if larger)."""
    w, h = img.size
    m = max(w, h)
    if m <= max_edge:
        return img
    s = max_edge / float(m)
    nw, nh = int(round(w * s)), int(round(h * s))
    return img.resize((nw, nh), resample=Image.BICUBIC)


def pil_to_tensor_rgb01(img: Image.Image) -> torch.Tensor:
    """Convert PIL RGB to float tensor (3,H,W) in [0,1]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def normalize_imagenet(t: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std to a (3,H,W) tensor."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device)[:, None, None]
    return (t - mean) / std


def pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad spatial dims (H,W) with zeros so they are divisible by `multiple`."""
    _, h, w = x.shape
    ph = (multiple - (h % multiple)) % multiple
    pw = (multiple - (w % multiple)) % multiple
    if ph == 0 and pw == 0:
        return x
    return F.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)


def pad_batch_to_patch_multiple(batch: torch.Tensor, patch: int = 14) -> torch.Tensor:
    """Pad a batch (B,3,H,W) so H and W are divisible by `patch`."""
    b, c, h, w = batch.shape
    ph = (patch - (h % patch)) % patch
    pw = (patch - (w % patch)) % patch
    if ph == 0 and pw == 0:
        return batch
    return F.pad(batch, (0, pw, 0, ph), mode="constant", value=0.0)
