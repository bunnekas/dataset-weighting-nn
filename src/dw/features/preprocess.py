from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pil_to_tensor_rgb01(img: Image.Image) -> torch.Tensor:
    """Convert PIL RGB to float tensor (3,H,W) in [0,1]."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def normalize_imagenet(t: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet mean/std to a (3,H,W) tensor."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device)[:, None, None]
    return (t - mean) / std