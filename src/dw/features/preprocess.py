"""
Image preprocessing utilities used before embedding extraction.

These helpers implement the minimum required steps for DINOv2:
- resize to a bounded resolution (cap max edge)
- crop to a patch-size multiple (ViT patch embedding requirement)
- convert to tensor and apply ImageNet normalization
"""

from __future__ import annotations
import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_and_crop_to_multiple(
    img: Image.Image, 
    max_edge: int = 672, 
    patch: int = 14
) -> Image.Image:
    """
    Resize image so max(H,W) = max_edge, then crop smaller edge 
    to next lower multiple of patch size (center crop).
    """
    w, h = img.size
    m = max(w, h)
    
    # Resize if larger than max_edge
    if m > max_edge:
        s = max_edge / float(m)
        nw, nh = int(round(w * s)), int(round(h * s))
        img = img.resize((nw, nh), resample=Image.BICUBIC)
        w, h = nw, nh
    
    # Crop to next lower multiple of patch size
    # DINOv2 ViT-g/14 expects H/W divisible by 14; we crop rather than pad.
    crop_w = w - (w % patch)
    crop_h = h - (h % patch)
    
    # Center crop
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    
    return img.crop((left, top, right, bottom))


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
