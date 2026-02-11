"""
Image preprocessing utilities used before embedding extraction.

These helpers implement the geometric part of preprocessing for DINOv2:
- resize to a bounded resolution (cap max edge)
- crop to a patch-size multiple (ViT patch embedding requirement)

Standard tensor conversion + normalization are handled via torchvision
transforms in the embedding CLI.
"""

from __future__ import annotations

from PIL import Image


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
    crop_w = w - (w % patch)
    crop_h = h - (h % patch)
    
    # Center crop
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h
    
    return img.crop((left, top, right, bottom))
