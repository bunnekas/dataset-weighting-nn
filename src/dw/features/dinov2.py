"""
DINOv2 embedding backend.

We keep a process-wide singleton model to avoid repeated (slow) model loads when
called from CLI utilities. The model runs on a configurable device/dtype and
returns the raw CLS token embedding for each image.
"""

from __future__ import annotations

import torch
from transformers import AutoModel


class DINOv2ViTG14Embedder:
    """Thin wrapper around `facebook/dinov2-giant` with cached model instance."""

    _instance: "DINOv2ViTG14Embedder | None" = None

    def __new__(cls, *, dtype: torch.dtype = torch.float16, device: str = "cuda"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_model(dtype=dtype, device=device)
        return cls._instance

    def _initialize_model(self, *, dtype: torch.dtype, device: str) -> None:
        print(f"Loading DINOv2 ViT-g/14 on {device} with {dtype}")

        # Model expects inputs already resized/cropped and ImageNet-normalized.
        self.model = AutoModel.from_pretrained(
            "facebook/dinov2-giant",
            torch_dtype=dtype,
        ).to(device)

        self.model.eval()

    @torch.inference_mode()
    def embed_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        Embed a batch of images (B,3,H,W) and return the raw CLS token.

        Autocast and L2-normalization (if desired) are handled by the caller.
        """
        outputs = self.model(images)
        # CLS token is at position 0 for ViT-like models.
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token