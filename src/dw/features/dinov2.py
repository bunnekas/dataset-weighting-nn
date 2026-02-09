"""
DINOv2 embedding backend.

We keep a process-wide singleton model to avoid repeated (slow) model loads when
called from CLI utilities. The model runs on CUDA in fp16 and returns the CLS
token embedding for each image.
"""

from __future__ import annotations
import torch
from transformers import AutoModel


class DINOv2ViTG14Embedder:
    """Thin wrapper around `facebook/dinov2-giant` with cached model instance."""
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_model()
        return cls._instance
    
    def _initialize_model(self):
        print("Loading DINOv2 ViT-g/14 on CUDA with float16")
        
        # Model expects inputs already resized/cropped and normalized.
        self.model = AutoModel.from_pretrained(
            "facebook/dinov2-giant",
            dtype=torch.float16,
        ).cuda()
        
        self.model.eval()
        
    @torch.no_grad()
    def embed_batch(self, images: torch.Tensor, l2_normalize: bool = True) -> torch.Tensor:
        """Embed a batch of images (B,3,H,W) and optionally L2-normalize outputs."""
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(images)
            # CLS token is at position 0 for ViT-like models.
            cls_token = outputs.last_hidden_state[:, 0]
        
        if l2_normalize:
            # Normalization enables cosine-similarity retrieval via inner product.
            cls_token = torch.nn.functional.normalize(cls_token, p=2, dim=-1)
        
        return cls_token