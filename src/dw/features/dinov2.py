from __future__ import annotations

import torch
import torch.nn.functional as F


class DINOv2ViTG14Embedder:
    """DINOv2 ViT-g/14 CLS embeddings via torch.hub."""

    def __init__(self, device: str = "cuda", amp_dtype: str = "bf16"):
        self.device = torch.device(device)
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def embed_batch(self, batch: torch.Tensor, l2_normalize: bool = True) -> torch.Tensor:
        """batch: (B,3,H,W) float tensor (already ImageNet normalized). Returns (B,d) float32."""
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            feats = self.model.forward_features(batch)
            cls = feats["x_norm_clstoken"]
        cls = cls.float()
        return F.normalize(cls, p=2, dim=1) if l2_normalize else cls
