from __future__ import annotations

import torch
from transformers import AutoModel


class DINOv2ViTG14Embedder:
    def __init__(self):
        print("Loading DINOv2 ViT-g/14 on CUDA with float16")
        
        self.model = AutoModel.from_pretrained(
            "facebook/dinov2-giant",
            dtype=torch.float16,
        ).cuda()
        
        self.model.eval()
        
    @torch.no_grad()
    def embed_batch(
        self, 
        images: torch.Tensor, 
        l2_normalize: bool = True
    ) -> torch.Tensor:
        """
        images: (B, 3, H, W) already normalized with ImageNet stats
        returns: (B, 1536) CLS token embeddings
        """
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = self.model(images)
            cls_token = outputs.last_hidden_state[:, 0]
        
        if l2_normalize:
            cls_token = torch.nn.functional.normalize(cls_token, p=2, dim=-1)
        
        return cls_token