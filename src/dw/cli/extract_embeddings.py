from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from PIL import Image

from dw.adapters.pattern_glob import PatternGlobAdapter
from dw.features.dinov2 import DINOv2ViTG14Embedder
from dw.features.preprocess import (
    pil_to_tensor_rgb01,
    normalize_imagenet,
)
from dw.npy import save_fp16


def _resize_and_pad_to_multiple(
    img: Image.Image, 
    max_edge: int = 672, 
    patch: int = 14
) -> Image.Image:
    """
    Resize image so max(H,W) = max_edge, then pad to make dimensions divisible by patch.
    """
    from PIL import Image
    
    w, h = img.size
    m = max(w, h)
    if m > max_edge:
        s = max_edge / float(m)
        nw, nh = int(round(w * s)), int(round(h * s))
        img = img.resize((nw, nh), resample=Image.BICUBIC)
        w, h = nw, nh
    
    pad_w = (patch - (w % patch)) % patch
    pad_h = (patch - (h % patch)) % patch
    
    if pad_w == 0 and pad_h == 0:
        return img
    
    new_w = w + pad_w
    new_h = h + pad_h
    new_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    new_img.paste(img, (0, 0))
    return new_img


def _pad_stack(tensors: list[torch.Tensor], patch: int = 14) -> torch.Tensor:
    """Stack list of (3,H,W) into (B,3,Hpad,Wpad), padding to max H/W + patch multiple."""
    if not tensors:
        return torch.empty((0, 3, patch, patch), dtype=torch.float32)

    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    
    max_h = max_h + (patch - (max_h % patch)) % patch
    max_w = max_w + (patch - (max_w % patch)) % patch

    padded = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1])) for t in tensors]
    return torch.stack(padded, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config file")
    ap.add_argument("--dataset", required=True, help="Dataset name for output folder")
    ap.add_argument("--root", required=True, help="Root directory for dataset")
    ap.add_argument("--pattern", required=True, help="Glob pattern for images")
    ap.add_argument("--max-frames-per-scene", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    
    max_edge = int(cfg.get("preprocess", {}).get("max_edge", 672))
    batch_size = int(args.batch_size or cfg.get("batch_size", 32))
    l2_norm = bool(cfg.get("l2_normalize", True))
    
    out_root = Path(cfg.get("artifacts", {}).get("root", "artifacts"))
    outdir = out_root / "embeddings" / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    embedder = DINOv2ViTG14Embedder()
    
    paths_fp = (outdir / "paths.jsonl").open("w")
    emb_chunks: list[np.ndarray] = []
    batch_imgs: list[torch.Tensor] = []
    batch_meta: list[dict] = []
    
    adapter = PatternGlobAdapter(
        root=Path(args.root),
        pattern=args.pattern,
        max_frames_per_scene=args.max_frames_per_scene,
        dataset_name=args.dataset,
    )
    
    total = 0
    for sample in tqdm(adapter, desc=f"Embedding {args.dataset}", unit="img"):
        if args.max_samples and total >= args.max_samples:
            break
            
        from PIL import Image
        img = _resize_and_pad_to_multiple(sample.image, max_edge=max_edge, patch=14)
        t = pil_to_tensor_rgb01(img)
        batch_imgs.append(t)
        batch_meta.append(sample.meta)
        
        if len(batch_imgs) < batch_size:
            continue
        
        batch = _pad_stack(batch_imgs, patch=14).cuda(non_blocking=True)
        batch = torch.stack([normalize_imagenet(batch[i]) for i in range(batch.shape[0])], dim=0)
        
        cls = embedder.embed_batch(batch, l2_normalize=l2_norm)
        emb_chunks.append(cls.detach().cpu().numpy().astype(np.float16, copy=False))
        
        for m in batch_meta:
            paths_fp.write(json.dumps(m) + "\n")
        
        total += len(batch_imgs)
        batch_imgs.clear()
        batch_meta.clear()
    
    if batch_imgs:
        batch = _pad_stack(batch_imgs, patch=14).cuda(non_blocking=True)
        batch = torch.stack([normalize_imagenet(batch[i]) for i in range(batch.shape[0])], dim=0)
        cls = embedder.embed_batch(batch, l2_normalize=l2_norm)
        emb_chunks.append(cls.detach().cpu().numpy().astype(np.float16, copy=False))
        for m in batch_meta:
            paths_fp.write(json.dumps(m) + "\n")
        total += len(batch_imgs)
    
    paths_fp.close()
    
    emb = np.concatenate(emb_chunks, axis=0) if emb_chunks else np.empty((0, 0), dtype=np.float16)
    save_fp16(outdir / "emb.npy", emb)
    
    meta = {
        "dataset": args.dataset,
        "root": str(Path(args.root).resolve()),
        "pattern": args.pattern,
        "max_frames_per_scene": args.max_frames_per_scene,
        "count": int(emb.shape[0]),
        "dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
        "max_edge": max_edge,
        "patch_multiple": 14,
        "imagenet_norm": True,
        "l2_normalize": l2_norm,
        "dtype_storage": "fp16",
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"outdir": str(outdir), "count": int(emb.shape[0])}, indent=2))


if __name__ == "__main__":
    main()