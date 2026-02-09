"""
Embedding extraction entrypoint.

This command walks a dataset (via an Adapter) and writes:
- `paths.jsonl`: one JSON object per embedded image (includes on-disk path + any extra metadata)
- `emb.npy`: stacked CLS embeddings (stored as fp16 on disk)
- `meta.json`: provenance + preprocessing settings for reproducibility
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm
from dw.adapters.pattern_glob import PatternGlobAdapter
from dw.features.dinov2 import DINOv2ViTG14Embedder
from dw.features.preprocess import (
    pil_to_tensor_rgb01,
    normalize_imagenet,
    resize_and_crop_to_multiple,
)
from dw.npy import save_fp16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config file")
    ap.add_argument("--dataset", required=True, help="Dataset name for output folder")
    ap.add_argument("--root", required=True, help="Root directory for dataset")
    ap.add_argument("--pattern", required=True, help="Glob pattern for images")
    ap.add_argument("--max_frames_per_scene", type=int, default=None)
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
            
        img = resize_and_crop_to_multiple(sample.image, max_edge=max_edge, patch=14)
        t = pil_to_tensor_rgb01(img)
        batch_imgs.append(t)
        batch_meta.append(sample.meta)
        
        if len(batch_imgs) < batch_size:
            continue
        
        batch = torch.stack(batch_imgs, dim=0).cuda(non_blocking=True)
        
        # Safety check for patch divisibility
        h, w = batch.shape[2], batch.shape[3]
        if h % 14 != 0 or w % 14 != 0:
            h_crop = h - (h % 14)
            w_crop = w - (w % 14)
            batch = batch[:, :, :h_crop, :w_crop]
        
        # Normalize per-image then embed the batch (CLS token).
        batch = torch.stack([normalize_imagenet(batch[i]) for i in range(batch.shape[0])], dim=0)
        cls = embedder.embed_batch(batch, l2_normalize=l2_norm)
        emb_chunks.append(cls.detach().cpu().numpy().astype(np.float16, copy=False))
        
        # Persist metadata in the same order as embeddings are appended.
        for m in batch_meta:
            paths_fp.write(json.dumps(m) + "\n")
        
        total += len(batch_imgs)
        batch_imgs.clear()
        batch_meta.clear()
    
    if batch_imgs:
        # Flush last partial batch.
        batch = torch.stack(batch_imgs, dim=0).cuda(non_blocking=True)
        h, w = batch.shape[2], batch.shape[3]
        if h % 14 != 0 or w % 14 != 0:
            h_crop = h - (h % 14)
            w_crop = w - (w % 14)
            batch = batch[:, :, :h_crop, :w_crop]
        batch = torch.stack([normalize_imagenet(batch[i]) for i in range(batch.shape[0])], dim=0)
        cls = embedder.embed_batch(batch, l2_normalize=l2_norm)
        emb_chunks.append(cls.detach().cpu().numpy().astype(np.float16, copy=False))
        for m in batch_meta:
            paths_fp.write(json.dumps(m) + "\n")
        total += len(batch_imgs)
    
    paths_fp.close()
    
    emb = np.concatenate(emb_chunks, axis=0) if emb_chunks else np.empty((0, 0), dtype=np.float16)
    save_fp16(outdir / "emb.npy", emb)
    
    # Meta is primarily used for sanity checking runs and for reproducibility.
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