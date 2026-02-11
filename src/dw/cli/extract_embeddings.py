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
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torchvision import transforms as T
from tqdm import tqdm

from dw.adapters.pattern_glob import PatternGlobAdapter
from dw.features.dinov2 import DINOv2ViTG14Embedder
from dw.features.preprocess import resize_and_crop_to_multiple
from dw.npy import save_fp16


def parse_args_and_config():
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

    preprocess_cfg = cfg.get("preprocess", {})
    features_cfg = cfg.get("features", {})
    artifacts_cfg = cfg.get("artifacts", {})

    max_edge = int(preprocess_cfg.get("max_edge", 672))
    batch_size = int(args.batch_size or features_cfg.get("batch_size", 32))
    amp_dtype_str = str(features_cfg.get("amp_dtype", "fp16")).lower()
    l2_norm = bool(features_cfg.get("l2_normalize", True))
    device = str(features_cfg.get("device", "cuda"))

    return args, cfg, {
        "max_edge": max_edge,
        "batch_size": batch_size,
        "amp_dtype": amp_dtype_str,
        "l2_normalize": l2_norm,
        "device": device,
        "artifacts_root": artifacts_cfg.get("root", "artifacts"),
    }


def run_embedding(args, cfg: dict, runtime_cfg: dict) -> None:
    max_edge = runtime_cfg["max_edge"]
    batch_size = runtime_cfg["batch_size"]
    amp_dtype_str = runtime_cfg["amp_dtype"]
    l2_norm = runtime_cfg["l2_normalize"]
    device = runtime_cfg["device"]
    artifacts_root = Path(runtime_cfg["artifacts_root"])

    outdir = artifacts_root / "embeddings" / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    # Map config string to torch dtype
    if amp_dtype_str == "fp16":
        model_dtype = torch.float16
    elif amp_dtype_str == "bf16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    embedder = DINOv2ViTG14Embedder(dtype=model_dtype, device=device)

    # Torchvision-based preprocessing: PIL → tensor (ImageNet norm on GPU for speed).
    to_tensor = T.ToTensor()

    adapter = PatternGlobAdapter(
        root=Path(args.root),
        pattern=args.pattern,
        max_frames_per_scene=args.max_frames_per_scene,
        dataset_name=args.dataset,
    )

    paths_fp = (outdir / "paths.jsonl").open("w")
    emb_chunks: List[np.ndarray] = []
    batch_imgs: List[torch.Tensor] = []
    batch_meta: List[Dict[str, Any]] = []
    total = 0

    if amp_dtype_str == "fp16":
        autocast_dtype = torch.float16
    elif amp_dtype_str == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None

    # Precompute ImageNet normalization tensors on the target device.
    imagenet_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

    for sample in tqdm(adapter, desc=f"Embedding {args.dataset}", unit="img"):
        if args.max_samples and total >= args.max_samples:
            break

        img = resize_and_crop_to_multiple(sample.image, max_edge=max_edge, patch=14)
        t = to_tensor(img)  # (3,H,W) on CPU
        batch_imgs.append(t)
        batch_meta.append(sample.meta)

        if len(batch_imgs) < batch_size:
            continue

        batch = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
        # Apply ImageNet normalization on GPU.
        batch = (batch - imagenet_mean) / imagenet_std

        if autocast_dtype is not None:
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                cls = embedder.embed_batch(batch)
        else:
            cls = embedder.embed_batch(batch)

        if l2_norm:
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)

        emb_chunks.append(cls.detach().cpu().numpy().astype(np.float16, copy=False))

        # Persist metadata in the same order as embeddings are appended.
        for m in batch_meta:
            paths_fp.write(json.dumps(m) + "\n")

        total += batch.shape[0]
        batch_imgs.clear()
        batch_meta.clear()

    # Flush last partial batch, if any.
    if batch_imgs:
        batch = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
        batch = (batch - imagenet_mean) / imagenet_std

        if autocast_dtype is not None:
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                cls = embedder.embed_batch(batch)
        else:
            cls = embedder.embed_batch(batch)

        if l2_norm:
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)

        emb_chunks.append(cls.detach().cpu().numpy().astype(np.float16, copy=False))
        for m in batch_meta:
            paths_fp.write(json.dumps(m) + "\n")
        total += batch.shape[0]

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


def main() -> None:
    args, cfg, runtime_cfg = parse_args_and_config()
    run_embedding(args, cfg, runtime_cfg)


if __name__ == "__main__":
    main()