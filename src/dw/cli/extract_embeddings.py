from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from dw.adapters.base import Sample
from dw.adapters.hypersim import HypersimAdapter
from dw.adapters.gta_sfm import GtaSfmAdapter
from dw.adapters.image_glob import ImageGlobAdapter
from dw.adapters.openimages_manifest import OpenImagesManifestAdapter
from dw.features.dinov2 import DINOv2ViTG14Embedder
from dw.features.preprocess import (
    resize_max_edge,
    pil_to_tensor_rgb01,
    normalize_imagenet,
    pad_batch_to_patch_multiple,
)
from dw.npy import save_bf16


def _iter_samples(args: argparse.Namespace) -> Iterator[Sample]:
    root = Path(args.root)

    if args.adapter == "hypersim":
        yield from HypersimAdapter(root=root)
        return

    if args.adapter == "gta_sfm":
        yield from GtaSfmAdapter(root=root)
        return

    if args.adapter == "image_glob":
        if not args.pattern:
            raise ValueError("--pattern is required for adapter=image_glob")
        yield from ImageGlobAdapter(
            root=root,
            pattern=args.pattern,
            max_frames_per_scene=args.max_frames_per_scene,
        )
        return

    if args.adapter == "openimages_manifest":
        if not args.manifest:
            raise ValueError("--manifest is required for adapter=openimages_manifest")
        images_dir = Path(args.images_dir) if args.images_dir else root
        yield from OpenImagesManifestAdapter(
            images_dir=images_dir,
            manifest_path=Path(args.manifest),
            missing=args.missing,
            max_samples=args.max_samples,
        )
        return

    raise ValueError(f"Unknown adapter: {args.adapter}")


def _pad_stack(tensors: list[torch.Tensor], patch: int = 14) -> torch.Tensor:
    """Stack list of (3,H,W) into (B,3,Hpad,Wpad), padding to max H/W + patch multiple."""
    if not tensors:
        return torch.empty((0, 3, patch, patch), dtype=torch.float32)

    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)

    padded = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1])) for t in tensors]
    batch = torch.stack(padded, dim=0)
    return pad_batch_to_patch_multiple(batch, patch=patch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML with artifacts.root + preprocess/features defaults")
    ap.add_argument("--dataset", required=True, help="name for output folder (e.g. hypersim)")
    ap.add_argument(
        "--adapter",
        required=True,
        choices=["hypersim", "gta_sfm", "image_glob", "openimages_manifest"],
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--pattern", default=None, help="glob pattern for adapter=image_glob")
    ap.add_argument("--max-frames-per-scene", type=int, default=None)

    ap.add_argument("--manifest", default=None, help="manifest txt for adapter=openimages_manifest")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--missing", default="skip", choices=["skip", "raise"])

    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-edge", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp-dtype", default=None, choices=["bf16", "fp16"])
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    max_edge = int(args.max_edge or cfg.get("preprocess", {}).get("max_edge", 512))
    feat_cfg = cfg.get("features", {})
    device = str(args.device or feat_cfg.get("device", "cuda"))
    batch_size = int(args.batch_size or feat_cfg.get("batch_size", 32))
    amp_dtype = str(args.amp_dtype or feat_cfg.get("amp_dtype", "bf16"))
    l2_norm = bool(feat_cfg.get("l2_normalize", True))

    out_root = Path(cfg.get("artifacts", {}).get("root", "artifacts"))
    outdir = Path(args.outdir) if args.outdir else (out_root / "embeddings" / args.dataset)
    outdir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device)
    embedder = DINOv2ViTG14Embedder(device=device, amp_dtype=amp_dtype)

    paths_fp = (outdir / "paths.jsonl").open("w")
    emb_chunks: list[np.ndarray] = []

    batch_imgs: list[torch.Tensor] = []
    batch_meta: list[dict] = []

    total = 0
    for sample in tqdm(_iter_samples(args), desc=f"Embedding {args.dataset}", unit="img"):
        img = resize_max_edge(sample.image, max_edge=max_edge)
        t = pil_to_tensor_rgb01(img)
        batch_imgs.append(t)
        batch_meta.append(sample.meta)

        if len(batch_imgs) < batch_size:
            continue

        batch = _pad_stack(batch_imgs, patch=14).to(dev, non_blocking=True)
        # normalize per-image
        batch = torch.stack([normalize_imagenet(batch[i]) for i in range(batch.shape[0])], dim=0)

        cls = embedder.embed_batch(batch, l2_normalize=l2_norm)
        emb_chunks.append(cls.detach().cpu().numpy().astype(np.float32, copy=False))

        for m in batch_meta:
            paths_fp.write(json.dumps(m) + "\n")

        total += len(batch_imgs)
        batch_imgs.clear()
        batch_meta.clear()

    if batch_imgs:
        batch = _pad_stack(batch_imgs, patch=14).to(dev, non_blocking=True)
        batch = torch.stack([normalize_imagenet(batch[i]) for i in range(batch.shape[0])], dim=0)
        cls = embedder.embed_batch(batch, l2_normalize=l2_norm)
        emb_chunks.append(cls.detach().cpu().numpy().astype(np.float32, copy=False))
        for m in batch_meta:
            paths_fp.write(json.dumps(m) + "\n")
        total += len(batch_imgs)

    paths_fp.close()

    emb = np.concatenate(emb_chunks, axis=0) if emb_chunks else np.empty((0, 0), dtype=np.float32)
    save_bf16(outdir / "emb_bf16.npy", emb)

    meta = {
        "dataset": args.dataset,
        "adapter": args.adapter,
        "root": str(Path(args.root).resolve()),
        "count": int(emb.shape[0]),
        "dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
        "max_edge": max_edge,
        "patch_multiple": 14,
        "imagenet_norm": True,
        "l2_normalize": l2_norm,
        "dtype_storage": "bf16",
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"outdir": str(outdir), "count": int(emb.shape[0])}, indent=2))


if __name__ == "__main__":
    main()
