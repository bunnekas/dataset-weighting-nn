#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random


def detect_id_column(fieldnames: list[str]) -> str:
    # try common names
    candidates = ["image_id", "ImageID", "imageid", "id", "ImageId"]
    lowered = {c.lower(): c for c in fieldnames}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    raise ValueError(f"Could not detect image id column. Columns: {fieldnames}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--split", default="train", choices=["train", "validation", "test", "challenge2018"])
    ap.add_argument("--fraction", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dedup", action="store_true")
    args = ap.parse_args()

    with args.csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header row")
        id_col = detect_id_column(reader.fieldnames)
        ids = []
        for row in reader:
            v = (row.get(id_col) or "").strip()
            if v:
                ids.append(v)

    if args.dedup:
        ids = sorted(set(ids))
    else:
        # stable order before sampling to keep reproducible across machines
        ids = sorted(ids)

    n_total = len(ids)
    if n_total == 0:
        raise RuntimeError("No image ids found")

    n_sample = max(1, int(round(n_total * args.fraction)))
    rng = random.Random(args.seed)
    sample = rng.sample(ids, n_sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for img_id in sample:
            f.write(f"{args.split}/{img_id}\n")

    print(f"Wrote manifest: {args.out}")
    print(f"split={args.split} total_ids={n_total} fraction={args.fraction} sample={n_sample} seed={args.seed}")


if __name__ == "__main__":
    main()
