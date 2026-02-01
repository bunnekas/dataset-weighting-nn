#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, List

import boto3
import botocore
import tqdm


BUCKET_NAME = "open-images-dataset"


@dataclass(frozen=True)
class Candidate:
    split: str
    image_id: str  # without extension


def _parse_candidate_line(line: str, default_split: str) -> Optional[Candidate]:
    s = line.strip()
    if not s or s.startswith("#"):
        return None

    # Accept formats:
    # - train/<id>
    # - train/<id>.jpg
    # - <id>           (uses default_split)
    # - <split> <id>   (whitespace separated)
    if " " in s or "\t" in s:
        parts = s.split()
        if len(parts) >= 2:
            split = parts[0].strip().strip("/")
            img = parts[1].strip()
        else:
            return None
    elif "/" in s:
        split, img = s.split("/", 1)
        split = split.strip().strip("/")
    else:
        split, img = default_split, s

    img = img.strip()
    if img.lower().endswith(".jpg") or img.lower().endswith(".jpeg") or img.lower().endswith(".png"):
        img = os.path.splitext(img)[0]

    if not split or not img:
        return None

    return Candidate(split=split, image_id=img)


def read_candidates(path: Path, default_split: str) -> list[Candidate]:
    out: list[Candidate] = []
    with path.open("r") as f:
        for line in f:
            c = _parse_candidate_line(line, default_split=default_split)
            if c is not None:
                out.append(c)
    return out


def _s3_bucket():
    # unsigned public bucket
    s3 = boto3.resource(
        "s3",
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    )
    return s3.Bucket(BUCKET_NAME)


def _try_download(bucket, cand: Candidate, out_dir: Path, count_existing: bool) -> Tuple[bool, str, Candidate]:
    """
    Returns (success, reason, candidate). reason is one of:
    - "ok"
    - "exists"
    - "missing" (404)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # CHANGED: Save as just image_id.jpg, not split_image_id.jpg
    out_path = out_dir / f"{cand.image_id}.jpg"

    if out_path.exists():
        return (True, "exists" if count_existing else "ok", cand)

    # Try both key styles to avoid subtle manifest/script mismatch:
    #   train/<id>.jpg  (common)
    #   train/<id>      (some manifests mistakenly omit .jpg, so try anyway)
    keys = [
        f"{cand.split}/{cand.image_id}.jpg",
        f"{cand.split}/{cand.image_id}",
    ]

    last_404 = False
    for key in keys:
        try:
            # download_file will overwrite partials; write to tmp then atomic rename
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

            bucket.download_file(key, str(tmp_path))
            tmp_path.replace(out_path)
            return (True, "ok", cand)
        except botocore.exceptions.ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                last_404 = True
                continue
            raise  # propagate non-404
        except Exception:
            raise

    if last_404:
        return (False, "missing", cand)
    return (False, "missing", cand)


def download_target(
    candidates: list[Candidate],
    download_folder: Path,
    target: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    count_existing: bool,
    success_manifest: Optional[Path],
    failed_manifest: Optional[Path],
) -> None:
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(candidates)

    bucket = _s3_bucket()

    successes = 0
    attempted = 0
    missing = 0
    
    # NEW: Track successful candidates for manifest
    successful_candidates: List[Candidate] = []

    # Producer index
    idx = 0
    n = len(candidates)

    pbar = tqdm.tqdm(total=target, desc="Downloaded", unit="img", leave=True)

    # Helper to submit next work item
    def submit_one(executor):
        nonlocal idx
        if idx >= n:
            return None
        cand = candidates[idx]
        idx += 1
        return executor.submit(_try_download, bucket, cand, download_folder, count_existing)

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = set()

        # prime the queue
        while len(futures) < num_workers and idx < n and successes < target:
            fut = submit_one(ex)
            if fut is not None:
                futures.add(fut)

        while futures and successes < target:
            for fut in as_completed(list(futures)):
                futures.remove(fut)

                ok, reason, cand = fut.result()
                attempted += 1

                if ok:
                    if count_existing or reason == "ok":
                        successes += 1
                        successful_candidates.append(cand)  # NEW: Track success
                        pbar.update(1)
                else:
                    missing += 1

                # submit more if needed
                while len(futures) < num_workers and idx < n and successes < target:
                    fut2 = submit_one(ex)
                    if fut2 is not None:
                        futures.add(fut2)

                if successes >= target:
                    break

    pbar.close()

    if successes < target:
        print(
            f"[WARN] Not enough downloadable images to reach target.\n"
            f"  target={target}\n"
            f"  successes={successes}\n"
            f"  attempted={attempted}\n"
            f"  missing(404)={missing}\n"
            f"  candidates_total={len(candidates)}\n"
            f"Provide a larger candidate manifest (oversample aggressively).",
            file=sys.stderr,
        )
        sys.exit(2)

    # NEW: Write manifest in correct format
    if success_manifest is not None:
        success_manifest.parent.mkdir(parents=True, exist_ok=True)
        with success_manifest.open("w") as f:
            for cand in successful_candidates[:target]:
                # Write in format: split/image_id
                f.write(f"{cand.split}/{cand.image_id}\n")
        print(f"[OK] wrote success manifest with {len(successful_candidates[:target])} entries: {success_manifest}")

    print(
        f"[OK] target reached.\n"
        f"  target={target}\n"
        f"  successes={successes}\n"
        f"  attempted={attempted}\n"
        f"  missing(404)={missing}\n"
        f"  out={download_folder}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_manifest", type=Path, help="Text file with candidate IDs (see formats in script).")
    ap.add_argument("--download_folder", type=Path, required=True)
    ap.add_argument("--target", type=int, default=100_000)
    ap.add_argument("--split", type=str, default="train", help="Used if manifest lines contain only <id>.")
    ap.add_argument("--num_workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_shuffle", action="store_true")
    ap.add_argument(
        "--count_existing",
        action="store_true",
        help="If files already exist in download_folder, count them toward target.",
    )
    ap.add_argument("--success_manifest_out", type=Path, default=None)

    args = ap.parse_args()

    candidates = read_candidates(args.candidate_manifest, default_split=args.split)
    if not candidates:
        print(f"[ERR] No candidates parsed from {args.candidate_manifest}", file=sys.stderr)
        sys.exit(1)

    download_target(
        candidates=candidates,
        download_folder=args.download_folder,
        target=args.target,
        num_workers=args.num_workers,
        seed=args.seed,
        shuffle=(not args.no_shuffle),
        count_existing=args.count_existing,
        success_manifest=args.success_manifest_out,
        failed_manifest=None,
    )


if __name__ == "__main__":
    main()