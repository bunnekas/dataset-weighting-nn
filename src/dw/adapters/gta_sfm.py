from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional
from collections import OrderedDict

from PIL import Image

from dw.adapters.base import Sample

def read_pair_frame_ids(pair_path: Path) -> List[int]:
    """
    Match vggt-s3m semantics: use the 'image_id' lines (1::2 after first line),
    convert 1-based -> 0-based.
    Adds validation and preserves order with uniqueness.
    """
    lines = pair_path.read_text().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"pair.txt too short: {pair_path}")

    # First line is number of images
    try:
        num_images = int(lines[0].strip())
    except Exception as e:
        raise RuntimeError(f"Failed to parse num_images from {pair_path}: {lines[0]!r}") from e

    ids_1based_raw: List[int] = []
    for s in lines[1::2]:
        s = s.strip()
        if not s:
            continue
        try:
            ids_1based_raw.append(int(s))
        except ValueError:
            raise RuntimeError(f"Non-integer image id line in {pair_path}: {s!r}")

    # Preserve order while making unique
    ids_1based = list(OrderedDict.fromkeys(ids_1based_raw))

    # Convert to 0-based and validate range [0, num_images-1]
    ids_0based: List[int] = []
    for i1 in ids_1based:
        i0 = i1 - 1
        if 0 <= i0 < num_images:
            ids_0based.append(i0)
        else:
            continue

    return ids_0based


@dataclass
class GtaSfmAdapter:
    root: Path
    dataset_name: str = "gta_sfm"  # .../gta_sfm_clean/train
    strict: bool = False  # if True, raise on missing files rather than skipping
    max_scenes: Optional[int] = None  # optional cap for debugging

    def __iter__(self) -> Iterator[Sample]:
        seq_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if self.max_scenes is not None:
            seq_dirs = seq_dirs[: int(self.max_scenes)]

        for seq in seq_dirs:
            pair_path = seq / "colmap" / "dense" / "0" / "pair.txt"
            img_dir = seq / "colmap" / "dense" / "0" / "processed" / "images"

            if not pair_path.is_file():
                if self.strict:
                    raise FileNotFoundError(f"Missing pair.txt: {pair_path}")
                continue
            if not img_dir.is_dir():
                if self.strict:
                    raise FileNotFoundError(f"Missing images dir: {img_dir}")
                continue

            try:
                frame_ids = read_pair_frame_ids(pair_path)
            except Exception:
                if self.strict:
                    raise
                continue

            for fid in frame_ids:
                img_path = img_dir / f"{fid:08}.png"
                if not img_path.is_file():
                    if self.strict:
                        raise FileNotFoundError(f"Missing image: {img_path}")
                    continue

                # Prevent leaking file handles
                with Image.open(img_path) as im:
                    img = im.convert("RGB")

                meta: Dict[str, Any] = {
                    "dataset": self.dataset_name,
                    "sequence": seq.name,
                    "frame_id": int(fid),
                    "path": str(img_path),
                    "pair_txt": str(pair_path),
                }

                yield Sample(image=img, meta=meta)
