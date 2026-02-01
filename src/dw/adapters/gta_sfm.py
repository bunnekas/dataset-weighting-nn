from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional
import os

from PIL import Image

from dw.adapters.base import Sample


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
            img_dir = seq / "colmap" / "dense" / "0" / "processed" / "images"

            if not img_dir.is_dir():
                if self.strict:
                    raise FileNotFoundError(f"Missing images dir: {img_dir}")
                continue

            # Get all PNG images in the directory
            img_paths = sorted(img_dir.glob("*.png"))
            
            for img_path in img_paths:
                if not img_path.is_file():
                    if self.strict:
                        raise FileNotFoundError(f"Missing image: {img_path}")
                    continue

                # Extract frame_id from filename (remove .png extension and parse as int)
                try:
                    fid = int(img_path.stem)
                except ValueError:
                    if self.strict:
                        raise ValueError(f"Invalid image filename format: {img_path.name}")
                    continue

                # Prevent leaking file handles
                with Image.open(img_path) as im:
                    img = im.convert("RGB")

                meta: Dict[str, Any] = {
                    "dataset": self.dataset_name,
                    "sequence": seq.name,
                    "frame_id": int(fid),
                    "path": str(img_path),
                }

                yield Sample(image=img, meta=meta)