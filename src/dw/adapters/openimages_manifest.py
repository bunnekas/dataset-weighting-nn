from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, Tuple

from PIL import Image

from dw.adapters.base import Sample


@dataclass
class OpenImagesManifestAdapter:
    """
    Iterates OpenImages images given a manifest file with lines like:
      train/<image_id>
    and a directory containing downloaded images named:
      <image_id>.jpg
    """

    images_dir: Path
    manifest_path: Path
    missing: str = "skip"  # "skip" | "raise"
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    max_samples: Optional[int] = None

    def __iter__(self) -> Iterator[Sample]:
        n = 0
        lines = self.manifest_path.read_text().splitlines()
        for line_idx, raw in enumerate(lines):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue

            # expected: split/image_id
            parts = s.split("/", 1)
            if len(parts) != 2:
                if self.missing == "raise":
                    raise ValueError(f"Bad manifest line {line_idx+1}: {raw!r}")
                continue

            split, image_id = parts[0].strip(), parts[1].strip()
            if not image_id:
                if self.missing == "raise":
                    raise ValueError(
                        f"Empty image_id in manifest line {line_idx+1}: {raw!r}"
                    )
                continue

            img_path = None
            for ext in self.exts:
                p = self.images_dir / f"{image_id}{ext}"
                if p.is_file():
                    img_path = p
                    break

            if img_path is None:
                if self.missing == "raise":
                    tried = [str(self.images_dir / f"{image_id}{ext}") for ext in self.exts]
                    raise FileNotFoundError(
                        f"Missing OpenImages file for {split}/{image_id}. Tried: {tried}"
                    )
                continue

            # Avoid leaking file handles
            with Image.open(img_path) as im:
                img = im.convert("RGB")

            meta: Dict[str, Any] = {
                "dataset": "openimages",
                "split": split,
                "image_id": image_id,
                "path": str(img_path),
                "manifest_path": str(self.manifest_path),
                "manifest_line": int(line_idx + 1),
            }

            yield Sample(image=img, meta=meta)
            n += 1
            if self.max_samples is not None and n >= int(self.max_samples):
                break
