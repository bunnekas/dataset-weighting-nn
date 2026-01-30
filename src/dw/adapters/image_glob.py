from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, DefaultDict
from collections import defaultdict
import glob

from PIL import Image
from dw.adapters.base import Sample


@dataclass
class ImageGlobAdapter:
    root: Path
    pattern: str
    max_frames_per_scene: Optional[int] = None
    dataset_name: str = "image_glob"

    def __iter__(self) -> Iterator[Sample]:
        paths = glob.glob(str(self.root / self.pattern), recursive=True)
        paths.sort()
        per_scene_count: DefaultDict[Optional[str], int] = defaultdict(int)
        for p in paths:
            # scene name for scannetpp is the immediate directory under root
            try:
                rel = Path(p).relative_to(self.root)
                scene = rel.parts[0]
            except Exception:
                scene = None

            if self.max_frames_per_scene is not None:
                if per_scene_count[scene] >= int(self.max_frames_per_scene):
                    continue

            # Avoid leaking file handles; skip unreadable/corrupt images
            try:
                with Image.open(p) as im:
                    img = im.convert("RGB")
            except Exception:
                continue

            meta: Dict[str, Any] = {
                "dataset": self.dataset_name,
                "scene": scene,
                "path": str(p),
            }
            yield Sample(image=img, meta=meta)
            per_scene_count[scene] += 1
