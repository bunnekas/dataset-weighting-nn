# src/dw/adapters/pattern_glob.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, DefaultDict
from collections import defaultdict
import glob

from PIL import Image

from dw.adapters.base import Sample


@dataclass
class PatternGlobAdapter:
    root: Path
    pattern: str
    max_frames_per_scene: Optional[int] = None
    dataset_name: str = "pattern_glob"

    def __iter__(self) -> Iterator[Sample]:
        paths = glob.glob(str(self.root / self.pattern), recursive=True)
        paths.sort()

        per_scene_count: DefaultDict[Optional[str], int] = defaultdict(int)

        for p in paths:
            try:
                rel = Path(p).relative_to(self.root)
                scene = rel.parts[0] if rel.parts else None
            except Exception:
                scene = None

            if self.max_frames_per_scene is not None and per_scene_count[scene] >= int(self.max_frames_per_scene):
                continue

            try:
                with Image.open(p) as im:
                    img = im.convert("RGB")
            except Exception:
                continue

            yield Sample(
                image=img,
                meta={"dataset": self.dataset_name, "scene": scene, "path": str(p)},
            )

            per_scene_count[scene] += 1
