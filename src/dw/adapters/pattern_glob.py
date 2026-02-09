from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, DefaultDict
from collections import defaultdict
import glob
from PIL import Image
from dw.adapters.base import Sample


@dataclass
class PatternGlobAdapter:
    """
    Simple adapter that discovers images via a glob pattern.

    The optional `max_frames_per_scene` provides a cheap way to downsample large
    datasets: we bucket by parent directory ("scene") and cap the
    number of yielded frames per bucket.
    """
    root: Path
    pattern: str
    max_frames_per_scene: Optional[int] = None
    dataset_name: str = "pattern_glob"

    def __iter__(self) -> Iterator[Sample]:
        # Use Python's glob for recursive patterns like "**/*.jpg".
        paths = glob.glob(str(self.root / self.pattern), recursive=True)
        paths.sort()

        per_scene_count: DefaultDict[str, int] = defaultdict(int)

        for p in paths:
            try:
                rel = Path(p).relative_to(self.root)
                # Treat the parent directory as a "scene" key when possible.
                scene = str(rel.parent) if str(rel.parent) != "." else "flat"
            except Exception:
                scene = "flat"

            # Check limit
            if (self.max_frames_per_scene is not None and 
                per_scene_count[scene] >= self.max_frames_per_scene):
                continue

            try:
                # Skip unreadable/corrupt files rather than failing the whole run.
                with Image.open(p) as im:
                    img = im.convert("RGB")
            except Exception:
                continue

            yield Sample(
                image=img,
                meta={"dataset": self.dataset_name, "scene": scene, "path": str(p)},
            )

            per_scene_count[scene] += 1