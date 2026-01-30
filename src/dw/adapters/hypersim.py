from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Set, Dict, Any, Optional, List
import numpy as np

from PIL import Image

from dw.adapters.base import Sample

# wai-clone provides this
from waifu.loader.base import GenericMultiViewDataset  # type: ignore


def _to_pil_rgb(img: object) -> Image.Image:
    """Convert waifu-loaded image (PIL / torch / numpy) to PIL RGB."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if hasattr(img, "detach"):
        # torch tensor [C,H,W] float in [0,1]
        t = img.detach()
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t[:3].permute(1, 2, 0).contiguous()
            arr = (t.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        raise ValueError(f"Unexpected torch image shape: {tuple(t.shape)}")

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Unsupported numpy image shape: {arr.shape}")

        # CHW -> HWC if needed
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.shape[2] == 4:
            arr = arr[:, :, :3]

        if arr.dtype != np.uint8:
            a = arr.astype(np.float32, copy=False)
            vmax = float(np.nanmax(a)) if a.size else 0.0
            if vmax <= 1.5:
                a = a * 255.0
            a = np.clip(a, 0.0, 255.0)
            arr = (a + 0.5).astype(np.uint8)

        return Image.fromarray(arr, mode="RGB")

    raise TypeError(f"Unsupported image type from waifu loader: {type(img)}")


def _list_color_images_for_scene(scene_root: Path) -> List[Path]:
    """Deterministically list all Hypersim RGB frames in a scene across all cams."""
    exts = (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP")
    paths: List[Path] = []
    for cam_dir in sorted(scene_root.glob("cam_*")):
        color_dir = cam_dir / "color"
        if not color_dir.is_dir():
            continue
        for ext in exts:
            paths.extend(color_dir.glob(f"*{ext}"))
    # deterministic
    return sorted(p.resolve() for p in paths if p.is_file())


@dataclass
class HypersimAdapter:
    root: Path
    dataset_name: str = "hypersim"
    modalities: Set[str] = None
    strict: bool = False  # if True, raise when path cannot be resolved

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = {"image"}
        self.root = Path(self.root)
        self.mvd = GenericMultiViewDataset(self.root, None, self.modalities)

    def __iter__(self) -> Iterator[Sample]:
        # Cache scene -> file list to avoid repeated globbing
        scene_files_cache: Dict[str, List[Path]] = {}

        for scene_id in range(len(self.mvd.scenes)):
            scene_meta = self.mvd.get_scene_meta(scene_id)
            scene_name = getattr(scene_meta, "scene_name", None) or (
                scene_meta.get("scene_name", None) if isinstance(scene_meta, dict) else None
            )
            if scene_name is None:
                scene_name = str(self.mvd.scenes[scene_id].name)

            frames = scene_meta.frames if hasattr(scene_meta, "frames") else (
                scene_meta.get("frames") if isinstance(scene_meta, dict) else None
            )
            if frames is None:
                continue

            if scene_name not in scene_files_cache:
                scene_root = self.root / scene_name
                scene_files_cache[scene_name] = _list_color_images_for_scene(scene_root)

            scene_files = scene_files_cache[scene_name]

            for frame_id in range(len(frames)):
                fr = self.mvd.load_frame(scene_id, frame_id)
                img = fr.get("image")
                if img is None:
                    continue
                img_pil = _to_pil_rgb(img)

                img_path: Optional[Path] = None
                if 0 <= frame_id < len(scene_files):
                    img_path = scene_files[frame_id]

                if img_path is None:
                    if self.strict:
                        raise RuntimeError(
                            f"HypersimAdapter: could not resolve path for scene={scene_name} frame_id={frame_id}. "
                            f"Found {len(scene_files)} color images under {self.root/scene_name}."
                        )
                    continue

                meta: Dict[str, Any] = {
                    "dataset": self.dataset_name,
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "scene_name": scene_name,
                    "path": str(img_path),
                }

                yield Sample(image=img_pil, meta=meta)