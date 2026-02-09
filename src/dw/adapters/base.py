from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from PIL import Image


__all__ = ["Sample"]

@dataclass
class Sample:
    image: Image.Image
    meta: Dict[str, Any]
