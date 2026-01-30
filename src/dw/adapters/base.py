from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple
from PIL import Image

__all__ = ["Sample"]

@dataclass
class Sample:
    image: Image.Image
    meta: Dict[str, Any]
