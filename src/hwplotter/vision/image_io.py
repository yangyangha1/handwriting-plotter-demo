from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image through bytes so Windows non-ASCII paths remain safe."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"图片文件不存在：{path}")
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, flags)
    if image is None:
        raise OSError(f"图片无法解码：{path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image through bytes so Windows non-ASCII paths remain safe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise OSError(f"图片无法编码为 {suffix}：{path}")
    path.write_bytes(encoded.tobytes())
