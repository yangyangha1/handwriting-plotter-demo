from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hwplotter.vision.grid import remove_frame_rules
from hwplotter.vision.image_io import read_image


def read_gray(path: Path) -> np.ndarray:
    return read_image(path, cv2.IMREAD_GRAYSCALE)


def binarize_ink(gray: np.ndarray) -> np.ndarray:
    """Return uint8 mask: ink=255, background=0."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return remove_frame_rules(mask)


def tight_crop(mask: np.ndarray, pad: int = 6) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("No ink found")
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + 1 + pad)
    return mask[y0:y1, x0:x1], (int(x0), int(y0), int(x1), int(y1))


def square_resize(mask: np.ndarray, size: int = 512, margin: int = 28) -> np.ndarray:
    h, w = mask.shape
    scale = (size - 2 * margin) / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y = (size - nh) // 2
    x = (size - nw) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def frame_square_resize(
    mask: np.ndarray, size: int = 768, margin: int = 10
) -> tuple[np.ndarray, dict[str, float]]:
    """Letterbox an entire character cell without discarding its whitespace.

    Unlike ``tight_crop`` this keeps the glyph's position inside the OCR-derived
    character region, which is required for facsimile page-layout replay.
    Returned metadata is sufficient to map normalized fitted points back into
    normalized coordinates of the original crop.
    """
    h, w = mask.shape
    if h < 1 or w < 1:
        raise ValueError("Empty character frame")
    usable = max(1, size - 2 * margin)
    scale = usable / float(max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas, {
        "source_w": float(w),
        "source_h": float(h),
        "scale": float(scale),
        "offset_x": float(ox),
        "offset_y": float(oy),
        "size": float(size),
    }


def unframe_points(points: np.ndarray, transform: dict[str, float]) -> np.ndarray:
    """Map normalized letterbox coordinates back to normalized source-crop coordinates."""
    if len(points) == 0:
        return points.astype(np.float32, copy=True)
    size = transform["size"]
    scale = transform["scale"]
    ox, oy = transform["offset_x"], transform["offset_y"]
    sw, sh = transform["source_w"], transform["source_h"]
    px = points.astype(np.float32).copy()
    # facsimile coordinates use 0..1 inclusive over the working image.
    px[:, 0] = (px[:, 0] * max(size - 1.0, 1.0) - ox) / max(scale, 1e-8)
    px[:, 1] = (px[:, 1] * max(size - 1.0, 1.0) - oy) / max(scale, 1e-8)
    px[:, 0] /= max(sw - 1.0, 1.0)
    px[:, 1] /= max(sh - 1.0, 1.0)
    return np.clip(px, 0.0, 1.0).astype(np.float32)
