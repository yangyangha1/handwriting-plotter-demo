from __future__ import annotations

import numpy as np

from hwplotter.geometry import resample_polyline
from hwplotter.model.entities import Character, Stroke


def point_features(stroke: Stroke, character: Character, n: int = 64) -> np.ndarray:
    p = resample_polyline(stroke.points, n)
    d = np.gradient(p, axis=0)
    dd = np.gradient(d, axis=0)
    speed = np.linalg.norm(d, axis=1, keepdims=True)
    cross = d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]
    curvature = (cross / np.maximum(speed[:, 0] ** 3, 1e-6))[:, None]
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    allp = np.concatenate([s.points for s in character.strokes], axis=0)
    lo, hi = allp.min(0), allp.max(0)
    bbox = np.array([lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]], np.float32)
    meta = np.tile(
        np.array(
            [
                stroke.index / max(1, character.stroke_count - 1),
                character.stroke_count / 40.0,
                *bbox,
            ],
            np.float32,
        ),
        (n, 1),
    )
    return np.concatenate([p, t, d, curvature, meta], axis=1).astype(np.float32)
