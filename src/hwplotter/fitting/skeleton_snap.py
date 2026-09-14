from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from skimage.morphology import skeletonize

from hwplotter.geometry import resample_polyline, smooth_polyline
from hwplotter.model.entities import Character, Stroke, all_points


@dataclass(slots=True)
class FitConfig:
    image_size: int = 512
    points_per_stroke: int = 64
    iterations: int = 35
    attraction: float = 0.28
    smoothness: float = 0.18
    anchor_strength: float = 0.06
    max_step_px: float = 5.0


def _normalize_standard_to_ink(character: Character, mask: np.ndarray) -> Character:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return character
    ink_min = np.array([xs.min(), ys.min()], dtype=np.float32) / np.array(
        [mask.shape[1] - 1, mask.shape[0] - 1], dtype=np.float32
    )
    ink_max = np.array([xs.max(), ys.max()], dtype=np.float32) / np.array(
        [mask.shape[1] - 1, mask.shape[0] - 1], dtype=np.float32
    )
    p = all_points(character)
    pmin, pmax = p.min(0), p.max(0)
    span = np.maximum(pmax - pmin, 1e-5)
    target_span = np.maximum(ink_max - ink_min, 1e-5)
    scale = float(min(target_span / span))
    src_center = (pmin + pmax) / 2
    dst_center = (ink_min + ink_max) / 2
    out: list[Stroke] = []
    for stroke in character.strokes:
        q = (stroke.points - src_center) * scale + dst_center
        out.append(
            Stroke(stroke.index, q.astype(np.float32), stroke.stroke_type, stroke.source_path)
        )
    return Character(
        character.char, out, source=character.source, metadata=dict(character.metadata)
    )


def _field_from_skeleton(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skel = skeletonize(mask > 0)
    # distance to nearest skeleton; gradient points away from skeleton.
    dist = distance_transform_edt(~skel).astype(np.float32)
    gy, gx = np.gradient(dist)
    return skel.astype(np.uint8), gx.astype(np.float32), gy.astype(np.float32)


def _sample(field: np.ndarray, xy_px: np.ndarray) -> np.ndarray:
    x = np.clip(xy_px[:, 0], 0, field.shape[1] - 1)
    y = np.clip(xy_px[:, 1], 0, field.shape[0] - 1)
    return map_coordinates(field, [y, x], order=1, mode="nearest")


def fit_character_to_mask(
    standard: Character,
    mask: np.ndarray,
    config: FitConfig | None = None,
) -> Character:
    """CPU baseline inspired by HST's lightweight skeleton/median snapping.

    Objective (discretized):
      E = λd * distance_to_skeleton + λs * curvature + λa * deviation_from_prior
    We optimize median points by gradient descent on the skeleton distance field,
    while Laplacian smoothing preserves stroke continuity.
    """
    cfg = config or FitConfig()
    if mask.shape != (cfg.image_size, cfg.image_size):
        mask = cv2.resize(mask, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_NEAREST)
    initialized = _normalize_standard_to_ink(standard, mask)
    _, gx, gy = _field_from_skeleton(mask)
    wh = np.array([cfg.image_size - 1, cfg.image_size - 1], dtype=np.float32)
    fitted: list[Stroke] = []

    for stroke in initialized.strokes:
        p0 = resample_polyline(stroke.points, cfg.points_per_stroke)
        p = p0.copy()
        for _ in range(cfg.iterations):
            px = p * wh
            grad = np.stack([_sample(gx, px), _sample(gy, px)], axis=1)
            norms = np.linalg.norm(grad, axis=1, keepdims=True)
            grad = grad / np.maximum(norms, 1e-6)
            step = -cfg.attraction * grad * (cfg.max_step_px / wh)

            lap = np.zeros_like(p)
            lap[1:-1] = (p[:-2] + p[2:]) * 0.5 - p[1:-1]
            anchor = p0 - p
            p += step + cfg.smoothness * lap + cfg.anchor_strength * anchor
            p = np.clip(p, 0.0, 1.0)

        p = smooth_polyline(p, cfg.points_per_stroke, smooth=0.0003)
        fitted.append(
            Stroke(
                index=stroke.index,
                points=p,
                stroke_type=stroke.stroke_type,
                source_path=stroke.source_path,
            )
        )
    return Character(standard.char, fitted, source="skeleton-snap")
