from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from hwplotter.fitting.skeleton_snap import FitConfig, fit_character_to_mask
from hwplotter.fitting.stroke_order import StrokeOrderVariant, infer_stroke_order
from hwplotter.geometry import resample_polyline, smooth_polyline
from hwplotter.model.entities import Character, Stroke
from hwplotter.plotter_paths import smooth_path


@dataclass(slots=True)
class FacsimileConfig:
    image_size: int = 768
    points_per_stroke: int = 96
    coarse_iterations: int = 45
    assignment_radius_px: float = 48.0
    bins: int = 96
    min_bin_pixels: int = 1
    smooth: float = 0.00015


def _polyline_distance_and_t(
    points_xy: np.ndarray, polyline: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate nearest distance and normalized arc-position on a polyline.

    points_xy is in normalized coordinates.  The implementation samples the prior
    densely; this is deterministic, dependency-light and fast enough for single
    character crops.
    """
    prior = resample_polyline(polyline, 256)
    from scipy.spatial import cKDTree

    distance, idx = cKDTree(prior).query(points_xy)
    return distance, idx.astype(np.float32) / max(len(prior) - 1, 1)


def _assign_ink_to_strokes(
    mask: np.ndarray, coarse: Character, radius_px: float
) -> list[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return [np.zeros_like(mask, dtype=np.uint8) for _ in coarse.strokes]
    wh = np.array([mask.shape[1] - 1, mask.shape[0] - 1], dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32) / wh
    distances = []
    for stroke in coarse.strokes:
        d, _ = _polyline_distance_and_t(pts, stroke.points)
        distances.append(d)
    D = np.stack(distances, axis=1)
    owner = np.argmin(D, axis=1)
    min_d_px = D[np.arange(len(pts)), owner] * float(max(mask.shape))

    out: list[np.ndarray] = [np.zeros_like(mask, dtype=np.uint8) for _ in coarse.strokes]
    for i in range(len(coarse.strokes)):
        keep = (owner == i) & (min_d_px <= radius_px)
        out[i][ys[keep], xs[keep]] = 1
        # Closing reconnects small gaps introduced at stroke crossings.
        out[i] = cv2.morphologyEx(out[i], cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return out


def _ordered_centerline(
    stroke_mask: np.ndarray, prior: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray, float]:
    skel = skeletonize(stroke_mask > 0)
    ys, xs = np.where(skel)
    if len(xs) < 3:
        p = resample_polyline(prior, bins)
        return p, np.zeros((bins,), dtype=np.float32), 0.0

    wh = np.array([stroke_mask.shape[1] - 1, stroke_mask.shape[0] - 1], dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32) / wh
    d, t = _polyline_distance_and_t(pts, prior)
    # Remove skeleton branches that are implausibly far from the stroke prior.
    threshold = max(np.quantile(d, 0.85) * 1.5, 0.015)
    keep = d <= threshold
    pts, t = pts[keep], t[keep]
    if len(pts) < 3:
        p = resample_polyline(prior, bins)
        return p, np.zeros((bins,), dtype=np.float32), 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    centers: list[np.ndarray | None] = []
    for i in range(bins):
        k = (t >= edges[i]) & (t <= edges[i + 1] if i == bins - 1 else t < edges[i + 1])
        centers.append(np.median(pts[k], axis=0) if np.any(k) else None)

    prior_r = resample_polyline(prior, bins)
    q = np.empty_like(prior_r)
    known = np.array([c is not None for c in centers])
    for i, c in enumerate(centers):
        if c is not None:
            q[i] = c
    if np.any(known):
        ki = np.where(known)[0]
        for dim in range(2):
            q[:, dim] = np.interp(
                np.arange(bins), ki, q[ki, dim], left=q[ki[0], dim], right=q[ki[-1], dim]
            )
        # Keep unobserved tails attached to the prior rather than extrapolating wild branches.
        first, last = ki[0], ki[-1]
        if first > 0:
            delta = q[first] - prior_r[first]
            q[:first] = prior_r[:first] + delta
        if last < bins - 1:
            delta = q[last] - prior_r[last]
            q[last + 1 :] = prior_r[last + 1 :] + delta
    else:
        q = prior_r

    q = np.clip(q, 0.0, 1.0)
    q = smooth_polyline(q, bins, smooth=0.00015)
    q = resample_polyline(np.asarray(smooth_path(q, tolerance=0.0015)), bins)

    # Local ink width sampled from the stroke-specific mask.
    dist_inside = distance_transform_edt(stroke_mask > 0).astype(np.float32)
    px = np.rint(q * wh).astype(int)
    px[:, 0] = np.clip(px[:, 0], 0, stroke_mask.shape[1] - 1)
    px[:, 1] = np.clip(px[:, 1], 0, stroke_mask.shape[0] - 1)
    widths_px = 2.0 * dist_inside[px[:, 1], px[:, 0]]
    widths = widths_px / float(max(stroke_mask.shape))

    # Coverage score: fraction of ordered bins supported by observed skeleton pixels.
    coverage = float(np.mean(known))
    return q.astype(np.float32), widths.astype(np.float32), coverage


def facsimile_character(
    standard: Character,
    mask: np.ndarray,
    config: FacsimileConfig | None = None,
    cursive_variants: list[StrokeOrderVariant] | None = None,
) -> Character:
    """Recover a high-fidelity centerline representation from an existing glyph image.

    The canonical MMH strokes supply stroke count/order/direction.  A coarse fit first
    aligns these semantics to the image.  Ink pixels are then partitioned by their
    nearest coarse stroke, each partition is skeletonized, and the observed skeleton
    is ordered by projection onto that stroke's canonical direction.  This preserves
    the photographed glyph geometry while avoiding ambiguous graph traversal at
    crossings.
    """
    cfg = config or FacsimileConfig()
    if mask.shape != (cfg.image_size, cfg.image_size):
        mask = cv2.resize(mask, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_NEAREST)
    coarse = fit_character_to_mask(
        standard,
        mask,
        FitConfig(
            image_size=cfg.image_size,
            points_per_stroke=cfg.points_per_stroke,
            iterations=cfg.coarse_iterations,
            attraction=0.30,
            smoothness=0.16,
            anchor_strength=0.05,
            max_step_px=6.0,
        ),
    )
    stroke_masks = _assign_ink_to_strokes(mask, coarse, cfg.assignment_radius_px)
    out: list[Stroke] = []
    coverages: list[float] = []
    for stroke, smask in zip(coarse.strokes, stroke_masks):
        q, widths, coverage = _ordered_centerline(smask, stroke.points, cfg.bins)
        coverages.append(coverage)
        out.append(
            Stroke(
                index=stroke.index,
                points=q,
                stroke_type=stroke.stroke_type,
                source_path=stroke.source_path,
                confidence=max(0.0, min(1.0, coverage)),
                widths=widths,
            )
        )
    quality = float(np.mean(coverages)) if coverages else 0.0
    result = Character(
        standard.char,
        out,
        source="facsimile",
        metadata={"facsimile_quality": quality, "stroke_coverages": coverages},
    )
    order = infer_stroke_order(result, mask, cursive_variants=cursive_variants)
    result.metadata.update(
        {
            "stroke_order": list(order.order),
            "pen_down_groups": [list(g) for g in order.pen_down_groups],
            "stroke_order_confidence": float(order.confidence),
            "stroke_order_mode": order.mode,
            "stroke_order_ambiguous": bool(order.ambiguous),
            "stroke_order_score_margin": float(order.score_margin),
        }
    )
    return result
