from __future__ import annotations

import numpy as np
from scipy.interpolate import splev, splprep


def arc_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=np.float32)
    if len(points) == 1:
        return np.array([0.0], dtype=np.float32)
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)]).astype(np.float32)


def resample_polyline(points: np.ndarray, n: int = 64) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= 1:
        return np.repeat(points[:1], n, axis=0) if len(points) else np.zeros((n, 2), np.float32)
    s = arc_lengths(points)
    total = float(s[-1])
    if total < 1e-8:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    x = np.interp(targets, s, points[:, 0])
    y = np.interp(targets, s, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def smooth_polyline(points: np.ndarray, n: int = 64, smooth: float = 0.001) -> np.ndarray:
    p = resample_polyline(points, max(8, min(len(points), n)))
    if len(p) < 4:
        return resample_polyline(p, n)
    try:
        tck, _ = splprep([p[:, 0], p[:, 1]], s=smooth, k=min(3, len(p) - 1))
        u = np.linspace(0.0, 1.0, n)
        x, y = splev(u, tck)
        return np.stack([x, y], axis=1).astype(np.float32)
    except (ValueError, TypeError, RuntimeError):
        return resample_polyline(p, n)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return A(2x2), t(2,) minimizing ||src @ A.T + t - dst||."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError("src/dst require equal length >= 2")
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    xs = src - src_mean
    xd = dst - dst_mean
    var = (xs * xs).sum() / len(src)
    if var < 1e-12:
        return np.eye(2), dst_mean - src_mean
    cov = xd.T @ xs / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    scale = float((s * d).sum() / var)
    a = scale * r
    t = dst_mean - a @ src_mean
    return a.astype(np.float32), t.astype(np.float32)


def apply_transform(points: np.ndarray, a: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (points @ a.T + t).astype(np.float32)
