from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hwplotter.geometry import apply_transform, resample_polyline, similarity_transform
from hwplotter.model.entities import Character, Stroke, StyleSample


@dataclass(slots=True)
class StyleProfile:
    residual_mean: list[list[float]]
    residual_std: list[list[float]]
    slant_mean_deg: float
    scale_mean: float
    samples: int
    points_per_stroke: int = 64

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> StyleProfile:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _pair_residual(std: Stroke, fit: Stroke, n: int) -> tuple[np.ndarray, float, float]:
    a = resample_polyline(std.points, n)
    b = resample_polyline(fit.points, n)
    mat, t = similarity_transform(a, b)
    base = apply_transform(a, mat, t)
    residual = b - base
    scale = float(np.sqrt(abs(np.linalg.det(mat))))
    angle = float(np.degrees(np.arctan2(mat[1, 0], mat[0, 0])))
    return residual, angle, scale


def learn_profile(samples: list[StyleSample], n: int = 64) -> StyleProfile:
    residuals: list[np.ndarray] = []
    angles: list[float] = []
    scales: list[float] = []
    for sample in samples:
        for s, f in zip(sample.standard.strokes, sample.fitted.strokes):
            r, a, sc = _pair_residual(s, f, n)
            residuals.append(r)
            angles.append(a)
            scales.append(sc)
    if not residuals:
        raise ValueError("No stroke pairs for style learning")
    r = np.stack(residuals)
    return StyleProfile(
        residual_mean=r.mean(0).tolist(),
        residual_std=r.std(0).tolist(),
        slant_mean_deg=float(np.mean(angles)),
        scale_mean=float(np.mean(scales)),
        samples=len(samples),
        points_per_stroke=n,
    )


def apply_profile(
    character: Character, profile: StyleProfile, variation: float = 0.15, seed: int = 0
) -> Character:
    rng = np.random.default_rng(seed)
    mean = np.asarray(profile.residual_mean, dtype=np.float32)
    std = np.asarray(profile.residual_std, dtype=np.float32)
    theta = np.radians(profile.slant_mean_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    center = np.array([0.5, 0.5], np.float32)
    out: list[Stroke] = []
    for stroke in character.strokes:
        p = resample_polyline(stroke.points, profile.points_per_stroke)
        p = ((p - center) @ rot.T) * profile.scale_mean + center
        noise = rng.normal(size=mean.shape).astype(np.float32)
        p = p + mean + variation * std * noise
        p = np.clip(p, 0.0, 1.0)
        out.append(Stroke(stroke.index, p, stroke.stroke_type, stroke.source_path))
    return Character(character.char, out, source="style-profile")
