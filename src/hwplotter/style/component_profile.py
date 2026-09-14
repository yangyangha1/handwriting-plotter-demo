from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from hwplotter.geometry import apply_transform, resample_polyline, similarity_transform
from hwplotter.model.entities import Character, Stroke, StyleSample
from hwplotter.stroke_data.components import MMHDictionarySource


@dataclass(slots=True)
class ComponentStrokeStyle:
    samples: int
    residual_mean: list[list[float]]
    residual_std: list[list[float]]
    slant_mean_deg: float
    scale_mean: float


@dataclass(slots=True)
class ComponentStyle:
    occurrences: int
    stroke_slots: dict[int, ComponentStrokeStyle] = field(default_factory=dict)


@dataclass(slots=True)
class ComponentStyleModel:
    points_per_stroke: int = 64
    components: dict[str, ComponentStyle] = field(default_factory=dict)
    aligned_characters: int = 0
    aligned_strokes: int = 0

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ComponentStyleModel:
        data = json.loads(path.read_text(encoding="utf-8"))
        components: dict[str, ComponentStyle] = {}
        for component, value in data.get("components", {}).items():
            slots = {
                int(slot): ComponentStrokeStyle(**style)
                for slot, style in value.get("stroke_slots", {}).items()
            }
            components[component] = ComponentStyle(
                occurrences=int(value.get("occurrences", 0)),
                stroke_slots=slots,
            )
        return cls(
            points_per_stroke=int(data.get("points_per_stroke", 64)),
            components=components,
            aligned_characters=int(data.get("aligned_characters", 0)),
            aligned_strokes=int(data.get("aligned_strokes", 0)),
        )


def _stroke_summary(std: Stroke, fit: Stroke, n: int) -> tuple[np.ndarray, float, float]:
    a = resample_polyline(std.points, n)
    b = resample_polyline(fit.points, n)
    mat, t = similarity_transform(a, b)
    base = apply_transform(a, mat, t)
    residual = b - base
    scale = float(np.sqrt(abs(np.linalg.det(mat))))
    angle = float(np.degrees(np.arctan2(mat[1, 0], mat[0, 0])))
    return residual.astype(np.float32), angle, scale


def learn_component_styles(
    samples: list[StyleSample],
    dictionary_path: Path,
    n: int = 64,
    min_samples: int = 2,
) -> ComponentStyleModel:
    """Learn exact component-owned stroke styles using MMH ``matches`` labels.

    ``dictionary.txt`` supplies one decomposition-tree path per character stroke.
    We resolve those paths to component labels, group the corresponding standard
    and fitted strokes, and learn statistics for each *component-relative stroke
    slot*.  A component model therefore no longer borrows residuals from unrelated
    strokes in the same character.
    """
    dictionary = MMHDictionarySource(dictionary_path)
    residuals: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    angles: dict[tuple[str, int], list[float]] = defaultdict(list)
    scales: dict[tuple[str, int], list[float]] = defaultdict(list)
    occurrences: dict[str, int] = defaultdict(int)
    aligned_chars = 0
    aligned_strokes = 0

    for sample in samples:
        info = dictionary.get(sample.char)
        if not info or not info.component_strokes:
            continue
        char_used = False
        for component, stroke_indices in info.component_strokes.items():
            valid_indices = [
                i
                for i in stroke_indices
                if i < len(sample.standard.strokes) and i < len(sample.fitted.strokes)
            ]
            if not valid_indices:
                continue
            occurrences[component] += 1
            for local_slot, stroke_index in enumerate(valid_indices):
                r, a, s = _stroke_summary(
                    sample.standard.strokes[stroke_index],
                    sample.fitted.strokes[stroke_index],
                    n,
                )
                key = (component, local_slot)
                residuals[key].append(r)
                angles[key].append(a)
                scales[key].append(s)
                aligned_strokes += 1
                char_used = True
        if char_used:
            aligned_chars += 1

    out: dict[str, ComponentStyle] = {}
    for component, occurrence_count in occurrences.items():
        slots: dict[int, ComponentStrokeStyle] = {}
        slot_ids = sorted(slot for comp, slot in residuals if comp == component)
        for slot in slot_ids:
            key = (component, slot)
            rs = residuals[key]
            if len(rs) < min_samples:
                continue
            stack = np.stack(rs, axis=0)
            slots[slot] = ComponentStrokeStyle(
                samples=len(rs),
                residual_mean=np.mean(stack, axis=0).tolist(),
                residual_std=np.std(stack, axis=0).tolist(),
                slant_mean_deg=float(np.mean(angles[key])),
                scale_mean=float(np.mean(scales[key])),
            )
        if slots:
            out[component] = ComponentStyle(occurrences=occurrence_count, stroke_slots=slots)

    return ComponentStyleModel(
        points_per_stroke=n,
        components=out,
        aligned_characters=aligned_chars,
        aligned_strokes=aligned_strokes,
    )


def _styled_component_stroke(
    standard: Stroke,
    style: ComponentStrokeStyle,
    n: int,
    variation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    p = resample_polyline(standard.points, n)
    center = np.mean(p, axis=0, keepdims=True)
    theta = np.radians(style.slant_mean_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    q = ((p - center) @ rot.T) * style.scale_mean + center
    residual = np.asarray(style.residual_mean, dtype=np.float32)
    if variation > 0.0:
        std = np.asarray(style.residual_std, dtype=np.float32)
        residual = (
            residual + rng.normal(0.0, 1.0, residual.shape).astype(np.float32) * std * variation
        )
    return np.clip(q + residual, 0.0, 1.0)


def apply_component_conditioning(
    character: Character,
    global_profile,
    component_model: ComponentStyleModel,
    dictionary_path: Path,
    blend: float = 0.65,
    variation: float = 0.08,
    seed: int = 0,
) -> Character:
    """Synthesize with exact component-to-stroke conditioning.

    Global writer style remains the fallback.  For strokes with an MMH ``matches``
    owner and a trained component-relative slot, a component-specific trajectory is
    blended only into that owned stroke.  Unmatched strokes are left at the global
    profile result.
    """
    from hwplotter.style.profile import apply_profile

    base = apply_profile(character, global_profile, variation=variation, seed=seed)
    dictionary = MMHDictionarySource(dictionary_path)
    info = dictionary.get(character.char)
    if not info or not info.component_strokes:
        return base

    rng = np.random.default_rng(seed)
    out = [stroke.copy() for stroke in base.strokes]
    n = component_model.points_per_stroke
    for component, stroke_indices in info.component_strokes.items():
        component_style = component_model.components.get(component)
        if component_style is None:
            continue
        for local_slot, stroke_index in enumerate(stroke_indices):
            if stroke_index >= len(character.strokes) or stroke_index >= len(out):
                continue
            slot_style = component_style.stroke_slots.get(local_slot)
            if slot_style is None:
                continue
            comp_points = _styled_component_stroke(
                character.strokes[stroke_index], slot_style, n, variation, rng
            )
            global_points = resample_polyline(out[stroke_index].points, n)
            q = (1.0 - blend) * global_points + blend * comp_points
            out[stroke_index] = Stroke(
                index=out[stroke_index].index,
                points=np.clip(q, 0.0, 1.0),
                stroke_type=out[stroke_index].stroke_type,
                source_path=out[stroke_index].source_path,
                confidence=out[stroke_index].confidence,
            )
    return Character(character.char, out, source="exact-component-conditioned-style")
