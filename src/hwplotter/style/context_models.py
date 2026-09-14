from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from hwplotter.geometry import apply_transform, resample_polyline, similarity_transform
from hwplotter.model.entities import Character, Stroke, StyleSample
from hwplotter.stroke_data.components import IDS_OPERATORS, MMHDictionarySource, parse_ids

POSITIONS = ("left", "right", "top", "bottom", "center", "other")


def _bbox_for_indices(
    character: Character, indices: tuple[int, ...]
) -> tuple[float, float, float, float] | None:
    chunks = [character.strokes[i].points for i in indices if 0 <= i < len(character.strokes)]
    if not chunks:
        return None
    p = np.concatenate(chunks, axis=0)
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def classify_position(bbox: tuple[float, float, float, float] | None) -> str:
    if bbox is None:
        return "other"
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    w, h = x1 - x0, y1 - y0
    if cx < 0.40 and w < 0.72:
        return "left"
    if cx > 0.60 and w < 0.72:
        return "right"
    if cy < 0.40 and h < 0.72:
        return "top"
    if cy > 0.60 and h < 0.72:
        return "bottom"
    if 0.35 <= cx <= 0.65 and 0.35 <= cy <= 0.65:
        return "center"
    return "other"


def _root_operator(decomposition: str) -> str:
    root = parse_ids(decomposition)
    return root.symbol if root and root.symbol in IDS_OPERATORS else "single"


def _stroke_residual(std: Stroke, fit: Stroke, n: int) -> tuple[np.ndarray, float, float]:
    a = resample_polyline(std.points, n)
    b = resample_polyline(fit.points, n)
    mat, t = similarity_transform(a, b)
    base = apply_transform(a, mat, t)
    residual = b - base
    scale = float(np.sqrt(abs(np.linalg.det(mat))))
    angle = float(np.degrees(np.arctan2(mat[1, 0], mat[0, 0])))
    return residual.astype(np.float32), angle, scale


@dataclass(slots=True)
class ResidualStyle:
    samples: int
    residual_mean: list[list[float]]
    residual_std: list[list[float]]
    slant_mean_deg: float
    scale_mean: float


@dataclass(slots=True)
class PositionStyleModel:
    points_per_stroke: int = 64
    styles: dict[str, dict[int, ResidualStyle]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> PositionStyleModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        styles = {}
        for key, slots in raw.get("styles", {}).items():
            styles[key] = {int(k): ResidualStyle(**v) for k, v in slots.items()}
        return cls(points_per_stroke=int(raw.get("points_per_stroke", 64)), styles=styles)


@dataclass(slots=True)
class StructureStyleModel:
    points_per_stroke: int = 64
    styles: dict[str, ResidualStyle] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> StructureStyleModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            int(raw.get("points_per_stroke", 64)),
            {k: ResidualStyle(**v) for k, v in raw.get("styles", {}).items()},
        )


@dataclass(slots=True)
class AdjacencyStyleModel:
    points_per_stroke: int = 64
    styles: dict[str, dict[int, ResidualStyle]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> AdjacencyStyleModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        styles = {
            k: {int(i): ResidualStyle(**v) for i, v in slots.items()}
            for k, slots in raw.get("styles", {}).items()
        }
        return cls(int(raw.get("points_per_stroke", 64)), styles)


def _build_style(residuals, angles, scales) -> ResidualStyle:
    stack = np.stack(residuals, axis=0)
    return ResidualStyle(
        samples=len(residuals),
        residual_mean=np.mean(stack, axis=0).tolist(),
        residual_std=np.std(stack, axis=0).tolist(),
        slant_mean_deg=float(np.mean(angles)),
        scale_mean=float(np.mean(scales)),
    )


def learn_context_models(
    samples: list[StyleSample], dictionary_path: Path, n: int = 64, min_samples: int = 2
):
    """Train position-, structure-, and adjacency-conditioned residual models.

    The labels are deterministic MMH structure labels; trajectories always come from
    the user's facsimile-recovered strokes.
    """
    dictionary = MMHDictionarySource(dictionary_path)
    pos_r = defaultdict(list)
    pos_a = defaultdict(list)
    pos_s = defaultdict(list)
    str_r = defaultdict(list)
    str_a = defaultdict(list)
    str_s = defaultdict(list)
    adj_r = defaultdict(list)
    adj_a = defaultdict(list)
    adj_s = defaultdict(list)

    for sample in samples:
        info = dictionary.get(sample.char)
        if not info:
            continue
        root_op = _root_operator(info.decomposition)
        component_positions: dict[str, str] = {}
        for comp, indices in info.component_strokes.items():
            component_positions[comp] = classify_position(
                _bbox_for_indices(sample.standard, indices)
            )

        # structure model: all strokes in same structural class
        for i, (std, fit) in enumerate(zip(sample.standard.strokes, sample.fitted.strokes)):
            r, a, sc = _stroke_residual(std, fit, n)
            str_r[root_op].append(r)
            str_a[root_op].append(a)
            str_s[root_op].append(sc)

        # component@position, with local slot
        for comp, indices in info.component_strokes.items():
            position = component_positions.get(comp, "other")
            key_base = f"{comp}@{position}"
            for slot, i in enumerate(indices):
                if i >= len(sample.standard.strokes) or i >= len(sample.fitted.strokes):
                    continue
                r, a, sc = _stroke_residual(sample.standard.strokes[i], sample.fitted.strokes[i], n)
                stroke_key = (key_base, slot)
                pos_r[stroke_key].append(r)
                pos_a[stroke_key].append(a)
                pos_s[stroke_key].append(sc)

        # adjacency is intentionally generalized by positions rather than exact neighbor character.
        # This avoids combinatorial sparsity while still learning context effects.
        comps = list(info.component_strokes)
        for comp in comps:
            p = component_positions.get(comp, "other")
            neighbors = [component_positions.get(o, "other") for o in comps if o != comp]
            if not neighbors:
                continue
            context = "+".join(sorted(set(neighbors)))
            key_base = f"{comp}@{p}|nbr:{context}|op:{root_op}"
            for slot, i in enumerate(info.component_strokes[comp]):
                if i >= len(sample.standard.strokes) or i >= len(sample.fitted.strokes):
                    continue
                r, a, sc = _stroke_residual(sample.standard.strokes[i], sample.fitted.strokes[i], n)
                stroke_key = (key_base, slot)
                adj_r[stroke_key].append(r)
                adj_a[stroke_key].append(a)
                adj_s[stroke_key].append(sc)

    pos_styles: dict[str, dict[int, ResidualStyle]] = defaultdict(dict)
    for (key, slot), rs in pos_r.items():
        if len(rs) >= min_samples:
            pos_styles[key][slot] = _build_style(rs, pos_a[(key, slot)], pos_s[(key, slot)])
    str_styles = {
        key: _build_style(rs, str_a[key], str_s[key])
        for key, rs in str_r.items()
        if len(rs) >= min_samples
    }
    adj_styles: dict[str, dict[int, ResidualStyle]] = defaultdict(dict)
    for (key, slot), rs in adj_r.items():
        if len(rs) >= min_samples:
            adj_styles[key][slot] = _build_style(rs, adj_a[(key, slot)], adj_s[(key, slot)])

    return (
        PositionStyleModel(n, dict(pos_styles)),
        StructureStyleModel(n, str_styles),
        AdjacencyStyleModel(n, dict(adj_styles)),
    )


def _styled_points(
    standard: Stroke, style: ResidualStyle, n: int, variation: float, rng: np.random.Generator
) -> np.ndarray:
    p = resample_polyline(standard.points, n)
    center = np.mean(p, axis=0, keepdims=True)
    theta = np.radians(style.slant_mean_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32
    )
    q = ((p - center) @ rot.T) * style.scale_mean + center
    residual = np.asarray(style.residual_mean, np.float32)
    if variation:
        residual = residual + variation * np.asarray(style.residual_std, np.float32) * rng.normal(
            size=residual.shape
        ).astype(np.float32)
    return np.clip(q + residual, 0.0, 1.0)


def apply_context_models(
    standard: Character,
    current: Character,
    dictionary_path: Path,
    position_model: PositionStyleModel | None = None,
    structure_model: StructureStyleModel | None = None,
    adjacency_model: AdjacencyStyleModel | None = None,
    variation: float = 0.05,
    seed: int = 0,
    position_blend: float = 0.45,
    structure_blend: float = 0.20,
    adjacency_blend: float = 0.35,
) -> Character:
    dictionary = MMHDictionarySource(dictionary_path)
    info = dictionary.get(standard.char)
    if not info:
        return current
    rng = np.random.default_rng(seed)
    out = [s.copy() for s in current.strokes]
    root_op = _root_operator(info.decomposition)
    positions = {
        comp: classify_position(_bbox_for_indices(standard, idxs))
        for comp, idxs in info.component_strokes.items()
    }

    # Structure is broadest context and therefore lowest blend.
    if structure_model and root_op in structure_model.styles:
        structure_style = structure_model.styles[root_op]
        for i in range(min(len(out), len(standard.strokes))):
            target = _styled_points(
                standard.strokes[i],
                structure_style,
                structure_model.points_per_stroke,
                variation,
                rng,
            )
            base = resample_polyline(out[i].points, structure_model.points_per_stroke)
            out[i].points = np.clip((1 - structure_blend) * base + structure_blend * target, 0, 1)

    for comp, idxs in info.component_strokes.items():
        pos = positions.get(comp, "other")
        pos_key = f"{comp}@{pos}"
        neighbors = [positions.get(o, "other") for o in info.component_strokes if o != comp]
        adj_key = (
            f"{comp}@{pos}|nbr:{'+'.join(sorted(set(neighbors)))}|op:{root_op}"
            if neighbors
            else None
        )
        for slot, i in enumerate(idxs):
            if i >= len(out) or i >= len(standard.strokes):
                continue
            if position_model:
                style = position_model.styles.get(pos_key, {}).get(slot)
                if style:
                    target = _styled_points(
                        standard.strokes[i], style, position_model.points_per_stroke, variation, rng
                    )
                    base = resample_polyline(out[i].points, position_model.points_per_stroke)
                    out[i].points = np.clip(
                        (1 - position_blend) * base + position_blend * target, 0, 1
                    )
            if adjacency_model and adj_key:
                style = adjacency_model.styles.get(adj_key, {}).get(slot)
                if style:
                    target = _styled_points(
                        standard.strokes[i],
                        style,
                        adjacency_model.points_per_stroke,
                        variation,
                        rng,
                    )
                    base = resample_polyline(out[i].points, adjacency_model.points_per_stroke)
                    out[i].points = np.clip(
                        (1 - adjacency_blend) * base + adjacency_blend * target, 0, 1
                    )

    return Character(
        standard.char, out, source="context-conditioned-style", metadata=dict(current.metadata)
    )
