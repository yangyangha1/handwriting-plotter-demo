from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hwplotter.model.entities import Character


@dataclass(slots=True)
class StrokeOrderVariant:
    order: tuple[int, ...]
    name: str = "variant"


@dataclass(slots=True)
class StrokeOrderResult:
    order: tuple[int, ...]
    pen_down_groups: tuple[tuple[int, ...], ...]
    confidence: float
    mode: str
    ambiguous: bool
    score_margin: float


class CursiveOrderSource:
    """Optional cursive/running-script order overrides.

    JSONL schema per line::

        {"character":"字", "variants":[
          {"name":"running-1", "order":[0,1,3,2,4]},
          {"name":"cursive-1", "order":[0,1,2,3,4]}
        ]}

    The indices refer to canonical Make-Me-A-Hanzi strokes.  This file is optional
    because there is no comprehensive open temporal cursive-order database.  The
    image evidence is still scored against the supplied variants and the standard
    order; a low-margin result is rejected by the workflow instead of being guessed.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._variants: dict[str, list[StrokeOrderVariant]] = {}
        if path is None or not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ch = str(row["character"])
            vals: list[StrokeOrderVariant] = []
            for item in row.get("variants", []):
                vals.append(
                    StrokeOrderVariant(
                        tuple(int(x) for x in item["order"]),
                        str(item.get("name", "cursive")),
                    )
                )
            if vals:
                self._variants[ch] = vals

    def get(self, char: str) -> list[StrokeOrderVariant]:
        return list(self._variants.get(char, ()))


def _sample_line_support(mask: np.ndarray, a: np.ndarray, b: np.ndarray, radius: int = 2) -> float:
    h, w = mask.shape
    n = max(8, int(np.linalg.norm((b - a) * np.array([w - 1, h - 1]))))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    pts = a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]
    xy = np.rint(pts * np.array([w - 1, h - 1], np.float32)).astype(int)
    hit = 0
    for x, y in xy:
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        hit += int(np.any(mask[y0:y1, x0:x1] > 0))
    return float(hit) / max(len(xy), 1)


def _sequence_cost(
    order: tuple[int, ...], fitted: Character, mask: np.ndarray, prior_penalty: float
) -> tuple[float, list[tuple[float, float]]]:
    links: list[tuple[float, float]] = []
    cost = prior_penalty
    for ia, ib in itertools.pairwise(order):
        sa, sb = fitted.strokes[ia], fitted.strokes[ib]
        if not len(sa.points) or not len(sb.points):
            links.append((1.0, 0.0))
            cost += 1.5
            continue
        gap = float(np.linalg.norm(sa.points[-1] - sb.points[0]))
        support = _sample_line_support(mask, sa.points[-1], sb.points[0])
        # Cursive/running writing tends to minimize pen travel; actual ink between
        # canonical strokes is strong evidence that they form one pen-down action.
        cost += 1.10 * gap + 0.20 * (1.0 - support)
        links.append((gap, support))
    return cost, links


def _candidate_orders(count: int, extra: list[StrokeOrderVariant]) -> list[StrokeOrderVariant]:
    standard = tuple(range(count))
    out = [StrokeOrderVariant(standard, "regular-standard")]
    seen = {standard}
    for variant in extra:
        if (
            len(variant.order) == count
            and set(variant.order) == set(range(count))
            and variant.order not in seen
        ):
            out.append(variant)
            seen.add(variant.order)
    # Static-image inference needs alternatives even when no curated cursive source
    # exists.  Explore permutations within two adjacent transpositions of standard
    # order.  This captures local running/cursive order flexibility while keeping the
    # hypothesis space bounded for large characters.
    frontier: list[tuple[tuple[int, ...], list[int]]] = [(standard, [])]
    for depth in range(2):
        nxt = []
        for base, history in frontier:
            for i in range(max(0, count - 1)):
                x = list(base)
                x[i], x[i + 1] = x[i + 1], x[i]
                v = tuple(x)
                h = history + [i]
                if v not in seen:
                    name = "image-local-permutation-" + "-".join(str(k) for k in h)
                    out.append(StrokeOrderVariant(v, name))
                    seen.add(v)
                    nxt.append((v, h))
        frontier = nxt
    return out


def infer_stroke_order(
    fitted: Character,
    mask: np.ndarray,
    cursive_variants: list[StrokeOrderVariant] | None = None,
    merge_gap: float = 0.095,
    merge_support: float = 0.52,
) -> StrokeOrderResult:
    """Infer a plausible physical writing order from a static glyph image.

    This cannot recover time uniquely from every bitmap.  We therefore compare the
    regular order, optional curated cursive orders and limited image-derived local
    alternatives.  Results with insufficient score separation are explicitly marked
    ambiguous so the acquisition workflow can reject/rewrite them.
    """
    n = len(fitted.strokes)
    if n <= 1:
        return StrokeOrderResult(
            tuple(range(n)), (tuple(range(n)),) if n else (), 1.0, "single-stroke", False, 1.0
        )
    variants = _candidate_orders(n, cursive_variants or [])
    standard = tuple(range(n))
    scored = []
    for v in variants:
        deviations = sum(int(a != b) for a, b in zip(v.order, standard))
        prior = 0.025 * deviations
        cost, links = _sequence_cost(v.order, fitted, mask, prior)
        scored.append((float(cost), v, links))
    scored.sort(key=lambda x: x[0])
    best_cost, best_variant, best_links = scored[0]
    second = scored[1][0] if len(scored) > 1 else best_cost + 1.0
    margin = max(0.0, second - best_cost)
    # Margin is evidence for order; per-stroke facsimile confidence is independent
    # evidence that the underlying centerlines themselves were well recovered.
    fit_conf = float(np.mean([s.confidence for s in fitted.strokes])) if fitted.strokes else 0.0
    margin_conf = float(1.0 - np.exp(-6.0 * margin))
    confidence = float(np.clip(0.60 * fit_conf + 0.40 * margin_conf, 0.0, 1.0))

    groups: list[list[int]] = [[best_variant.order[0]]]
    for next_idx, (gap, support) in zip(best_variant.order[1:], best_links):
        if gap <= merge_gap and support >= merge_support:
            groups[-1].append(next_idx)
        else:
            groups.append([next_idx])
    ambiguous = confidence < 0.58 or margin < 0.012
    mode = best_variant.name
    if any(len(g) > 1 for g in groups):
        mode += "+connected"
    return StrokeOrderResult(
        best_variant.order,
        tuple(tuple(g) for g in groups),
        confidence,
        mode,
        ambiguous,
        margin,
    )


def build_pen_paths(character: Character) -> list[np.ndarray]:
    """Build physical pen-down paths from metadata while keeping canonical strokes intact."""
    groups = character.metadata.get("pen_down_groups")
    order = character.metadata.get("stroke_order")
    if not isinstance(groups, list):
        groups = [
            [int(i)] for i in (order if isinstance(order, list) else range(len(character.strokes)))
        ]
    paths: list[np.ndarray] = []
    for group in groups:
        chunks: list[np.ndarray] = []
        for pos, idx in enumerate(group):
            s = character.strokes[int(idx)]
            if not len(s.points):
                continue
            if chunks:
                a = chunks[-1][-1]
                b = s.points[0]
                bridge = np.linspace(a, b, 8, dtype=np.float32)[1:-1]
                chunks.append(bridge)
            chunks.append(s.points)
        if chunks:
            paths.append(np.concatenate(chunks, axis=0).astype(np.float32))
    return paths
