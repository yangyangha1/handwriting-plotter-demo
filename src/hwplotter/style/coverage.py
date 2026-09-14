from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hwplotter.stroke_data.components import IDS_OPERATORS, MMHDictionarySource, parse_ids
from hwplotter.stroke_data.mmh import MakeMeAHanziSource
from hwplotter.style.common_chars import COMMON_CHARACTERS

CORE_COMPONENTS = tuple(
    "亻人氵水扌手口木日月女子心忄言讠纟糹艹土火灬金钅山石禾竹米虫足车車马馬门門阝耳辶走宀广厂疒衣衤示礻贝貝页頁力刀刂又大小王玉犬犭目田皿白立穴雨食饣弓尸巾欠攵攴文方酉鱼魚鸟鳥"
)


@dataclass(slots=True)
class CoverageReport:
    sample_characters: int
    unique_characters: int
    stroke_count_seen: int
    component_mode: bool
    covered_components: list[str]
    missing_components: list[str]
    insufficient_components: list[str]
    component_occurrences: dict[str, int]
    aligned_component_strokes: int
    component_coverage: float
    geometry_coverage: float
    overall_completion: float
    suggested_text: str
    suggested_characters: list[str]
    notes: list[str]
    structure_occurrences: dict[str, int] = field(default_factory=dict)
    position_occurrences: dict[str, int] = field(default_factory=dict)
    adjacency_contexts: int = 0
    context_suggested_text: str = ""

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def _geometry_signature(points) -> str:
    import numpy as np

    if len(points) < 2:
        return "dot"
    p = np.asarray(points, dtype=np.float32)
    d = p[-1] - p[0]
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    if length < 0.08:
        return "dot"
    bucket = int(((angle + 180.0) % 360.0) // 45.0)
    chord = max(float(np.linalg.norm(d)), 1e-5)
    curve_ratio = length / chord
    curve = "curve" if curve_ratio > 1.22 else "straight"
    return f"d{bucket}_{curve}"


def _component_position(character, indices: tuple[int, ...]) -> str:
    import numpy as np

    chunks = [character.strokes[i].points for i in indices if i < len(character.strokes)]
    if not chunks:
        return "other"
    p = np.concatenate(chunks, axis=0)
    lo, hi = p.min(axis=0), p.max(axis=0)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    w, h = hi[0] - lo[0], hi[1] - lo[1]
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


def _recommend_context_chars(
    dictionary,
    graphics,
    existing: set[str],
    missing_ops: set[str],
    missing_positions: set[str],
    limit: int = 24,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for char in COMMON_CHARACTERS:
        if char in existing:
            continue
        info = dictionary.get(char)
        if not info:
            continue
        root = parse_ids(info.decomposition)
        op = root.symbol if root and root.symbol in IDS_OPERATORS else "single"
        try:
            canonical = graphics.get(char)
        except KeyError:
            continue
        positions = {
            _component_position(canonical, idxs) for idxs in info.component_strokes.values()
        }
        score = (2.0 if op in missing_ops else 0.0) + len(positions & missing_positions)
        if score > 0:
            scored.append((score, char))
    scored.sort(key=lambda x: (-x[0], ord(x[1])))
    return [char for _, char in scored[:limit]]


def _recommend_chars_for_deficits(
    deficits: dict[str, int],
    dictionary: MMHDictionarySource,
    existing: set[str],
    limit: int,
) -> list[str]:
    """Greedy multi-cover recommendation using exact MMH stroke ownership.

    Unlike plain set-cover, a component that has been seen once but needs two
    samples still has a remaining deficit and is preferentially recommended.
    """
    remaining = {k: v for k, v in deficits.items() if v > 0}
    if not remaining:
        return []

    candidate_components: dict[str, set[str]] = {}
    for char in COMMON_CHARACTERS:
        if char in existing:
            continue
        info = dictionary.get(char)
        if not info:
            continue
        exact = set(info.component_strokes)
        gains = exact & set(remaining)
        if gains:
            candidate_components[char] = gains

    chosen: list[str] = []
    while remaining and len(chosen) < limit:
        best_char: str | None = None
        best_score = 0.0
        best_gain: set[str] = set()
        for char, comps in candidate_components.items():
            gain = {c for c in comps if remaining.get(c, 0) > 0}
            if not gain:
                continue
            # Prioritize wholly missing components slightly over under-sampled ones.
            score = sum(1.25 if remaining[c] >= 2 else 1.0 for c in gain)
            if score > best_score:
                best_char, best_score, best_gain = char, score, gain
        if best_char is None:
            break
        chosen.append(best_char)
        for component in best_gain:
            remaining[component] -= 1
            if remaining[component] <= 0:
                remaining.pop(component, None)
        candidate_components.pop(best_char, None)
    return chosen


def analyze_coverage(
    sample_chars: list[str],
    graphics_path: Path,
    dictionary_path: Path | None = None,
    target_components: tuple[str, ...] = CORE_COMPONENTS,
    suggestion_limit: int = 48,
    min_component_samples: int = 2,
) -> CoverageReport:
    graphics = MakeMeAHanziSource(graphics_path)
    geometry: Counter[str] = Counter()
    stroke_count = 0
    for char in sample_chars:
        try:
            c = graphics.get(char)
        except KeyError:
            continue
        for stroke in c.strokes:
            geometry[_geometry_signature(stroke.points)] += 1
            stroke_count += 1

    geometry_coverage = min(1.0, len(geometry) / 17.0)
    covered: set[str] = set()
    missing: set[str] = set()
    insufficient: set[str] = set()
    occurrences: Counter[str] = Counter()
    aligned_component_strokes = 0
    suggested: list[str] = []
    notes: list[str] = []
    structure_occurrences: Counter[str] = Counter()
    position_occurrences: Counter[str] = Counter()
    adjacency_keys: set[str] = set()
    context_suggested: list[str] = []
    component_mode = bool(dictionary_path and Path(dictionary_path).exists())

    if component_mode:
        assert dictionary_path is not None
        dictionary = MMHDictionarySource(Path(dictionary_path))
        for char in sample_chars:
            info = dictionary.get(char)
            if not info:
                continue
            root = parse_ids(info.decomposition)
            op = root.symbol if root and root.symbol in IDS_OPERATORS else "single"
            structure_occurrences[op] += 1
            try:
                canonical = graphics.get(char)
            except KeyError:
                canonical = None
            positions_for_char = {}
            for component, stroke_indices in info.component_strokes.items():
                occurrences[component] += 1
                aligned_component_strokes += len(stroke_indices)
                if canonical is not None:
                    pos = _component_position(canonical, stroke_indices)
                    positions_for_char[component] = pos
                    position_occurrences[pos] += 1
            comps = list(positions_for_char)
            for component in comps:
                pos = positions_for_char[component]
                nbr = "+".join(sorted({positions_for_char[o] for o in comps if o != component}))
                if nbr:
                    adjacency_keys.add(f"{component}@{pos}|nbr:{nbr}|op:{op}")

        target = set(target_components)
        covered = {c for c in target if occurrences[c] >= min_component_samples}
        missing = {c for c in target if occurrences[c] == 0}
        insufficient = {c for c in target if 0 < occurrences[c] < min_component_samples}
        component_coverage = len(covered) / max(len(target), 1)
        deficits = {c: max(0, min_component_samples - occurrences[c]) for c in target}
        suggested = _recommend_chars_for_deficits(
            deficits, dictionary, set(sample_chars), suggestion_limit
        )
        required_ops = {"⿰", "⿱", "⿴", "⿵", "⿸", "⿺"}
        required_positions = {"left", "right", "top", "bottom"}
        missing_ops = {op for op in required_ops if structure_occurrences[op] < 2}
        missing_positions = {pos for pos in required_positions if position_occurrences[pos] < 3}
        context_suggested = _recommend_context_chars(
            dictionary,
            graphics,
            set(sample_chars) | set(suggested),
            missing_ops,
            missing_positions,
            24,
        )
        # keep one ordered list; component deficits have priority, then context deficits
        suggested = (suggested + [c for c in context_suggested if c not in suggested])[
            :suggestion_limit
        ]
        notes.append(
            "偏旁覆盖使用 dictionary.txt 的 matches 逐笔映射；只有达到最小样本数的组件计为已完成。"
        )
    else:
        component_coverage = 0.0
        notes.append("未提供 dictionary.txt：无法执行偏旁到笔画的精确对齐，仅统计笔画几何覆盖。")

    overall = (
        0.70 * component_coverage + 0.30 * geometry_coverage
        if component_mode
        else geometry_coverage * 0.5
    )

    if len(set(sample_chars)) < 50:
        notes.append("样本字数偏少；建议先采集至少 50 个不同汉字，再评价个人风格稳定度。")
    if stroke_count < 200:
        notes.append("有效笔画样本不足 200，局部变形统计仍不稳定。")
    if component_mode and aligned_component_strokes < stroke_count:
        notes.append(
            f"MMH matches 已对齐 {aligned_component_strokes}/{stroke_count} 个样本笔画；"
            "未映射笔画仍参与全局风格，但不进入偏旁专属模型。"
        )

    return CoverageReport(
        sample_characters=len(sample_chars),
        unique_characters=len(set(sample_chars)),
        stroke_count_seen=stroke_count,
        component_mode=component_mode,
        covered_components=sorted(covered),
        missing_components=sorted(missing),
        insufficient_components=sorted(insufficient),
        component_occurrences=dict(sorted(occurrences.items())),
        aligned_component_strokes=aligned_component_strokes,
        component_coverage=component_coverage,
        geometry_coverage=geometry_coverage,
        overall_completion=overall,
        suggested_text="".join(suggested),
        suggested_characters=suggested,
        notes=notes,
        structure_occurrences=dict(sorted(structure_occurrences.items())),
        position_occurrences=dict(sorted(position_occurrences.items())),
        adjacency_contexts=len(adjacency_keys),
        context_suggested_text="".join(context_suggested),
    )
