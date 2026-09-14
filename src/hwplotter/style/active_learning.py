from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from hwplotter.stroke_data.components import IDS_OPERATORS, MMHDictionarySource, parse_ids
from hwplotter.style.common_chars import COMMON_CHARACTERS
from hwplotter.style.coverage import CoverageReport


@dataclass(slots=True)
class ActiveLearningPlan:
    status: str
    suggested_text: str
    ranked_characters: list[dict]
    reasons: list[str]

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def recommend_active_samples(
    report: CoverageReport,
    dictionary_path: Path,
    max_chars: int = 24,
    hard_operators: list[str] | None = None,
) -> ActiveLearningPlan:
    d = MMHDictionarySource(dictionary_path)
    missing = set(report.missing_components)
    insufficient = set(report.insufficient_components)
    hard = set(hard_operators or [])
    position_counts = report.position_occurrences
    need_positions = {
        p for p in ("left", "right", "top", "bottom") if position_counts.get(p, 0) < 3
    }
    rows = []
    for ch in COMMON_CHARACTERS:
        info = d.get(ch)
        if not info or not info.decomposition_valid or not info.component_strokes:
            continue
        root = parse_ids(info.decomposition)
        op = root.symbol if root and root.symbol in IDS_OPERATORS else "single"
        score = 0.0
        why = []
        hit_m = missing.intersection(info.component_strokes)
        hit_i = insufficient.intersection(info.component_strokes)
        if hit_m:
            score += 4 * len(hit_m)
            why.append("缺失偏旁:" + "".join(sorted(hit_m)))
        if hit_i:
            score += 2 * len(hit_i)
            why.append("不足偏旁:" + "".join(sorted(hit_i)))
        if op in hard:
            score += 3
            why.append("高误差结构:" + op)
        # Root operator supplies reliable position opportunities without needing glyph geometry.
        if op in {"⿰", "⿲"} and need_positions.intersection({"left", "right"}):
            score += 2
        if op in {"⿱", "⿳"} and need_positions.intersection({"top", "bottom"}):
            score += 2
        if score > 0:
            rows.append({"char": ch, "score": score, "reasons": why, "operator": op})
    rows.sort(key=lambda x: (-float(str(x["score"])), str(x["char"])))
    chosen = []
    seen = set()
    for r in rows:
        if r["char"] in seen:
            continue
        chosen.append(r)
        seen.add(r["char"])
        if len(chosen) >= max_chars:
            break
    text = "".join(str(r["char"]) for r in chosen) or report.suggested_text
    reasons = ["按当前模型缺口主动选择信息量较高的汉字，而不是固定抄写表。"]
    if hard:
        reasons.append("优先覆盖验证误差较高的结构：" + "、".join(sorted(hard)))
    return ActiveLearningPlan("NEED_MORE_SAMPLES" if text else "READY", text, chosen, reasons)
