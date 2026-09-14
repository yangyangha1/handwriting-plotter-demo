from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from hwplotter.style.coverage import CoverageReport


@dataclass(slots=True)
class TrainingThresholds:
    min_unique_characters: int = 50
    min_strokes: int = 300
    min_component_coverage: float = 0.65
    min_geometry_coverage: float = 0.70
    min_tiny_points: int = 2000
    min_structure_types: int = 4
    min_position_types: int = 4
    min_adjacency_contexts: int = 12


@dataclass(slots=True)
class TrainingReadiness:
    status: str
    ready: bool
    global_ready: bool
    component_ready: bool
    context_ready: bool
    tiny_ready: bool
    reasons: list[str]
    required_more_text: str
    thresholds: dict[str, float | int]
    active_model_version: int = 0
    validation_error: float | None = None
    validation_ready: bool = False

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TrainingReadiness:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def assess_readiness(
    report: CoverageReport, thresholds: TrainingThresholds | None = None
) -> TrainingReadiness:
    t = thresholds or TrainingThresholds()
    reasons: list[str] = []
    global_ready = (
        report.unique_characters >= t.min_unique_characters
        and report.stroke_count_seen >= t.min_strokes
    )
    component_ready = (
        report.component_mode and report.component_coverage >= t.min_component_coverage
    )
    learned_structure_types = sum(
        1 for _, count in report.structure_occurrences.items() if count >= 2
    )
    learned_position_types = sum(
        1
        for pos, count in report.position_occurrences.items()
        if pos in {"left", "right", "top", "bottom"} and count >= 3
    )
    context_ready = (
        global_ready
        and component_ready
        and report.geometry_coverage >= t.min_geometry_coverage
        and learned_structure_types >= t.min_structure_types
        and learned_position_types >= t.min_position_types
        and report.adjacency_contexts >= t.min_adjacency_contexts
    )
    point_count = report.stroke_count_seen * 64
    tiny_ready = context_ready and point_count >= t.min_tiny_points

    if report.unique_characters < t.min_unique_characters:
        reasons.append(f"不同汉字 {report.unique_characters}/{t.min_unique_characters}")
    if report.stroke_count_seen < t.min_strokes:
        reasons.append(f"有效笔画 {report.stroke_count_seen}/{t.min_strokes}")
    if not report.component_mode:
        reasons.append("缺少 dictionary.txt，无法建立偏旁/位置/邻接模型")
    elif report.component_coverage < t.min_component_coverage:
        reasons.append(
            f"核心偏旁覆盖 {report.component_coverage:.1%}/{t.min_component_coverage:.0%}"
        )
    if report.geometry_coverage < t.min_geometry_coverage:
        reasons.append(f"笔画几何覆盖 {report.geometry_coverage:.1%}/{t.min_geometry_coverage:.0%}")
    if learned_structure_types < t.min_structure_types:
        reasons.append(f"主要汉字结构类型 {learned_structure_types}/{t.min_structure_types}")
    if learned_position_types < t.min_position_types:
        reasons.append(f"组件位置类型 {learned_position_types}/{t.min_position_types}")
    if report.adjacency_contexts < t.min_adjacency_contexts:
        reasons.append(f"邻接上下文 {report.adjacency_contexts}/{t.min_adjacency_contexts}")
    if point_count < t.min_tiny_points:
        reasons.append(f"Tiny模型训练点约 {point_count}/{t.min_tiny_points}")

    ready = global_ready and component_ready and context_ready and tiny_ready
    return TrainingReadiness(
        status="READY" if ready else "NEED_MORE_SAMPLES",
        ready=ready,
        global_ready=global_ready,
        component_ready=component_ready,
        context_ready=context_ready,
        tiny_ready=tiny_ready,
        reasons=reasons,
        required_more_text=report.suggested_text,
        thresholds=asdict(t),
    )
