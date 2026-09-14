import json
from pathlib import Path

import numpy as np
import pytest

from hwplotter.model.entities import Character, Stroke, StyleSample
from hwplotter.pipeline import hierarchical_synthesize
from hwplotter.style.context_models import classify_position, learn_context_models
from hwplotter.style.coverage import CoverageReport
from hwplotter.style.profile import StyleProfile
from hwplotter.style.readiness import TrainingReadiness, TrainingThresholds, assess_readiness


def _char(ch: str, shift=0.0):
    s0 = Stroke(0, np.array([[0.15 + shift, 0.20], [0.25 + shift, 0.80]], np.float32))
    s1 = Stroke(1, np.array([[0.55 + shift, 0.30], [0.85 + shift, 0.30]], np.float32))
    return Character(ch, [s0, s1])


def _dictionary(tmp_path: Path) -> Path:
    p = tmp_path / "dictionary.txt"
    rows = [
        {"character": "什", "decomposition": "⿰亻十", "radical": "亻", "matches": [[0], [1]]},
        {"character": "休", "decomposition": "⿰亻木", "radical": "亻", "matches": [[0], [1]]},
    ]
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    return p


def test_position_classifier():
    assert classify_position((0.05, 0.1, 0.30, 0.9)) == "left"
    assert classify_position((0.70, 0.1, 0.95, 0.9)) == "right"
    assert classify_position((0.2, 0.05, 0.8, 0.30)) == "top"


def test_context_models_learn_position_structure_adjacency(tmp_path):
    d = _dictionary(tmp_path)
    samples = []
    for k in range(3):
        std = _char("什")
        fit = _char("什", shift=0.01 * (k + 1))
        samples.append(StyleSample("什", std, fit))
    pos, structure, adjacency = learn_context_models(samples, d, min_samples=2)
    assert any(key.startswith("亻@left") for key in pos.styles)
    assert "⿰" in structure.styles
    assert any("|op:⿰" in key for key in adjacency.styles)


def _report(unique=10, strokes=80, comp=0.2, geom=0.4):
    return CoverageReport(
        sample_characters=unique,
        unique_characters=unique,
        stroke_count_seen=strokes,
        component_mode=True,
        covered_components=[],
        missing_components=["亻"],
        insufficient_components=[],
        component_occurrences={},
        aligned_component_strokes=strokes,
        component_coverage=comp,
        geometry_coverage=geom,
        overall_completion=0.3,
        suggested_text="你他们",
        suggested_characters=list("你他们"),
        notes=[],
        structure_occurrences={"⿰": 3, "⿱": 3, "⿴": 2, "⿵": 2},
        position_occurrences={"left": 3, "right": 3, "top": 3, "bottom": 3},
        adjacency_contexts=15,
        context_suggested_text="",
    )


def test_readiness_requires_more_samples():
    r = assess_readiness(_report())
    assert not r.ready
    assert r.status == "NEED_MORE_SAMPLES"
    assert r.required_more_text == "你他们"
    assert r.reasons


def test_readiness_can_be_ready_with_thresholds():
    r = assess_readiness(_report(unique=60, strokes=400, comp=0.8, geom=0.9), TrainingThresholds())
    assert r.ready
    assert r.tiny_ready


def test_hierarchical_generation_blocks_incomplete(tmp_path):
    # Gate is checked before MMH files are read, so an incomplete model must fail fast.
    profile = StyleProfile([[0, 0]] * 64, [[0, 0]] * 64, 0.0, 1.0, 1, 64)
    readiness = TrainingReadiness(
        "NEED_MORE_SAMPLES", False, False, False, False, False, ["样本不足"], "你他", {}
    )
    with pytest.raises(RuntimeError, match="NEED_MORE_SAMPLES"):
        hierarchical_synthesize(
            "休",
            tmp_path / "graphics.txt",
            profile,
            tmp_path / "dictionary.txt",
            readiness=readiness,
        )
