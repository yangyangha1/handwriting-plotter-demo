import json
from pathlib import Path

import numpy as np
import pytest

from hwplotter.model.continual_learning import (
    ContinualState,
    accept_or_rollback,
    split_by_character,
)
from hwplotter.model.entities import Character, Stroke, StyleSample
from hwplotter.model.structure_network import apply_structure_network, train_structure_network
from hwplotter.style.active_learning import recommend_active_samples
from hwplotter.style.coverage import CoverageReport


def _char(ch: str, scale=1.0, dx=0.0):
    p = np.asarray([[0.2, 0.2], [0.8, 0.8]], np.float32)
    q = (p - 0.5) * scale + 0.5
    q[:, 0] += dx
    return Character(ch, [Stroke(0, q)])


def test_split_by_character_has_no_character_leakage():
    samples = []
    for ch in "天地玄黄宇宙洪荒":
        samples += [StyleSample(ch, _char(ch), _char(ch, 1.02), None) for _ in range(2)]
    tr, va = split_by_character(samples, val_fraction=0.25, seed=1)
    assert {s.char for s in tr}.isdisjoint({s.char for s in va})
    assert va


def test_continual_model_accepts_then_rolls_back(tmp_path: Path):
    model = tmp_path / "style"
    model.mkdir()
    (model / "weights.bin").write_text("good")
    v1 = accept_or_rollback(model, 0.01, 0.01, 10, 2)
    assert v1.accepted
    assert ContinualState.load(model / "continual_state.json").active_version == 1
    (model / "weights.bin").write_text("bad")
    v2 = accept_or_rollback(model, 0.5, 0.5, 12, 3)
    assert not v2.accepted
    assert (model / "weights.bin").read_text() == "good"
    assert ContinualState.load(model / "continual_state.json").active_version == 1


def test_active_learning_targets_missing_components(tmp_path: Path):
    d = tmp_path / "dictionary.txt"
    rows = [
        {
            "character": "休",
            "decomposition": "⿰亻木",
            "radical": "亻",
            "matches": [[0], [0], [1], [1], [1], [1]],
        },
        {
            "character": "河",
            "decomposition": "⿰氵可",
            "radical": "氵",
            "matches": [[0], [0], [0], [1], [1], [1], [1], [1]],
        },
    ]
    d.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    r = CoverageReport(
        1,
        1,
        5,
        True,
        [],
        ["氵"],
        ["亻"],
        {"亻": 1},
        2,
        0.1,
        0.2,
        0.1,
        "",
        [],
        [],
        {"⿰": 1},
        {"left": 1},
        1,
        "",
    )
    plan = recommend_active_samples(r, d, max_chars=4)
    assert "河" in plan.suggested_text


def test_structure_nn_train_and_apply(tmp_path: Path):
    pytest.importorskip("torch")
    d = tmp_path / "dictionary.txt"
    chars = "天地玄黄宇宙"
    d.write_text(
        "\n".join(
            json.dumps({"character": c, "decomposition": c, "matches": [[]]}, ensure_ascii=False)
            for c in chars
        ),
        encoding="utf-8",
    )
    samples = [StyleSample(c, _char(c), _char(c, 1.08, 0.02), None) for c in chars]
    meta = train_structure_network(
        samples[:4], samples[4:], d, tmp_path / "s.pt", tmp_path / "s.json", epochs=5
    )
    assert meta.validation_mse is not None
    out = apply_structure_network(
        samples[4].standard, samples[4].standard, d, tmp_path / "s.pt", tmp_path / "s.json"
    )
    assert out.metadata.get("structure_nn") is True
