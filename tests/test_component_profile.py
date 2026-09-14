import json
from pathlib import Path

import numpy as np

from hwplotter.model.entities import Character, Stroke, StyleSample
from hwplotter.style.component_profile import learn_component_styles


def _stroke(i, p):
    return Stroke(i, np.asarray(p, dtype=np.float32))


def _sample(char: str, offset: float) -> StyleSample:
    standard = Character(
        char,
        [
            _stroke(0, [[0.15, 0.15], [0.25, 0.45]]),
            _stroke(1, [[0.28, 0.20], [0.28, 0.75]]),
            _stroke(2, [[0.55, 0.35], [0.85, 0.35]]),
            _stroke(3, [[0.70, 0.20], [0.70, 0.70]]),
        ],
    )
    fitted = Character(
        char,
        [
            _stroke(0, [[0.15 + offset, 0.15], [0.27 + offset, 0.45]]),
            _stroke(1, [[0.28 + offset, 0.20], [0.30 + offset, 0.75]]),
            _stroke(2, [[0.55, 0.35], [0.85, 0.36]]),
            _stroke(3, [[0.70, 0.20], [0.70, 0.70]]),
        ],
    )
    return StyleSample(char=char, standard=standard, fitted=fitted)


def test_component_model_uses_exact_owned_strokes(tmp_path: Path):
    dictionary = tmp_path / "dictionary.txt"
    dictionary.write_text(
        json.dumps(
            {
                "character": "什",
                "decomposition": "⿰亻十",
                "radical": "亻",
                "matches": [[0], [0], [1], [1]],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    model = learn_component_styles(
        [_sample("什", 0.01), _sample("什", 0.02)],
        dictionary,
        n=16,
        min_samples=2,
    )
    assert model.aligned_characters == 2
    assert model.aligned_strokes == 8
    assert set(model.components["亻"].stroke_slots) == {0, 1}
    assert set(model.components["十"].stroke_slots) == {0, 1}
    assert model.components["亻"].stroke_slots[0].samples == 2


from hwplotter.style.component_profile import (
    ComponentStrokeStyle,
    ComponentStyle,
    ComponentStyleModel,
    apply_component_conditioning,
)
from hwplotter.style.profile import StyleProfile


def test_component_conditioning_only_changes_owned_strokes(tmp_path: Path):
    dictionary = tmp_path / "dictionary.txt"
    dictionary.write_text(
        json.dumps(
            {
                "character": "什",
                "decomposition": "⿰亻十",
                "radical": "亻",
                "matches": [[0], [0], [1], [1]],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    standard = _sample("什", 0.0).standard
    n = 16
    zero = np.zeros((n, 2), dtype=np.float32)
    shift = zero.copy()
    shift[:, 0] = 0.04
    global_profile = StyleProfile(
        residual_mean=zero.tolist(),
        residual_std=zero.tolist(),
        slant_mean_deg=0.0,
        scale_mean=1.0,
        samples=1,
        points_per_stroke=n,
    )
    cstyle = ComponentStrokeStyle(
        samples=3,
        residual_mean=shift.tolist(),
        residual_std=zero.tolist(),
        slant_mean_deg=0.0,
        scale_mean=1.0,
    )
    model = ComponentStyleModel(
        points_per_stroke=n,
        components={"亻": ComponentStyle(occurrences=3, stroke_slots={0: cstyle, 1: cstyle})},
    )
    out = apply_component_conditioning(
        standard,
        global_profile,
        model,
        dictionary,
        blend=1.0,
        variation=0.0,
    )
    assert np.mean(out.strokes[0].points[:, 0]) > np.mean(standard.strokes[0].points[:, 0]) + 0.02
    assert np.mean(out.strokes[1].points[:, 0]) > np.mean(standard.strokes[1].points[:, 0]) + 0.02
    # The 十 strokes have no component model and should stay on the global baseline.
    assert (
        abs(np.mean(out.strokes[2].points[:, 0]) - np.mean(standard.strokes[2].points[:, 0])) < 0.01
    )
    assert (
        abs(np.mean(out.strokes[3].points[:, 0]) - np.mean(standard.strokes[3].points[:, 0])) < 0.01
    )
