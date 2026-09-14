from pathlib import Path

import cv2
import numpy as np

from hwplotter.fitting.facsimile import FacsimileConfig, facsimile_character
from hwplotter.model.entities import Character, Stroke
from hwplotter.sample_library import FacsimileLibrary


def _ten_standard():
    return Character(
        "十",
        [
            Stroke(0, np.array([[0.2, 0.5], [0.8, 0.5]], dtype=np.float32)),
            Stroke(1, np.array([[0.5, 0.2], [0.5, 0.8]], dtype=np.float32)),
        ],
        source="test",
    )


def test_facsimile_recovers_observed_centerlines_and_widths():
    size = 256
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.line(mask, (45, 130), (210, 118), 13, 1)
    cv2.line(mask, (132, 42), (125, 218), 11, 1)
    out = facsimile_character(
        _ten_standard(),
        mask,
        FacsimileConfig(image_size=size, points_per_stroke=48, bins=48, assignment_radius_px=40),
    )
    assert out.source == "facsimile"
    assert len(out.strokes) == 2
    assert all(s.widths is not None and len(s.widths) == 48 for s in out.strokes)
    assert float(out.metadata["facsimile_quality"]) > 0.35
    # Recovered horizontal/vertical means should stay close to observed ink centers.
    assert abs(float(np.mean(out.strokes[0].points[:, 1])) - 0.49) < 0.12
    assert abs(float(np.mean(out.strokes[1].points[:, 0])) - 0.50) < 0.12


def test_facsimile_library_replays_real_variant(tmp_path: Path):
    image = tmp_path / "x.png"
    image.write_bytes(b"not-an-image-but-stable-sample-bytes")
    c = _ten_standard()
    c.source = "facsimile"
    c.metadata["facsimile_quality"] = 0.91
    for s in c.strokes:
        s.widths = np.full(len(s.points), 0.03, dtype=np.float32)
    lib = FacsimileLibrary(tmp_path / "lib")
    lib.add("十", image, c)
    replay = lib.replay("十")
    assert replay is not None
    assert replay.source == "facsimile-replay"
    assert replay.strokes[0].widths is not None
    assert lib.replay("未") is None
