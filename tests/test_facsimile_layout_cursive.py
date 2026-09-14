from pathlib import Path

import cv2
import numpy as np

from hwplotter.exporters.page_gcode import to_page_gcode
from hwplotter.exporters.page_svg import save_character_page_svg
from hwplotter.fitting.stroke_order import infer_stroke_order
from hwplotter.model.entities import Character, OCRToken, Stroke
from hwplotter.ocr.cropper import split_token_to_char_crops
from hwplotter.vision.preprocess import frame_square_resize, unframe_points


def test_letterbox_roundtrip_preserves_character_cell_position():
    mask = np.zeros((80, 120), np.uint8)
    mask[20:65, 12:52] = 255  # deliberately left-shifted glyph in its OCR cell
    _, tr = frame_square_resize(mask, size=400, margin=10)
    # Convert known source normalized points into letterbox normalized coordinates,
    # then verify the inverse used by facsimile page replay recovers them.
    src = np.array([[0.10, 0.25], [0.40, 0.75]], np.float32)
    px = src.copy()
    px[:, 0] *= tr["source_w"] - 1
    px[:, 1] *= tr["source_h"] - 1
    px[:, 0] = px[:, 0] * tr["scale"] + tr["offset_x"]
    px[:, 1] = px[:, 1] * tr["scale"] + tr["offset_y"]
    norm = px / (tr["size"] - 1)
    back = unframe_points(norm, tr)
    assert np.allclose(back, src, atol=2e-3)


def test_light_inter_character_connection_is_preserved(tmp_path: Path):
    page = np.full((150, 360, 3), 255, np.uint8)
    cv2.line(page, (50, 45), (110, 95), (0, 0, 0), 7)
    cv2.line(page, (80, 30), (78, 115), (0, 0, 0), 6)
    cv2.line(page, (230, 40), (290, 90), (0, 0, 0), 7)
    cv2.line(page, (260, 30), (258, 115), (0, 0, 0), 6)
    # Thin pen-down bridge between two otherwise distinct characters.
    cv2.line(page, (108, 96), (230, 42), (0, 0, 0), 2)
    image = tmp_path / "connected.png"
    cv2.imwrite(str(image), page)
    token = OCRToken(
        "天地", 0.99, np.array([[30, 20], [310, 20], [310, 125], [30, 125]], np.float32)
    )
    crops = split_token_to_char_crops(image, token, tmp_path / "crops")
    assert len(crops) == 2
    assert crops[0].connector_to_next is not None
    assert len(crops[0].connector_to_next) >= 3


def test_static_image_order_inference_detects_connected_pen_down_group():
    s0 = Stroke(0, np.array([[0.10, 0.50], [0.45, 0.50]], np.float32), confidence=0.95)
    s1 = Stroke(1, np.array([[0.48, 0.50], [0.80, 0.50]], np.float32), confidence=0.95)
    ch = Character("示", [s0, s1])
    mask = np.zeros((100, 100), np.uint8)
    cv2.line(mask, (10, 50), (80, 50), 255, 3)
    result = infer_stroke_order(ch, mask)
    assert result.order == (0, 1)
    assert result.pen_down_groups == ((0, 1),)
    assert not result.ambiguous


def test_image_evidence_can_select_local_cursive_order_change():
    # Standard order is 0,1,2. Geometry strongly supports 0,2,1, a local running/
    # cursive alternative that minimizes pen travel and follows visible ink bridges.
    s0 = Stroke(0, np.array([[0.10, 0.20], [0.30, 0.20]], np.float32), confidence=0.98)
    s1 = Stroke(1, np.array([[0.62, 0.60], [0.85, 0.60]], np.float32), confidence=0.98)
    s2 = Stroke(2, np.array([[0.32, 0.22], [0.60, 0.58]], np.float32), confidence=0.98)
    ch = Character("示", [s0, s1, s2])
    mask = np.zeros((120, 120), np.uint8)
    for s in (s0, s1, s2):
        pts = np.rint(s.points * 119).astype(np.int32)
        cv2.polylines(mask, [pts], False, 255, 3)
    # Visible joins 0->2 and 2->1.
    cv2.line(
        mask,
        tuple(np.rint(s0.points[-1] * 119).astype(int)),
        tuple(np.rint(s2.points[0] * 119).astype(int)),
        255,
        2,
    )
    cv2.line(
        mask,
        tuple(np.rint(s2.points[-1] * 119).astype(int)),
        tuple(np.rint(s1.points[0] * 119).astype(int)),
        255,
        2,
    )
    result = infer_stroke_order(ch, mask)
    assert result.order == (0, 2, 1)
    assert "permutation" in result.mode


def test_page_svg_uses_quad_and_page_gcode_keeps_connector_pen_down(tmp_path: Path):
    a = Character(
        "甲",
        [Stroke(0, np.array([[0.2, 0.5], [0.8, 0.5]], np.float32))],
        metadata={
            "stroke_order": [0],
            "pen_down_groups": [[0]],
            "line_id": 0,
        },
    )
    b = Character(
        "乙",
        [Stroke(0, np.array([[0.2, 0.5], [0.8, 0.5]], np.float32))],
        metadata={
            "stroke_order": [0],
            "pen_down_groups": [[0]],
            "line_id": 0,
        },
    )
    qa = np.array([[10, 10], [100, 15], [95, 100], [5, 95]], np.float32)
    qb = np.array([[120, 14], [210, 18], [205, 103], [115, 99]], np.float32)
    # Connector endpoints are deliberately close to mapped end/start points.
    conn = np.array([[78, 58], [100, 55], [122, 57], [138, 58]], np.float32)
    entries = [(a, qa, conn), (b, qb, None)]
    out = tmp_path / "page.svg"
    save_character_page_svg(entries, (240, 120), out)
    text = out.read_text(encoding="utf-8")
    assert 'data-inter-char-connector="1"' in text
    g = to_page_gcode(entries, (240, 120))
    # One merged A-connector-B pen-down action: only one drawing pen-down command.
    assert g.count("M3 S1000") == 1
