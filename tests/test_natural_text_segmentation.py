from pathlib import Path

import cv2
import numpy as np

from hwplotter.model.entities import OCRToken
from hwplotter.ocr.cropper import split_token_to_char_crops


def _page_with_irregular_glyphs(path: Path, angle_deg: float = -8.0):
    canvas = np.full((260, 760, 3), 255, np.uint8)
    # Four synthetic "handwritten characters" with deliberately irregular widths/gaps.
    x_ranges = [(70, 150), (185, 300), (342, 420), (485, 625)]
    for i, (a, b) in enumerate(x_ranges):
        cv2.line(canvas, (a + 8, 105 + i * 2), (b - 10, 150 - i), (0, 0, 0), 9)
        cv2.line(canvas, ((a + b) // 2, 78), ((a + b) // 2 - 8, 181), (0, 0, 0), 7)
    center = (canvas.shape[1] / 2, canvas.shape[0] / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]), borderValue=(255, 255, 255))
    cv2.imwrite(str(path), rot)
    # Transform a loose source quad around the unrotated text band.
    q = np.array([[[50, 55], [650, 55], [650, 195], [50, 195]]], np.float32)
    qh = np.concatenate([q[0], np.ones((4, 1), np.float32)], axis=1)
    qq = (M @ qh.T).T.astype(np.float32)
    return qq


def test_normal_handwriting_photo_is_rectified_and_split_without_grid(tmp_path: Path):
    image = tmp_path / "page.png"
    poly = _page_with_irregular_glyphs(image)
    token = OCRToken(text="天地玄黄", confidence=0.96, polygon=poly)
    crops = split_token_to_char_crops(image, token, tmp_path / "crops")
    assert [c.char for c in crops] == list("天地玄黄")
    assert len(crops) == 4
    assert all(c.image_path.exists() for c in crops)
    # Natural spacing was intentionally not equal; the splitter must not require equal cells.
    widths = [c.bbox[2] - c.bbox[0] for c in crops]
    assert max(widths) - min(widths) > 10
    assert all(0.0 < c.confidence <= 0.96 for c in crops)


def test_punctuation_has_narrower_soft_width_prior(tmp_path: Path):
    page = np.full((160, 500, 3), 255, np.uint8)
    cv2.rectangle(page, (40, 40), (120, 125), (0, 0, 0), 5)
    cv2.circle(page, (165, 115), 5, (0, 0, 0), -1)
    cv2.rectangle(page, (210, 42), (305, 127), (0, 0, 0), 5)
    cv2.imwrite(str(tmp_path / "p.png"), page)
    token = OCRToken(
        "你，好", 0.9, np.array([[30, 30], [330, 30], [330, 140], [30, 140]], np.float32)
    )
    crops = split_token_to_char_crops(tmp_path / "p.png", token, tmp_path / "c")
    assert len(crops) == 3
    # punctuation should not be forced to a full CJK cell by the prior
    ws = [c.bbox[2] - c.bbox[0] for c in crops]
    assert ws[1] < max(ws[0], ws[2])
