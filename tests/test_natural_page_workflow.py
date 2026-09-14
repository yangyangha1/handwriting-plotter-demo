from pathlib import Path

import cv2
import numpy as np

from hwplotter.model.entities import OCRToken
from hwplotter.workflow import create_ocr_session, save_review_manifest


class FakeNaturalPageOCR:
    def recognize(self, image: Path):
        return [
            OCRToken(
                "天地", 0.95, np.array([[35, 30], [260, 38], [258, 125], [32, 118]], np.float32)
            ),
            OCRToken(
                "玄黄", 0.93, np.array([[55, 145], [315, 151], [312, 242], [51, 235]], np.float32)
            ),
        ]


def test_normal_multiline_handwritten_page_requires_no_grid_or_manual_character_boxes(
    tmp_path: Path,
):
    page = np.full((280, 360, 3), 255, np.uint8)
    # Synthetic ink with irregular spacing on two normal text lines.
    for x in (55, 185):
        cv2.line(page, (x, 55), (x + 55, 100), (0, 0, 0), 7)
        cv2.line(page, (x + 30, 40), (x + 22, 115), (0, 0, 0), 6)
    for x in (75, 220):
        cv2.line(page, (x, 170), (x + 60, 218), (0, 0, 0), 7)
        cv2.line(page, (x + 28, 158), (x + 24, 230), (0, 0, 0), 6)
    image = tmp_path / "normal_page.png"
    cv2.imwrite(str(image), page)

    session = create_ocr_session(image, tmp_path / "session", ocr_engine=FakeNaturalPageOCR())
    assert [x.char for x in session.items] == list("天地玄黄")
    assert len(session.items) == 4
    assert all(x.image_path.exists() for x in session.items)

    # Reject instead of relabelling: the OCR-derived label remains immutable and
    # the rejected character is explicitly queued for the user to rewrite.
    session.items[1].included = False
    session.items[1].rejection_reason = "manual_exclude_rewrite"
    save_review_manifest(session)
    assert session.items[1].char == "地"
    assert (session.work_dir / "rewrite_required.txt").read_text(encoding="utf-8") == "地"
