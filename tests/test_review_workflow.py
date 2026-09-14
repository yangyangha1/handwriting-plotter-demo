from pathlib import Path

import cv2
import numpy as np

from hwplotter.model.session import ReviewItem, ReviewSession
from hwplotter.workflow import vectorize_reviewed


def test_screened_items_are_not_vectorized(tmp_path: Path):
    # Synthetic cross resembles 十 and exercises the actual fitting pipeline.
    img = np.full((160, 160, 3), 255, np.uint8)
    cv2.line(img, (25, 70), (135, 70), (0, 0, 0), 12)
    cv2.line(img, (80, 20), (80, 145), (0, 0, 0), 12)
    page = tmp_path / "page.png"
    crop1 = tmp_path / "a.png"
    crop2 = tmp_path / "b.png"
    cv2.imwrite(str(page), img)
    cv2.imwrite(str(crop1), img)
    cv2.imwrite(str(crop2), img)

    session = ReviewSession(
        source_image=page,
        work_dir=tmp_path / "session",
        items=[
            ReviewItem(0, "十", crop1, (0, 0, 80, 160), 0.9, included=True),
            ReviewItem(1, "十", crop2, (80, 0, 160, 160), 0.9, included=False),
        ],
    )
    graphics = Path(__file__).parents[1] / "data" / "sample_graphics.txt"
    samples, page_svg = vectorize_reviewed(session, graphics, tmp_path / "vectors")
    assert len(samples) == 1
    assert session.items[0].vector_svg is not None
    assert session.items[1].vector_svg is None
    assert page_svg.exists()
