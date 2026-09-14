from pathlib import Path

from hwplotter.model.session import ReviewItem, ReviewSession


def test_rejected_items_are_queued_for_rewrite_without_relabel(tmp_path: Path):
    items = [
        ReviewItem(
            0, "你", tmp_path / "a.png", (0, 0, 10, 10), 0.9, included=True, original_char="你"
        ),
        ReviewItem(
            1,
            "好",
            tmp_path / "b.png",
            (10, 0, 20, 10),
            0.8,
            included=False,
            original_char="好",
            rejection_reason="manual_exclude_rewrite",
        ),
        ReviewItem(
            2,
            "字",
            tmp_path / "c.png",
            (20, 0, 30, 10),
            0.7,
            included=True,
            original_char="字",
            error="fit failed",
        ),
        ReviewItem(
            3, "好", tmp_path / "d.png", (30, 0, 40, 10), 0.7, included=False, original_char="好"
        ),
    ]
    s = ReviewSession(tmp_path / "page.png", tmp_path / "session", items)
    assert [x.char for x in s.accepted()] == ["你", "字"]
    assert s.rewrite_text() == "好字"
    assert items[1].char == items[1].original_char == "好"
