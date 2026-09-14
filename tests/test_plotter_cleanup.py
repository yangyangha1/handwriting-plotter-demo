import cv2
import numpy as np

from hwplotter.plotter_paths import join_continuations, smooth_path
from hwplotter.vision.grid import remove_frame_rules


def test_staircase_removed_without_moving_endpoints():
    x = np.arange(200, dtype=float)
    p = np.column_stack([x, np.round(x * 0.32)])
    q = np.asarray(smooth_path(p))
    assert len(q) < len(p) / 3
    np.testing.assert_array_equal(q[[0, -1]], p[[0, -1]])
    assert np.max(np.abs(q[:, 1] - q[:, 0] * 0.32)) < 1


def test_fold_loop_and_disconnected_strokes_preserved():
    p = [[0, 0], [30, 0], [30, 30]]
    assert smooth_path(p) == p
    loop = smooth_path([[0, 0], [30, 0], [30, 30], [0, 30], [0, 0]])
    assert loop[0] == loop[-1]
    cross = [[[0, 10], [10, 10]], [[10, 10], [20, 10]], [[10, 0], [10, 10]], [[10, 10], [10, 20]]]
    joined = join_continuations(cross)
    assert len(joined) == 2
    assert len(join_continuations([[[0, 0], [10, 0]], [[11, 0], [20, 0]]])) == 2


def test_frame_cleanup_keeps_central_cross_and_box_character():
    m = np.zeros((100, 100), np.uint8)
    cv2.rectangle(m, (2, 2), (97, 97), 1, 1)
    cv2.line(m, (20, 50), (80, 50), 1, 3)
    cv2.line(m, (50, 20), (50, 80), 1, 3)
    clean = remove_frame_rules(m)
    assert not clean[2, 20:80].any()
    assert clean[50, 20:81].all()
    box = np.zeros_like(m)
    cv2.rectangle(box, (20, 20), (80, 80), 1, 3)
    np.testing.assert_array_equal(remove_frame_rules(box), box)
