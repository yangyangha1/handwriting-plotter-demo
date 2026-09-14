import numpy as np

from hwplotter.geometry import apply_transform, resample_polyline, similarity_transform


def test_resample_shape():
    p = np.array([[0, 0], [1, 0], [1, 1]], np.float32)
    q = resample_polyline(p, 17)
    assert q.shape == (17, 2)
    assert np.allclose(q[0], p[0])
    assert np.allclose(q[-1], p[-1])


def test_similarity_transform():
    src = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    dst = src * 2 + np.array([3, 4], np.float32)
    a, t = similarity_transform(src, dst)
    pred = apply_transform(src, a, t)
    assert np.allclose(pred, dst, atol=1e-5)
