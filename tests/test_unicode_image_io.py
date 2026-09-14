from pathlib import Path

import numpy as np

from hwplotter.vision.image_io import read_image, write_image


def test_image_io_supports_non_ascii_paths(tmp_path: Path):
    path = tmp_path / "中文路径" / "手写图.png"
    source = np.zeros((12, 18, 3), dtype=np.uint8)
    source[2:10, 4:14] = (240, 240, 240)

    write_image(path, source)
    loaded = read_image(path)

    assert loaded.shape == source.shape
    assert int(loaded[5, 8, 0]) == 240
