from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hwplotter.geometry import smooth_polyline
from hwplotter.model.entities import Character, Stroke


class MakeMeAHanziSource:
    """Reader for Make Me A Hanzi graphics.txt JSON-lines format.

    MMH medians use a 1024-ish coordinate system with y increasing upward.
    We normalize into image coordinates [0,1] with y increasing downward.
    """

    def __init__(self, graphics_path: Path):
        self.graphics_path = Path(graphics_path)
        self._index: dict[str, dict] | None = None

    def _load(self) -> None:
        idx: dict[str, dict] = {}
        with self.graphics_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx[obj["character"]] = obj
        self._index = idx

    def get(self, char: str, points_per_stroke: int = 64) -> Character:
        if self._index is None:
            self._load()
        assert self._index is not None
        if char not in self._index:
            raise KeyError(f"Character not found in MMH: {char}")
        obj = self._index[char]
        strokes: list[Stroke] = []
        paths = obj.get("strokes", [])
        for i, median in enumerate(obj["medians"]):
            p = np.asarray(median, dtype=np.float32)
            # MMH: display transform is y_img = 900 - y.
            p[:, 0] = p[:, 0] / 1024.0
            p[:, 1] = (900.0 - p[:, 1]) / 1024.0
            p = np.clip(p, 0.0, 1.0)
            p = smooth_polyline(p, n=points_per_stroke, smooth=0.0002)
            strokes.append(
                Stroke(
                    index=i,
                    points=p,
                    source_path=paths[i] if i < len(paths) else None,
                )
            )
        return Character(char=char, strokes=strokes, source="makemeahanzi")
