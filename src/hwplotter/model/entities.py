from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PointArray = np.ndarray  # shape=(N, 2), float32, normalized to [0, 1]


@dataclass(slots=True)
class Stroke:
    index: int
    points: PointArray
    stroke_type: str = "unknown"
    source_path: str | None = None
    confidence: float = 1.0
    widths: np.ndarray | None = None  # shape=(N,), normalized local ink width

    def copy(self) -> Stroke:
        return Stroke(
            index=self.index,
            points=self.points.copy(),
            stroke_type=self.stroke_type,
            source_path=self.source_path,
            confidence=self.confidence,
            widths=None if self.widths is None else self.widths.copy(),
        )


@dataclass(slots=True)
class Character:
    char: str
    strokes: list[Stroke]
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def stroke_count(self) -> int:
        return len(self.strokes)


@dataclass(slots=True)
class OCRToken:
    text: str
    confidence: float
    polygon: PointArray  # pixel coordinates, shape=(4, 2) generally


@dataclass(slots=True)
class CharacterCrop:
    char: str
    image_path: Path
    bbox: tuple[int, int, int, int]
    confidence: float = 1.0
    polygon: PointArray | None = None  # original-page quad TL,TR,BR,BL
    line_id: int | None = None
    char_position: int | None = None
    connector_to_next: PointArray | None = None  # original-page coordinates


@dataclass(slots=True)
class StyleSample:
    char: str
    standard: Character
    fitted: Character
    image_path: Path | None = None


def all_points(character: Character) -> PointArray:
    chunks: list[np.ndarray] = [s.points for s in character.strokes if len(s.points)]
    if not chunks:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def iter_strokes(chars: Iterable[Character]) -> Iterable[Stroke]:
    for char in chars:
        yield from char.strokes
