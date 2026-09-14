from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class ReviewItem:
    index: int
    char: str
    image_path: Path
    bbox: tuple[int, int, int, int]
    confidence: float
    included: bool = True
    original_char: str | None = None
    vector_svg: Path | None = None
    vector_json: Path | None = None
    error: str | None = None
    facsimile_quality: float | None = None
    rejection_reason: str | None = None
    polygon: np.ndarray | None = None
    line_id: int | None = None
    char_position: int | None = None
    stroke_order_confidence: float | None = None
    stroke_order_mode: str | None = None
    connector_to_next: np.ndarray | None = None

    @property
    def effective_char(self) -> str:
        return self.char.strip()[:1]


@dataclass(slots=True)
class ReviewSession:
    source_image: Path
    work_dir: Path
    items: list[ReviewItem] = field(default_factory=list)

    def accepted(self) -> list[ReviewItem]:
        return [x for x in self.items if x.included and x.effective_char]

    def rejected(self) -> list[ReviewItem]:
        return [x for x in self.items if not x.included or x.error]

    def rewrite_text(self) -> str:
        """Characters that must be written again because the image/OCR/crop was rejected.

        We intentionally keep the OCR label immutable during review.  A bad candidate is
        excluded rather than manually relabelled, so no human-corrected pseudo-label can
        enter the facsimile/style training set.
        """
        seen: set[str] = set()
        chars: list[str] = []
        for item in self.items:
            if item.included and not item.error:
                continue
            ch = (item.original_char or item.char).strip()[:1]
            if ch and ch not in seen:
                seen.add(ch)
                chars.append(ch)
        return "".join(chars)
