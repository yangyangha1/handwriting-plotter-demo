from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hwplotter.model.entities import Character, Stroke


def save_json(character: Character, path: Path) -> None:
    obj = {
        "char": character.char,
        "source": character.source,
        "metadata": character.metadata,
        "strokes": [
            {
                "index": s.index,
                "stroke_type": s.stroke_type,
                "confidence": s.confidence,
                "points": s.points.round(6).tolist(),
                "widths": None if s.widths is None else s.widths.round(6).tolist(),
            }
            for s in character.strokes
        ],
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Character:
    obj = json.loads(path.read_text(encoding="utf-8"))
    strokes = []
    for row in obj["strokes"]:
        widths = row.get("widths")
        strokes.append(
            Stroke(
                index=int(row["index"]),
                points=np.asarray(row["points"], dtype=np.float32),
                stroke_type=row.get("stroke_type", "unknown"),
                confidence=float(row.get("confidence", 1.0)),
                widths=None if widths is None else np.asarray(widths, dtype=np.float32),
            )
        )
    return Character(
        char=obj["char"],
        strokes=strokes,
        source=obj.get("source", "json"),
        metadata=dict(obj.get("metadata", {})),
    )
