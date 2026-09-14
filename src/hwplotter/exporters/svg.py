from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from hwplotter.model.entities import Character


def character_svg(character: Character, size: int = 512, stroke_width: float = 4.0) -> str:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    order = character.metadata.get("stroke_order")
    draw_order = order if isinstance(order, list) else list(range(len(character.strokes)))
    for idx in draw_order:
        stroke = character.strokes[int(idx)]
        if not len(stroke.points):
            continue
        if (
            stroke.widths is not None
            and len(stroke.widths) == len(stroke.points)
            and len(stroke.points) > 1
        ):
            for i in range(len(stroke.points) - 1):
                a, b = stroke.points[i], stroke.points[i + 1]
                sw = max(0.6, float((stroke.widths[i] + stroke.widths[i + 1]) * 0.5 * size))
                lines.append(
                    f'<line x1="{a[0] * size:.2f}" y1="{a[1] * size:.2f}" '
                    f'x2="{b[0] * size:.2f}" y2="{b[1] * size:.2f}" fill="none" stroke="black" '
                    f'stroke-width="{sw:.2f}" stroke-linecap="round"/>'
                )
        else:
            pts = " ".join(f"{x * size:.2f},{y * size:.2f}" for x, y in stroke.points)
            lines.append(
                f'<polyline points="{escape(pts)}" fill="none" stroke="black" '
                f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
    lines.append("</svg>")
    return "\n".join(lines)


def save_svg(character: Character, path: Path, size: int = 512) -> None:
    path.write_text(character_svg(character, size=size), encoding="utf-8")
