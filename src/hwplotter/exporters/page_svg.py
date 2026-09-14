from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np


def _map_points_to_layout(points: np.ndarray, layout) -> np.ndarray:
    if isinstance(layout, tuple) and len(layout) == 4:
        x0, y0, x1, y1 = layout
        out = points.astype(np.float32).copy()
        out[:, 0] = x0 + out[:, 0] * max(1, x1 - x0)
        out[:, 1] = y0 + out[:, 1] * max(1, y1 - y0)
        return out
    poly = np.asarray(layout, dtype=np.float32).reshape(4, 2)
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    h = cv2.getPerspectiveTransform(src, poly)
    return cv2.perspectiveTransform(points.astype(np.float32).reshape(1, -1, 2), h)[0]


def save_character_page_svg(
    characters: list[tuple],
    canvas_size: tuple[int, int],
    path: Path,
    stroke_width: float = 2.0,
) -> None:
    """Export recovered trajectories into their original photographed page layout.

    Each entry may be ``(Character, bbox)`` for backward compatibility or
    ``(Character, quad, connector_to_next)``.  Quads preserve perspective/baseline
    geometry; connector polylines preserve light pen-down bridges between characters.
    Physical pen-down groups are used when available so connected cursive strokes are
    represented as continuous paths rather than artificial pen lifts.
    """
    width, height = canvas_size
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<g fill="none" stroke="black" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for entry in characters:
        if len(entry) == 2:
            character, layout = entry
            connector = None
        else:
            character, layout, connector = entry[:3]
        lines.append(
            f'<g data-char="{escape(character.char)}" data-order="{escape(str(character.metadata.get("stroke_order_mode", "unknown")))}">'
        )
        order = character.metadata.get("stroke_order")
        draw_order = order if isinstance(order, list) else list(range(len(character.strokes)))
        if isinstance(layout, tuple):
            local_scale = float(max(layout[2] - layout[0], layout[3] - layout[1], 1))
        else:
            lp = np.asarray(layout, dtype=np.float32).reshape(4, 2)
            local_scale = float(
                max(float(np.linalg.norm(lp[1] - lp[0])), float(np.linalg.norm(lp[3] - lp[0])), 1.0)
            )
        for idx in draw_order:
            stroke = character.strokes[int(idx)]
            if not len(stroke.points):
                continue
            mapped = _map_points_to_layout(stroke.points, layout)
            if (
                stroke.widths is not None
                and len(stroke.widths) == len(stroke.points)
                and len(mapped) > 1
            ):
                for j in range(len(mapped) - 1):
                    a, b = mapped[j], mapped[j + 1]
                    sw = max(
                        0.45, float((stroke.widths[j] + stroke.widths[j + 1]) * 0.5 * local_scale)
                    )
                    lines.append(
                        f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke-width="{sw:.2f}"/>'
                    )
            else:
                pts = " ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in mapped)
                lines.append(f'<polyline points="{pts}" stroke-width="{stroke_width}"/>')
        if connector is not None and len(connector) >= 2:
            cp = np.asarray(connector, dtype=np.float32)
            pts = " ".join(f"{float(x):.2f},{float(y):.2f}" for x, y in cp)
            lines.append(
                f'<polyline data-inter-char-connector="1" points="{pts}" stroke-width="{stroke_width}"/>'
            )
        lines.append("</g>")
    lines.extend(["</g>", "</svg>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
