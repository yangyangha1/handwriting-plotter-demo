from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hwplotter.exporters.page_svg import _map_points_to_layout
from hwplotter.fitting.stroke_order import build_pen_paths


@dataclass(slots=True)
class PageGCodeOptions:
    page_width_mm: float = 210.0
    travel_feed: int = 4000
    draw_feed: int = 1800
    pen_up: str = "M3 S0"
    pen_down: str = "M3 S1000"
    invert_y: bool = False
    connector_join_tolerance_px: float = 28.0


def to_page_gcode(
    characters: list[tuple], canvas_size: tuple[int, int], opts: PageGCodeOptions | None = None
) -> str:
    """Write photographed text while preserving page spacing and light cross-glyph joins.

    When a recovered inter-character connector is present, the last physical pen path
    of the current glyph, the connector, and the first physical path of the next glyph
    are emitted as one pen-down action if their endpoints are spatially compatible.
    """
    o = opts or PageGCodeOptions()
    cw, ch = canvas_size
    scale = o.page_width_mm / max(float(cw), 1.0)
    lines = ["G21 ; millimeters", "G90 ; absolute positioning", o.pen_up]

    entries = []
    for entry in characters:
        if len(entry) == 2:
            character, layout = entry
            connector = None
        else:
            character, layout, connector = entry[:3]
        paths = [_map_points_to_layout(p, layout) for p in build_pen_paths(character)]
        entries.append(
            (
                character,
                paths,
                None if connector is None else np.asarray(connector, dtype=np.float32),
            )
        )

    def to_mm(page_pts: np.ndarray) -> np.ndarray:
        pts = page_pts.astype(np.float32).copy()
        if o.invert_y:
            pts[:, 1] = float(ch) - pts[:, 1]
        return pts * scale

    def emit(page_pts: np.ndarray) -> None:
        if len(page_pts) < 2:
            return
        pts = to_mm(page_pts)
        x0, y0 = pts[0]
        lines.append(f"G0 X{x0:.3f} Y{y0:.3f} F{o.travel_feed}")
        lines.append(o.pen_down)
        for x, y in pts[1:]:
            lines.append(f"G1 X{x:.3f} Y{y:.3f} F{o.draw_feed}")
        lines.append(o.pen_up)

    skip_first: set[int] = set()
    for i, (character, paths, connector) in enumerate(entries):
        start = 1 if i in skip_first else 0
        if not paths:
            continue
        # All paths except the last are independent pen-down groups.
        for p in paths[start:-1]:
            emit(p)
        last = paths[-1] if start < len(paths) else None
        if last is None:
            continue
        joined = False
        if connector is not None and len(connector) >= 2 and i + 1 < len(entries):
            next_char, next_paths, _ = entries[i + 1]
            same_line = character.metadata.get("line_id") == next_char.metadata.get("line_id")
            if next_paths and same_line:
                a = float(np.linalg.norm(last[-1] - connector[0]))
                b = float(np.linalg.norm(connector[-1] - next_paths[0][0]))
                if a <= o.connector_join_tolerance_px and b <= o.connector_join_tolerance_px:
                    merged = np.concatenate([last, connector[1:], next_paths[0][1:]], axis=0)
                    emit(merged)
                    skip_first.add(i + 1)
                    joined = True
        if not joined:
            emit(last)
            if connector is not None and len(connector) >= 2:
                emit(connector)
    return "\n".join(lines) + "\n"


def save_page_gcode(
    characters: list[tuple],
    canvas_size: tuple[int, int],
    path: Path,
    opts: PageGCodeOptions | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_page_gcode(characters, canvas_size, opts), encoding="utf-8")
