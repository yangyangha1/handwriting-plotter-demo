from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hwplotter.model.entities import Character


@dataclass(slots=True)
class GCodeOptions:
    char_size_mm: float = 20.0
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0
    travel_feed: int = 4000
    draw_feed: int = 1800
    min_draw_feed: int = 900
    pen_up: str = "M3 S0"
    pen_down: str = "M3 S1000"
    invert_y: bool = False
    dynamic_pressure: bool = False
    pressure_min: int = 650
    pressure_max: int = 1000
    dynamic_speed: bool = False


def _pressure_commands(widths: np.ndarray, o: GCodeOptions) -> np.ndarray:
    w = np.asarray(widths, dtype=np.float32)
    lo, hi = np.quantile(w, [0.10, 0.90]) if len(w) > 3 else (float(w.min()), float(w.max()))
    norm = (w - lo) / max(float(hi - lo), 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    return np.rint(o.pressure_min + norm * (o.pressure_max - o.pressure_min)).astype(int)


def _curvature_feed(points: np.ndarray, o: GCodeOptions) -> np.ndarray:
    feed = np.full(len(points), o.draw_feed, dtype=np.int32)
    if len(points) < 3:
        return feed
    d1 = np.diff(points, axis=0)
    a = np.arctan2(d1[:, 1], d1[:, 0])
    turn = np.abs(np.diff(np.unwrap(a)))
    turn = np.pad(turn, (1, 1), mode="edge")[: len(points)]
    # Slow down at corners/turns, preserve normal feed on straight segments.
    strength = np.clip(turn / (np.pi / 2), 0.0, 1.0)
    feed = np.rint(o.draw_feed - strength * (o.draw_feed - o.min_draw_feed)).astype(np.int32)
    return feed


def _xy_mm(points: np.ndarray, o: GCodeOptions) -> np.ndarray:
    xy = points.astype(np.float32).copy()
    if o.invert_y:
        xy[:, 1] = 1.0 - xy[:, 1]
    xy[:, 0] = o.origin_x_mm + xy[:, 0] * o.char_size_mm
    xy[:, 1] = o.origin_y_mm + xy[:, 1] * o.char_size_mm
    return xy


def _emit_stroke_motion(lines: list[str], stroke, o: GCodeOptions, pen_already_down: bool) -> None:
    if not len(stroke.points):
        return
    xy_norm = stroke.points
    xy = _xy_mm(xy_norm, o)
    pressures = None
    if (
        o.dynamic_pressure
        and stroke.widths is not None
        and len(stroke.widths) == len(stroke.points)
    ):
        pressures = _pressure_commands(stroke.widths, o)
        if not pen_already_down:
            lines.append(f"M3 S{pressures[0]}")
    elif not pen_already_down:
        lines.append(o.pen_down)
    feeds = _curvature_feed(xy_norm, o) if o.dynamic_speed else np.full(len(xy), o.draw_feed)
    last_pressure = pressures[0] if pressures is not None else None
    for i, (x, y) in enumerate(xy[1:], start=1):
        if (
            pressures is not None
            and last_pressure is not None
            and abs(int(pressures[i]) - int(last_pressure)) >= 12
        ):
            last_pressure = int(pressures[i])
            lines.append(f"M3 S{last_pressure}")
        lines.append(f"G1 X{x:.3f} Y{y:.3f} F{int(feeds[i])}")


def to_gcode(character: Character, opts: GCodeOptions | None = None) -> str:
    """Emit physical writing order, including inferred cursive pen-down groups."""
    o = opts or GCodeOptions()
    lines = ["G21 ; millimeters", "G90 ; absolute positioning", o.pen_up]
    groups = character.metadata.get("pen_down_groups")
    order = character.metadata.get("stroke_order")
    if not isinstance(groups, list):
        groups = [
            [int(i)] for i in (order if isinstance(order, list) else range(len(character.strokes)))
        ]
    for group in groups:
        valid = [
            character.strokes[int(i)]
            for i in group
            if 0 <= int(i) < len(character.strokes) and len(character.strokes[int(i)].points)
        ]
        if not valid:
            continue
        first_xy = _xy_mm(valid[0].points, o)[0]
        lines.append(f"G0 X{first_xy[0]:.3f} Y{first_xy[1]:.3f} F{o.travel_feed}")
        pen_down = False
        previous_end = None
        for stroke in valid:
            if previous_end is not None:
                target = _xy_mm(stroke.points, o)[0]
                # A pen-down bridge is intentional: running/cursive merged strokes
                # must not acquire an artificial lift between canonical MMH strokes.
                lines.append(
                    f"G1 X{target[0]:.3f} Y{target[1]:.3f} F{o.min_draw_feed if o.dynamic_speed else o.draw_feed}"
                )
            _emit_stroke_motion(lines, stroke, o, pen_already_down=pen_down)
            pen_down = True
            previous_end = stroke.points[-1]
        lines.append(o.pen_up)
    return "\n".join(lines) + "\n"


def save_gcode(character: Character, path: Path, opts: GCodeOptions | None = None) -> None:
    path.write_text(to_gcode(character, opts), encoding="utf-8")
