import numpy as np

from hwplotter.exporters.gcode import to_gcode
from hwplotter.model.entities import Character, Stroke


def test_gcode_pen_sequence():
    c = Character("一", [Stroke(0, np.array([[0.1, 0.2], [0.9, 0.2]], np.float32))])
    g = to_gcode(c)
    assert "G21" in g
    assert "M3 S1000" in g
    assert "G1 X" in g


def test_dynamic_pressure_uses_recovered_widths():
    import numpy as np

    from hwplotter.exporters.gcode import GCodeOptions, to_gcode
    from hwplotter.model.entities import Character, Stroke

    c = Character(
        "一",
        [
            Stroke(
                0,
                np.array([[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]], dtype=np.float32),
                widths=np.array([0.01, 0.05, 0.02], dtype=np.float32),
            )
        ],
    )
    text = to_gcode(c, GCodeOptions(dynamic_pressure=True, pressure_min=500, pressure_max=1000))
    assert "M3 S" in text
    assert text.count("M3 S") >= 3  # initial pen-up plus pressure commands
