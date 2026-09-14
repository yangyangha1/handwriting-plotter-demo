import json
from pathlib import Path

from hwplotter.stroke_data.components import (
    MMHDictionarySource,
    extract_components,
    parse_ids,
    resolve_stroke_components,
)
from hwplotter.style.coverage import analyze_coverage


def test_extract_components_and_ids_paths():
    assert extract_components("⿰亻尔", "你") == frozenset({"亻", "尔"})
    tree = parse_ids("⿰亻⿱夂彡")
    assert tree is not None
    assert tree.at_path([0]).symbol == "亻"
    assert tree.at_path([1, 0]).symbol == "夂"
    assert tree.at_path([1, 1]).symbol == "彡"


def test_resolve_stroke_matches():
    owners, grouped = resolve_stroke_components(
        "⿰亻⿱夂彡",
        ((0,), (0,), (1, 0), (1, 0), (1, 1), (1, 1), (1, 1)),
    )
    assert owners[2] == "夂"
    assert grouped["亻"] == (0, 1)
    assert grouped["夂"] == (2, 3)
    assert grouped["彡"] == (4, 5, 6)


def test_dictionary_and_exact_coverage(tmp_path: Path):
    dictionary = tmp_path / "dictionary.txt"
    rows = [
        {"character": "十", "decomposition": "十", "radical": "十", "matches": [[], []]},
        {
            "character": "什",
            "decomposition": "⿰亻十",
            "radical": "亻",
            "matches": [[0], [0], [1], [1]],
        },
    ]
    dictionary.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8"
    )
    source = MMHDictionarySource(dictionary)
    info = source.get("什")
    assert info is not None
    assert info.components == frozenset({"亻", "十"})
    assert info.component_strokes["亻"] == (0, 1)
    assert info.component_strokes["十"] == (2, 3)

    graphics = Path(__file__).parents[1] / "data" / "sample_graphics.txt"
    report = analyze_coverage(
        ["十"],
        graphics,
        dictionary,
        target_components=("十", "亻"),
        suggestion_limit=4,
        min_component_samples=2,
    )
    assert report.component_mode
    assert "亻" in report.missing_components
    assert "十" in report.insufficient_components
    assert "什" in report.suggested_characters
