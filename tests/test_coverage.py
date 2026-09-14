from pathlib import Path

from hwplotter.style.coverage import analyze_coverage


def test_coverage_without_dictionary():
    p = Path(__file__).parents[1] / "data" / "sample_graphics.txt"
    report = analyze_coverage(["十"], p)
    assert report.sample_characters == 1
    assert 0 <= report.overall_completion <= 1
    assert report.component_mode is False
