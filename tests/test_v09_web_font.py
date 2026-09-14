import io
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from hwplotter.font_export import save_ttf
from hwplotter.model.continual_learning import accept_or_rollback
from hwplotter.vector_ink import center_paths, gcode, outlines
from hwplotter.web_service import Studio
from hwplotter.webui import make_server


def sample_bytes():
    im = Image.new("RGB", (150, 150), "white")
    d = ImageDraw.Draw(im)
    d.line((30, 75, 120, 75), fill="black", width=9)
    d.line((75, 20, 75, 130), fill="black", width=9)
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def test_outline_holes_and_ttf_raster(tmp_path):
    m = np.zeros((100, 100), np.uint8)
    m[10:90, 10:90] = 1
    m[30:70, 30:70] = 0
    p = outlines(m)
    assert len(p) == 2
    item = {"polygons": [[[x * 8, y * 8] for x, y in r] for r in p], "advance": 1000}
    save_ttf({"口": item}, tmp_path / "test.ttf")
    with TTFont(tmp_path / "test.ttf") as f:
        assert f.getBestCmap()[ord("口")]
    font = ImageFont.truetype(str(tmp_path / "test.ttf"), 100)
    im = Image.new("L", (130, 150), 255)
    ImageDraw.Draw(im).text((0, 0), "口", font=font, fill=0)
    a = np.array(im)
    ys, xs = np.where(a < 128)
    assert len(xs) > 100
    assert a[round((ys.min() + ys.max()) / 2), round((xs.min() + xs.max()) / 2)] == 255


def test_skeleton_loop_and_junction_edges():
    m = np.zeros((30, 30), np.uint8)
    cv2.rectangle(m, (4, 4), (25, 25), 1, 1)
    paths = center_paths(m)
    assert len(paths) == 1 and paths[0][0] == paths[0][-1]
    g = gcode(paths, 30)
    assert g.endswith("M3 S0\n") and "G21" in g
    assert "nan" not in g.lower()


def test_unvalidated_model_cannot_replace_valid(tmp_path):
    (tmp_path / "a").write_text("valid")
    assert accept_or_rollback(tmp_path, 0.1, 0.1, 10, 3).accepted
    (tmp_path / "a").write_text("unvalidated")
    assert not accept_or_rollback(tmp_path, None, None, 10, 3).accepted
    assert (tmp_path / "a").read_text() == "valid"


def test_studio_replay_unseen_and_restart(tmp_path):
    graphics = tmp_path / "graphics.txt"
    rows = [
        json.loads(x)
        for x in Path("data/sample_graphics.txt").read_text(encoding="utf-8").splitlines()
    ]
    extra = dict(rows[0])
    extra["character"] = "人"
    graphics.write_text("\n".join(json.dumps(x) for x in rows + [extra]))
    s = Studio(tmp_path, graphics)
    page = s.ingest(sample_bytes())
    p = tmp_path / page["id"] / "source.png"
    page["items"] = [
        {
            "index": 0,
            "char": "十",
            "crop": s.asset(p),
            "bbox": [0, 0, 150, 150],
            "confidence": 1,
            "included": False,
        }
    ]
    s.learn(page["id"], [0])
    s = Studio(tmp_path, graphics)
    r = s.generate("十 十\n十")
    assert r["report"]["observed_count"] == 1
    with pytest.raises(ValueError, match="未提供字样"):
        s.generate("人")
    r = s.generate("十人", draft=True)
    assert r["report"]["experimental_count"] == 1
    package = s.generate_font_package("observed")
    assert package["report"]["charset_count"] == 1
    with pytest.raises(ValueError, match="完整字库不可用"):
        s.generate_font_package("all")
    with pytest.raises(ValueError, match="不包含"):
        s.generate("😀", draft=True)
    with pytest.raises(ValueError):
        s.generate("十", pen_up="M3\nG0 X1000")


def test_http_bootstrap_csrf_and_path_traversal(tmp_path):
    server = make_server(tmp_path, Path("data/sample_graphics.txt"), 0)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        html = urllib.request.urlopen(url).read().decode()
        assert "墨迹" in html
        env = json.load(urllib.request.urlopen(url + "/api/environment"))
        assert env["graphics_available"]
        req = urllib.request.Request(
            url + "/api/upload", data=b"{}", headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        assert err.value.code == 403
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + "/files/%2e%2e/pyproject.toml")
    finally:
        server.shutdown()
        server.server_close()
        server.executor.shutdown()


def test_paddle_numpy_arrays(monkeypatch, tmp_path):
    from hwplotter.ocr.paddle_adapter import PaddleOCRAdapter

    adapter = object.__new__(PaddleOCRAdapter)

    class Fake:
        def predict(self, p):
            return [
                {
                    "rec_texts": ["十"],
                    "rec_scores": [0.99],
                    "rec_polys": np.array([[[0, 0], [10, 0], [10, 10], [0, 10]]]),
                }
            ]

    adapter._ocr = Fake()
    assert adapter.recognize(tmp_path / "x")[0].text == "十"
