"""Recompute visual metrics from the bundled real sample and verified glyph session."""

import io
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import cairosvg
from hwplotter.vector_ink import read_mask, outlines, svg
from hwplotter.web_service import Studio

root = Path(__file__).resolve().parents[1]
known = json.loads((root / "examples/verified_session/studio.json").read_text("utf-8"))["glyphs"]
_, mask = read_mask(root / "evidence/source_0.jpg", 85)


def bitmap(polygons, w, h, target=None):
    opts = {"output_width": target, "output_height": target} if target else {}
    raw = cairosvg.svg2png(
        bytestring=svg(polygons, w, h).encode(), background_color="white", **opts
    )
    return np.array(Image.open(io.BytesIO(raw)).convert("L")) < 128


r = bitmap(outlines(mask), 500, 779)
t = mask > 0
page_iou = float(np.sum(r & t) / np.sum(r | t))
holdout = "雪晴陰張"
errors = {}
with tempfile.TemporaryDirectory() as tmp:
    s = Studio(tmp, root / "data/graphics.txt")
    s.state["glyphs"] = {k: v for k, v in known.items() if k not in holdout}
    result = s.generate(holdout, draft=True)
    path = s.root / result["bundle"][7:]
    for line in path.with_name("glyphs.jsonl").read_text().splitlines():
        row = json.loads(line)
        c = row["char"]
        a = bitmap(known[c]["polygons"], 1000, 1000, 256)
        b = bitmap(row["polygons"], 1000, 1000, 256)
        errors[c] = float(np.sum(a & b) / max(1, np.sum(a | b)))
report = {
    "page_outline_vs_threshold_mask_iou": page_iou,
    "threshold": 85,
    "holdout_training_characters": [c for c in known if c not in holdout],
    "holdout_iou": errors,
    "holdout_mean_iou": float(np.mean(list(errors.values()))),
    "note": "IoU只测归一坐标墨迹重合，不是书法风格评分；原稿指标针对二值墨迹。",
}
(root / "evidence/metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), "utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
