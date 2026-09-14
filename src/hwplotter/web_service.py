import importlib.util
import json
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hwplotter.font_export import save_ttf
from hwplotter.plotter_paths import smooth_path, smooth_paths
from hwplotter.vector_ink import center_paths, gcode, outlines, paths_svg, read_mask, svg
from hwplotter.vision.grid import remove_frame_rules


class DiskGlyphMap:
    """Bounded-memory glyph store: full charsets must not retain millions of Python points."""

    def __init__(self, path):
        self.path = path
        self.offsets = {}
        path.write_bytes(b"")

    def __setitem__(self, key, value):
        value = dict(value)
        for field in ("polygons", "paths"):
            if field in value:
                value[field] = [
                    [[round(float(x), 3), round(float(y), 3)] for x, y in ring]
                    for ring in value[field]
                ]
        with self.path.open("ab") as f:
            self.offsets[key] = f.tell()
            f.write((json.dumps({"char": key, **value}, ensure_ascii=False) + "\n").encode())

    def __getitem__(self, key):
        with self.path.open("rb") as f:
            f.seek(self.offsets[key])
            result = json.loads(f.readline())
            result.pop("char", None)
            return result

    def __iter__(self):
        return iter(self.offsets)

    def __len__(self):
        return len(self.offsets)


class Studio:
    def __init__(self, root, graphics):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.graphics = Path(graphics).resolve()
        self.progress = {}
        self.state_path = self.root / "studio.json"
        self.state = (
            json.loads(self.state_path.read_text("utf-8"))
            if self.state_path.exists()
            else {"pages": {}, "glyphs": {}}
        )

    def save(self):
        p = self.state_path.with_suffix(".tmp")
        p.write_text(json.dumps(self.state, ensure_ascii=False), "utf-8")
        p.replace(self.state_path)

    @staticmethod
    def check_engine(engine):
        labels = {
            "rapid": "RapidOCR · CPU",
            "paddle": "PaddleOCR · CPU",
            "gpu": "GPU 引擎 · PaddlePaddle GPU",
            "cuda": "CUDA 引擎 · ONNX Runtime CUDA",
        }
        requirements = {
            "rapid": ("rapidocr", "onnxruntime"),
            "paddle": ("paddleocr", "paddle"),
            "gpu": ("paddleocr", "paddle"),
            "cuda": ("rapidocr", "onnxruntime"),
        }
        if engine not in requirements:
            return {
                "ok": False,
                "engine": engine,
                "label": "未知引擎",
                "message": f"不支持的识别引擎：{engine}",
                "reasons": [f"不支持的识别引擎：{engine}"],
            }

        missing = [name for name in requirements[engine] if importlib.util.find_spec(name) is None]
        reasons = [f"缺少 {name} 模块" for name in missing]
        if engine == "gpu" and not missing:
            try:
                import paddle

                if not paddle.device.is_compiled_with_cuda():
                    reasons.append("当前 PaddlePaddle 不是 CUDA/GPU 版本")
            except (ImportError, AttributeError) as exc:
                reasons.append(f"无法检查 Paddle GPU 状态：{exc}")
        if engine == "cuda" and not missing:
            try:
                import onnxruntime as ort

                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    reasons.append("ONNX Runtime 未提供 CUDAExecutionProvider")
            except (ImportError, AttributeError) as exc:
                reasons.append(f"无法检查 ONNX Runtime CUDA 状态：{exc}")
        if reasons:
            return {
                "ok": False,
                "engine": engine,
                "label": labels[engine],
                "message": "；".join(reasons),
                "reasons": reasons,
            }
        speed = (
            "较快" if engine == "rapid" else "中等" if engine == "paddle" else "较快（GPU/CUDA）"
        )
        return {
            "ok": True,
            "engine": engine,
            "label": labels[engine],
            "message": f"{labels[engine]}可用，预计速度：{speed}。实际速度取决于图片大小和硬件。",
            "reasons": [],
        }

    def asset(self, p):
        return "/files/" + Path(p).relative_to(self.root).as_posix()

    def _glyph_from_crop(self, char, crop, page, threshold=0):
        """Build an observed glyph after frame removal and plotter cleanup."""
        p = self.root / str(crop).removeprefix("/files/")
        _, m = read_mask(p, threshold)
        m = remove_frame_rules(m)
        ys, xs = np.where(m)
        if not len(xs):
            return None
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        cropped = m[y0:y1, x0:x1]
        h, w = cropped.shape
        scale = 800 / max(h, w)
        ox = (1000 - w * scale) / 2
        oy = 50 + (800 - h * scale) / 2
        polygons = [
            [[x * scale + ox, y * scale + oy] for x, y in ring] for ring in outlines(cropped)
        ]
        paths = [
            [[x * scale + ox, y * scale + oy] for x, y in line] for line in center_paths(cropped)
        ]
        paths = smooth_paths(paths, tolerance=8.0)
        dist = cv2.distanceTransform(cropped, cv2.DIST_L2, 3)
        positive = dist[(dist > 0) & np.isfinite(dist)].astype(np.float64, copy=False)
        measured_width = float(np.median(positive) * 3 * scale) if len(positive) else 28.0
        width = float(np.clip(measured_width, 8.0, 160.0))
        return {
            "polygons": polygons,
            "paths": paths,
            "advance": 1000,
            "source": "observed-outline",
            "page": page,
            "crop": crop,
            "aspect": w / max(h, 1),
            "ink_width": width,
            "plotter_cleanup": "grid-safe-smoothing-v2",
        }

    def _refresh_observed_glyph(self, char, item):
        if item.get("plotter_cleanup") == "grid-safe-smoothing-v2":
            return item
        crop = item.get("crop")
        page = self.state.get("pages", {}).get(item.get("page"), {})
        if not crop:
            candidates = [x for x in page.get("items", []) if x.get("char") == char]
            crop = next((x.get("crop") for x in candidates if x.get("included")), None)
        if not crop:
            return item
        refreshed = self._glyph_from_crop(char, crop, item.get("page"), page.get("threshold", 0))
        return refreshed or item

    def ingest(self, image_bytes, threshold=0):
        ident = uuid.uuid4().hex[:12]
        d = self.root / ident
        d.mkdir()
        p = d / "source.png"
        import io

        from PIL import ImageOps

        with Image.open(io.BytesIO(image_bytes)) as im:
            if im.width * im.height > 24_000_000:
                raise ValueError("图片超过2400万像素")
            ImageOps.exif_transpose(im).convert("RGB").save(p)
        im, mask = read_mask(p, threshold)
        if not mask.any():
            raise ValueError("未检测到墨迹，请调整阈值")
        poly = outlines(mask)
        (d / "outline.svg").write_text(svg(poly, *im.size), "utf-8")
        Image.fromarray((255 - mask * 255).astype("uint8")).save(d / "binary.png")
        paths = center_paths(mask)
        (d / "center.svg").write_text(paths_svg(paths, *im.size), "utf-8")
        (d / "paths.json").write_text(
            json.dumps(
                {
                    "width": im.width,
                    "height": im.height,
                    "paths": paths,
                    "order": "graph-not-temporal",
                }
            ),
            "utf-8",
        )
        (d / "page.gcode").write_text(gcode(paths, im.width), "utf-8")
        row = {
            "id": ident,
            "width": im.width,
            "height": im.height,
            "threshold": threshold,
            "items": [],
            "source": self.asset(p),
            "outline": self.asset(d / "outline.svg"),
            "center": self.asset(d / "center.svg"),
            "binary": self.asset(d / "binary.png"),
            "gcode": self.asset(d / "page.gcode"),
            "paths": len(paths),
        }
        self.state["pages"][ident] = row
        self.save()
        return row

    def recognize(self, ident, engine="rapid"):
        from hwplotter.model.entities import OCRToken
        from hwplotter.workflow import create_ocr_session

        if engine in {"paddle", "gpu"}:
            from hwplotter.ocr.paddle_adapter import PaddleOCRAdapter

            adapter = PaddleOCRAdapter(device="gpu" if engine == "gpu" else "cpu")
        elif engine in {"rapid", "cuda"}:
            from rapidocr import RapidOCR

            params = {"EngineConfig.onnxruntime.intra_op_num_threads": 2}
            if engine == "cuda":
                try:
                    import onnxruntime as ort
                except ImportError as exc:
                    raise RuntimeError(
                        "CUDA 引擎不可用：请在项目虚拟环境中安装 onnxruntime-gpu。"
                    ) from exc
                if "CUDAExecutionProvider" not in ort.get_available_providers():
                    raise RuntimeError(
                        "CUDA 引擎不可用：当前 ONNX Runtime 没有 CUDAExecutionProvider。"
                    )
                params["EngineConfig.onnxruntime.use_cuda"] = True
            cache_name = "_rapid_cuda" if engine == "cuda" else "_rapid"
            if not hasattr(self, cache_name):
                setattr(self, cache_name, RapidOCR(params=params))
            rapid = getattr(self, cache_name)

            class Adapter:
                def recognize(self, image):
                    result = rapid(str(image))
                    if result.txts is None:
                        return []
                    return [
                        OCRToken(
                            text=str(t),
                            polygon=np.asarray(b, dtype=np.float32),
                            confidence=float(s),
                        )
                        for t, b, s in zip(result.txts, result.boxes, result.scores)
                    ]

            adapter = Adapter()
        else:
            raise ValueError(f"不支持的识别引擎：{engine}")
        d = self.root / ident
        session = create_ocr_session(d / "source.png", d / "ocr", ocr_engine=adapter)
        items = []
        for item in session.items:
            items.append(
                {
                    "index": item.index,
                    "char": item.char,
                    "confidence": item.confidence,
                    "crop": self.asset(item.image_path),
                    "bbox": list(item.bbox),
                    "included": False,
                }
            )
        self.state["pages"][ident]["items"] = items
        self.save()
        return self.state["pages"][ident]

    def learn(self, ident, indices):
        row = self.state["pages"][ident]
        accepted = set(indices)
        added = []
        self.progress = {
            "message": "正在去除字框、平滑并保存保留字样",
            "completed": 0,
            "total": len(accepted),
        }
        for item in row["items"]:
            item["included"] = item["index"] in accepted
            if not item["included"]:
                continue
            c = item["char"]
            if len(c) != 1 or c.isspace():
                continue
            glyph = self._glyph_from_crop(c, item["crop"], ident, row["threshold"])
            if glyph is None:
                continue
            self.state["glyphs"][c] = glyph
            added.append(c)
            self.progress = {
                "message": "正在去除字框、平滑并保存保留字样",
                "completed": len(added),
                "total": len(accepted),
            }
        self.save()
        return {
            "added": added,
            "observed": len(self.state["glyphs"]),
            "status": "EXPERIMENTAL_STYLE",
            "message": "已保存核对字形；未写字仅为几何风格估计，未通过相似度验收",
        }

    def train_legacy(self, ident, indices):
        """Expose the existing cumulative model workflow without relabelling OCR."""
        from dataclasses import asdict

        from hwplotter.model.session import ReviewItem, ReviewSession
        from hwplotter.workflow import train_reviewed_style, vectorize_reviewed

        d = self.root / ident
        rows = json.loads((d / "ocr/review_manifest.json").read_text("utf-8"))
        items = []
        for row in rows:
            for key in ("image_path", "vector_svg", "vector_json"):
                if row.get(key) is not None:
                    row[key] = Path(row[key])
            for key in ("polygon", "connector_to_next"):
                if row.get(key) is not None:
                    row[key] = np.asarray(row[key], dtype=np.float32)
            row["included"] = row["index"] in indices
            items.append(ReviewItem(**row))
        session = ReviewSession(d / "source.png", d / "ocr", items)

        def update_progress(completed, total):
            self.progress = {
                "message": "正在恢复有序笔画并检查样本",
                "completed": completed,
                "total": total,
            }

        samples, _ = vectorize_reviewed(
            session, self.graphics, self.root / "legacy/vectors", progress=update_progress
        )
        if not samples:
            errors = [i.error for i in items if i.included and i.error]
            raise ValueError("没有可用于训练的有效笔画样本：" + "；".join(errors[:8]))
        dictionary = self.graphics.with_name("dictionary.txt")
        self.progress = {
            "message": f"已检查 {len(items)} 项，正在训练 {len(samples)} 个有效样本并分析偏旁覆盖",
            "completed": None,
        }
        _, report, readiness = train_reviewed_style(
            session,
            samples,
            self.graphics,
            self.root / "legacy/style",
            dictionary if dictionary.exists() else None,
        )
        return {
            "status": readiness.status,
            "accepted": len(samples),
            "coverage": asdict(report),
            "readiness": asdict(readiness),
        }

    def generate(
        self,
        text,
        draft=False,
        all_chars=False,
        gap=40,
        line_height=1250,
        page_mm=180,
        pen_up="M3 S0",
        pen_down="M3 S1000",
        feed=1200,
        use_model=None,
        output_dir=None,
    ):
        if not text and not all_chars:
            raise ValueError("请输入文字")
        if not self.state["glyphs"]:
            raise ValueError("请先OCR并确认至少一个正确样本")
        if not 0 <= gap <= 500 or not 900 <= line_height <= 3000:
            raise ValueError("间距参数超出范围")
        known = self.state["glyphs"]
        src = None
        if all_chars or any(c not in known and not c.isspace() for c in text):
            from hwplotter.stroke_data.mmh import MakeMeAHanziSource

            src = MakeMeAHanziSource(self.graphics)
            src._load()
        if all_chars:
            text = "".join(src._index)
        if len(text) > 20000:
            raise ValueError("单次最多20000字符")
        missing = sorted({c for c in text if not c.isspace() and c not in known})
        if missing and not draft:
            raise ValueError(
                "未提供字样：" + "".join(missing[:80]) + "。实验生成须勾选“未写字实验估计”。"
            )
        unsupported = [c for c in missing if c not in src._index]
        if unsupported:
            raise ValueError(
                "基础笔画库不包含：" + "".join(unsupported[:80]) + "；请提供这些字符的手写样本"
            )
        model_args = None
        if use_model is None:
            use_model = (self.root / "legacy/style/style.json").exists()
        if use_model:
            from hwplotter.style.component_profile import ComponentStyleModel
            from hwplotter.style.context_models import (
                AdjacencyStyleModel,
                PositionStyleModel,
                StructureStyleModel,
            )
            from hwplotter.style.profile import StyleProfile
            from hwplotter.style.readiness import TrainingReadiness

            md = self.root / "legacy/style"
            model_args = {
                "profile": StyleProfile.load(md / "style.json"),
                "readiness": TrainingReadiness.load(md / "training_readiness.json"),
            }
            for key, filename, cls in [
                ("component_model", "component_style.json", ComponentStyleModel),
                ("position_model", "position_style.json", PositionStyleModel),
                ("structure_model", "structure_style.json", StructureStyleModel),
                ("adjacency_model", "adjacency_style.json", AdjacencyStyleModel),
            ]:
                if (md / filename).exists():
                    model_args[key] = cls.load(md / filename)
            model_args.update(
                tiny_model_path=md / "writer_stroke.pt",
                tiny_meta_path=md / "writer_stroke_meta.json",
                structure_nn_path=md / "writer_structure.pt",
                structure_nn_meta_path=md / "writer_structure_meta.json",
            )
        d = (
            Path(output_dir)
            if output_dir is not None
            else self.root / ("export_" + uuid.uuid4().hex[:12])
        )
        d.mkdir(parents=True, exist_ok=True)
        generated = DiskGlyphMap(d / "glyphs.jsonl")
        provenance = {}
        unique_text = list(dict.fromkeys(text))
        for position, c in enumerate(unique_text, start=1):
            if c.isspace():
                continue
            if c in known:
                item = self._refresh_observed_glyph(c, dict(known[c]))
                known[c] = item
            else:
                # CPU baseline: measured ink width and aspect, NOT a pretrained font generator.
                standard = src.get(c)
                if model_args is not None:
                    from hwplotter.pipeline import hierarchical_synthesize

                    standard = hierarchical_synthesize(
                        c,
                        self.graphics,
                        dictionary=self.graphics.with_name("dictionary.txt"),
                        require_ready=not draft,
                        variation=0.0,
                        **model_args,
                    )
                width = float(np.clip(np.median([g["ink_width"] for g in known.values()]), 12, 100))
                aspect = float(np.clip(np.median([g["aspect"] for g in known.values()]), 0.65, 1.2))
                canvas = np.zeros((256, 256), np.uint8)
                paths = []
                for s in standard.strokes:
                    p = np.asarray(smooth_path(s.points.astype(float), tolerance=0.006)) * 800 + 50
                    p[:, 0] = (p[:, 0] - 450) * (aspect if model_args is None else 1) + 500
                    p = np.clip(p, 10, 990)
                    paths.append(p.tolist())
                    cv2.polylines(
                        canvas,
                        [np.rint(p * 0.256).astype("int32")],
                        False,
                        1,
                        max(1, round(width * 0.256)),
                        cv2.LINE_8,
                    )
                    for endpoint in (p[0], p[-1]):
                        cv2.circle(
                            canvas,
                            tuple(np.rint(endpoint * 0.256).astype(int)),
                            max(1, round(width * 0.128)),
                            1,
                            -1,
                        )
                paths = smooth_paths(paths, tolerance=8.0)
                item = {
                    "polygons": [
                        [[x / 0.256, y / 0.256] for x, y in ring] for ring in outlines(canvas)
                    ],
                    "paths": paths,
                    "advance": 1000,
                    "source": "experimental-hierarchical-model"
                    if model_args
                    else "experimental-mmh-geometry",
                }
            generated[c] = item
            provenance[c] = item["source"]
            self.progress = {
                "message": "正在生成完整字库" if all_chars else "正在生成轨迹文件",
                "completed": position,
                "total": len(unique_text),
            }
        save_ttf(
            generated,
            d / "handwriting.ttf",
            family="Handwriting Experimental" if missing else "Handwriting Observed",
        )
        # Preview wraps independently of font charset; newlines and spaces retained.
        show = text[:160] if all_chars else text
        polygons = []
        paths = []
        x = y = 40
        max_x = 0
        for c in show:
            if c == "\n":
                x = 40
                y += line_height
                continue
            advance = 500 if c.isspace() else generated[c]["advance"]
            if x + advance > 10000:
                x = 40
                y += line_height
            if not c.isspace():
                item = generated[c]
                polygons.extend([[[a + x, b + y] for a, b in p] for p in item["polygons"]])
                paths.extend([[[a + x, b + y] for a, b in p] for p in item["paths"]])
            x += advance + gap
            max_x = max(max_x, x)
        w = max(1000, max_x + 40)
        h = y + 1050
        (d / "preview.svg").write_text(svg(polygons, w, h, "已见字与未写字实验预览"), "utf-8")
        (d / "center.svg").write_text(paths_svg(paths, w, h), "utf-8")
        (d / "text.gcode").write_text(gcode(paths, w, page_mm, pen_up, pen_down, feed), "utf-8")
        report = {
            "charset_count": len(generated),
            "observed_count": len(generated) - len(missing),
            "experimental_count": len(missing),
            "complete_for_requested_charset": True,
            "universal_font": False,
            "style_verified": False,
            "preview_truncated": all_chars and len(text) > 160,
            "glyph_provenance": provenance,
            "note": "TTF存轮廓，JSON存路径；图遍历不是真实笔顺；实验字不代表学会本人草书。",
        }
        (d / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        with zipfile.ZipFile(d / "font_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for p in d.iterdir():
                if p.suffix != ".zip":
                    z.write(p, p.name)
        return {
            "preview": self.asset(d / "preview.svg"),
            "center": self.asset(d / "center.svg"),
            "ttf": self.asset(d / "handwriting.ttf"),
            "gcode": self.asset(d / "text.gcode"),
            "bundle": self.asset(d / "font_bundle.zip"),
            "report": report,
        }

    def generate_font_package(self, kind="observed"):
        """Export observed glyphs, or the complete inferred font after readiness."""
        if kind not in {"observed", "all"}:
            raise ValueError("字库类型只能是 observed 或 all")
        if not self.state["glyphs"]:
            raise ValueError("请先OCR并确认至少一个正确样本")
        package_root = self.root / (
            "font_package_observed" if kind == "observed" else "font_package_all"
        )
        if kind == "all":
            from hwplotter.style.readiness import TrainingReadiness

            readiness_path = self.root / "legacy/style/training_readiness.json"
            if not readiness_path.exists():
                raise ValueError("完整字库不可用：尚未完成训练验收")
            readiness = TrainingReadiness.load(readiness_path)
            if not readiness.ready:
                detail = "；".join(readiness.reasons[:4]) or "样本覆盖和验证尚未达到输出门槛"
                raise ValueError(f"完整字库不可用：模型已处理完成，但数据量或覆盖不足。{detail}")
            result = self.generate(
                "", all_chars=True, draft=False, use_model=True, output_dir=package_root
            )
            result["kind"] = "all"
            result["title"] = "所有完整字库（已通过训练验收）"
            return result
        observed_text = "".join(sorted(self.state["glyphs"]))
        result = self.generate(observed_text, draft=False, use_model=False, output_dir=package_root)
        result["kind"] = "observed"
        result["title"] = "已有字样完整字库（全部已确认字样）"
        return result
