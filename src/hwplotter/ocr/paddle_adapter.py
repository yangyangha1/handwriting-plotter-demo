from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hwplotter.model.entities import OCRToken


class PaddleOCRAdapter:
    """Thin adapter around PaddleOCR 3.x. Import is lazy so core demo stays lightweight."""

    def __init__(self, device: str = "cpu") -> None:
        if device not in {"cpu", "gpu"}:
            raise ValueError(f"Unsupported PaddleOCR device: {device}")
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install OCR extras: pip install -e '.[ocr]'") from exc
        if device == "gpu":
            try:
                import paddle  # type: ignore
            except ImportError as exc:
                raise RuntimeError("GPU 引擎需要安装 paddlepaddle-gpu。") from exc
            if not paddle.device.is_compiled_with_cuda():
                raise RuntimeError("GPU 引擎需要支持 CUDA 的 PaddlePaddle GPU 版本。")
        # Natural handwritten page/photo mode.  PP-OCRv5 explicitly targets
        # handwriting and uncommon/vertical text; server models prioritize
        # recognition quality over model size.  Page orientation/unwarping and
        # line orientation are intentionally enabled because input is an ordinary
        # handwritten photo/scan, not a pre-aligned grid/template.
        self._ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device=device,
        )

    @staticmethod
    def _as_dict(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result.get("res", result)
        # PaddleX result objects expose json in current releases; keep duck-typed fallback.
        if hasattr(result, "json"):
            value = result.json
            value = value() if callable(value) else value
            if isinstance(value, dict):
                return value.get("res", value)
        if hasattr(result, "to_dict"):
            value = result.to_dict()
            return value.get("res", value)
        raise TypeError(f"Unsupported PaddleOCR result type: {type(result)!r}")

    def recognize(self, image: Path) -> list[OCRToken]:
        output = self._ocr.predict(str(image))
        tokens: list[OCRToken] = []
        for page in output:
            data = self._as_dict(page)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [1.0] * len(texts))
            polys = data.get("rec_polys")
            if polys is None:
                polys = data.get("dt_polys")
            if polys is None:
                polys = []
            for text, score, poly in zip(texts, scores, polys):
                if not text:
                    continue
                tokens.append(
                    OCRToken(
                        text=str(text),
                        confidence=float(score),
                        polygon=np.asarray(poly, dtype=np.float32),
                    )
                )
        return tokens
