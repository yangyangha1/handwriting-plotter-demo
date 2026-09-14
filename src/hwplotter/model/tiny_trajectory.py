from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hwplotter.geometry import resample_polyline
from hwplotter.model.entities import Character, Stroke, StyleSample
from hwplotter.stroke_data.components import IDS_OPERATORS, MMHDictionarySource, parse_ids

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]  # optional dependency
    nn = None  # type: ignore[assignment]  # optional dependency


def _hash01(text: str) -> float:
    b = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(b, "little") / 4294967295.0


def _position_code(points: np.ndarray) -> tuple[float, float]:
    c = points.mean(axis=0)
    return float(c[0]), float(c[1])


def _curvature(p: np.ndarray) -> np.ndarray:
    d = np.gradient(p, axis=0)
    dd = np.gradient(d, axis=0)
    cross = d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]
    denom = np.power(np.sum(d * d, axis=1) + 1e-6, 1.5)
    return cross / denom


def _features_for_stroke(
    points: np.ndarray,
    stroke_index: int,
    stroke_count: int,
    component: str | None,
    operator: str,
) -> np.ndarray:
    n = len(points)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    d = np.gradient(points, axis=0)
    norm = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
    tangent = d / norm
    curv = _curvature(points).astype(np.float32)
    cx, cy = _position_code(points)
    comp_hash = _hash01(component or "")
    op_hash = _hash01(operator)
    feats = np.column_stack(
        [
            points[:, 0],
            points[:, 1],
            t,
            tangent[:, 0],
            tangent[:, 1],
            curv,
            np.full(n, stroke_index / max(stroke_count - 1, 1), np.float32),
            np.full(n, stroke_count / 40.0, np.float32),
            np.full(n, cx, np.float32),
            np.full(n, cy, np.float32),
            np.full(n, comp_hash, np.float32),
            np.full(n, op_hash, np.float32),
        ]
    ).astype(np.float32)
    return feats


if nn is not None:

    class TinyTrajectoryNet(nn.Module):
        def __init__(self, in_dim: int = 12, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 2),
            )

        def forward(self, x):
            return self.net(x)
else:

    class TinyTrajectoryNet:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Tiny trajectory model requires torch; install .[ml]")


@dataclass(slots=True)
class TinyTrajectoryMeta:
    points_per_stroke: int = 64
    input_dim: int = 12
    hidden: int = 128
    training_points: int = 0
    training_strokes: int = 0
    training_characters: int = 0
    final_loss: float = 0.0
    epochs: int = 0
    validation_mse: float | None = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TinyTrajectoryMeta:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _dataset(samples: list[StyleSample], dictionary_path: Path, n: int):
    dictionary = MMHDictionarySource(dictionary_path)
    xs, ys = [], []
    stroke_total = 0
    for sample in samples:
        info = dictionary.get(sample.char)
        operator = "single"
        if info:
            root = parse_ids(info.decomposition)
            if root and root.symbol in IDS_OPERATORS:
                operator = root.symbol
        owners = info.stroke_components if info else ()
        for i, (std, fit) in enumerate(zip(sample.standard.strokes, sample.fitted.strokes)):
            a = resample_polyline(std.points, n)
            b = resample_polyline(fit.points, n)
            comp = owners[i] if i < len(owners) else None
            xs.append(_features_for_stroke(a, i, len(sample.standard.strokes), comp, operator))
            ys.append((b - a).astype(np.float32))
            stroke_total += 1
    if not xs:
        raise ValueError("没有 TinyTrajectory 可训练轨迹")
    return np.concatenate(xs), np.concatenate(ys), stroke_total


def train_tiny_trajectory(
    samples: list[StyleSample],
    dictionary_path: Path,
    model_path: Path,
    meta_path: Path,
    n: int = 64,
    epochs: int = 80,
    lr: float = 2e-3,
    seed: int = 0,
    val_samples: list[StyleSample] | None = None,
) -> TinyTrajectoryMeta:
    if torch is None:
        raise RuntimeError("Tiny trajectory model requires torch; install .[ml]")
    torch.manual_seed(seed)
    x, y, stroke_total = _dataset(samples, dictionary_path, n)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    model = TinyTrajectoryNet(in_dim=x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    last = 0.0
    for _ in range(epochs):
        pred = model(xt)
        loss = loss_fn(pred, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach().cpu())
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    validation_mse = None
    if val_samples:
        vx, vy, _ = _dataset(val_samples, dictionary_path, n)
        model.eval()
        with torch.no_grad():
            validation_mse = float(
                torch.mean((model(torch.from_numpy(vx)) - torch.from_numpy(vy)) ** 2).cpu()
            )
    meta = TinyTrajectoryMeta(
        n, x.shape[1], 128, len(x), stroke_total, len(samples), last, epochs, validation_mse
    )
    meta.save(meta_path)
    return meta


def apply_tiny_trajectory(
    standard: Character,
    current: Character,
    dictionary_path: Path,
    model_path: Path,
    meta_path: Path,
    blend: float = 0.25,
) -> Character:
    if torch is None:
        return current
    meta = TinyTrajectoryMeta.load(meta_path)
    model = TinyTrajectoryNet(meta.input_dim, meta.hidden)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    dictionary = MMHDictionarySource(dictionary_path)
    info = dictionary.get(standard.char)
    operator = "single"
    if info:
        root = parse_ids(info.decomposition)
        if root and root.symbol in IDS_OPERATORS:
            operator = root.symbol
    owners = info.stroke_components if info else ()
    out: list[Stroke] = []
    with torch.no_grad():
        for i, std in enumerate(standard.strokes):
            a = resample_polyline(std.points, meta.points_per_stroke)
            comp = owners[i] if i < len(owners) else None
            f = _features_for_stroke(a, i, len(standard.strokes), comp, operator)
            delta = model(torch.from_numpy(f)).numpy()
            neural_target = np.clip(a + delta, 0.0, 1.0)
            base = resample_polyline(current.strokes[i].points, meta.points_per_stroke)
            q = np.clip((1.0 - blend) * base + blend * neural_target, 0.0, 1.0)
            out.append(
                Stroke(
                    i,
                    q,
                    current.strokes[i].stroke_type,
                    current.strokes[i].source_path,
                    current.strokes[i].confidence,
                )
            )
    return Character(
        standard.char, out, source="tiny-trajectory-refined", metadata=dict(current.metadata)
    )
