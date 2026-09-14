from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hwplotter.model.entities import Character, StyleSample
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


def _bbox(c: Character) -> np.ndarray:
    pts = [s.points for s in c.strokes if len(s.points)]
    if not pts:
        return np.asarray([0, 0, 1, 1], np.float32)
    p = np.concatenate(pts, axis=0)
    lo, hi = p.min(axis=0), p.max(axis=0)
    return np.asarray([lo[0], lo[1], hi[0], hi[1]], np.float32)


def _features(c: Character, operator: str, component_count: int) -> np.ndarray:
    b = _bbox(c)
    w = b[2] - b[0]
    h = b[3] - b[1]
    return np.asarray(
        [
            len(c.strokes) / 40.0,
            component_count / 8.0,
            _hash01(operator),
            b[0],
            b[1],
            b[2],
            b[3],
            w,
            h,
        ],
        np.float32,
    )


def _target(std: Character, fit: Character) -> np.ndarray:
    a, b = _bbox(std), _bbox(fit)
    aw, ah = max(a[2] - a[0], 1e-5), max(a[3] - a[1], 1e-5)
    bw, bh = max(b[2] - b[0], 1e-5), max(b[3] - b[1], 1e-5)
    ac = np.asarray([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2], np.float32)
    bc = np.asarray([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32)
    return np.asarray([bw / aw - 1.0, bh / ah - 1.0, bc[0] - ac[0], bc[1] - ac[1]], np.float32)


if nn is not None:

    class WriterStructureNet(nn.Module):
        def __init__(self, in_dim: int = 9, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 4),
            )

        def forward(self, x):
            return self.net(x)
else:

    class WriterStructureNet:  # type: ignore
        def __init__(self, *a, **k):
            raise RuntimeError("WriterStructureNet requires torch; install .[ml]")


@dataclass(slots=True)
class StructureNNMeta:
    input_dim: int = 9
    hidden: int = 64
    training_characters: int = 0
    validation_characters: int = 0
    final_loss: float = 0.0
    validation_mse: float | None = None
    epochs: int = 0

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def _xy(samples: list[StyleSample], dictionary_path: Path):
    d = MMHDictionarySource(dictionary_path)
    xs = []
    ys = []
    for s in samples:
        info = d.get(s.char)
        op = "single"
        cc = 0
        if info:
            r = parse_ids(info.decomposition)
            op = r.symbol if r and r.symbol in IDS_OPERATORS else "single"
            cc = len(info.component_strokes)
        xs.append(_features(s.standard, op, cc))
        ys.append(_target(s.standard, s.fitted))
    if not xs:
        raise ValueError("没有 Structure NN 可训练样本")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def train_structure_network(
    train_samples: list[StyleSample],
    val_samples: list[StyleSample],
    dictionary_path: Path,
    model_path: Path,
    meta_path: Path,
    epochs: int = 100,
    lr: float = 2e-3,
    seed: int = 0,
) -> StructureNNMeta:
    if torch is None:
        raise RuntimeError("WriterStructureNet requires torch; install .[ml]")
    torch.manual_seed(seed)
    x, y = _xy(train_samples, dictionary_path)
    model = WriterStructureNet(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    last = 0.0
    for _ in range(epochs):
        pred = model(xt)
        loss = loss_fn(pred, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
    vmse = None
    if val_samples:
        vx, vy = _xy(val_samples, dictionary_path)
        with torch.no_grad():
            vmse = float(torch.mean((model(torch.from_numpy(vx)) - torch.from_numpy(vy)) ** 2))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    meta = StructureNNMeta(x.shape[1], 64, len(train_samples), len(val_samples), last, vmse, epochs)
    meta.save(meta_path)
    return meta


def apply_structure_network(
    standard: Character,
    current: Character,
    dictionary_path: Path,
    model_path: Path,
    meta_path: Path,
    blend: float = 0.5,
) -> Character:
    if torch is None or not model_path.exists() or not meta_path.exists():
        return current
    meta = StructureNNMeta.load(meta_path)
    model = WriterStructureNet(meta.input_dim, meta.hidden)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    info = MMHDictionarySource(dictionary_path).get(standard.char)
    op = "single"
    cc = 0
    if info:
        r = parse_ids(info.decomposition)
        op = r.symbol if r and r.symbol in IDS_OPERATORS else "single"
        cc = len(info.component_strokes)
    with torch.no_grad():
        pred = model(torch.from_numpy(_features(standard, op, cc))[None, :]).numpy()[0]
    sx, sy, dx, dy = 1 + pred[0] * blend, 1 + pred[1] * blend, pred[2] * blend, pred[3] * blend
    out = []
    from hwplotter.model.entities import Stroke

    for s in current.strokes:
        q = s.points.copy()
        q[:, 0] = (q[:, 0] - 0.5) * sx + 0.5 + dx
        q[:, 1] = (q[:, 1] - 0.5) * sy + 0.5 + dy
        q = np.clip(q, 0, 1)
        out.append(
            Stroke(
                s.index,
                q,
                s.stroke_type,
                s.source_path,
                s.confidence,
                None if s.widths is None else s.widths.copy(),
            )
        )
    c = Character(current.char, out, source="structure-nn-refined", metadata=dict(current.metadata))
    c.metadata["structure_nn"] = True
    return c
