from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from hwplotter.geometry import resample_polyline
from hwplotter.model.entities import StyleSample
from hwplotter.sample_library import FacsimileLibrary, SampleRecord
from hwplotter.stroke_data.mmh import MakeMeAHanziSource


@dataclass(slots=True)
class ModelVersion:
    version: int
    accepted: bool
    composite_validation_error: float | None
    structure_validation_mse: float | None
    stroke_validation_mse: float | None
    train_characters: int
    validation_characters: int
    created_at: float
    note: str = ""


@dataclass(slots=True)
class ContinualState:
    active_version: int = 0
    best_validation_error: float | None = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(path.read_text(encoding="utf-8"))) if path.exists() else cls()


def load_cumulative_samples(library_root: Path, graphics_path: Path) -> list[StyleSample]:
    lib = FacsimileLibrary(library_root)
    src = MakeMeAHanziSource(graphics_path)
    out = []
    for row in lib._rows:
        try:
            fitted = lib.load_variant(SampleRecord(**row))
            standard = src.get(row["char"])
            out.append(StyleSample(row["char"], standard, fitted, lib.root / row["image"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Skipping invalid sample %s: %s", row.get("sample_id"), exc
            )
            continue
    return out


def split_by_character(samples: list[StyleSample], val_fraction: float = 0.2, seed: int = 17):
    chars = sorted({s.char for s in samples})
    if len(chars) < 5:
        return samples, []
    rng = np.random.default_rng(seed)
    arr = np.asarray(chars, dtype=object)
    rng.shuffle(arr)
    n = max(1, round(len(arr) * val_fraction))
    val = set(arr[:n].tolist())
    return [s for s in samples if s.char not in val], [s for s in samples if s.char in val]


def stroke_validation_mse(
    samples: list[StyleSample], dictionary_path: Path, model_path: Path, meta_path: Path
) -> float | None:
    if not samples or not model_path.exists() or not meta_path.exists():
        return None
    try:
        import torch

        from hwplotter.model.tiny_trajectory import (
            TinyTrajectoryMeta,
            TinyTrajectoryNet,
            _features_for_stroke,
        )
        from hwplotter.stroke_data.components import IDS_OPERATORS, MMHDictionarySource, parse_ids
    except ImportError:
        return None
    meta = TinyTrajectoryMeta.load(meta_path)
    model = TinyTrajectoryNet(meta.input_dim, meta.hidden)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    d = MMHDictionarySource(dictionary_path)
    errs = []
    with torch.no_grad():
        for s in samples:
            info = d.get(s.char)
            op = "single"
            owners: tuple[str | None, ...] = ()
            if info:
                r = parse_ids(info.decomposition)
                op = r.symbol if r and r.symbol in IDS_OPERATORS else "single"
                owners = info.stroke_components
            for i, (a0, b0) in enumerate(zip(s.standard.strokes, s.fitted.strokes)):
                a = resample_polyline(a0.points, meta.points_per_stroke)
                b = resample_polyline(b0.points, meta.points_per_stroke)
                comp = owners[i] if i < len(owners) else None
                f = _features_for_stroke(a, i, len(s.standard.strokes), comp, op)
                pred = a + model(torch.from_numpy(f)).numpy()
                errs.append(float(np.mean((pred - b) ** 2)))
    return float(np.mean(errs)) if errs else None


def composite_error(structure_mse: float | None, stroke_mse: float | None) -> float | None:
    vals = []
    if structure_mse is not None:
        vals.append((0.35, structure_mse))
    if stroke_mse is not None:
        vals.append((0.65, stroke_mse))
    if not vals:
        return None
    total = sum(w for w, _ in vals)
    return sum(w * v for w, v in vals) / total


def snapshot_dir(model_dir: Path, versions_dir: Path, version: int):
    dst = versions_dir / f"v{version:04d}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for p in model_dir.iterdir():
        if p.name in {"versions", "continual_state.json"}:
            continue
        if p.is_file():
            shutil.copy2(p, dst / p.name)
    return dst


def restore_snapshot(snapshot: Path, model_dir: Path):
    for p in model_dir.iterdir():
        if p.name in {"versions", "continual_state.json"} or p.is_dir():
            continue
        p.unlink()
    for p in snapshot.iterdir():
        if p.is_file():
            shutil.copy2(p, model_dir / p.name)


def accept_or_rollback(
    model_dir: Path,
    struct_mse: float | None,
    stroke_mse: float | None,
    train_chars: int,
    val_chars: int,
    min_improvement: float = 1e-6,
) -> ModelVersion:
    state_path = model_dir / "continual_state.json"
    state = ContinualState.load(state_path)
    versions = model_dir / "versions"
    versions.mkdir(exist_ok=True)
    new_version = (
        max(
            [state.active_version]
            + [
                int(p.name[1:])
                for p in versions.glob("v[0-9][0-9][0-9][0-9]")
                if p.name[1:].isdigit()
            ]
        )
        + 1
    )
    score = composite_error(struct_mse, stroke_mse)
    accepted = (
        score is not None
        and np.isfinite(score)
        and (
            state.best_validation_error is None
            or score <= state.best_validation_error - min_improvement
        )
    ) or (state.active_version == 0 and score is None)
    note = (
        "accepted: validation improved or baseline"
        if accepted
        else "rejected: validation regressed; rolled back to active version"
    )
    if accepted:
        snapshot_dir(model_dir, versions, new_version)
        state.active_version = new_version
        if score is not None:
            state.best_validation_error = score
    elif state.active_version:
        restore_snapshot(versions / f"v{state.active_version:04d}", model_dir)
    mv = ModelVersion(
        new_version,
        accepted,
        score,
        struct_mse,
        stroke_mse,
        train_chars,
        val_chars,
        time.time(),
        note,
    )
    state.history.append(asdict(mv))
    state.save(state_path)
    return mv
