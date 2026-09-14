from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from hwplotter.exporters.json_polyline import save_json
from hwplotter.model.entities import Character


@dataclass(slots=True)
class SampleRecord:
    char: str
    sample_id: str
    image: str
    trajectory: str
    quality: float


class FacsimileLibrary:
    """Persistent library of actually observed glyph variants.

    One character may have many samples.  Generation can replay a real sample instead
    of synthesizing a new glyph whenever an observed variant exists.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self._rows = self._load()

    def _load(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.index_path.write_text(
            json.dumps(self._rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, char: str, image_path: Path, trajectory: Character) -> SampleRecord:
        digest = hashlib.sha1(image_path.read_bytes()).hexdigest()[:12]
        sample_id = f"{ord(char):04x}_{digest}"
        d = self.root / f"U+{ord(char):04X}" / sample_id
        d.mkdir(parents=True, exist_ok=True)
        image_dst = d / image_path.name
        if not image_dst.exists():
            shutil.copy2(image_path, image_dst)
        traj_dst = d / "trajectory.json"
        save_json(trajectory, traj_dst)
        quality = float(trajectory.metadata.get("facsimile_quality", 0.0))
        row = {
            "char": char,
            "sample_id": sample_id,
            "image": str(image_dst.relative_to(self.root)),
            "trajectory": str(traj_dst.relative_to(self.root)),
            "quality": quality,
        }
        if not any(x["sample_id"] == sample_id for x in self._rows):
            self._rows.append(row)
            self._save()
        return SampleRecord(char, sample_id, str(row["image"]), str(row["trajectory"]), quality)

    def variants(self, char: str) -> list[SampleRecord]:
        return [SampleRecord(**r) for r in self._rows if r["char"] == char]

    def load_variant(self, record: SampleRecord) -> Character:
        from hwplotter.exporters.json_polyline import load_json

        return load_json(self.root / record.trajectory)

    def replay(self, char: str, variant_index: int = 0) -> Character | None:
        variants = sorted(self.variants(char), key=lambda x: x.quality, reverse=True)
        if not variants:
            return None
        record = variants[variant_index % len(variants)]
        c = self.load_variant(record)
        c.source = "facsimile-replay"
        c.metadata = dict(c.metadata)
        c.metadata["sample_id"] = record.sample_id
        return c
