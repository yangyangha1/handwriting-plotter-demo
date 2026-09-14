from __future__ import annotations

import json
from pathlib import Path

import typer

from hwplotter.exporters.gcode import GCodeOptions, save_gcode
from hwplotter.exporters.json_polyline import save_json
from hwplotter.exporters.svg import save_svg
from hwplotter.ocr.cropper import split_token_to_char_crops
from hwplotter.ocr.paddle_adapter import PaddleOCRAdapter
from hwplotter.pipeline import (
    fit_one,
    hierarchical_synthesize,
    hybrid_generate,
    train_style,
)
from hwplotter.sample_library import FacsimileLibrary
from hwplotter.style.component_profile import ComponentStyleModel
from hwplotter.style.context_models import (
    AdjacencyStyleModel,
    PositionStyleModel,
    StructureStyleModel,
)
from hwplotter.style.profile import StyleProfile
from hwplotter.style.readiness import TrainingReadiness

app = typer.Typer(
    no_args_is_help=True, help="Handwriting image -> stroke trajectories -> plotter files"
)


@app.command()
def ocr(
    image: Path = typer.Option(..., exists=True),
    out_dir: Path = typer.Option(Path("out/ocr_crops")),
    min_confidence: float = typer.Option(0.5, min=0.0, max=1.0),
) -> None:
    """Run PaddleOCR and create approximate per-character crops + manifest."""
    engine = PaddleOCRAdapter()
    tokens = engine.recognize(image)
    rows = []
    for token in tokens:
        if token.confidence < min_confidence:
            continue
        for crop in split_token_to_char_crops(image, token, out_dir):
            rows.append(
                {
                    "char": crop.char,
                    "image": str(crop.image_path),
                    "bbox": crop.bbox,
                    "confidence": crop.confidence,
                }
            )
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"OCR tokens={len(tokens)}, character crops={len(rows)}, manifest={manifest}")


@app.command()
def fit(
    char: str = typer.Option(..., help="Single Hanzi represented by the image"),
    image: Path = typer.Option(..., exists=True),
    graphics: Path = typer.Option(..., exists=True, help="Make Me A Hanzi graphics.txt"),
    out: Path = typer.Option(Path("out/fitted")),
) -> None:
    """Fit canonical MMH stroke medians to one handwritten character image."""
    _, fitted = fit_one(char, image, graphics)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_svg(fitted, out.with_suffix(".svg"))
    save_json(fitted, out.with_suffix(".json"))
    save_gcode(fitted, out.with_suffix(".gcode"), GCodeOptions())
    typer.echo(f"wrote: {out}.svg/.json/.gcode")


@app.command("train-style")
def train_style_cmd(
    manifest: Path = typer.Option(
        ..., exists=True, help='JSON list: [{"char":"永","image":"..."}]'
    ),
    graphics: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(Path("out/style.json")),
) -> None:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    samples = [(r["char"], Path(r["image"])) for r in rows]
    profile = train_style(samples, graphics)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile.save(out)
    typer.echo(f"style profile: {out}")


@app.command()
def generate(
    text: str = typer.Option(..., help="Hanzi text; exports one file per character"),
    graphics: Path = typer.Option(..., exists=True),
    style: Path = typer.Option(..., exists=True),
    dictionary: Path | None = typer.Option(
        None, exists=True, help="MMH dictionary.txt for exact component ownership"
    ),
    component_style: Path | None = typer.Option(
        None, exists=True, help="component_style.json learned from reviewed samples"
    ),
    facsimile_library: Path | None = typer.Option(
        None, exists=True, help="Observed glyph library; replay exact samples before synthesis"
    ),
    out_dir: Path = typer.Option(Path("out/generated")),
    variation: float = typer.Option(0.12, min=0.0, max=1.0),
) -> None:
    profile = StyleProfile.load(style)
    component_model = ComponentStyleModel.load(component_style) if component_style else None
    if (dictionary is None) != (component_model is None):
        raise typer.BadParameter("--dictionary 与 --component-style 必须同时提供，或同时省略")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, char in enumerate(text):
        try:
            c = hybrid_generate(
                char,
                graphics,
                profile,
                library_root=facsimile_library,
                variation=variation,
                seed=i,
                dictionary=dictionary,
                component_model=component_model,
            )
        except KeyError:
            typer.echo(f"skip unsupported character: {char!r}")
            continue
        stem = out_dir / f"{i:04d}_{ord(char):04x}_{char}"
        save_svg(c, stem.with_suffix(".svg"))
        save_json(c, stem.with_suffix(".json"))
        save_gcode(c, stem.with_suffix(".gcode"), GCodeOptions())
    mode = (
        "hybrid-replay"
        if facsimile_library
        else ("exact-component" if component_model else "global-style")
    )
    typer.echo(f"generated files under: {out_dir} (mode={mode})")


@app.command("generate-v08")
def generate_v05(
    text: str = typer.Option(...),
    graphics: Path = typer.Option(..., exists=True),
    dictionary: Path = typer.Option(..., exists=True),
    model_dir: Path = typer.Option(..., exists=True, help="style 训练输出目录"),
    facsimile_library: Path | None = typer.Option(None, exists=True),
    out_dir: Path = typer.Option(Path("out/generated_v05")),
    variation: float = typer.Option(0.08, min=0.0, max=0.5),
    allow_incomplete: bool = typer.Option(False, help="仅调试；允许样本不足时生成"),
) -> None:
    profile = StyleProfile.load(model_dir / "style.json")
    readiness = TrainingReadiness.load(model_dir / "training_readiness.json")
    component = (
        ComponentStyleModel.load(model_dir / "component_style.json")
        if (model_dir / "component_style.json").exists()
        else None
    )
    position = (
        PositionStyleModel.load(model_dir / "position_style.json")
        if (model_dir / "position_style.json").exists()
        else None
    )
    structure = (
        StructureStyleModel.load(model_dir / "structure_style.json")
        if (model_dir / "structure_style.json").exists()
        else None
    )
    adjacency = (
        AdjacencyStyleModel.load(model_dir / "adjacency_style.json")
        if (model_dir / "adjacency_style.json").exists()
        else None
    )
    tiny_pt = model_dir / "writer_stroke.pt"
    tiny_meta = model_dir / "writer_stroke_meta.json"
    structure_nn_pt = model_dir / "writer_structure.pt"
    structure_nn_meta = model_dir / "writer_structure_meta.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, char in enumerate(text):
        replay = (
            FacsimileLibrary(facsimile_library).replay(char, variant_index=i)
            if facsimile_library
            else None
        )
        if replay is not None:
            c = replay
        else:
            c = hierarchical_synthesize(
                char,
                graphics,
                profile,
                dictionary,
                component,
                position,
                structure,
                adjacency,
                tiny_pt if tiny_pt.exists() else None,
                tiny_meta if tiny_meta.exists() else None,
                readiness=readiness,
                variation=variation,
                seed=i,
                require_ready=not allow_incomplete,
                structure_nn_path=structure_nn_pt if structure_nn_pt.exists() else None,
                structure_nn_meta_path=structure_nn_meta if structure_nn_meta.exists() else None,
            )
        stem = out_dir / f"{i:04d}_{ord(char):04x}_{char}"
        save_svg(c, stem.with_suffix(".svg"))
        save_json(c, stem.with_suffix(".json"))
        save_gcode(c, stem.with_suffix(".gcode"), GCodeOptions())
    typer.echo(f"generated under {out_dir}; readiness={readiness.status}")


if __name__ == "__main__":
    app()
