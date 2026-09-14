from __future__ import annotations

from pathlib import Path

from hwplotter.exporters.gcode import GCodeOptions, save_gcode
from hwplotter.exporters.json_polyline import save_json
from hwplotter.exporters.svg import save_svg
from hwplotter.fitting.facsimile import FacsimileConfig, facsimile_character
from hwplotter.model.entities import Character, StyleSample
from hwplotter.model.structure_network import apply_structure_network
from hwplotter.model.tiny_trajectory import apply_tiny_trajectory
from hwplotter.sample_library import FacsimileLibrary
from hwplotter.stroke_data.mmh import MakeMeAHanziSource
from hwplotter.style.component_profile import ComponentStyleModel, apply_component_conditioning
from hwplotter.style.context_models import (
    AdjacencyStyleModel,
    PositionStyleModel,
    StructureStyleModel,
    apply_context_models,
)
from hwplotter.style.profile import StyleProfile, apply_profile, learn_profile
from hwplotter.style.readiness import TrainingReadiness
from hwplotter.vision.preprocess import binarize_ink, frame_square_resize, read_gray, unframe_points


def fit_one(char: str, image: Path, graphics: Path, size: int = 768) -> tuple[Character, Character]:
    source = MakeMeAHanziSource(graphics)
    standard = source.get(char)
    gray = read_gray(image)
    mask = binarize_ink(gray)
    framed, transform = frame_square_resize(mask, size=size, margin=10)
    fitted = facsimile_character(standard, framed, FacsimileConfig(image_size=size))
    for stroke in fitted.strokes:
        stroke.points = unframe_points(stroke.points, transform)
        if stroke.widths is not None:
            stroke.widths = (
                stroke.widths
                * transform["size"]
                / max(transform["scale"], 1e-8)
                / max(transform["source_w"], transform["source_h"])
            ).astype("float32")
    fitted.metadata["coordinate_space"] = "source-character-frame"
    return standard, fitted


def train_style(samples: list[tuple[str, Path]], graphics: Path) -> StyleProfile:
    pairs: list[StyleSample] = []
    for char, image in samples:
        standard, fitted = fit_one(char, image, graphics)
        pairs.append(StyleSample(char=char, standard=standard, fitted=fitted, image_path=image))
    return learn_profile(pairs)


def synthesize(
    char: str,
    graphics: Path,
    profile: StyleProfile,
    variation: float = 0.12,
    seed: int = 0,
    dictionary: Path | None = None,
    component_model: ComponentStyleModel | None = None,
) -> Character:
    standard = MakeMeAHanziSource(graphics).get(char)
    if dictionary is not None and component_model is not None:
        return apply_component_conditioning(
            standard,
            profile,
            component_model,
            dictionary,
            blend=0.9,
            variation=variation,
            seed=seed,
        )
    return apply_profile(standard, profile, variation=variation, seed=seed)


def hierarchical_synthesize(
    char: str,
    graphics: Path,
    profile: StyleProfile,
    dictionary: Path,
    component_model: ComponentStyleModel | None = None,
    position_model: PositionStyleModel | None = None,
    structure_model: StructureStyleModel | None = None,
    adjacency_model: AdjacencyStyleModel | None = None,
    tiny_model_path: Path | None = None,
    tiny_meta_path: Path | None = None,
    readiness: TrainingReadiness | None = None,
    variation: float = 0.08,
    seed: int = 0,
    require_ready: bool = True,
    structure_nn_path: Path | None = None,
    structure_nn_meta_path: Path | None = None,
) -> Character:
    """Generate an unseen glyph with the full v0.5 hierarchical model.

    Order: global -> exact component -> component-position/structure/adjacency ->
    tiny trajectory residual.  If `require_ready` is true, incomplete training is
    rejected instead of emitting a low-confidence result.
    """
    if require_ready and readiness is None:
        raise RuntimeError("NEED_MORE_SAMPLES：缺少训练验收状态")
    if require_ready and readiness is not None and not readiness.ready:
        detail = "；".join(readiness.reasons)
        extra = (
            f"；建议补写：{readiness.required_more_text}" if readiness.required_more_text else ""
        )
        raise RuntimeError(f"NEED_MORE_SAMPLES：{detail}{extra}")
    standard = MakeMeAHanziSource(graphics).get(char)
    current = apply_profile(standard, profile, variation=min(variation, 0.03), seed=seed)
    if component_model is not None:
        current = apply_component_conditioning(
            standard,
            profile,
            component_model,
            dictionary,
            blend=0.9,
            variation=variation,
            seed=seed,
        )
    context_allowed = readiness is None or readiness.context_ready
    neural_allowed = readiness is None or readiness.tiny_ready
    if context_allowed:
        current = apply_context_models(
            standard,
            current,
            dictionary,
            position_model,
            structure_model,
            adjacency_model,
            variation=min(variation, 0.03),
            seed=seed,
        )
    elif structure_model is not None:
        # A partial model may still have reliable broad structure samples even when
        # component coverage is below the full-font gate. Keep this blend conservative.
        current = apply_context_models(
            standard,
            current,
            dictionary,
            position_model,
            structure_model,
            None,
            variation=0.0,
            seed=seed,
            position_blend=0.20,
            structure_blend=0.18,
            adjacency_blend=0.0,
        )
    if neural_allowed and (
        structure_nn_path
        and structure_nn_meta_path
        and structure_nn_path.exists()
        and structure_nn_meta_path.exists()
    ):
        current = apply_structure_network(
            standard, current, dictionary, structure_nn_path, structure_nn_meta_path, blend=0.50
        )
    if (
        neural_allowed
        and tiny_model_path
        and tiny_meta_path
        and tiny_model_path.exists()
        and tiny_meta_path.exists()
    ):
        current = apply_tiny_trajectory(
            standard, current, dictionary, tiny_model_path, tiny_meta_path, blend=0.25
        )
    current.source = "hierarchical-v0.8"
    return current


def hybrid_generate(
    char: str,
    graphics: Path,
    profile: StyleProfile,
    library_root: Path | None = None,
    variation: float = 0.12,
    seed: int = 0,
    dictionary: Path | None = None,
    component_model: ComponentStyleModel | None = None,
) -> Character:
    """Replay an observed glyph whenever possible; synthesize only unseen glyphs."""
    if library_root is not None:
        replay = FacsimileLibrary(library_root).replay(char, variant_index=seed)
        if replay is not None:
            return replay
    return synthesize(
        char,
        graphics,
        profile,
        variation=variation,
        seed=seed,
        dictionary=dictionary,
        component_model=component_model,
    )


def export_all(character: Character, output_prefix: Path, char_size_mm: float = 20.0) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_svg(character, output_prefix.with_suffix(".svg"))
    save_json(character, output_prefix.with_suffix(".json"))
    save_gcode(
        character,
        output_prefix.with_suffix(".gcode"),
        GCodeOptions(char_size_mm=char_size_mm),
    )
