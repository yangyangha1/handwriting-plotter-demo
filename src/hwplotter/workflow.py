from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from hwplotter.exporters.json_polyline import save_json
from hwplotter.exporters.page_gcode import save_page_gcode
from hwplotter.exporters.page_svg import save_character_page_svg
from hwplotter.exporters.svg import save_svg
from hwplotter.fitting.facsimile import FacsimileConfig, facsimile_character
from hwplotter.fitting.stroke_order import CursiveOrderSource
from hwplotter.model.continual_learning import (
    accept_or_rollback,
    load_cumulative_samples,
    split_by_character,
)
from hwplotter.model.entities import StyleSample
from hwplotter.model.session import ReviewItem, ReviewSession
from hwplotter.model.structure_network import train_structure_network
from hwplotter.model.tiny_trajectory import train_tiny_trajectory
from hwplotter.ocr.cropper import split_token_to_char_crops
from hwplotter.ocr.paddle_adapter import PaddleOCRAdapter
from hwplotter.sample_library import FacsimileLibrary
from hwplotter.stroke_data.mmh import MakeMeAHanziSource
from hwplotter.style.active_learning import recommend_active_samples
from hwplotter.style.component_profile import learn_component_styles
from hwplotter.style.context_models import learn_context_models
from hwplotter.style.coverage import analyze_coverage
from hwplotter.style.profile import StyleProfile, learn_profile
from hwplotter.style.readiness import assess_readiness
from hwplotter.vision.image_io import read_image
from hwplotter.vision.preprocess import binarize_ink, frame_square_resize, read_gray, unframe_points


def create_ocr_session(
    image_path: Path,
    work_dir: Path,
    min_confidence: float = 0.4,
    ocr_engine: PaddleOCRAdapter | None = None,
) -> ReviewSession:
    engine = ocr_engine or PaddleOCRAdapter()
    tokens = engine.recognize(image_path)
    crop_dir = work_dir / "crops"
    items: list[ReviewItem] = []
    for line_id, token in enumerate(tokens):
        if token.confidence < min_confidence:
            continue
        for crop in split_token_to_char_crops(image_path, token, crop_dir):
            items.append(
                ReviewItem(
                    index=len(items),
                    char=crop.char,
                    original_char=crop.char,
                    image_path=crop.image_path,
                    bbox=crop.bbox,
                    confidence=crop.confidence,
                    polygon=crop.polygon,
                    line_id=line_id,
                    char_position=crop.char_position,
                    connector_to_next=crop.connector_to_next,
                )
            )
    session = ReviewSession(source_image=image_path, work_dir=work_dir, items=items)
    save_review_manifest(session)
    return session


def save_review_manifest(session: ReviewSession) -> Path:
    session.work_dir.mkdir(parents=True, exist_ok=True)
    path = session.work_dir / "review_manifest.json"
    rows = []
    for item in session.items:
        row = asdict(item)
        for key in ("image_path", "vector_svg", "vector_json"):
            if row.get(key) is not None:
                row[key] = str(row[key])
        for key in ("polygon", "connector_to_next"):
            value = row.get(key)
            if value is not None:
                row[key] = value.tolist()
        rows.append(row)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    # Separate rewrite queue for the next acquisition round.  It is deliberately
    # generated from rejected/failed image-derived labels, never from manual relabelling.
    rewrite_path = session.work_dir / "rewrite_required.txt"
    rewrite_path.write_text(session.rewrite_text(), encoding="utf-8")
    return path


def _fit_from_source(
    source: MakeMeAHanziSource,
    char: str,
    image: Path,
    size: int = 768,
    cursive_orders: CursiveOrderSource | None = None,
):
    standard = source.get(char)
    raw_mask = binarize_ink(read_gray(image))
    mask, frame = frame_square_resize(raw_mask, size=size, margin=10)
    variants = cursive_orders.get(char) if cursive_orders is not None else None
    fitted = facsimile_character(
        standard, mask, FacsimileConfig(image_size=size), cursive_variants=variants
    )
    # Critical for page facsimile: put the recovered points back into the original
    # OCR character-cell coordinate system instead of keeping a tight/square crop.
    for stroke in fitted.strokes:
        stroke.points = unframe_points(stroke.points, frame)
        if stroke.widths is not None:
            # Convert working-square normalized ink width to source-cell normalized
            # width so SVG/G-code pressure mapping survives letterbox inversion.
            stroke.widths = (
                stroke.widths
                * frame["size"]
                / max(frame["scale"], 1e-8)
                / max(frame["source_w"], frame["source_h"])
            ).astype("float32")
    fitted.metadata["coordinate_space"] = "ocr-character-cell"
    fitted.metadata["source_frame_w"] = int(raw_mask.shape[1])
    fitted.metadata["source_frame_h"] = int(raw_mask.shape[0])
    return standard, fitted


def vectorize_reviewed(
    session: ReviewSession,
    graphics_path: Path,
    output_dir: Path,
    cursive_order_path: Path | None = None,
    min_stroke_order_confidence: float = 0.58,
    progress=None,
) -> tuple[list[StyleSample], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted_page = []
    samples: list[StyleSample] = []
    source = MakeMeAHanziSource(graphics_path)
    library = FacsimileLibrary(output_dir.parent / "facsimile_library")
    cursive_orders = CursiveOrderSource(cursive_order_path)
    accepted = list(session.accepted())
    for position, item in enumerate(accepted):
        if progress:
            progress(position, len(accepted))
        char = item.effective_char
        item.error = None
        try:
            standard, fitted = _fit_from_source(
                source, char, item.image_path, cursive_orders=cursive_orders
            )
        except Exception as exc:  # noqa: BLE001 - report per-job failure to user
            item.error = str(exc)
            continue
        item.stroke_order_confidence = float(fitted.metadata.get("stroke_order_confidence", 0.0))
        item.stroke_order_mode = str(fitted.metadata.get("stroke_order_mode", "unknown"))
        if (
            bool(fitted.metadata.get("stroke_order_ambiguous", False))
            or item.stroke_order_confidence < min_stroke_order_confidence
        ):
            item.error = f"笔顺时序歧义，需重写（confidence={item.stroke_order_confidence:.3f}）"
            item.rejection_reason = "ambiguous_stroke_order_rewrite"
            continue
        stem = output_dir / f"{item.index:05d}_{ord(char):04x}"
        item.vector_svg = stem.with_suffix(".svg")
        item.vector_json = stem.with_suffix(".json")
        save_svg(fitted, item.vector_svg)
        save_json(fitted, item.vector_json)
        samples.append(
            StyleSample(char=char, standard=standard, fitted=fitted, image_path=item.image_path)
        )
        library.add(char, item.image_path, fitted)
        item.facsimile_quality = float(fitted.metadata.get("facsimile_quality", 0.0))
        fitted.metadata["line_id"] = item.line_id
        fitted.metadata["char_position"] = item.char_position
        fitted_page.append(
            (
                fitted,
                item.polygon if item.polygon is not None else item.bbox,
                item.connector_to_next,
            )
        )

    image = read_image(session.source_image)
    h, w = image.shape[:2]
    page_path = output_dir / "page_vectors.svg"
    save_character_page_svg(fitted_page, (w, h), page_path)
    save_page_gcode(fitted_page, (w, h), output_dir / "page_vectors.gcode")
    save_review_manifest(session)
    return samples, page_path


def train_reviewed_style(
    session: ReviewSession,
    samples: list[StyleSample],
    graphics_path: Path,
    output_dir: Path,
    dictionary_path: Path | None = None,
):
    """Train/update one personal handwriting profile with validation and rollback.

    v0.8 uses every accepted facsimile sample accumulated for this profile, splits
    by character identity (not by stroke), trains deterministic/context models plus
    Structure NN and Stroke NN, validates on held-out characters, and only promotes
    a new model version when validation does not regress.
    """
    if not samples:
        raise ValueError("没有可用于训练的有效矢量样本")
    output_dir.mkdir(parents=True, exist_ok=True)
    library_root = output_dir.parent / "facsimile_library"
    cumulative = load_cumulative_samples(library_root, graphics_path)
    if not cumulative:
        cumulative = list(samples)
    train_samples, val_samples = split_by_character(cumulative)

    profile = learn_profile(train_samples)
    profile.save(output_dir / "style.json")
    chars = [s.char for s in cumulative]
    report = analyze_coverage(chars, graphics_path, dictionary_path)
    report.save(output_dir / "coverage.json")
    readiness = assess_readiness(report)

    structure_mse = None
    stroke_mse = None
    if dictionary_path and dictionary_path.exists():
        learn_component_styles(train_samples, dictionary_path).save(
            output_dir / "component_style.json"
        )
        position, structure, adjacency = learn_context_models(train_samples, dictionary_path)
        position.save(output_dir / "position_style.json")
        structure.save(output_dir / "structure_style.json")
        adjacency.save(output_dir / "adjacency_style.json")

        if readiness.context_ready and len(train_samples) >= 4:
            smeta = train_structure_network(
                train_samples,
                val_samples,
                dictionary_path,
                output_dir / "writer_structure.pt",
                output_dir / "writer_structure_meta.json",
            )
            structure_mse = smeta.validation_mse

        if readiness.tiny_ready:
            tmeta = train_tiny_trajectory(
                train_samples,
                dictionary_path,
                output_dir / "writer_stroke.pt",
                output_dir / "writer_stroke_meta.json",
                val_samples=val_samples,
            )
            # Backward-compatible aliases for existing generation code/tools.
            import shutil

            shutil.copy2(output_dir / "writer_stroke.pt", output_dir / "tiny_trajectory.pt")
            shutil.copy2(
                output_dir / "writer_stroke_meta.json", output_dir / "tiny_trajectory_meta.json"
            )
            stroke_mse = tmeta.validation_mse

        # Active learning uses current coverage plus held-out validation difficulty.
        hard_ops: list[str] = []
        if structure_mse is not None and structure_mse > 0.002:
            hard_ops = [
                k for k, v in sorted(report.structure_occurrences.items(), key=lambda kv: kv[1])[:2]
            ]
        plan = recommend_active_samples(report, dictionary_path, hard_operators=hard_ops)
        plan.save(output_dir / "active_learning_plan.json")
        if plan.suggested_text:
            report.suggested_text = plan.suggested_text
            report.save(output_dir / "coverage.json")

    # Validation is mandatory once a hold-out set exists. With too few unique
    # characters the profile remains NEED_MORE_SAMPLES even if fixed count gates pass.
    if not val_samples:
        readiness.ready = False
        readiness.status = "NEED_MORE_SAMPLES"
        readiness.reasons.append("不同汉字不足以建立独立留出验证集（至少建议5个不同汉字）")
        readiness.validation_ready = False
    else:
        readiness.validation_ready = structure_mse is not None or stroke_mse is not None

    if not readiness.validation_ready:
        readiness.ready = False
        readiness.status = "NEED_MORE_SAMPLES"
        readiness.reasons.append("尚无独立验证分数，不能宣称未写字生成已通过验收")

    version = accept_or_rollback(
        output_dir,
        structure_mse,
        stroke_mse,
        len({s.char for s in train_samples}),
        len({s.char for s in val_samples}),
    )
    readiness.active_model_version = version.version if version.accepted else 0
    # Reload active version number after rollback/rejection.
    from hwplotter.model.continual_learning import ContinualState

    state = ContinualState.load(output_dir / "continual_state.json")
    readiness.active_model_version = state.active_version
    readiness.validation_error = state.best_validation_error
    if not version.accepted:
        readiness.reasons.append("本轮增量训练验证误差变差，已自动回滚到上一模型版本")
    readiness.save(output_dir / "training_readiness.json")
    return profile, report, readiness


def generate_text_v05(
    text: str,
    graphics_path: Path,
    dictionary_path: Path,
    model_dir: Path,
    library_root: Path,
    output_dir: Path,
    variation: float = 0.08,
) -> list[Path]:
    """GUI-facing strict generator. Unseen characters require READY training state."""
    from hwplotter.exporters.gcode import GCodeOptions, save_gcode
    from hwplotter.pipeline import hierarchical_synthesize
    from hwplotter.style.component_profile import ComponentStyleModel
    from hwplotter.style.context_models import (
        AdjacencyStyleModel,
        PositionStyleModel,
        StructureStyleModel,
    )
    from hwplotter.style.readiness import TrainingReadiness

    profile = StyleProfile.load(model_dir / "style.json")
    readiness = TrainingReadiness.load(model_dir / "training_readiness.json")
    comp_p = model_dir / "component_style.json"
    pos_p = model_dir / "position_style.json"
    str_p = model_dir / "structure_style.json"
    adj_p = model_dir / "adjacency_style.json"
    component = ComponentStyleModel.load(comp_p) if comp_p.exists() else None
    position = PositionStyleModel.load(pos_p) if pos_p.exists() else None
    structure = StructureStyleModel.load(str_p) if str_p.exists() else None
    adjacency = AdjacencyStyleModel.load(adj_p) if adj_p.exists() else None
    tiny_pt = model_dir / "writer_stroke.pt"
    tiny_meta = model_dir / "writer_stroke_meta.json"
    structure_nn_pt = model_dir / "writer_structure.pt"
    structure_nn_meta = model_dir / "writer_structure_meta.json"
    library = FacsimileLibrary(library_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for i, char in enumerate(text):
        if char.isspace():
            continue
        c = library.replay(char, variant_index=i)
        if c is None:
            c = hierarchical_synthesize(
                char,
                graphics_path,
                profile,
                dictionary_path,
                component,
                position,
                structure,
                adjacency,
                tiny_pt if tiny_pt.exists() else None,
                tiny_meta if tiny_meta.exists() else None,
                readiness=readiness,
                variation=variation,
                seed=i,
                require_ready=True,
                structure_nn_path=structure_nn_pt if structure_nn_pt.exists() else None,
                structure_nn_meta_path=structure_nn_meta if structure_nn_meta.exists() else None,
            )
        stem = output_dir / f"{i:04d}_{ord(char):04x}_{char}"
        save_svg(c, stem.with_suffix(".svg"))
        save_json(c, stem.with_suffix(".json"))
        save_gcode(c, stem.with_suffix(".gcode"), GCodeOptions())
        outputs.append(stem.with_suffix(".svg"))
    return outputs
