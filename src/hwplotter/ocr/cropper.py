from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from hwplotter.model.entities import CharacterCrop, OCRToken
from hwplotter.vision.image_io import read_image, write_image


@dataclass(slots=True)
class RectifiedToken:
    image: np.ndarray
    polygon: np.ndarray
    inverse_h: np.ndarray
    horizontal: bool


def _order_quad(poly: np.ndarray) -> np.ndarray:
    """Return quad as TL, TR, BR, BL. Works for mildly rotated OCR boxes."""
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(p) != 4:
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        p = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    s = p.sum(axis=1)
    d = np.diff(p, axis=1).ravel()
    return np.array(
        [p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], np.float32
    )


def _rectify_token(image: np.ndarray, token: OCRToken) -> RectifiedToken:
    quad = _order_quad(token.polygon)
    tl, tr, br, bl = quad
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    width = max(2, round(max(top, bottom)))
    height = max(2, round(max(left, right)))
    horizontal = width >= height
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    h = cv2.getPerspectiveTransform(quad, dst)
    inv = cv2.getPerspectiveTransform(dst, quad)
    warped = cv2.warpPerspective(
        image,
        h,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    if not horizontal:
        # Normalize vertical OCR tokens to a top-to-bottom strip.  Segmentation below
        # uses the long axis; the saved crops remain upright as detected.
        pass
    return RectifiedToken(warped, quad, inv, horizontal)


def _char_width_weight(ch: str) -> float:
    """Loose width prior for natural text; CJK remains full-width, punctuation narrower."""
    if not ch or ch.isspace():
        return 0.0
    cat = unicodedata.category(ch)
    east = unicodedata.east_asian_width(ch)
    if cat.startswith("P"):
        return 0.45 if east in {"W", "F", "A"} else 0.35
    if ch.isascii():
        return 0.58
    return 1.0


def _ink_projection(gray: np.ndarray, horizontal: bool) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Remove isolated camera/scan noise without joining neighboring glyphs.
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    projection = ink.sum(axis=0 if horizontal else 1).astype(np.float32)
    if len(projection) >= 7:
        projection = cv2.GaussianBlur(projection.reshape(1, -1), (7, 1), 0).ravel()
    return projection


def _global_projection_cuts(
    gray: np.ndarray, chars: list[str], horizontal: bool
) -> tuple[list[int], list[float]]:
    """Globally choose character boundaries for a normal handwritten text token.

    OCR supplies the character sequence.  We do *not* assume grid cells.  Expected
    positions are only soft width priors; each separator is selected from a broad
    neighborhood using low-ink evidence and a global dynamic program so irregular
    handwriting spacing is preserved.
    """
    n = len(chars)
    length = gray.shape[1] if horizontal else gray.shape[0]
    if n <= 1:
        return [0, length], [1.0]
    projection = _ink_projection(gray, horizontal)
    pmax = float(max(projection.max(), 1.0))
    pn = projection / pmax

    weights = np.array([max(_char_width_weight(c), 0.25) for c in chars], np.float32)
    cum = np.cumsum(weights) / float(weights.sum())
    targets = np.rint(cum[:-1] * length).astype(int)
    nominal = max(4.0, length / max(float(weights.sum()), 1.0))
    min_gap = max(2, round(nominal * 0.26))
    radius = max(5, round(nominal * 0.60))

    candidates: list[np.ndarray] = []
    costs: list[np.ndarray] = []
    for target in targets:
        lo = max(min_gap, int(target - radius))
        hi = min(length - min_gap, int(target + radius))
        pos = np.arange(lo, hi + 1, dtype=np.int32)
        if len(pos) == 0:
            pos = np.array([int(np.clip(target, 1, length - 1))], np.int32)
        dist_cost = np.abs(pos - target) / max(radius, 1)
        # Ink valley dominates; target distance is a soft prior only.
        c = pn[pos] + 0.22 * dist_cost
        candidates.append(pos)
        costs.append(c.astype(np.float32))

    # Dynamic program enforces ordered, non-collapsing cells while letting each
    # boundary move independently according to real handwriting spacing.
    dp: list[np.ndarray] = []
    prev_idx: list[np.ndarray] = []
    dp.append(costs[0].copy())
    prev_idx.append(np.full(len(candidates[0]), -1, np.int32))
    for i in range(1, n - 1):
        cur = np.full(len(candidates[i]), np.inf, np.float32)
        back = np.full(len(candidates[i]), -1, np.int32)
        for j, x in enumerate(candidates[i]):
            valid = np.where(candidates[i - 1] <= x - min_gap)[0]
            if len(valid) == 0:
                continue
            vals = dp[i - 1][valid]
            krel = int(np.argmin(vals))
            k = int(valid[krel])
            cur[j] = costs[i][j] + vals[krel]
            back[j] = k
        dp.append(cur)
        prev_idx.append(back)

    if not np.isfinite(dp[-1]).any():
        # Defensive fallback; still uses soft weighted positions, never a grid UI.
        cuts = [0] + [int(x) for x in targets] + [length]
    else:
        j = int(np.nanargmin(dp[-1]))
        chosen = [int(candidates[-1][j])]
        for i in range(n - 2, 0, -1):
            j = int(prev_idx[i][j])
            if j < 0:
                break
            chosen.append(int(candidates[i - 1][j]))
        chosen.reverse()
        if len(chosen) != n - 1:
            chosen = [int(x) for x in targets]
        cuts = [0] + chosen + [length]

    # Per-character segmentation confidence.  A good separator has little ink and
    # reasonable cell width; low scores are surfaced to the review UI.
    confs: list[float] = []
    expected_widths = weights / weights.sum() * length
    for i in range(n):
        a, b = cuts[i], cuts[i + 1]
        width_score = np.exp(-abs((b - a) - expected_widths[i]) / max(expected_widths[i], 1.0))
        boundary_ink = 0.0
        count = 0
        for c in (a if i else None, b if i < n - 1 else None):
            if c is not None:
                lo = max(0, c - 1)
                hi = min(length, c + 2)
                boundary_ink += float(np.mean(pn[lo:hi]))
                count += 1
        valley_score = 1.0 - min(1.0, boundary_ink / max(count, 1))
        confs.append(float(np.clip(0.68 * valley_score + 0.32 * width_score, 0.0, 1.0)))
    return cuts, confs


def _polygon_from_rectified(
    inv_h: np.ndarray, a: int, b: int, short: int, horizontal: bool
) -> np.ndarray:
    if horizontal:
        q = np.array([[[a, 0], [b, 0], [b, short - 1], [a, short - 1]]], np.float32)
    else:
        q = np.array([[[0, a], [short - 1, a], [short - 1, b], [0, b]]], np.float32)
    return cv2.perspectiveTransform(q, inv_h)[0].astype(np.float32)


def _axis_bbox_from_polygon(
    src: np.ndarray, image_w: int, image_h: int
) -> tuple[int, int, int, int]:
    x0 = max(0, int(np.floor(src[:, 0].min())))
    x1 = min(image_w, int(np.ceil(src[:, 0].max())))
    y0 = max(0, int(np.floor(src[:, 1].min())))
    y1 = min(image_h, int(np.ceil(src[:, 1].max())))
    return x0, y0, x1, y1


def _extract_boundary_connector(
    gray: np.ndarray, cut: int, horizontal: bool, inv_h: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Recover a light inter-character pen-down bridge around a segmentation cut.

    A straight character crop must cut any real connection somewhere.  Rather than
    discarding that ink, detect a thin connected component traversing a narrow band
    around the separator and preserve it as an independent page-space polyline.
    This is intentionally conservative: broad components are treated as ambiguous
    glyph ink, not as a connector.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    short = gray.shape[0] if horizontal else gray.shape[1]
    long_len = gray.shape[1] if horizontal else gray.shape[0]
    radius = max(4, min(14, round(short * 0.14)))
    lo, hi = max(0, cut - radius), min(long_len, cut + radius + 1)
    band = ink[:, lo:hi] if horizontal else ink[lo:hi, :]
    if band.size == 0 or int(band.sum()) < 4:
        return None
    num, labels, _stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
    best = None
    best_score = -1.0
    for lab in range(1, num):
        ys, xs = np.where(labels == lab)
        if not len(xs):
            continue
        long_axis = xs if horizontal else ys
        cross_axis = ys if horizontal else xs
        span = int(long_axis.max() - long_axis.min() + 1)
        if span < max(4, int(0.70 * (hi - lo))):
            continue
        # A bridge should be thin relative to line height/width; wide blobs are
        # likely a character body cut in the wrong place and should be reviewed.
        thickness = float(len(xs)) / max(span, 1)
        cross_span = float(cross_axis.max() - cross_axis.min() + 1) / max(short, 1)
        if thickness > max(8.0, short * 0.18) or cross_span > 0.62:
            continue
        score = span / max(hi - lo, 1) - 0.35 * cross_span
        if score > best_score:
            best_score = score
            best = (ys, xs)
    if best is None:
        return None
    ys, xs = best
    coords = []
    if horizontal:
        for x in range(int(xs.min()), int(xs.max()) + 1):
            yy = ys[xs == x]
            if len(yy):
                coords.append((lo + x, float(np.median(yy))))
    else:
        for y in range(int(ys.min()), int(ys.max()) + 1):
            xx = xs[ys == y]
            if len(xx):
                coords.append((float(np.median(xx)), lo + y))
    if len(coords) < 3:
        return None
    rect_pts = np.asarray(coords, np.float32)
    q = rect_pts.reshape(1, -1, 2)
    page_pts = cv2.perspectiveTransform(q, inv_h)[0].astype(np.float32)
    return page_pts, rect_pts


def split_token_to_char_crops(
    image_path: Path, token: OCRToken, out_dir: Path
) -> list[CharacterCrop]:
    """Split a detected *natural handwritten text* token into character crops.

    No grid/template coordinates are required.  The OCR polygon is perspective-rectified,
    boundaries are estimated from actual ink valleys with only soft character-width priors,
    and each crop is saved from the rectified token.  The original-image bbox is retained
    for page reconstruction.
    """
    image = read_image(image_path, cv2.IMREAD_COLOR)
    chars = [c for c in token.text if not c.isspace()]
    if not chars:
        return []

    rect = _rectify_token(image, token)
    gray = cv2.cvtColor(rect.image, cv2.COLOR_BGR2GRAY)
    cuts, seg_conf = _global_projection_cuts(gray, chars, rect.horizontal)

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.png")))
    h, w = image.shape[:2]
    rh, rw = rect.image.shape[:2]
    crops: list[CharacterCrop] = []
    for i, char in enumerate(chars):
        a, b = cuts[i], cuts[i + 1]
        if rect.horizontal:
            crop = rect.image[:, a:b]
            poly = _polygon_from_rectified(rect.inverse_h, a, b, rh, True)
            box = _axis_bbox_from_polygon(poly, w, h)
        else:
            crop = rect.image[a:b, :]
            poly = _polygon_from_rectified(rect.inverse_h, a, b, rw, False)
            box = _axis_bbox_from_polygon(poly, w, h)
        if crop.size == 0:
            continue
        path = out_dir / f"{existing + i:05d}_{ord(char):04x}.png"
        write_image(path, crop)
        confidence = float(np.clip(token.confidence * (0.55 + 0.45 * seg_conf[i]), 0.0, 1.0))
        crops.append(CharacterCrop(char, path, box, confidence, polygon=poly, char_position=i))

    # Preserve light pen-down bridges across adjacent characters.  They are not
    # assigned to either glyph model, because doing so would contaminate component
    # training; page replay/export owns them separately.
    for i, cut in enumerate(cuts[1:-1]):
        if i >= len(crops):
            continue
        detected = _extract_boundary_connector(gray, int(cut), rect.horizontal, rect.inverse_h)
        if detected is None:
            continue
        page_pts, rect_pts = detected
        crops[i].connector_to_next = page_pts
        # Remove only the narrow transition segment from the neighboring training
        # crops.  It remains available as a page-level pen-down trajectory, so replay
        # keeps the original appearance without teaching the bridge as glyph anatomy.
        for crop_index in (i, i + 1):
            if crop_index >= len(crops):
                continue
            a, b = cuts[crop_index], cuts[crop_index + 1]
            img = read_image(crops[crop_index].image_path, cv2.IMREAD_COLOR)
            pts = rect_pts.copy()
            if rect.horizontal:
                pts[:, 0] -= float(a)
                keep = (pts[:, 0] >= -2) & (pts[:, 0] < img.shape[1] + 2)
            else:
                pts[:, 1] -= float(a)
                keep = (pts[:, 1] >= -2) & (pts[:, 1] < img.shape[0] + 2)
            pts = np.rint(pts[keep]).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, (255, 255, 255), 5, lineType=cv2.LINE_AA)
                write_image(crops[crop_index].image_path, img)
    return crops
