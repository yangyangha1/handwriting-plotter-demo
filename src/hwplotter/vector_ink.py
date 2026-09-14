"""Image-faithful outlines and graph paths; neither claims temporal stroke recovery."""

import html

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.measure import find_contours
from skimage.morphology import skeletonize

from hwplotter.plotter_paths import join_continuations, smooth_paths


def read_mask(path, threshold=0, remove_red=True):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if im.width * im.height > 24_000_000:
        raise ValueError("图片超过2400万像素，请先缩小")
    a = np.asarray(im)
    # Red seals should not become handwriting; use red channel on antique paper.
    gray = a[:, :, 0] if remove_red else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if threshold:
        mask = (gray < threshold).astype(np.uint8)
    else:
        _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return im, mask


def outlines(mask, tolerance=0.35):
    # Marching squares uses half-pixel boundaries, preserving holes and border ink.
    return [
        cv2.approxPolyDP((p[:, ::-1] - 0.5).astype(np.float32), tolerance, True)
        .reshape(-1, 2)
        .tolist()
        for p in find_contours(np.pad(mask, 1), 0.5)
        if len(p) >= 4
    ]


def svg(polygons, width, height, title="原稿轮廓"):
    paths = []
    for p in polygons:
        if len(p) > 2:
            paths.append("M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in p) + " Z")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        f'<title>{html.escape(title)}</title><path fill="black" fill-rule="evenodd" d="'
        + " ".join(paths)
        + '"/></svg>'
    )


def center_paths(mask):
    """Visit each skeleton edge once. Junctions lift; cycles remain closed."""
    skel = skeletonize(mask > 0)
    nodes = set(map(tuple, np.argwhere(skel)))
    adj = {}
    for y, x in sorted(nodes):
        near = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            q = (y + dy, x + dx)
            if q in nodes:
                if dy and dx and ((y + dy, x) in nodes or (y, x + dx) in nodes):
                    continue
                near.append(q)
        adj[(y, x)] = near
    seen = set()
    result = []

    def walk(start, nxt):
        line = [start, nxt]
        seen.add(tuple(sorted((start, nxt))))
        while len(adj[line[-1]]) == 2:
            p = line[-1]
            candidates = [q for q in adj[p] if tuple(sorted((p, q))) not in seen]
            if not candidates:
                break
            q = candidates[0]
            seen.add(tuple(sorted((p, q))))
            line.append(q)
        return [[float(x) + 0.5, float(y) + 0.5] for y, x in line]

    for p in sorted(nodes):
        if len(adj[p]) != 2:
            for q in adj[p]:
                if tuple(sorted((p, q))) not in seen:
                    result.append(walk(p, q))
            if not adj[p]:
                result.append([[p[1] + 0.5, p[0] + 0.5], [p[1] + 0.51, p[0] + 0.5]])
    for p in sorted(nodes):
        for q in adj[p]:
            if tuple(sorted((p, q))) not in seen:
                result.append(walk(p, q))
    return smooth_paths(join_continuations(result))


def paths_svg(paths, width, height):
    ds = ["M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in p) for p in paths if p]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
        '<path fill="none" stroke="black" stroke-width="1" stroke-linecap="round" d="'
        + " ".join(ds)
        + '"/></svg>'
    )


def gcode(paths, width, page_mm=180, pen_up="M3 S0", pen_down="M3 S1000", feed=1200):
    if not 1 <= page_mm <= 1000 or not 1 <= feed <= 20000:
        raise ValueError("尺寸或速度超出范围")
    if "\n" in pen_up or "\n" in pen_down or "\r" in pen_up or "\r" in pen_down:
        raise ValueError("抬落笔命令只允许单行")
    scale = page_mm / max(width, 1)
    out = ["; graph traversal / not recovered handwriting order", "G21", "G90", pen_up]
    for path in paths:
        if not path:
            continue
        x, y = path[0]
        out.extend([f"G0 X{x * scale:.3f} Y{y * scale:.3f} F3000", pen_down])
        previous = (round(x * scale, 3), round(y * scale, 3))
        for x, y in path[1:]:
            point = (round(x * scale, 3), round(y * scale, 3))
            if point != previous:
                out.append(f"G1 X{point[0]:.3f} Y{point[1]:.3f} F{feed}")
                previous = point
        out.append(pen_up)
    return "\n".join(out) + "\n"
