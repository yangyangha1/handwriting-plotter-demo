"""Conservative ruling cleanup: only long rules touching the frame margins."""

import cv2
import numpy as np


def remove_frame_rules(mask):
    ink = (mask > 0).astype(np.uint8)
    h, w = ink.shape
    if min(h, w) < 20:
        return mask.copy()
    remove = np.zeros_like(ink)
    min_span = max(10, round(min(h, w) * 0.16))
    edge_band = max(4, round(min(h, w) * 0.11))
    # Morphological opening catches continuous rules even when the character touches them.
    for axis, length in ((0, w), (1, h)):
        kernel = (
            (max(10, round(length * 0.16)), 1) if axis == 0 else (1, max(10, round(length * 0.16)))
        )
        rules = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(rules)
        for label in range(1, count):
            x, y, bw, bh, _ = stats[label]
            near_edge = (
                (y < edge_band or y + bh > h - edge_band)
                if axis == 0
                else (x < edge_band or x + bw > w - edge_band)
            )
            span = bw if axis == 0 else bh
            thin = bh <= max(5, h * 0.08) if axis == 0 else bw <= max(5, w * 0.08)
            if near_edge and thin and span >= min_span:
                remove[labels == label] = 1

    # Hough tolerates gaps caused by the glyph crossing a ruling line.
    lines = cv2.HoughLinesP(
        (ink * 255),
        1,
        np.pi / 180.0,
        threshold=max(8, round(min(h, w) * 0.12)),
        minLineLength=min_span,
        maxLineGap=max(3, round(min(h, w) * 0.07)),
    )
    if lines is not None:
        # OpenCV returns either (N, 1, 4) or (N, 4) depending on the
        # platform/build; normalize both forms before unpacking.
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
            horizontal = dx >= min_span and dy <= max(2, round(dx * 0.035))
            vertical = dy >= min_span and dx <= max(2, round(dy * 0.035))
            near_edge = (
                horizontal and (min(y1, y2) <= edge_band or max(y1, y2) >= h - edge_band)
            ) or (vertical and (min(x1, x2) <= edge_band or max(x1, x2) >= w - edge_band))
            if near_edge and (horizontal or vertical):
                cv2.line(remove, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness=5)

    # A ruling may be broken into short dashes. Remove only dominant edge rows/columns.
    for y in list(range(edge_band)) + list(range(max(edge_band, h - edge_band), h)):
        if int(ink[y].sum()) >= max(8, round(w * 0.16)):
            remove[max(0, y - 2) : min(h, y + 3), :] |= ink[y : y + 1]
    for x in list(range(edge_band)) + list(range(max(edge_band, w - edge_band), w)):
        if int(ink[:, x].sum()) >= max(8, round(h * 0.16)):
            remove[:, max(0, x - 2) : min(w, x + 3)] |= ink[:, x : x + 1]

    out = mask.copy()
    out[remove > 0] = 0
    return out
