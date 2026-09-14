"""Bounded, endpoint-preserving path cleanup in the input coordinate system."""

import cv2
import numpy as np


def smooth_path(points, tolerance=0.75):
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(p) < 3:
        return p.tolist()
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-7]]
    if len(p) < 3:
        return p.tolist()
    closed = bool(np.linalg.norm(p[0] - p[-1]) < 1e-6)
    # Remove subpixel reversals first. RDP bounds deviation from the original ink.
    q = cv2.approxPolyDP(p, tolerance, closed).reshape(-1, 2)
    if closed:
        if len(q) < 3:
            return p.tolist()
        q = np.vstack([q, q[0]])
    if len(q) < 3:
        return q.tolist()
    result = [q[0]]
    for i in range(1, len(q) - 1):
        a, b, c = q[i - 1 : i + 2]
        u, v = b - a, c - b
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        cosine = float(np.dot(u, v) / max(lu * lv, 1e-9))
        # Genuine hooks/folds remain exact; only small direction changes are rounded.
        if cosine < 0.5 or min(lu, lv) < 1e-6:
            result.append(b)
            continue
        radius = min(float(lu) * 0.25, float(lv) * 0.25, tolerance)
        entry, leave = b - u / lu * radius, b + v / lv * radius
        result.append(entry)
        for t in (0.25, 0.5, 0.75, 1.0):
            result.append((1 - t) ** 2 * entry + 2 * (1 - t) * t * b + t * t * leave)
    result.append(q[-1])
    return np.asarray(result).tolist()


def smooth_paths(paths, tolerance=0.75):
    return [smooth_path(p, tolerance) for p in paths if len(p)]


def join_continuations(paths):
    """Pair near-collinear branches sharing an exact junction, never bridge air."""
    lines = [np.asarray(p, dtype=float) for p in paths if len(p) >= 2]
    endpoints = {}
    for i, p in enumerate(lines):
        if np.array_equal(p[0], p[-1]):
            continue
        for end in (0, 1):
            endpoints.setdefault(tuple(p[-1 if end else 0]), []).append((i, end))
    links = {}
    for ports in endpoints.values():
        candidates = []
        for a, (i, end) in enumerate(ports):
            p = lines[i] if end == 0 else lines[i][::-1]
            u = p[min(4, len(p) - 1)] - p[0]
            for j, other in ports[a + 1 :]:
                if i == j:
                    continue
                q = lines[j] if other == 0 else lines[j][::-1]
                v = q[min(4, len(q) - 1)] - q[0]
                cosine = float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9))
                if cosine < -0.9:
                    candidates.append((cosine, (i, end), (j, other)))
        for _, a, b in sorted(candidates):
            if a not in links and b not in links:
                links[a], links[b] = b, a
    used, output = set(), []
    starts = [(i, end) for i in range(len(lines)) for end in (0, 1) if (i, end) not in links]
    starts += [(i, 0) for i in range(len(lines))]
    for i, end in starts:
        if i in used:
            continue
        joined = []
        while i not in used:
            used.add(i)
            line = lines[i] if end == 0 else lines[i][::-1]
            joined.extend(line.tolist() if not joined else line[1:].tolist())
            nxt = links.get((i, 1 - end))
            if nxt is None:
                break
            i, end = nxt
        output.append(joined)
    return output
