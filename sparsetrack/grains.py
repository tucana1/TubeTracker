"""Grain census on the pre-germination reference image."""

from __future__ import annotations

import cv2
import numpy as np


def to_uint8(image: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.8) -> np.ndarray:
    lo, hi = np.percentile(image, [lo_pct, hi_pct])
    return np.clip((image.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def _ring_scores(image: np.ndarray, x: float, y: float, r: float) -> tuple[float, float]:
    """(ring contrast, body contrast): outside-annulus mean minus rim / disc mean."""
    h, w = image.shape
    pad = int(np.ceil(r + 10))
    x0, x1 = max(0, int(x) - pad), min(w, int(x) + pad + 1)
    y0, y1 = max(0, int(y) - pad), min(h, int(y) + pad + 1)
    sub = image[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    outside = sub[(d > r + 4) & (d < r + 8)]
    ring = sub[(d > r - 1.5) & (d < r + 1.0)]
    disc = sub[d < r - 1.0]
    if not len(outside) or not len(ring) or not len(disc):
        return 0.0, 0.0
    base = float(np.median(outside))
    return base - float(ring.mean()), base - float(disc.mean())


def rim_fit(image: np.ndarray, cx: float, cy: float, r: float, search: float = 8.0,
            step: float = 0.5) -> tuple[float, float, float]:
    """Best grain-centre offset near (cx, cy) for a dark rim of radius r: (dx, dy, score).

    Score = mean just inside and outside the rim (+/- 3.5 px) minus the rim mean; about 0
    where there is no grain.
    """
    phis = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    offs = np.arange(-search, search + 1e-9, step)
    ox, oy = (a.ravel() for a in np.meshgrid(offs, offs))

    def ring(rad):
        mx = (cx + ox[:, None] + rad * np.cos(phis)[None]).astype(np.float32)
        my = (cy + oy[:, None] + rad * np.sin(phis)[None]).astype(np.float32)
        return cv2.remap(image.astype(np.float32), mx, my, cv2.INTER_LINEAR).mean(axis=1)

    score = 0.5 * (ring(r - 3.5) + ring(r + 3.5)) - ring(r)
    i = int(np.argmax(score))
    return float(ox[i]), float(oy[i]), float(score[i])


def flat_field(image: np.ndarray, sigma: float = 40.0) -> np.ndarray:
    """Divide out slow illumination changes (vignetting), keeping the median brightness."""
    image = image.astype(np.float64)
    background = cv2.GaussianBlur(image, (0, 0), sigma)
    return image / np.maximum(background, 1.0) * float(np.median(image))


def detect(reference: np.ndarray, r_min: int = 9, r_max: int = 18, ring_min: float = 10.0,
           body_min: float = 25.0, flatfield: bool = False) -> list[dict]:
    """Detect grains as dark-rimmed (or dark-filled) circles; returns dicts sorted row-major."""
    if flatfield:
        reference = flat_field(reference)
    blur = cv2.GaussianBlur(to_uint8(reference), (0, 0), 1.2)
    found = []
    for param2 in (18, 13):
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=14, param1=60,
                                   param2=param2, minRadius=r_min, maxRadius=r_max)
        if circles is not None:
            found.extend(circles[0].tolist())
    candidates = []
    for x, y, r in found:
        ring, body = _ring_scores(reference, x, y, r)
        if ring >= ring_min or body >= body_min:
            candidates.append({"x": float(x), "y": float(y), "r": float(r),
                               "ring_contrast": round(ring, 2), "body_contrast": round(body, 2)})
    # suppress duplicate circles on one grain (touching grains are ~2r apart and survive)
    candidates.sort(key=lambda c: -max(c["ring_contrast"], c["body_contrast"] / 2.5))
    kept: list[dict] = []
    for c in candidates:
        if all(np.hypot(c["x"] - k["x"], c["y"] - k["y"]) >= 1.2 * min(c["r"], k["r"]) for k in kept):
            kept.append(c)
    return annotate_layout(kept, reference.shape)


def annotate_layout(grains: list[dict], shape: tuple[int, int], clump_factor: float = 2.4,
                    border_margin: float = 8.0) -> list[dict]:
    """Add id, nearest-neighbour distance, clump size and border flag; sort row-major."""
    h, w = shape
    grains = sorted(grains, key=lambda g: (round(g["y"] / 40.0), g["x"]))
    pts = np.array([[g["x"], g["y"]] for g in grains]) if grains else np.zeros((0, 2))
    radii = np.array([g["r"] for g in grains]) if grains else np.zeros(0)
    n = len(grains)
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1]) if n else np.zeros((0, 0))
    np.fill_diagonal(dist, np.inf)
    # clumps: single-linkage over near-touching grains
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < clump_factor * 0.5 * (radii[i] + radii[j]):
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    sizes = {r: roots.count(r) for r in set(roots)}
    for i, g in enumerate(grains):
        g["id"] = g.get("id") or f"g{i + 1:03d}"
        g["nn_dist"] = round(float(dist[i].min()), 1) if n > 1 else None
        g["clump_size"] = sizes[roots[i]]
        edge = min(g["x"], g["y"], w - 1 - g["x"], h - 1 - g["y"])
        g["border"] = bool(edge < g["r"] + border_margin)
        g["isolated"] = bool(g["clump_size"] == 1 and not g["border"])
        g["source"] = g.get("source", "auto")
        for k in ("x", "y", "r"):
            g[k] = round(g[k], 2)
    return grains
