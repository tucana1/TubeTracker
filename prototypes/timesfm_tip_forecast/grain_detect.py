"""Precision grain-center detection via multi-scale radial symmetry (H177).

Pollen grains are dark-rimmed bright disks (~13-20 px radius).  The
existing recall path (Hough-flavored ``detect_grain_candidates`` +
Cellpose census) leaves centers ~18 px off the rim-fitted middle
(P58: template (535.7, 473.7) vs rim fit (535, 473) — close — but the
series median (536.5, 472.1) vs bright centroid (539, 483) disagree by
~11 px, and drifted centers reach 40 px+).  This module implements the
Fast Radial Symmetry Transform (Loy & Zelinsky 2003) for the bright
disk interior, multi-scale over the grain-radius band, with
radius-scaled non-maximum suppression.  No weights, no training data:
pure image structure.  Intended consumer: grain anchors for overlay
crops and (with TRUE radii, still open) grain-masked root support.
"""

from __future__ import annotations

import cv2
import numpy as np


def _frst_single(
    gray: np.ndarray, radius: int, alpha: float = 2.0, dark: bool = False
) -> np.ndarray:
    """Radial symmetry contribution of one radius.

    Bright centers (grain interiors): vote opposite the gradient.
    Dark centers (rim arcs as seen from outside): vote along it.
    """
    g = gray.astype(np.float64)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    # Ignore flat background (bottom decile of gradient energy).
    thr = np.percentile(mag, 90) * 0.1
    ys, xs = np.nonzero(mag > thr)
    if len(xs) == 0:
        return np.zeros_like(g)
    mx = mag[ys, xs]
    ux = gx[ys, xs] / np.maximum(mx, 1e-9)
    uy = gy[ys, xs] / np.maximum(mx, 1e-9)
    # Bright disk: gradient points outward, center lies opposite.
    # Dark ring (dark=True): center lies along the gradient.
    sgn = 1.0 if dark else -1.0
    px = np.round(xs + sgn * ux * radius).astype(int)
    py = np.round(ys + sgn * uy * radius).astype(int)
    h, w = g.shape
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px, py, mx = px[ok], py[ok], mx[ok]
    acc = np.zeros_like(g)
    np.add.at(acc, (py, px), mx)
    # Strict radiality: sharpen by local competition (alpha power
    # normalized by a uniform vote count).
    cnt = np.zeros_like(g)
    np.add.at(cnt, (py, px), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(cnt > 0, (acc / np.maximum(cnt, 1.0)) ** alpha * cnt, 0.0)
    return out


def grain_symmetry_maps(
    gray: np.ndarray,
    radii: tuple[int, ...] = (11, 14, 17, 20),
    alpha: float = 2.0,
    blur: float = 1.0,
    dark_weight: float = 0.5,
) -> dict[int, np.ndarray]:
    """Per-radius FRST maps (H188): argmax over radii sizes detections."""
    k = max(3, int(round(blur * 2)) * 2 + 1)
    maps = {}
    for r in radii:
        m = _frst_single(gray, r, alpha) + dark_weight * _frst_single(
            gray, r, alpha, dark=True
        )
        maps[r] = cv2.GaussianBlur(m, (k, k), blur)
    return maps


def grain_symmetry_map(
    gray: np.ndarray,
    radii: tuple[int, ...] = (11, 14, 17, 20),
    alpha: float = 2.0,
    blur: float = 1.0,
    dark_weight: float = 0.5,
) -> np.ndarray:
    """Sum single-radius FRST maps, smoothed at the grain scale.

    Bright-interior votes plus ``dark_weight``-scaled dark-center votes
    (rim arcs, germinated grains whose tube breaks bright symmetry).
    """
    maps = grain_symmetry_maps(gray, radii, alpha, blur, dark_weight)
    total = None
    for m in maps.values():
        total = m if total is None else total + m
    assert total is not None
    return total


def peak_radii(
    maps: dict[int, np.ndarray],
    centers_xy: list[tuple[float, float]],
) -> list[float]:
    """Best-response radius per center, parabolic sub-step refined."""
    rs = sorted(maps)
    out = []
    for (x, y) in centers_xy:
        xi, yi = int(round(x)), int(round(y))
        vals = []
        for r in rs:
            m = maps[r]
            if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1]:
                vals.append(float(m[yi, xi]))
            else:
                vals.append(0.0)
        j = int(np.argmax(vals))
        if 0 < j < len(rs) - 1 and vals[j] > 0:
            # Parabolic refinement in radius.
            y0, y1, y2 = vals[j - 1], vals[j], vals[j + 1]
            denom = (y0 - 2 * y1 + y2)
            dj = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
            dj = float(np.clip(dj, -1.0, 1.0))
            step = (rs[j + 1] - rs[j - 1]) / 2.0
            out.append(rs[j] + dj * step)
        else:
            out.append(float(rs[j]))
    return out


def split_mergers(
    gray: np.ndarray,
    dets: list[tuple[float, float, float]],
    radii: tuple[int, ...] = (11, 14, 17, 20),
    merge_radius: float = 19.0,
    split_radii: tuple[int, ...] = (8, 10, 12),
    split_nms: float = 6.0,
    zone_frac: float = 0.5,
    zone_footprint: int = 11,
    max_splits: int = 4,
) -> list[tuple[float, float, float]]:
    """Second finer pass inside oversized detections (H189).

    Cluster mergers read peak_radius >= ``merge_radius`` (one circle
    spanning several grains, position unreliable).  Re-detect the zone
    with small radii + tight NMS; keep the split set iff it finds >=2
    (else the original stands).  Returns a new list, score order kept.
    """
    maps = grain_symmetry_maps(gray, radii)
    pr = peak_radii(maps, [(x, y) for (x, y, _s) in dets])
    small = grain_symmetry_maps(gray, split_radii)
    small_sum = None
    for m in small.values():
        small_sum = m if small_sum is None else small_sum + m
    assert small_sum is not None
    peak = float(small_sum.max())
    out: list[tuple[float, float, float]] = []
    for (x, y, s), r in zip(dets, pr):
        if r < merge_radius:
            out.append((x, y, s))
            continue
        R = r + 8.0
        x0, y0 = int(max(0, x - R)), int(max(0, y - R))
        zone = small_sum[y0:int(y + R), x0:int(x + R)]
        if zone.size == 0:
            out.append((x, y, s))
            continue
        # Local maxima (9px footprint) at 0.35 of the ZONE max: one big
        # grain yields one maximum (kept whole); a true cluster yields
        # several (split).  Global thresholds starve zones; zone-local
        # ones oversplit singles — maxima count decides, not energy.
        zmax = float(zone.max())
        dil = cv2.dilate(zone, cv2.getStructuringElement(
            cv2.MORPH_RECT, (zone_footprint, zone_footprint)))
        ys, xs = np.nonzero((zone == dil) & (zone >= zone_frac * zmax))
        cand = sorted(
            ((float(xs[i] + x0), float(ys[i] + y0),
              float(zone[ys[i], xs[i]]) / peak) for i in range(len(xs))),
            key=lambda t: -t[2],
        )
        kept: list[tuple[float, float, float]] = []
        for (cx, cy, cs) in cand:
            if all((cx - kx) ** 2 + (cy - ky) ** 2 >= split_nms**2
                   for (kx, ky, _ks) in kept):
                kept.append((cx, cy, cs))
            if len(kept) >= max_splits:
                break
        out.extend(kept if len(kept) >= 2 else [(x, y, s)])
    return sorted(out, key=lambda d: -d[2])


def detect_grains(
    gray: np.ndarray,
    radii: tuple[int, ...] = (11, 14, 17, 20),
    threshold_frac: float = 0.35,
    nms_radius: float = 12.0,
    tiles: tuple[int, int] = (4, 3),
    tile_floor_frac: float = 0.08,
) -> list[tuple[float, float, float]]:
    """Return (x, y, score) grain centers, strongest first.

    Threshold is relative to the frame maximum (self-normalizing across
    illumination); NMS suppresses within ``nms_radius`` px.  With
    ``tiles=(nx, ny)``, each tile thresholds at ``threshold_frac`` of
    its own maximum (floor: ``tile_floor_frac`` of the global maximum),
    so hot grains stop suppressing weaker neighbors; scores stay
    global (peak-normalized) for comparability.
    """
    sm = grain_symmetry_map(gray, radii)
    peak = float(sm.max())
    if peak <= 0:
        return []
    H, W = sm.shape
    nx, ny = tiles
    keep_xy: list[tuple[int, int]] = []
    for ty in range(ny):
        for tx in range(nx):
            t = sm[ty * H // ny:(ty + 1) * H // ny, tx * W // nx:(tx + 1) * W // nx]
            thr = max(threshold_frac * float(t.max()), tile_floor_frac * peak)
            ys, xs = np.nonzero(t >= thr)
            keep_xy.extend((xs + tx * W // nx, ys + ty * H // ny) for xs, ys in zip(xs, ys))
    if not keep_xy:
        return []
    xs = np.array([p[0] for p in keep_xy])
    ys = np.array([p[1] for p in keep_xy])
    order = np.argsort(-sm[ys, xs])
    kept: list[tuple[float, float, float]] = []
    taken = np.zeros(len(xs), dtype=bool)
    for i in order:
        if taken[i]:
            continue
        x, y = float(xs[i]), float(ys[i])
        kept.append((x, y, float(sm[ys[i], xs[i]]) / peak))
        d2 = (xs.astype(float) - x) ** 2 + (ys.astype(float) - y) ** 2
        taken |= d2 < nms_radius**2
    return kept


def angular_coverage(
    gray: np.ndarray,
    centers_xy: list[tuple[float, float]],
    rmin: float = 9.0,
    rmax: float = 22.0,
    bins: int = 12,
) -> list[float]:
    """Fraction of angular bins with gradient-voter support (H187).

    For each center, voters are above-background gradient pixels in the
    [rmin, rmax] annulus; a voter's angle is its position angle around
    the center.  Whole grains read ~1.0; tube bends read ~0.3-0.5
    (one-sided arcs); debris reads low.  Pure function of the image.
    """
    g = np.asarray(gray, dtype=float)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    thr = np.percentile(mag, 90) * 0.1
    ys, xs = np.nonzero(mag > thr)
    out = []
    for (cx, cy) in centers_xy:
        dx = xs.astype(float) - cx
        dy = ys.astype(float) - cy
        rr = np.hypot(dx, dy)
        sel = (rr >= rmin) & (rr <= rmax)
        if not sel.any():
            out.append(0.0)
            continue
        b = np.floor((np.arctan2(dy[sel], dx[sel]) + np.pi)
                     / (2 * np.pi) * bins).astype(int) % bins
        out.append(float(len(np.unique(b))) / bins)
    return out


def detect_grains_covered(
    gray: np.ndarray,
    min_coverage: float = 0.60,
    edge_margin: float = 24.0,
    **kwargs,
) -> list[tuple[float, float, float]]:
    """detect_grains filtered to full-circle supporters (H187).

    Detections within ``edge_margin`` of the frame border keep their
    votes unfiltered: the annulus clips there and coverage reads
    artificially low.
    """
    dets = detect_grains(gray, **kwargs)
    if not dets:
        return dets
    h, w = np.asarray(gray).shape[:2]
    cov = angular_coverage(gray, [(x, y) for (x, y, _s) in dets])
    return [
        d for d, c in zip(dets, cov)
        if c >= min_coverage
        or d[0] < edge_margin or d[1] < edge_margin
        or d[0] > w - edge_margin or d[1] > h - edge_margin
    ]


def confirm_grains(
    frames_xy: list[list[tuple[float, float, float]]],
    match_px: float = 8.0,
) -> list[tuple[float, float, float]]:
    """Keep detections that persist across frames (H186).

    Grains are near-static (per-sample drift is a few px); debris
    flicker, tube-bend transients, and threshold-edge noise are not.
    A detection in any frame survives iff another frame holds one
    within ``match_px``.  Score = max over matched frames.  Pure
    function on detection lists (no images) so it stays testable.
    """
    if not frames_xy:
        return []
    # A detection survives iff another frame holds one within match_px.
    kept: list[tuple[float, float, float]] = []
    for i, dets in enumerate(frames_xy):
        others = [d for j, ds in enumerate(frames_xy) for d in ds if j != i]
        for (x, y, s) in dets:
            if any((x - x2) ** 2 + (y - y2) ** 2 <= match_px**2
                   for (x2, y2, _s2) in others):
                kept.append((x, y, s))
    # Greedy clustering across frames: centroid + max score per cluster.
    clusters: list[list[tuple[float, float, float]]] = []
    for det in kept:
        for cl in clusters:
            cx = sum(d[0] for d in cl) / len(cl)
            cy = sum(d[1] for d in cl) / len(cl)
            if (det[0] - cx) ** 2 + (det[1] - cy) ** 2 <= match_px**2:
                cl.append(det)
                break
        else:
            clusters.append([det])
    out = [
        (sum(d[0] for d in cl) / len(cl),
         sum(d[1] for d in cl) / len(cl),
         max(d[2] for d in cl))
        for cl in clusters
    ]
    return sorted(out, key=lambda d: -d[2])
