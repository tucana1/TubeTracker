"""Decoder v2, the per-bin decoder: read the tube where it is in every bin, then make its length monotone.

SparseTrack traces one centreline on the end-state change map and reads every bin along that
path (rotated rigidly). With a tube-probability map for every bin that is no longer necessary:

1. per grain, follow its drift (SparseTrack's ``local_shifts`` on the image cache);
2. per bin, keep P > ``thr`` outside grain bodies, take the connected region attached to the
   grain's rim, and split it from regions reaching other grains' rims by geodesic ownership
   (SparseTrack's ``geodesic_owner``);
3. the bin's raw length is the geodesic length along that region's medial axis, from its end
   nearest the grain (plus that end's distance from the rim), plus 1 px;
4. over bins, the length is the monotone curve (growth <= ``vmax`` px per bin) closest to the raw
   lengths in L1, so single-bin misses and flickers do not move it; onset is where it passes
   ``onset_px``.

Writes predictions in SparseTrack's schema, so ``sparsetrack.evaluate.score`` scores them.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

import sparsetrack.analyze as A
from sparsetrack import stack
from sparsetrack.evaluate import PRED_SCHEMA
from sparsetrack.render import Renderer


def monotone_l1(raw: np.ndarray, vmax: float, step: float = 0.5) -> np.ndarray:
    """Non-decreasing sequence with increments <= vmax minimising sum |x - raw| (grid DP)."""
    levels = np.arange(0.0, float(raw.max()) + 2 * step, step)
    k = max(1, int(round(vmax / step)))
    n, m = len(raw), len(levels)
    cost = np.abs(levels[None, :] - raw[:, None])
    best = cost[0].copy()
    back = np.zeros((n, m), np.int32)
    idx = np.arange(m)
    for t in range(1, n):
        pad = np.concatenate([np.full(k, np.inf), best])
        win = np.lib.stride_tricks.sliding_window_view(pad, k + 1)  # win[j] covers levels j-k..j
        j = np.argmin(win, axis=1)
        best = cost[t] + win[idx, j]
        back[t] = idx - (k - j)
    out = np.zeros(n, np.int32)
    out[-1] = int(np.argmin(best))
    for t in range(n - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return levels[out]


def burst_cut(raw: np.ndarray, min_px: float = 8.0, frac: float = 0.3, hold: float = 0.9,
              min_bins: int = 5) -> int | None:
    """First bin of a burst: the tube had been read at ``min_px`` or more, and from this bin on its
    reading stays below ``frac`` of the longest reading so far in at least ``hold`` of the remaining
    bins (at least ``min_bins`` of them). A burst tube leaves nothing to trace, so its reading collapses
    for good; a monotone fit over the whole movie would instead pull its growth down."""
    raw = np.asarray(raw, float)
    peak = np.maximum.accumulate(raw)
    for b in range(1, raw.size - min_bins + 1):
        if peak[b - 1] >= min_px and raw[b] < frac * peak[b - 1]:
            if np.mean(raw[b:] < frac * peak[b - 1]) >= hold:
                return b
    return None


def grain_gone(img: np.ndarray, ls: np.ndarray, centre: float, gr: float, rg: np.ndarray, run: int = 5) -> int | None:
    """First bin from which the grain is no longer where it was, for ``run`` bins or more: its inner disc
    (registered on the grain) has lost more than half its early contrast against the ground round it (the
    contrast, so that the whole movie brightening or dimming does not count)."""
    inner, back = rg < 0.7 * gr, (rg >= gr + 6.0) & (rg <= gr + 14.0)
    disc = np.empty(len(img))
    ground = np.empty(len(img))
    for i, (im, (dx, dy)) in enumerate(zip(img, ls)):
        w = cv2.warpAffine(im.astype(np.float32), np.float32([[1, 0, -dx], [0, 1, -dy]]), im.shape[::-1],
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        disc[i], ground[i] = float(w[inner].mean()), float(np.median(w[back]))
    c0 = float((ground[:3] - disc[:3]).mean())
    if abs(c0) < 1e-6:
        return None
    away = (ground - disc) / c0 < 0.5
    for i in range(len(away) - run + 1):
        if away[i:i + run].all():
            return i
    return None


def reach_grain(RP: Renderer, R_img: Renderer, meta: dict, grain: dict, others: list[dict], thr: float = 0.5,
                scale: float = 16.0, half: int = 150, vmax: float = 4.0, onset_px: float = 2.0,
                min_tube_px: float = 8.0, rim_band: float = 5.0, seed: str = "skeleton", end_px: float = 1.0,
                tip: str = "const", big: int | None = None, burst: bool = False, keep: tuple = (),
                paths: tuple = ()) -> dict:
    """Length per bin, then a monotone fit. ``seed`` chooses how the bin's length is read:
    ``"skeleton"`` (the default, frozen on the development seed): geodesic length along the region's
    medial axis from its pixels nearest the grain centre, plus their distance from the rim, so a
    wide tube's half-width does not count; ``"nearest"``: the same over the whole region;
    ``"rim"`` (the first sketch): from every region pixel within 1.5 px of the rim, which reads
    ~2 px short. ``end_px`` is added to every non-zero reach: the skeleton stops short of the
    tube's end. With ``big`` set, a grain whose region reaches the crop's edge in any bin is read
    again with ``half = big`` (movies longer than the dev movie; ``pipeline.py`` sets 300).
    ``tip="dt"`` adds the region's half-width at the end of the axis instead: it helped with exact
    truth masks and hurt with learned evidence on fresh held-out movies, so it is off. With
    ``burst``, a reading that collapses for good (``burst_cut``) ends the growth fit there: a burst
    tube leaves nothing to trace (``pipeline.py`` sets it; its burst frame is not a measurement).
    ``keep`` lists bins (counted from the first analysed bin) whose registered image, region and
    medial axis are returned under ``_views`` for review pictures (``review.py``). ``paths`` lists
    bins whose reading is returned as a path under ``_paths``: from the grain's rim along the medial
    axis to its far end, in reference coordinates at that bin (the grain's own drift included, as
    the labelling tool's ``path_xy_ref``), for pre-filling review labels (``prefill.py``).

    With learned evidence it is ahead of SparseTrack's decoder on synthetic movies: +84 lengths in
    tolerance on seven development movies, +12 on eight held-out and +44 on four untouched test
    movies. No label-free per-grain switch between the two decoders did better than this alone."""
    fpb, rs, nb = int(meta["frames_per_bin"]), int(meta.get("ref_start", 0)), int(meta["n_bins"])
    gx, gy, gr = float(grain["x"]), float(grain["y"]), float(grain["r"])
    centre = half - 0.5
    img = np.stack([R_img.crop(b, gx, gy, half) for b in range(rs, nb)])
    if np.isnan(img).any():
        img = np.nan_to_num(img, nan=float(np.nanmedian(img)))
    ls = A.local_shifts(img, centre, gr, 12.0, 3)
    yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float64)
    rg = np.hypot(xx - centre, yy - centre)
    blocked = rg < gr - 1.0
    # pixels whose source leaves the movie in any bin are not evidence (as in SparseTrack): near an
    # edge, the padded border shows up as straight streaks that the network can take for tubes
    shifts = R_img.shifts[rs:nb] + ls
    ref_x, ref_y = gx - half + xx + 0.5, gy - half + yy + 0.5
    blocked |= ~((ref_x + shifts[:, 0].min() >= 1) & (ref_x + shifts[:, 0].max() < R_img.width - 1) &
                 (ref_y + shifts[:, 1].min() >= 1) & (ref_y + shifts[:, 1].max() < R_img.height - 1))
    rings = []
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        if -o["r"] - 10 < ox < 2 * half + o["r"] + 10 and -o["r"] - 10 < oy < 2 * half + o["r"] + 10:
            d = np.hypot(xx - ox, yy - oy)
            blocked |= d < o["r"] + 1.0
            rings.append((d >= o["r"] + 1.0) & (d <= o["r"] + rim_band))
    own_ring = (rg >= gr - 1.0) & (rg <= gr + rim_band)
    rim = (rg >= gr - 1.0) & (rg <= gr + 1.5)
    raw = np.zeros(nb - rs)
    width = np.full(nb - rs, np.nan)  # region area per px of length: the tube's full width
    edge = False  # the tube reaches the crop's edge: read the grain again with a bigger crop
    views, routes = {}, {}
    for i, b in enumerate(range(rs, nb)):
        p = RP.crop(b, gx, gy, half) / scale
        dx, dy = ls[i]
        shift = np.float32([[1, 0, -dx], [0, 1, -dy]])
        p = cv2.warpAffine(np.nan_to_num(p).astype(np.float32), shift,
                           (2 * half, 2 * half), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if i in keep:
            views[i] = [cv2.warpAffine(img[i].astype(np.float32), shift, (2 * half, 2 * half),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), None, None]
        m = (p > thr) & ~blocked
        if not (m & own_ring).any():
            continue
        _, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        ids = np.unique(lab[m & own_ring])
        comp = np.isin(lab, ids[ids > 0])
        rivals = [r for r in rings if (comp & r).any()]
        if rivals:
            comp = A.geodesic_owner(comp, [own_ring] + rivals) == 0
        k = 3 + int(np.ceil(np.abs(ls).max()))  # the registration shift leaves an empty band at the border
        edge = edge or bool(comp[:k].any() or comp[-k:].any() or comp[:, :k].any() or comp[:, -k:].any())
        region = comp
        if seed == "skeleton":  # along the medial axis: a wide tube's half-width does not count
            sk = skeletonize(comp)
            comp = sk if sk.sum() >= 2 else comp
        if i in views:
            views[i][1], views[i][2] = region, comp
        seeds = comp & rim if seed == "rim" else np.zeros_like(comp)
        offset = 0.0
        if not seeds.any():  # nearest pixels (or a faint base that starts beyond the rim)
            near = comp & own_ring if (comp & own_ring).any() else comp
            rmin = float(rg[near].min())
            seeds = comp & (rg <= rmin + (1.0 if seed == "rim" else 0.5))
            offset = max(0.0, rmin - gr)
        if not seeds.any():
            continue
        mcp = MCP_Geometric(np.where(comp, 1.0, np.inf))
        cum, _ = mcp.find_costs(list(zip(*np.nonzero(seeds))))
        ok = comp & np.isfinite(cum)
        if not ok.any():
            continue
        far = np.unravel_index(int(np.argmax(np.where(ok, cum, -1.0))), cum.shape)
        if i in paths:  # the exit on the rim, then the medial axis to the far end (crop cols, rows -> reference x, y)
            route = np.asarray(mcp.traceback(far), float)[:, ::-1]
            # where the region crosses the grain's ring (a wide tube's medial axis wanders along the rim there)
            ey, ex = np.nonzero(region & own_ring)
            v = (np.array([ex.mean(), ey.mean()]) if len(ex) else route[0]) - centre
            rim_pt = centre + v * gr / max(float(np.hypot(*v)), 1e-9)
            j = int(np.argmin(np.hypot(*(route - rim_pt).T)))
            routes[i] = np.vstack([rim_pt, route[j:]]) + [gx - half + dx + 0.5, gy - half + dy + 0.5]
        end = end_px
        if tip == "dt":  # the medial axis stops about a half-width short of the tube's end
            end += float(cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)[far])
        raw[i] = max(0.0, float(cum[far]) + offset + end)  # a negative end_px must not make lengths negative
        width[i] = float(region.sum()) / max(float(cum[far]) + 1.0, 1.0)
    if edge and big and half < big:
        res = reach_grain(RP, R_img, meta, grain, others, thr=thr, scale=scale, half=big, vmax=vmax,
                          onset_px=onset_px, min_tube_px=min_tube_px, rim_band=rim_band, seed=seed, end_px=end_px,
                          tip=tip, big=big, burst=burst, keep=keep, paths=paths)
        res["flags"].append(f"crop_grown:{big}")
        return res
    frames = [b * fpb + fpb // 2 for b in range(rs, nb)]
    cut = burst_cut(raw, min_px=min_tube_px) if burst else None
    if cut is not None:  # growth up to the burst, then the length it reached (nothing is left to trace)
        head = monotone_l1(raw[:cut], vmax)
        fit = np.concatenate([head, np.full(raw.size - cut, head[-1])])
    else:
        fit = monotone_l1(raw, vmax) if raw.max() > 0 else raw
    on = np.nonzero(fit >= onset_px)[0]
    flags = []
    if fit[-1] < min_tube_px:
        status, onset, fit = "no_emergence_by_end", None, np.zeros_like(fit)
    elif on[0] == 0:
        status, onset = "emerged_at_start", frames[0]
    else:
        status, onset = "emerged_within", frames[int(on[0])]
    interval = None if onset is None or on[0] == 0 else [frames[int(on[0]) - 1], onset]
    if cut is not None and status != "no_emergence_by_end":
        flags.append(f"burst_after:{frames[cut - 1]}")
    # review hints (the readings are unchanged): the grain has left its place, so what is read there is
    # not its tube; or the reading keeps jumping off the fit, as when a crossing tube takes over the region
    gone = grain_gone(img, ls, centre, gr, rg)
    if gone is not None:
        flags.append(f"no_grain_after:{frames[gone]}")
    if status != "no_emergence_by_end":
        live = (fit >= min_tube_px / 2) & (np.arange(raw.size) < (cut if cut is not None else raw.size))
        off = live & (np.abs(raw - fit) > np.maximum(10.0, 0.3 * fit))
        if live.sum() >= 5 and off.sum() >= 0.3 * live.sum():
            flags.append(f"unsteady:{int(off.sum())}/{int(live.sum())}")
    return {"id": grain["id"], "x": gx, "y": gy, "r": gr, "status": status, "onset_frame": onset,
            "onset_interval": interval, "final_length_px": round(float(fit[-1]), 2),
            "length": {"frames": frames, "px": [round(float(v), 2) for v in fit]},
            "raw_reach_px": [round(float(v), 2) for v in raw], "flags": flags,
            "burst_frame": frames[cut] if cut is not None and status != "no_emergence_by_end" else None,
            "width_px": (round(float(np.nanmedian(width[raw >= min_tube_px])), 2)
                         if np.any((raw >= min_tube_px) & np.isfinite(width)) else None),
            **({"_views": views, "_centre": centre} if keep else {}),
            **({"_paths": {i: routes[i].round(2).tolist() for i in routes}} if paths else {})}


def analyze(pcache: str | Path, image_cache: str | Path, grains_path: str | Path | None = None, log=print,
            **kw) -> dict:
    import json
    bins_p, meta = stack.load(pcache)
    RP = Renderer(bins_p, meta)
    R_img = Renderer(*stack.load(image_cache))
    src = Path(grains_path) if grains_path else Path(pcache) / "grains.json"
    doc = json.loads(src.read_text())
    census = list(doc["grains"].values()) if isinstance(doc["grains"], dict) else doc["grains"]
    grains = [g for g in census if not g.get("excluded")]
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    started, out = time.time(), []
    for g in grains:
        others = [o for o in physical if o["id"] != g["id"]]
        out.append(reach_grain(RP, R_img, meta, g, others, **kw))
    log(f"reach decoder: {len(out)} grains in {time.time() - started:.0f} s")
    return {"schema": PRED_SCHEMA, "method": "reach decoder (learned-evidence prototype)",
            "frames_per_bin": meta["frames_per_bin"], "grains": out}
