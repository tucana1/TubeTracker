"""Tube growth read by an arrival-time flood from the grain's rim (prototype; see README.md).

    python -m prototypes.learned_flood.flood MOVIE all|GID ... [--half H] [--ridge | --prob PROB_CACHE] [--png DIR]

Each pixel's arrival bin = first bin from which it stays changed (>= 70% of the next P bins).
The grain's tube is flooded in arrival order: a newly arrived component joins it only if it
touches the tube's most recently joined pixels (its tip), or the rim before the tube exists.
Material that was already there (a foreign tube) or appears beside old tube (sway) never joins.
Length at bin b = geodesic distance from the rim to the farthest tube pixel arrived by b.
"""
import argparse
import heapq
import json
import math
import sys

import cv2
import numpy as np

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from sparsetrack import stack  # noqa: E402
from sparsetrack.analyze import local_shifts, plausible_drift  # noqa: E402
from sparsetrack.render import Renderer  # noqa: E402


def geodesic_dist(mask, seeds):
    """Dijkstra distance (8-connected, diagonal sqrt 2) inside mask from seeds."""
    h, w = mask.shape
    dist = np.full((h, w), np.inf)
    pq = []
    for y, x in zip(*np.nonzero(seeds & mask)):
        dist[y, x] = 0.0
        pq.append((0.0, int(y), int(x)))
    heapq.heapify(pq)
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > dist[y, x]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if (dy or dx) and 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                    nd = d + math.hypot(dy, dx)
                    if nd < dist[yy, xx]:
                        dist[yy, xx] = nd
                        heapq.heappush(pq, (nd, yy, xx))
    return dist


def stack_for(renderer, meta, g, half, others, prob=None):
    rs = int(meta.get("ref_start", 0))
    n = int(meta["n_bins"]) - rs
    crops = np.stack([renderer.crop(b, g["x"], g["y"], half) for b in range(rs, rs + n)])
    centre = half - 0.5
    ls = local_shifts(crops, centre, g["r"], 12.0, 3)
    if not plausible_drift(ls):
        ls = np.zeros_like(ls)
    if prob is not None:  # read the tube-probability movie, registered on the real images
        crops = np.nan_to_num(np.stack([prob.crop(b, g['x'], g['y'], half) for b in range(rs, rs + n)]), nan=0.0)
    reg = np.stack([cv2.warpAffine(c, np.float32([[1, 0, -dx], [0, 1, -dy]]), (2 * half, 2 * half),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for c, (dx, dy) in zip(crops, ls)])
    del crops
    yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float32)
    rg = np.hypot(xx - centre, yy - centre)
    blocked = rg < g["r"] - 1.0
    for o in others:
        ox, oy = o["x"] - g["x"] + centre, o["y"] - g["y"] + centre
        if -o["r"] - 5 < ox < 2 * half + o["r"] + 5 and -o["r"] - 5 < oy < 2 * half + o["r"] + 5:
            blocked |= np.hypot(xx - ox, yy - oy) < o["r"] + 2.0
    shifts = np.asarray(meta["shifts"])[rs:] + ls
    ref_x, ref_y = g["x"] - half + xx + 0.5, g["y"] - half + yy + 0.5
    valid = ((ref_x + shifts[:, 0].min() >= 0) & (ref_x + shifts[:, 0].max() < renderer.width) &
             (ref_y + shifts[:, 1].min() >= 0) & (ref_y + shifts[:, 1].max() < renderer.height))
    return reg, rg, blocked | ~valid, centre, rs


def arrival_map(reg, rg, blocked, gr, P=10, frac=0.7, k=5.0, floor=5.0, ridge=False, prob=False):
    if prob:  # probability of tube x 16: present where P >= 0.5, persistently
        return _persist(reg >= 8.0, blocked, P, frac), 8.0
    early = reg[:3].mean(axis=0)
    sm = (reg[:-2] + reg[1:-1] + reg[2:]) / 3.0          # 3-bin moving average (bins 1..n-2)
    sm = np.concatenate([sm[:1], sm, sm[-1:]])
    D = np.abs(sm - early[None])
    if ridge:  # line-ness of the change: tubes are 2-6 px wide, focus/illumination blobs far wider
        from skimage.filters import sato
        for b in range(len(D)):
            D[b] = sato(D[b], sigmas=(1.5, 2.5), black_ridges=False, mode="reflect")
    else:
        for b in range(len(D)):
            D[b] = cv2.GaussianBlur(D[b], (0, 0), 1.0)
    late = D[-4:-1].mean(axis=0)
    bgm = (rg > gr + 30) & ~blocked
    bg = late[bgm]
    bg = bg[bg < np.percentile(bg, 95)]
    sigma = 1.4826 * float(np.median(np.abs(bg - np.median(bg))))
    thr = max(floor if not ridge else 0.0, k * sigma)
    return _persist(D > thr, blocked, P, frac), thr


def _persist(ex, blocked, P, frac):
    ex = ex & ~blocked[None]
    n = len(ex)
    cs = np.concatenate([np.zeros((1,) + ex.shape[1:], np.int16), np.cumsum(ex, axis=0, dtype=np.int16)])
    idx = np.arange(n)
    hi = np.minimum(idx + P, n)
    held = (cs[hi] - cs[idx]) / (hi - idx)[:, None, None]
    ok = ex & (held >= frac)
    return np.where(ok.any(axis=0), ok.argmax(axis=0), n)


def _extend_dist(dist, comp, seeds_mask, bridge):
    """Distances for a joining component: from the tube pixels (with their distances) within
    ``bridge`` px, then geodesically inside the component."""
    ys, xs = np.nonzero(comp)
    y0, y1 = max(ys.min() - bridge - 1, 0), min(ys.max() + bridge + 2, dist.shape[0])
    x0, x1 = max(xs.min() - bridge - 1, 0), min(xs.max() + bridge + 2, dist.shape[1])
    sub_d = dist[y0:y1, x0:x1]
    sub_c = comp[y0:y1, x0:x1]
    src = np.isfinite(sub_d) & seeds_mask[y0:y1, x0:x1]
    h, w = sub_c.shape
    out = np.full((h, w), np.inf)
    pq = []
    sy, sx = np.nonzero(src)
    cy, cx = np.nonzero(sub_c)
    for y, x in zip(cy, cx):  # each component pixel near the tube: straight-line hop across the gap
        dd = np.hypot(sy - y, sx - x)
        near = dd <= bridge + 0.5
        if near.any():
            v = float(np.min(sub_d[sy[near], sx[near]] + dd[near]))
            if v < out[y, x]:
                out[y, x] = v
                pq.append((v, int(y), int(x)))
    heapq.heapify(pq)
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > out[y, x]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if (dy or dx) and 0 <= yy < h and 0 <= xx < w and sub_c[yy, xx]:
                    nd = d + math.hypot(dy, dx)
                    if nd < out[yy, xx]:
                        out[yy, xx] = nd
                        heapq.heappush(pq, (nd, yy, xx))
    region = dist[y0:y1, x0:x1]
    upd = sub_c & (out < region)
    region[upd] = out[upd]


HALO = 3.0   # the rim's own change (focus, swelling) reaches this far out: the tube is read beyond it


def flood(arr, rg, blocked, gr, R=12, bridge=4, min_len=8.0, give_up=40, ang=None, start_band=4.0):
    """Returns (tube mask, arrival bin per tube pixel, emergence bin, length per bin)."""
    n = int(arr.max())
    blocked = blocked | (rg < gr + HALO)
    # a tube may start anywhere in this band: probability maps often miss the few px next to the rim
    ring = (rg >= gr + HALO) & (rg <= gr + HALO + start_band)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge + 1, 2 * bridge + 1))
    our = np.zeros(arr.shape, bool)
    t_our = np.full(arr.shape, -1, np.int32)
    dist = np.full(arr.shape, np.inf)
    L = np.zeros(n)
    t_emerge = None
    for b in range(n):
        new = (arr == b) & ~blocked
        if new.any():
            nl, lab = cv2.connectedComponents(new.astype(np.uint8), connectivity=8)
            if not our.any():
                sofar = ((arr <= b) & ~blocked).astype(np.uint8)
                _, lab_all = cv2.connectedComponents(sofar, connectivity=8)
                old_far = (arr < b - 3) & (rg > gr + 10.0)
                for l in range(1, nl):
                    c = lab == l
                    if not (c & ring).any():
                        continue
                    a = np.angle(np.exp(1j * (ang[c] - np.angle(np.mean(np.exp(1j * ang[c]))))))
                    if np.ptp(a) > np.deg2rad(60):      # an arc round the rim, not a stub leaving it
                        continue
                    whole = np.isin(lab_all, np.unique(lab_all[c]))
                    if (whole & old_far).any():          # the leading end of a structure already there
                        continue
                    our |= c
                    t_our[c] = b
                    dist[c & ring] = rg[c & ring] - gr  # lengths count from the rim, gap included
                    _extend_dist(dist, c, c & ring, 1)
                if our.any():
                    t_emerge = b
            else:
                t_last = t_our.max()
                recent = our & (t_our >= t_last - R)
                seeds = cv2.dilate(recent.astype(np.uint8), ker).astype(bool)
                for l in range(1, nl):
                    c = lab == l
                    if (c & seeds).any():
                        _extend_dist(dist, c, recent, bridge)
                        our |= c
                        t_our[c] = b
                # a start that never grows into a tube was noise at the rim: allow a real one later
                if b - t_emerge > give_up and not (our & (rg > gr + min_len)).any():
                    our[:] = False
                    t_our[:] = -1
                    dist[:] = np.inf
                    t_emerge = None
        fin = our & np.isfinite(dist)
        L[b] = float(dist[fin].max()) if fin.any() else 0.0
    return our, t_our, t_emerge, np.maximum.accumulate(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("gids", nargs="+")
    ap.add_argument("--half", type=int, default=150)
    ap.add_argument("--png")
    ap.add_argument("--R", type=int, default=12)
    ap.add_argument("--bridge", type=int, default=4)
    ap.add_argument("--ridge", action="store_true")
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--prob", help="tube-probability cache to flood instead of change")
    ap.add_argument("--start-band", type=float, default=4.0, help="px beyond the halo where a tube may start")
    ap.add_argument("--tip", type=float, default=2.5, help="subtracted from the flood's reach (tip blur)")
    args = ap.parse_args()
    labels = json.load(open(f"{REPO}/benchmark/labels/{args.movie}_v1.json"))
    if args.gids == ["all"]:
        args.gids = [g for g, v in labels["grains"].items() if not v.get("excluded") and v.get("isolated", True)]
    bins, meta = stack.load(f"{REPO}/runs/sparsetrack/{args.movie}")
    renderer = Renderer(bins, meta)
    prob = Renderer(*stack.load(args.prob)) if args.prob else None
    census = list(labels["grains"].values())
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    fpb = int(meta["frames_per_bin"])
    for gid in args.gids:
        g = labels["grains"][gid]
        others = [o for o in physical if o["id"] != gid]
        reg, rg, blocked, centre, rs = stack_for(renderer, meta, g, args.half, others, prob)
        arr, thr = arrival_map(reg, rg, blocked, g["r"], ridge=args.ridge, k=args.k, prob=prob is not None)
        yy, xx = np.mgrid[0:rg.shape[0], 0:rg.shape[1]]
        ang = np.arctan2(yy - centre, xx - centre)
        our, t_our, t_em, L = flood(arr, rg, blocked, g["r"], R=args.R, bridge=args.bridge, ang=ang,
                                    start_band=args.start_band)
        L = np.concatenate([L, np.full(len(reg) - len(L), L[-1] if len(L) else 0.0)])
        lab = labels["labels"].get(gid, {})
        on = lab.get("onset") or {}
        rows = []
        for b, t in sorted(lab.get("traces", {}).items(), key=lambda kv: int(kv[0])):
            i = int(b) - rs
            if t["state"] == "full" and not t.get("contact") and 0 <= i < len(L):
                est = max(L[i] - args.tip, 0.0) if L[i] > 0 else 0.0
                ok = abs(est - t["length_px"]) <= max(2.0, 0.1 * t["length_px"])
                TOTAL.append(ok)
                rows.append(f"b{b}: {est:.1f}/{t['length_px']:.1f} ({est - t['length_px']:+.1f}{' ok' if ok else ''})")
        em = None if t_em is None else (t_em + rs) * fpb + fpb // 2
        print(f"{gid}: thr {thr:.1f} | emerge {em} vs human ({on.get('last_absent_frame')}, {on.get('first_visible_frame')}] | "
              + " ; ".join(rows), flush=True)
        if args.png:
            n = len(reg)
            a = np.where(arr < n, arr, -1).astype(np.float32)
            col = cv2.applyColorMap(np.clip(a / max(n, 1) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            col[a < 0] = 30
            col[our] = (255, 255, 255)
            late = reg[-2]
            lo, hi = np.percentile(late, [0.5, 99.5])
            gray = cv2.cvtColor(np.clip((late - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for t in lab.get("traces", {}).values():
                if t["state"] == "full" and t.get("path_xy_view"):
                    hp = (np.asarray(t["path_xy_view"]) - [g["x"] - centre, g["y"] - centre]).astype(np.int32)
                    for im in (gray, col):
                        cv2.polylines(im, [hp.reshape(-1, 1, 2)], False, (40, 200, 40), 1, cv2.LINE_AA)
            cv2.imwrite(f"{args.png}/{args.movie}_{gid}_flood.png", np.hstack([gray, col]))
    print(f"lengths in tolerance: {sum(TOTAL)}/{len(TOTAL)}")


TOTAL = []
if __name__ == "__main__":
    main()
