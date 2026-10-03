"""Usefulness for tracking, first look: link the detector's peaks over time per grain (Viterbi) and compare the linked
tip's straight-line distance from the trace's exit point with the human's, at the human's FULL traces.

    python -m prototypes.tip_detector.link m2 --net runs/research/tip_detector/tip2_ldm1.pt --tag v2

Streaming over the movie, one bin at a time: the detector's map of the whole registered frame (512 px tiles with
48 px overlap; inputs as in common.py but with the local medians of the whole frame), then for every labelled grain
the top 6 peaks within its search disc (the grain where the labelling tool follows it; radius r + 1.25 x the grain's
longest human length + 30 px - the region's scale is taken from the labels, as in the per-trace check; the grain's
inside, r - 3 px, left out). Candidates at bin b: the peaks of bins b-2..b (relative to the grain), scored by bin b's
map value there. Viterbi over the candidates plus "no tube yet" (scores as a peak of TAU): a tube starts within
r + 12 px of the grain's centre; a step longer than 4 px per bin and a move back towards the grain of more than 2 px
cost 0.5 per px; a tube never disappears. TAU = 0.2 was set by eye from movie 2's value scale (young apexes median
0.25, maxima before the onset median 0.10), not tuned.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque

import numpy as np

from sparsetrack.learned import _registered

from .common import K, OUT, SCALE, Movie, heat, length_class, load_net, medbg
from .evaluate import peaks

TAU = 0.2
STEP0, SHRINK0, COST = 4.0, 2.0, 0.5


def heat_full(net, x: np.ndarray, tile: int = 512, pad: int = 48) -> np.ndarray:
    _, H, W = x.shape
    out = np.zeros((H, W), np.float32)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            ya, yb, xa, xb = max(0, y0 - pad), min(H, y0 + tile + pad), max(0, x0 - pad), min(W, x0 + tile + pad)
            h = heat(net, np.ascontiguousarray(x[:, ya:yb, xa:xb]))[0]
            hh, ww = min(tile, H - y0), min(tile, W - x0)
            out[y0:y0 + hh, x0:x0 + ww] = h[y0 - ya:y0 - ya + hh, x0 - xa:x0 - xa + ww]
    return out


def viterbi(bins, cands, vals, r):
    score, back, prev = {}, {}, None
    for b in bins:
        pool, em = cands[b], np.array([math.log(TAU)] + [math.log(max(v, 1e-4)) for v in vals[b]])
        n1 = len(em)
        if prev is None:
            s = np.full(n1, -np.inf)
            s[0] = em[0]
            for j, q in enumerate(pool, 1):
                if math.hypot(*q) <= r + 12:
                    s[j] = em[j]
            score[b], back[b], prev = s, np.zeros(n1, int), b
            continue
        ps, ppool = score[prev], cands[prev]
        trans = np.full((len(ps), n1), -np.inf)
        trans[0, 0] = 0.0
        for j, q in enumerate(pool, 1):
            rq = math.hypot(*q)
            if rq <= r + 12:
                trans[0, j] = 0.0
            for i, p in enumerate(ppool, 1):
                step = math.hypot(q[0] - p[0], q[1] - p[1])
                shrink = math.hypot(*p) - rq
                trans[i, j] = -COST * max(0.0, step - STEP0) - COST * max(0.0, shrink - SHRINK0)
        tot = ps[:, None] + trans
        back[b] = np.argmax(tot, axis=0)
        score[b] = tot[back[b], np.arange(n1)] + em
        prev = b
    path, j = {}, int(np.argmax(score[bins[-1]]))
    for b in reversed(bins):
        path[b] = None if j == 0 else cands[b][j - 1]
        j = int(back[b][j])
    return path


def run(movie: str, net_path: str, tag: str, log=print):
    mv = Movie(movie)
    net = load_net(net_path)
    chans = ["ADC".index(c) for c in net.channels]
    shifts = np.asarray(mv.meta["shifts"], np.float64)
    gids = [g for g in sorted(mv.labels) if any(mv.lo <= b <= mv.hi for b, _ in mv.traces(g, ("full",)))]
    info = {}
    for g in gids:
        Lmax = max([t.get("length_px") or 0.0 for _, t in mv.traces(g, ("full", "partial"))] + [10.0])
        info[g] = (float(mv.grains[g]["r"]), float(mv.grains[g]["r"]) + 1.25 * Lmax + 30)
    reg = {}
    E = np.mean([_registered(mv.bins, shifts, k) for k in range(mv.rs, mv.rs + 3)], axis=0)
    hist = {g: deque(maxlen=3) for g in gids}
    cands = {g: {} for g in gids}
    vals = {g: {} for g in gids}
    top1 = {g: {} for g in gids}
    bins = list(range(mv.lo, mv.hi + 1))
    H, W = E.shape
    t0 = time.time()
    for b in bins:
        for k in range(max(b - K - 1, mv.rs), min(b + 1, mv.nb - 1) + 1):
            if k not in reg:
                reg[k] = _registered(mv.bins, shifts, k)
        for k in [k for k in reg if k < b - K - 1]:
            del reg[k]
        mean = lambda a, z: np.mean([reg[k] for k in range(max(a, mv.rs), min(z, mv.nb - 1) + 1)], axis=0)
        M, P = mean(b - 1, b + 1), mean(b - K - 1, b - K + 1)
        A = M - medbg(M)
        D = M - P
        D -= medbg(D)
        C = M - E
        C -= medbg(C)
        x = (np.stack([A, D, C]) / SCALE[:, None, None])[chans].astype(np.float32)
        hm = heat_full(net, x)
        for g in gids:
            r, R = info[g]
            gx, gy, _ = mv.grain_at(g, b)
            px, py = gx - 0.5, gy - 0.5  # pixel-index position of the grain centre
            x0, x1 = max(0, int(px - R - 2)), min(W, int(px + R + 3))
            y0, y1 = max(0, int(py - R - 2)), min(H, int(py + R + 3))
            sub = hm[y0:y1, x0:x1]
            jj, ii = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            d = np.hypot(jj - px, ii - py)
            region = (d <= R) & (d > r - 3)
            pk = peaks(sub, region, 6)
            rel = [(x0 + u - px, y0 + v - py) for u, v, _ in pk]
            top1[g][b] = rel[0] if rel else None
            hist[g].append(rel)
            pool = []
            for prev in hist[g]:
                for q in prev:
                    if all(math.hypot(q[0] - w[0], q[1] - w[1]) > 2 for w in pool):
                        pool.append(q)
            vv = []
            for q in pool:
                xi, yi = int(round(px + q[0])), int(round(py + q[1]))
                ok = 0 <= xi < W and 0 <= yi < H and r - 3 < math.hypot(*q) <= R
                vv.append(float(hm[yi, xi]) if ok else 0.0)
            cands[g][b], vals[g][b] = pool, vv
        if b % 50 == 0:
            log(f"  bin {b}/{bins[-1]} ({time.time() - t0:.0f} s)")
    rows = []
    for g in gids:
        r, R = info[g]
        path = viterbi(bins, cands[g], vals[g], r)
        onset_lab = (mv.labels[g].get("onset") or {}).get("first_visible_bin")
        started = next((b for b in bins if path[b] is not None), None)
        for b, t in mv.traces(g, ("full",)):
            if not mv.lo <= b <= mv.hi:
                continue
            gx, gy, _ = mv.grain_at(g, b, t)
            p = np.asarray(t["path_xy_ref"], float)
            ex, apex = p[0], p[-1]
            L = float(t.get("length_px") or 0)
            out = {"grain": g, "bin": b, "L": L, "cls": length_class(L), "human_lin": float(np.hypot(*(apex - ex))),
                   "start_bin": started, "human_onset": onset_lab}
            for name, src in (("linked", path), ("top1", top1[g])):
                q = src.get(b)
                if q is None:
                    out[name], out[name + "_apex_err"] = 0.0, None
                else:
                    gxb, gyb, _ = mv.grain_at(g, b)  # the peaks are relative to the followed grain
                    tip = np.array([gxb + q[0], gyb + q[1]])
                    out[name] = float(np.hypot(*(tip - ex)))
                    out[name + "_apex_err"] = float(np.hypot(*(tip - apex)))
            rows.append(out)
    (OUT / f"link_{movie}_{tag}.json").write_text(json.dumps(rows))
    summarize(rows, movie, log)


def summarize(rows, movie, log=print):
    log(f"\n{movie}: {len(rows)} FULL traces; straight-line tip-to-exit vs the human's |apex - exit|")
    cls = np.array([r["cls"] for r in rows])
    for name in ("linked", "top1"):
        err = np.array([abs(r[name] - r["human_lin"]) for r in rows])
        tol = np.array([max(4.0, 0.1 * r["human_lin"]) for r in rows])
        ok = err <= tol
        ap = np.array([(r[name + "_apex_err"] is not None and r[name + "_apex_err"] <= 4) for r in rows])
        by = " ".join(f"{c} {int(ok[cls == c].sum())}/{int((cls == c).sum())}" for c in ("young", "mid", "long"))
        log(f"  {name:7s}: length within max(4 px, 10%) {int(ok.sum())}/{len(rows)} ({by}); tip within 4 px of the "
            f"apex {int(ap.sum())}/{len(rows)}; |error| median {np.median(err):.1f} px")
    per = {r["grain"]: (r["start_bin"], r["human_onset"]) for r in rows}
    st = [(a, b) for a, b in per.values() if b is not None]
    if st:
        d = np.array([(a if a is not None else np.nan) - b for a, b in st], float)
        log(f"  linked start bin - human first visible bin: median {np.nanmedian(d):+.0f}, within 3 bins "
            f"{int((np.abs(d) <= 3).sum())}/{len(d)}, never started {int(np.isnan(d).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--net", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run(a.movie, a.net, a.tag)
