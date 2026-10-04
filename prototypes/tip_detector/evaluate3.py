"""Judge several tip detectors (version 2 and 3 checkpoints) on one movie's FULL human traces, on identical crops.

    python -m prototypes.tip_detector.evaluate3 m1 --nets v2=runs/research/tip_detector/tip2_ldm2.pt \
        v3=runs/research/tip_detector/tip3_ldm2.pt --tag v3

The version 2 protocol (evaluate.py): per FULL trace at bin b, the grain at that bin (census + the trace's view
offset), human length L, apex = the trace's last point, search region = the disc r + L + 25 px round the grain inside
the frame; peaks = local maxima of 9 x 9, best first, >= 4 px apart. Here up to 10 peaks are kept (top-1 / 3 / 8), and
one crop of half max(304, r + L + 37) px serves every detector (version 2's inputs are version 3's A, D6, C unclipped).

Also the reader's view ("reader"): the candidates sparsetrack/tiptraj.py takes from a detector map round a grain - the
3 strongest peaks (>= 0.05) within r + 30 px of the grain centre, then the 6 strongest anywhere from r - 2 to 296 px
(>= 4 px apart) - and whether one lies within 4 px of the apex (the reader places the grain by its own drift; here
the human's grain position is used, as everywhere in this protocol).
Before the onsets (bins last_absent_bin - 1, 3, 6, 12, 24): each map's best value within r + 25 px of the grain.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
from scipy.ndimage import maximum_filter

from .common import OUT, heat, length_class, load_net
from .common3 import CHANNELS3, Movie3, scale3
from .evaluate import NEG, peaks, raw_peaks

V2_IDX = [CHANNELS3.index(c) for c in ("A", "D6", "C")]
V2_SCALE = np.array([20.0, 8.0, 20.0], np.float32)
READER_HALF = 300


def net_input(net, raw5: np.ndarray) -> np.ndarray:
    if isinstance(net.channels, str):  # version 2 ("ADC"): no clipping
        return (raw5[V2_IDX] / V2_SCALE[:, None, None]).astype(np.float32)
    return scale3(raw5[[CHANNELS3.index(c) for c in net.channels]], net.channels)


def reader_cands(m: np.ndarray, rg: np.ndarray, r: float, inside: np.ndarray) -> list[tuple[int, int, float]]:
    mm = np.where((rg >= r - 2) & (rg <= READER_HALF - 4) & inside, m, -1.0).astype(np.float32)
    pk = (mm == maximum_filter(mm, size=9, mode="constant", cval=-1.0)) & (mm >= 0.05)
    ys, xs = np.nonzero(pk)
    order = np.argsort(-mm[ys, xs], kind="stable")
    near = rg[ys, xs] <= r + 30
    cand = []
    for sel, extra in ((order[near[order]], 3), (order, 6)):
        kmax = len(cand) + extra
        for o in sel:
            if len(cand) >= kmax:
                break
            q = (int(xs[o]), int(ys[o]), float(mm[ys[o], xs[o]]))
            if all(math.hypot(q[0] - a, q[1] - c) >= 4 for a, c, _ in cand):
                cand.append(q)
    return cand


def heat_tta(net, x: np.ndarray) -> np.ndarray:
    """The mean of the maps of the input and its three flips (test-time augmentation)."""
    acc = np.zeros(x.shape[1:], np.float32)
    for fy, fx in ((False, False), (True, False), (False, True), (True, True)):
        xi = x[:, ::-1] if fy else x
        xi = xi[:, :, ::-1] if fx else xi
        m = heat(net, np.ascontiguousarray(xi))[0]
        m = m[::-1] if fy else m
        acc += m[:, ::-1] if fx else m
    return acc / 4


def run(movie: str, nets: dict, tag: str, log=print):
    mv = Movie3(movie)
    loaded = {k: load_net(p.split("+")[0]) for k, p in nets.items()}
    tta = {k: p.endswith("+tta") for k, p in nets.items()}
    hm = lambda k, x: heat_tta(loaded[k], x) if tta[k] else heat(loaded[k], x)[0]
    res = {"movie": movie, "nets": nets, "traces": [], "pre_onset": [], "skipped": []}
    t0 = time.time()
    for gid in sorted(mv.labels):
        for b, t in mv.traces(gid, ("full",)):
            if not mv.lo <= b <= mv.hi:
                res["skipped"].append([gid, b])
                continue
            gx, gy, r = mv.grain_at(gid, b, t)
            path = np.asarray(t["path_xy_ref"], float)
            L = float(t.get("length_px") or np.hypot(*np.diff(path, axis=0).T).sum())
            R = r + L + 25
            half = max(READER_HALF + 4, int(math.ceil((R + 12) / 8)) * 8)
            raw5 = mv.raw3(b, gx, gy, half)
            S = 2 * half
            x0, y0 = gx - half + 0.5, gy - half + 0.5
            jj, ii = np.meshgrid(np.arange(S), np.arange(S))
            dist = np.hypot(jj - (gx - x0), ii - (gy - y0))
            inside = ~mv.outside(b, gx, gy, half)
            region = (dist <= R) & inside
            ax, ay = path[-1][0] - x0, path[-1][1] - y0
            ent = {"grain": gid, "bin": b, "L": L, "cls": length_class(L), "r": r, "apex": [ax, ay],
                   "contact": bool(t.get("contact")), "half": half, "methods": {}, "rank": {}, "apex_val": {},
                   "reader": {}}
            near = np.hypot(jj - ax, ii - ay) <= 4
            for name, net in loaded.items():
                m = hm(name, net_input(net, raw5))
                pk = peaks(m, region, 10)
                ent["methods"][name] = [(x, y, v, float(math.hypot(x - ax, y - ay))) for x, y, v in pk]
                av = float(m[near & region].max()) if (near & region).any() else -1.0
                xs, ys, vs = raw_peaks(m, region)
                ent["rank"][name] = 1 + int(((vs > av) & (np.hypot(xs - ax, ys - ay) > 4)).sum())
                ent["apex_val"][name] = av
                rc = reader_cands(m, dist, r, inside)
                ent["reader"][name] = [(x, y, v, float(math.hypot(x - ax, y - ay))) for x, y, v in rc]
                # every peak of the reader's region (r - 2 .. 296 px), best first: how many the reader would need
                rr = (dist >= r - 2) & (dist <= READER_HALF - 4) & inside
                ent.setdefault("reader_all", {})[name] = [float(math.hypot(x - ax, y - ay))
                                                          for x, y, v in peaks(m, rr, 30)]
            res["traces"].append(ent)
        on = mv.labels[gid].get("onset") or {}
        if on.get("verdict") == "emerged_within":
            for k in NEG:
                b = int(on["last_absent_bin"]) - k
                if b < mv.lo:
                    continue
                gx, gy, r = mv.grain_at(gid, b)
                half = 64
                raw5 = mv.raw3(b, gx, gy, half)
                S = 2 * half
                jj, ii = np.meshgrid(np.arange(S), np.arange(S))
                dist = np.hypot(jj - (half - 0.5), ii - (half - 0.5))
                region = (dist <= r + 25) & ~mv.outside(b, gx, gy, half)
                ent = {"grain": gid, "bin": b, "back": k, "r": r, "methods": {}}
                for name, net in loaded.items():
                    m = hm(name, net_input(net, raw5))
                    pk = peaks(m, region, 1)
                    if pk:
                        x, y, v = pk[0]
                        ent["methods"][name] = [v, float(dist[y, x] - r)]
                res["pre_onset"].append(ent)
    out = OUT / f"eval3_{movie}_{tag}.json"
    out.write_text(json.dumps(res))
    log(f"{movie} [{tag}]: {len(res['traces'])} FULL traces ({len(res['skipped'])} outside the input range), "
        f"{len(res['pre_onset'])} pre-onset grain-bins, {time.time() - t0:.0f} s -> {out.name}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--nets", nargs="+", required=True, help="name=checkpoint[+tta] ...")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run(a.movie, dict(s.split("=", 1) for s in a.nets), a.tag)
