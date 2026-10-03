"""Judge a tip detector (and the raw evidence) on one movie's FULL human traces and its bins before the onsets.

    python -m prototypes.tip_detector.evaluate m1 --net runs/research/tip_detector/tip_ldm2.pt --tag ldm2

Per FULL trace at bin b: the grain at that bin (census + the trace's view offset; its exit points lie within a few px
of radius r of it), human length L, apex = the trace's last point. Search region = the disc r + L + 25 px round the
grain, inside the frame. Methods (all on the same crop):
  det        the detector's heatmap
  absD_s     |short-interval difference| (mean of bins b-1..b+1 minus b-7..b-5), Gaussian-smoothed sigma s = 0, 1, 2
  absC_s     |change from the first bins|, smoothed the same way
  tubenet    (m1 only) the existing tube network's map: P >= 0.5 pieces reaching the grain's rim (r + 6 px), their
             farthest pixel from the grain centre (top-1 only)
Peaks: local maxima in 9 x 9 windows, best first, at least 4 px apart. Also the same with the grain's inside
(within r + 1 px of its centre) left out of the search ("_out").
Before the onset: bins last_absent_bin - (1, 3, 6, 12, 24) of grains whose tube emerged within the movie, the grain
where the labelling tool follows it, disc r + 25 px: each map's best value (and where).
Also per trace and map: "rank", 1 + the number of peaks in the disc higher than the map's best value within 4 px of
the apex (how many distractors outrank the tip), and "local", the top peak's distance from the apex within a window
of radius 25 px centred on the human path 15 px behind the apex (the route given, as a route-carrying tracker would
have it: only where along the tube the tip is, and what lies just past it, is asked).
"""
from __future__ import annotations

import argparse
import json
import math
import time

import cv2
import numpy as np
from scipy.ndimage import label, maximum_filter

from sparsetrack import stack
from sparsetrack.render import Renderer

from .common import OUT, REPO, Movie, heat, length_class, load_net

NEG = (1, 3, 6, 12, 24)
SIGMAS = (0, 1, 2)


def raw_peaks(m: np.ndarray, region: np.ndarray):
    mm = np.where(region, m, -np.inf).astype(np.float32)
    pk = (mm == maximum_filter(mm, size=9, mode="constant", cval=-np.inf)) & region
    ys, xs = np.nonzero(pk)
    return xs, ys, mm[ys, xs]


def peaks(m: np.ndarray, region: np.ndarray, k: int = 5) -> list[tuple[int, int, float]]:
    xs, ys, v = raw_peaks(m, region)
    out = []
    for o in np.argsort(-v, kind="stable"):
        x, y = int(xs[o]), int(ys[o])
        if all(math.hypot(x - a, y - b) >= 4 for a, b, _ in out):
            out.append((x, y, float(v[o])))
            if len(out) == k:
                break
    return out


def maps_for(mv: Movie, b: int, gx: float, gy: float, half: int, net, chans, prr=None, r=None):
    raw = mv.raw(b, gx, gy, half)
    maps = {}
    if net is not None:
        maps["det"] = heat(net, (raw / mv.scale[:, None, None])[chans].astype(np.float32))[0]
    for s in SIGMAS:
        for name, ch in (("absD", raw[1]), ("absC", raw[2])):
            maps[f"{name}_{s}"] = np.abs(cv2.GaussianBlur(ch, (0, 0), s) if s else ch)
    if prr is not None:
        maps["P"] = np.nan_to_num(prr.crop(b, gx, gy, half)) / 250.0
    return maps


def tubenet_point(P: np.ndarray, region: np.ndarray, dist: np.ndarray, r: float):
    hi = (P >= 0.5) & region
    lab, n = label(hi)
    touch = np.unique(lab[(dist <= r + 6) & hi])
    touch = touch[touch > 0]
    if len(touch):
        cand = np.isin(lab, touch)
        i = int(np.argmax(np.where(cand, dist, -1)))
    else:
        i = int(np.argmax(np.where(region, P, -1)))
    y, x = divmod(i, P.shape[1])
    return [(x, y, float(P[y, x]))]


def run(movie: str, net_path: str | None, tag: str, log=print):
    mv = Movie(movie)
    net = load_net(net_path) if net_path else None
    chans = ["ADC".index(c) for c in net.channels] if net is not None else None
    prr = None
    if movie == "m1":
        pb, pm = stack.load(REPO / "runs/sparsetrack/m1/prob_tubes_bn_real_ld_m2")
        prr = Renderer(pb, pm)
    res = {"movie": movie, "net": net_path, "traces": [], "pre_onset": [], "skipped": []}
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
            half = max(64, int(math.ceil((R + 12) / 8)) * 8)
            maps = maps_for(mv, b, gx, gy, half, net, chans, prr)
            S = 2 * half
            x0, y0 = gx - half + 0.5, gy - half + 0.5
            jj, ii = np.meshgrid(np.arange(S), np.arange(S))
            dist = np.hypot(jj - (gx - x0), ii - (gy - y0))
            inside = ~mv.outside(b, gx, gy, half)
            region = (dist <= R) & inside
            ax, ay = path[-1][0] - x0, path[-1][1] - y0
            ent = {"grain": gid, "bin": b, "L": L, "cls": length_class(L), "r": r, "apex": [ax, ay],
                   "contact": bool(t.get("contact")), "apex_dist": float(math.hypot(ax - (gx - x0), ay - (gy - y0))),
                   "half": half, "methods": {}, "rank": {}, "local": {}}
            near = np.hypot(jj - ax, ii - ay) <= 4
            sp = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
            u = max(0.0, sp[-1] - 15.0)
            wx, wy = np.interp(u, sp, path[:, 0]) - x0, np.interp(u, sp, path[:, 1]) - y0
            window = region & (np.hypot(jj - wx, ii - wy) <= 25)
            for name, m in maps.items():
                for suffix, reg in (("", region), ("_out", region & (dist > r + 1))):
                    if name == "P":
                        pk = tubenet_point(m, reg, dist, r)
                        key = "tubenet" + suffix
                    else:
                        pk = peaks(m, reg, 5)
                        key = name + suffix
                        av = float(m[near & reg].max()) if (near & reg).any() else -1.0
                        xs, ys, vs = raw_peaks(m, reg)
                        ent["rank"][key] = 1 + int(((vs > av) & (np.hypot(xs - ax, ys - ay) > 4)).sum())
                        ent.setdefault("apex_val", {})[key] = av
                    ent["methods"][key] = [(x, y, v, float(math.hypot(x - ax, y - ay))) for x, y, v in pk]
                if name == "P":  # the tube's end within the window: its farthest P >= 0.5 pixel from the grain
                    hi = (m >= 0.5) & window
                    i = int(np.argmax(np.where(hi, dist, -1))) if hi.any() else int(np.argmax(np.where(window, m, -1)))
                    y, x = divmod(i, m.shape[1])
                    ent["local"]["tubenet"] = float(math.hypot(x - ax, y - ay))
                else:
                    pk = peaks(m, window, 1)
                    ent["local"][name] = float(math.hypot(pk[0][0] - ax, pk[0][1] - ay)) if pk else 99.0
            res["traces"].append(ent)
        on = mv.labels[gid].get("onset") or {}
        if on.get("verdict") == "emerged_within":
            for k in NEG:
                b = int(on["last_absent_bin"]) - k
                if b < mv.lo:
                    continue
                gx, gy, r = mv.grain_at(gid, b)
                R = r + 25
                half = 64
                maps = maps_for(mv, b, gx, gy, half, net, chans, prr)
                S = 2 * half
                jj, ii = np.meshgrid(np.arange(S), np.arange(S))
                dist = np.hypot(jj - (half - 0.5), ii - (half - 0.5))
                region = (dist <= R) & ~mv.outside(b, gx, gy, half)
                ent = {"grain": gid, "bin": b, "back": k, "r": r, "methods": {}}
                for name, m in maps.items():
                    if name == "P":
                        continue
                    for suffix, reg in (("", region), ("_out", region & (dist > r + 1))):
                        pk = peaks(m, reg, 1)
                        if pk:
                            x, y, v = pk[0]
                            ent["methods"][name + suffix] = [v, float(dist[y, x] - r)]
                res["pre_onset"].append(ent)
    out = OUT / f"eval_{movie}_{tag}.json"
    out.write_text(json.dumps(res))
    log(f"{movie} [{tag}]: {len(res['traces'])} FULL traces ({len(res['skipped'])} outside the input range), "
        f"{len(res['pre_onset'])} pre-onset grain-bins, {time.time() - t0:.0f} s -> {out.name}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--net")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run(a.movie, a.net, a.tag)
