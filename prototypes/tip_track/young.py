"""Young tubes' lengths from the detector's tip, scored against the human traces (leave one movie out).

    python -m prototypes.tip_track.young [--onsets rel|late]

Where the tube is young at a bin - the 0.8.8 reading is between 0 and `young` px, or (with --onsets) it reads no tube
yet although the detector's onset (onsets.py) has passed - the tip is the strongest detector peak (>= vmin) in the
band r - 2 .. r + young + 5 px round where the grain is then (within `sector` degrees of the reading's exit where it
has one). Length (est):
  "radial":      its distance from the grain's centre - r + k (a young tube is short and nearly straight; k takes up
                 where annotators start a trace, ~1 px inside the census circle, and the peak's lag behind the apex);
  "radial_edge": the same from the grain's visible edge in that direction (edges.py) + k;
  "route":       arc length along the reading's own route to the peak's projection (within 6 px of it), + k.
Lengths are made non-decreasing again (pool adjacent violators from the onset on); a reading with no tube at all
holds the detector's last young length. Settings chosen on two movies, applied to the third; every predicted
document is scored with sparsetrack.evaluate.score and compared with 0.8.8 per grain (paired bootstrap).
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import sys

import numpy as np

from sparsetrack import routes
from sparsetrack.evaluate import score

from .common import OUT, baseline, labels
from .onsets import Movie as OnsetMovie, angdiff, boot, combine, onset

EXT = 10.0
_RIM: dict = {}
_BASE: dict = {}
_EDGE: dict = {}


def rim_of(movie: str) -> dict:
    if movie not in _RIM:
        _RIM[movie] = json.loads((OUT / f"rim_{movie}.json").read_text())
    return _RIM[movie]


def base_of(movie: str) -> dict:
    if movie not in _BASE:
        _BASE[movie] = baseline(movie)
    return _BASE[movie]


def edge_of(movie: str) -> dict:
    if movie not in _EDGE:
        _EDGE[movie] = json.loads((OUT / f"edges_{movie}.json").read_text())
    return _EDGE[movie]


def route_field(res: dict, i: int) -> np.ndarray | None:
    """The route the reading draws at frame index i, in the field (turned about its pivot, moved by the drift)."""
    rt = routes.route_at(res, i)
    if len(rt) < 2:
        return None
    n = len(res["length"]["frames"])
    rot = res.get("rotation_deg") or []
    th = math.radians(float(rot[i])) if len(rot) == n else 0.0
    pivot = np.asarray(res["exit_xy"] if res.get("exit_xy") else [res["x"], res["y"]], float)
    turn = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    d = np.asarray(res["drift"]["xy"][i], float) if res.get("drift") else np.zeros(2)
    return (rt - pivot) @ turn.T + pivot + np.nan_to_num(d)


def project(route: np.ndarray, q, ext: float = EXT) -> tuple[float, float]:
    """(arc length of q's projection on the route or its straight extension by ext px, distance from it)."""
    p = np.asarray(route, float)
    s = routes.arc(p)
    back = next((k for k in range(len(p) - 2, -1, -1) if np.hypot(*(p[-1] - p[k])) >= 3.0), 0)
    dvec = p[-1] - p[back]
    dvec = dvec / max(float(np.hypot(*dvec)), 1e-9)
    pts = np.vstack([p, p[-1] + ext * dvec])
    ss = np.append(s, s[-1] + ext)
    a, b = pts[:-1], pts[1:]
    v = b - a
    L2 = np.maximum((v ** 2).sum(axis=1), 1e-12)
    t = np.clip(((q[0] - a[:, 0]) * v[:, 0] + (q[1] - a[:, 1]) * v[:, 1]) / L2, 0, 1)
    proj = a + t[:, None] * v
    d = np.hypot(proj[:, 0] - q[0], proj[:, 1] - q[1])
    k = int(np.argmin(d))
    return float(ss[k] + t[k] * math.sqrt(L2[k])), float(d[k])


def pav(y: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    vals, wts, cnt = [], [], []
    for v in y:
        vals.append(float(v)); wts.append(1.0); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            c = cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v2], [w], [c]
    return np.repeat(vals, cnt)


def exit_angle(res: dict) -> float | None:
    if not res.get("exit_xy") or not res.get("status", "").startswith("emerged"):
        return None
    return math.atan2(res["exit_xy"][1] - res["y"], res["exit_xy"][0] - res["x"])


def young_doc(movie: str, cfg: dict, onset_frames: dict | None = None) -> tuple[dict, dict]:
    """0.8.8's predictions with young lengths (and onsets) from the detector; also {(gid, i): kind} of changed bins."""
    rim, edges = rim_of(movie), edge_of(movie)
    lo, hi, rs = rim["lo"], rim["hi"], rim["rs"]
    doc = copy.deepcopy(base_of(movie))
    changed = {}
    Y = cfg["young"]
    for res in doc["grains"]:
        g = res["id"]
        G = rim["grains"].get(g)
        if G is None:
            continue
        pos = np.asarray(G["pos"], float)
        r = float(G["r"])
        frames = res["length"]["frames"]
        n = len(frames)
        px = np.asarray(res["length"]["px"], float)
        tips = [list(t) if t is not None else None for t in (res.get("tip") or {}).get("xy", [None] * n)]
        if not res.get("tip"):
            res["tip"] = {"frames": frames, "xy": tips}
        drift = np.nan_to_num(np.asarray(res["drift"]["xy"], float)) if res.get("drift") else np.zeros((n, 2))
        onset_new = None if onset_frames is None else onset_frames.get(g)
        old_on = res.get("onset_frame") if res.get("status") == "emerged_within" else None
        start = None
        if onset_new is not None and (old_on is None or onset_new < old_on):
            start = int(np.argmin(np.abs(np.asarray(frames) - onset_new)))
        a_exit = exit_angle(res)
        no_tube = not res.get("status", "").startswith("emerged")
        found = np.zeros(n, bool)
        for i in range(n):
            b = rs + i
            if not lo <= b <= hi:
                continue
            pre = start is not None and i >= start and px[i] <= 0
            if not ((0 < px[i] < Y and not cfg.get("only_pre")) or pre):
                continue
            c = pos[b]
            cands = []
            rt = route_field(res, i) if cfg["est"] == "route" and not no_tube else None
            for v, X, Yy, d in G["peaks"][b - lo]:
                if v < cfg["vmin"] or not -2.0 <= d <= Y + 5.0:
                    continue
                a = math.atan2(Yy - c[1], X - c[0])
                if cfg.get("sector") and a_exit is not None and angdiff(a, a_exit) > math.radians(cfg["sector"]):
                    continue
                if cfg["est"] == "route":
                    if rt is None:
                        continue
                    s, perp = project(rt, (X, Yy))
                    if perp > 6.0:
                        continue
                    L = s + cfg["k"]
                elif cfg["est"] == "radial_edge":
                    e = edges[g][int(round(math.degrees(a) % 360 / 5.0)) % 72]
                    L = math.hypot(X - c[0], Yy - c[1]) - (r + e) + cfg["k"]
                else:
                    L = math.hypot(X - c[0], Yy - c[1]) - r + cfg["k"]
                cands.append((v, L, X, Yy))
            if not cands:
                continue
            v, L, X, Yy = max(cands)
            px[i] = max(L, 0.0)
            tips[i] = [round(X - drift[i][0], 2), round(Yy - drift[i][1], 2)]
            found[i] = True
            changed[(g, i)] = "pre" if pre else "young"
        if start is not None:
            first = next((k for k in range(start, n) if px[k] > 0), None)
            if first is not None:
                for k in range(start, first):
                    px[k] = px[first] * (k - start + 1) / (first - start + 1)
                    tips[k] = tips[first]
                res["status"] = "emerged_within"
                res["onset_frame"] = frames[start]
                res["onset_interval"] = [frames[start - 1], frames[start]] if start > 0 else None
                if no_tube:  # no reading at all: the detector's young lengths, held after its last one
                    last = None
                    for k in range(start, n):
                        if found[k] or k < first:
                            last = k
                        elif last is not None:
                            px[k], tips[k] = px[last], tips[last]
        if found.any() or start is not None:
            on = int(np.argmax(px > 0)) if (px > 0).any() else n
            px[on:] = pav(px[on:])
            res["length"]["px"] = [round(float(v), 2) for v in px]
            res["final_length_px"] = round(float(px[-1]), 2)
            res["tip"]["xy"] = tips
    return doc, changed


def per_grain(rep: dict, young_only: bool = False, young_px: float = 15.0) -> dict:
    out = {}
    for r in rep["rows"]:
        full = [f for f in r.get("full", []) if not young_only or f["human"] < young_px]
        ok = [abs(f["error"]) <= max(2.0, 0.1 * f["human"]) for f in full]
        both = [o and f.get("tip_error", 1e9) <= max(5.0, 0.1 * f["human"]) for o, f in zip(ok, full)]
        out[r["grain"]] = (sum(ok), sum(both), len(full))
    return out


def compare(movie: str, doc: dict, log=print, tag: str = "") -> dict:
    L = labels(movie)
    rb, rn = score(L, base_of(movie)), score(L, doc)
    res = {"movie": movie, "tag": tag}
    for name, yo in (("all", False), ("young", True)):
        pb, pn = per_grain(rb, yo), per_grain(rn, yo)
        g = sorted(pb)
        dl = np.array([pn[k][0] - pb[k][0] for k in g])
        db = np.array([pn[k][1] - pb[k][1] for k in g])
        n = sum(pb[k][2] for k in g)
        res[name] = {"n": n, "len": int(sum(pn[k][0] for k in g)), "len_base": int(sum(pb[k][0] for k in g)),
                     "both": int(sum(pn[k][1] for k in g)), "both_base": int(sum(pb[k][1] for k in g)),
                     "len_ci": boot(dl), "both_ci": boot(db)}
        x = res[name]
        log(f"  {movie} {tag} {name:5s}: lengths {x['len']}/{n} vs 0.8.8 {x['len_base']} ({x['len'] - x['len_base']:+d}, "
            f"CI {x['len_ci'][0]:+.0f} to {x['len_ci'][1]:+.0f}); length+tip {x['both']} vs {x['both_base']} "
            f"({x['both'] - x['both_base']:+d}, CI {x['both_ci'][0]:+.0f} to {x['both_ci'][1]:+.0f})")
    o_b, o_n = rb["onset"], rn["onset"]
    res["onset"] = {"hits": o_n["hits"], "timed": o_n["n_timed"], "base_hits": o_b["hits"], "base_timed": o_b["n_timed"]}
    res["population"] = {"base": rb.get("population"), "new": rn.get("population")}
    pb, pn = rb.get("population") or {}, rn.get("population") or {}
    log(f"  {movie} {tag} onsets {o_n['hits']}/{o_n['n_timed']} (0.8.8 {o_b['hits']}/{o_b['n_timed']}); T50 human "
        f"{pb.get('t50_human')}, 0.8.8 {pb.get('t50_model')}, new {pn.get('t50_model')}; germinated human "
        f"{pb.get('germinated_human')}, 0.8.8 {pb.get('germinated_model')}, new {pn.get('germinated_model')}; max gap "
        f"{pb.get('max_gap')} -> {pn.get('max_gap')}")
    return res


GRID = [dict(est=e, k=k, vmin=v, young=y, sector=s) for e in ("radial", "radial_edge", "route")
        for k in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5) for v in (0.1, 0.2) for y in (15.0, 20.0, 25.0) for s in (None, 40.0)]


# onset rules for the bins before a reading's own onset (as chosen leave one movie out in onsets.py / README)
REL = dict(alpha=0.5, W=3, frac=1.0, q=98)


def rel_onset(R, alpha, floor, W, frac, q=98):
    from scipy.ndimage import median_filter
    Rs = median_filter(R, size=3, mode="nearest")
    thr = max(floor, alpha * np.percentile(Rs, q))
    above = Rs >= thr
    for t in np.flatnonzero(above):
        if t + W > len(R):
            break
        if above[t:t + W].sum() >= frac * W - 1e-9:
            return int(t)
    return None


def onset_frames_for(movie: str, rule: str, floor: float = 0.2, N: int = 10) -> dict:
    mv = OnsetMovie(movie)
    out = {}
    lvl = dict(thr=0.3, W=3, frac=1.0, thr_lo=None, back=6, ang_tol=None)
    for g in mv.rim["grains"]:
        if mv.rim["grains"][g]["kind"] != "graded":
            continue
        if rule == "rel":
            R, _ = mv.S[g]
            out[g] = mv.frame(rel_onset(R, floor=floor, **REL))
        elif rule == "late":
            out[g] = combine(mv, g, lvl, "late", N)
        elif rule == "late_nofill":  # only where the reading has an onset, later by more than N bins
            fb = mv.base_frame(g)
            out[g] = combine(mv, g, lvl, "late", N) if fb is not None else None
        elif rule == "rel_late":  # the plateau-relative onset where the reading's is later by more than N bins
            fb = mv.base_frame(g)
            R, _ = mv.S[g]
            fd = mv.frame(rel_onset(R, floor=floor, **REL))
            out[g] = fb if fd is None else (fd if fb is None or fb - fd > N * mv.fpb else fb)
    return out


def main(log=print, onset_rule: str | None = None):
    names = ["ld", "m2", "m1"]
    onset_frames = {m: (onset_frames_for(m, onset_rule) if onset_rule else None) for m in names}
    cache = {}
    for m, (k, cfg) in itertools.product(names, enumerate(GRID)):
        doc, _ = young_doc(m, cfg, onset_frames[m])
        cache[(m, k)] = score(labels(m), doc)["length_full"]["within_tolerance"]
    results = []
    for test in names:
        train = [m for m in names if m != test]
        k = max(range(len(GRID)), key=lambda k: (sum(cache[(m, k)] for m in train), GRID[k]["vmin"]))
        best = GRID[k]
        log(f"held out {test}: chosen on {train} ({sum(cache[(m, k)] for m in train)} there): {best}")
        doc, changed = young_doc(test, best, onset_frames[test])
        r = compare(test, doc, log, tag=f"young[{onset_rule or 'no onset change'}]")
        r["cfg"] = best
        r["changed"] = len(changed)
        results.append(r)
    (OUT / f"young_lomo_{onset_rule or 'none'}.json").write_text(json.dumps(results, default=str))
    return results


if __name__ == "__main__":
    rule = sys.argv[sys.argv.index("--onsets") + 1] if "--onsets" in sys.argv else None
    main(onset_rule=rule)
