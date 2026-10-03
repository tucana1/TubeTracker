"""O1: read lengths from the oracle-route kymographs (build.py) with a globally optimal monotone front; calibrate on
one movie, apply unchanged to the others; write predictions (the 0.8.8 baseline's, with labelled grains' length, tip
and onset replaced) and score them with sparsetrack.evaluate.score, paired against 0.8.8 over grains.

Reader: evidence E[b, s] = K[b, s] - theta (K: tube map maxed over the route normal within +/- w px; the first
``skip`` px of route neutral), front f[b] = sparsetrack-style dp_front with speed cap vmax px/bin (the tip can stay
put), length = material arc length to the front minus a tip offset c. Onset = first bin with a length > 0.
"Human onset forced": f = 0 up to the human's last absent bin, length >= 1 px from the first visible bin.

    python -m prototypes.carry_front.o1 --tune ld --apply ld m2 m1 --tag tune_ld
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from prototypes.carry_front.carry import OUT, REPO, Movie, arrival_bins, dp_arrival, dp_bounded

sys.path.insert(0, str(REPO / "scripts"))
from sparsetrack.evaluate import score  # noqa: E402

BASE = Path("/private/tmp/claude-501/-Users-joshjiang-Documents-TubeTracker/eaa2b708-9f78-4565-aa0c-9af3763da8fd/"
            "scratchpad/bt/base088")
VARIANTS = ("a", "a2", "braw", "b", "c")
GRID = {"w": (1.0, 2.0, 3.0), "theta": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7), "vmax": (1, 2, 3, 4, 6), "skip": (0, 3, 6),
        "arrival": ((0.0, 0),)}
ARRIVAL_GRID = {"w": (1.0, 2.0), "theta": (0.3, 0.4, 0.5), "vmax": (1, 2, 3, 4), "skip": (0, 3),
                "arrival": tuple((mu, d) for mu in (1.0, 10.0, 1000.0) for d in (5, 10, 20))}
OFFSETS = tuple(np.arange(-3.0, 4.01, 0.5))


def baseline(movie: str) -> dict:
    return json.loads((BASE / f"{movie}_real_0/predictions.json").read_text())


class Grain:
    """One labelled grain: its kymographs, routes and traces."""

    def __init__(self, mv: Movie, gid: str, drift: dict | None, root: Path = OUT, only: set | None = None):
        z = np.load(root / mv.name / f"{gid}.npz")
        meta = json.loads((root / mv.name / f"{gid}.json").read_text())
        self.gid, self.rs, self.nb, self.bT = gid, mv.rs, mv.nb, int(meta["bT"])
        self.S = z["S"].astype(np.float64)
        self.origin = np.asarray(meta["origin"])
        self.cover = meta["cover"]
        keep = lambda k: only is None or k.split("_")[0] in only  # noqa: E731 (variants to load: less memory)
        self.K = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("K_") and keep(k[2:])}
        for k in [k for k in self.K if k.startswith("a_")]:
            self.K.setdefault("a2" + k[1:], self.K[k])  # no 0.8.8 drift for this grain: a2 = a
        self.X = {k[2:]: z[k] for k in z.files if k.startswith("X_") and (keep(k[2:]) or k == "X_b")}
        self.F = self.X["b"][self.bT - self.rs].astype(np.float64) + self.origin  # the final route (reference)
        self.drift = None
        if drift is not None:
            fr = np.asarray(drift["frames"])
            bin_of = (fr - mv.fpb // 2) // mv.fpb
            dxy = np.asarray(drift["xy"], float)
            d = np.array([dxy[int(np.argmin(np.abs(bin_of - b)))] for b in range(mv.rs, mv.nb)])
            self.drift = d - d[self.bT - self.rs]
        lab = mv.labels["labels"][gid]
        self.onset = lab.get("onset") or {}
        self.traces = []  # scored FULL traces: (bin, human length, apex, contact)
        for b, t in sorted(((int(b), t) for b, t in (lab.get("traces") or {}).items()), key=lambda x: x[0]):
            if t["state"] == "full":
                self.traces.append((b, float(t["length_px"]), np.asarray(t["path_xy_ref"][-1], float),
                                    bool(t.get("contact")), t))

    def idx(self, b: int) -> int:
        return min(max(b, self.rs), self.nb - 1) - self.rs

    def route(self, v: str, i: int) -> np.ndarray:
        """Variant v's route at series index i, reference coordinates."""
        if v == "a":
            return self.F
        if v == "a2":
            return self.F + (self.drift[i] if self.drift is not None else 0.0)
        return self.X[v][i].astype(np.float64) + self.origin

    def bounds(self, c: float, vmax: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Front bounds that force the human onset: f = 0 through the last absent bin; a tube from the first visible
        bin (front >= jmin; lengths_from_front then reports >= 1 px). jmin is capped at vmax: with jmin > vmax the
        step from the forced 0 is infeasible and the DP returns garbage (the bug that voided vmax = 1 in the forced
        grid of the first O1 runs)."""
        n, N = self.nb - self.rs, len(self.S)
        lo, hi = np.zeros(n, int), np.full(n, N, int)
        v = self.onset.get("verdict")
        jmin = min(N, int(np.searchsorted(self.S, c + 1.0)) + 1, max(1, vmax))
        if v == "emerged_within":
            la, fv = self.onset.get("last_absent_bin"), self.onset.get("first_visible_bin")
            if la is not None:
                hi[: max(0, la - self.rs + 1)] = 0
            if fv is not None:
                lo[max(0, fv - self.rs):] = jmin
        elif v == "emerged_at_start":
            lo[:] = jmin
        elif v == "no_emergence_by_end":
            hi[:] = 0
        return lo, hi


def read_front(K: np.ndarray, S: np.ndarray, theta: float, vmax: int, skip: float, bounds=None, mu: float = 0.0,
               delta: int = 10) -> np.ndarray:
    ev = K - theta
    ns = int(np.searchsorted(S, skip, side="right")) if skip > 0 else 0
    ev[:, :ns] = 0.0
    lo, hi = bounds if bounds is not None else (None, None)
    if mu > 0:  # ownership by arrival (carry.dp_arrival); the first 3 px (the rim) exempt
        return dp_arrival(ev, vmax, arrival_bins(K, theta), mu, delta, int(np.searchsorted(S, 3.0)), lo, hi)
    return dp_bounded(ev, vmax, lo, hi)


def lengths_from_front(f: np.ndarray, S: np.ndarray, c: float, lo: np.ndarray | None = None) -> np.ndarray:
    """Material arc length to the front minus the tip offset; where the human onset is forced (``lo`` > 0) at
    least 1 px."""
    L = np.where(f >= 1, S[np.maximum(f - 1, 0)], 0.0)
    L = np.where(L > 0, np.maximum(L - c, 0.0), 0.0)
    if lo is not None:
        L = np.where(lo > 0, np.maximum(L, 1.0), L)
    return L


def tip_xy(route: np.ndarray, S: np.ndarray, L: float) -> np.ndarray:
    return np.array([np.interp(L, S, route[:, 0]), np.interp(L, S, route[:, 1])])


def trace_eval(g: Grain, v: str, L: np.ndarray) -> list[dict]:
    """Per scored FULL trace (not in contact): hit, length-and-tip hit, error."""
    rows = []
    for b, h, apex, contact, _ in g.traces:
        if contact:
            continue
        i = g.idx(b)
        p = float(L[i])
        tol = max(2.0, 0.1 * h)
        hit = abs(p - h) <= tol
        te = float(np.hypot(*(tip_xy(g.route(v, i), g.S, p) - apex))) if p > 0 else None
        both = hit and te is not None and te <= max(5.0, 0.1 * h)
        rows.append({"bin": b, "h": h, "pred": p, "err": p - h, "hit": hit, "both": both, "tip_err": te,
                     "cover": g.cover.get(str(b), {}).get(v)})
    return rows


CAP_FACTORS = (1.0, 1.5, 2.0, 3.0)
LOOSE_VMAX = 8


def movie_rate(grains: list[Grain], key: str, theta: float, skip: float, window: int = 20) -> float:
    """A movie's growth speed without labels: per grain the fastest 20-bin growth of a loosely capped front (8 px/bin)
    along its route; the median over the movie's grains (px/bin)."""
    rates = []
    for g in grains:
        L = lengths_from_front(read_front(g.K[key], g.S, theta, LOOSE_VMAX, skip), g.S, 0.0)
        if len(L) > window and L[-1] > 5:
            rates.append(float(np.max((L[window:] - L[:-window]) / window)))
    return float(np.median(rates)) if rates else 1.0


def auto_vmax(rate: float, factor: float) -> int:
    return max(1, int(np.ceil(factor * rate)))


def grid_search(grains: list[Grain], v: str, forced: bool, cap: str = "fixed") -> list[tuple]:
    """Every grid setting and offset: (hits, both, -median |err|, params). With ``cap="auto"`` the speed cap is
    ``factor`` x the movie's own label-free growth speed (movie_rate) instead of a fixed px/bin."""
    out = []
    caps = CAP_FACTORS if cap == "auto" else GRID["vmax"]
    rate_cache: dict = {}
    for w, theta, cv, skip, (mu, dl) in itertools.product(GRID["w"], GRID["theta"], caps, GRID["skip"],
                                                           GRID["arrival"]):
        key = f"{v}_{w:g}"
        if cap == "auto":
            if (key, theta, skip) not in rate_cache:
                rate_cache[(key, theta, skip)] = movie_rate(grains, key, theta, skip)
            vmax, extra = auto_vmax(rate_cache[(key, theta, skip)], cv), {"factor": cv}
        else:
            vmax, extra = cv, {}
        if mu > 0:
            extra = {**extra, "mu": mu, "delta": dl}
        fronts = []
        for g in grains:
            bd = g.bounds(0.0, vmax) if forced else None
            fronts.append((read_front(g.K[key], g.S, theta, vmax, skip, bd, mu, dl), bd))
        for c in OFFSETS:
            hits = both = 0
            errs = []
            for g, (f, bd) in zip(grains, fronts):
                L = lengths_from_front(f, g.S, c, bd[0] if bd is not None else None)
                for r in trace_eval(g, v, L):
                    hits += r["hit"]
                    both += r["both"]
                    errs.append(abs(r["err"]))
            out.append((hits, both, -float(np.median(errs)) if errs else 0.0,
                        {"w": w, "theta": theta, "vmax": vmax, "skip": skip, "c": float(c), **extra}))
    out.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return out


def load(movie: str, root: Path = OUT, only: set | None = None, all_scored: bool = False) -> tuple[Movie, list[Grain]]:
    """The movie's grains with kymographs under ``root`` (scored grains with a FULL trace; ``all_scored``: every
    scored grain, as O3 builds them); ``only``: the variants to load."""
    mv = Movie(movie)
    drifts = {g["id"]: g.get("drift") for g in baseline(movie)["grains"]}
    ids = ([gid for gid, g in sorted(mv.labels["grains"].items()) if not g.get("excluded") and g.get("isolated", True)]
           if all_scored else mv.scored_grains())
    gs = [Grain(mv, gid, drifts.get(gid), root, only) for gid in ids if (root / movie / f"{gid}.npz").exists()]
    return mv, gs


def predict(mv: Movie, grains: list[Grain], v: str, prm: dict, forced: bool, method: str) -> tuple[dict, dict]:
    """The baseline's predictions with the labelled grains' length, tip, drift (zero: tips are in reference
    coordinates) and onset replaced; and per grain the per-trace rows."""
    pred = copy.deepcopy(baseline(mv.name))
    pred["method"] = method
    byid = {g["id"]: g for g in pred["grains"]}
    rows = {}
    vmax = prm["vmax"]
    if "factor" in prm:  # the speed cap from this movie's own growth speed
        vmax = auto_vmax(movie_rate(grains, f"{v}_{prm['w']:g}", prm["theta"], prm["skip"]), prm["factor"])
        pred["vmax_px_per_bin"] = vmax
    for g in grains:
        K = g.K[f"{v}_{prm['w']:g}"]
        bd = g.bounds(0.0, vmax) if forced else None
        f = read_front(K, g.S, prm["theta"], vmax, prm["skip"], bd, prm.get("mu", 0.0), prm.get("delta", 10))
        L = lengths_from_front(f, g.S, prm["c"], bd[0] if bd is not None else None)
        rows[g.gid] = trace_eval(g, v, L)
        res = byid.get(g.gid)
        if res is None:
            continue
        frames = list(res["length"]["frames"])
        bins = [(fr - mv.fpb // 2) // mv.fpb for fr in frames]
        px, tips = [], []
        for b in bins:
            i = g.idx(b)
            px.append(round(float(L[i]), 2))
            tips.append([round(float(x), 2) for x in tip_xy(g.route(v, i), g.S, float(L[i]))])
        res["length"] = {"frames": frames, "px": px}
        res["tip"] = {"frames": frames, "xy": tips}
        res["drift"] = {"frames": frames, "xy": [[0.0, 0.0]] * len(frames)}
        grown = [k for k, p in enumerate(px) if p > 0]
        if not grown:
            res.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None)
        elif grown[0] == 0:
            res.update(status="emerged_at_start", onset_frame=frames[0], onset_interval=None)
        else:
            k = grown[0]
            res.update(status="emerged_within", onset_frame=frames[k], onset_interval=[frames[k - 1], frames[k]])
        for key in ("path_by_bin", "bend"):
            res.pop(key, None)
    return pred, rows


def per_grain(rep: dict) -> dict[str, tuple[int, int, int]]:
    out = {}
    for r in rep["rows"]:
        on = int(abs(r["onset_error"]) <= rep["onset"]["tolerance_frames"]) if "onset_error" in r else 0
        full = r.get("full", [])
        ln = sum(abs(f["error"]) <= max(2.0, 0.1 * f["human"]) for f in full)
        both = sum(abs(f["error"]) <= max(2.0, 0.1 * f["human"]) and f.get("tip_error", 1e9) <= max(5.0, 0.1 * f["human"])
                   for f in full)
        out[r["grain"]] = (on, ln, both)
    return out


def paired(base: dict, new: dict, n_boot: int = 4000, seed: int = 0) -> dict:
    """As scripts/compare_predictions.paired: summed per-grain differences with a 95% bootstrap over grains."""
    g = sorted(set(base) & set(new))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    out = {}
    for k, name in enumerate(("onsets", "lengths", "both")):
        d = np.array([new[x][k] - base[x][k] for x in g])
        lo, hi = np.percentile(d[idx].sum(axis=1), [2.5, 97.5])
        out[name] = (int(d.sum()), float(lo), float(hi))
    return out


def reasons(rows: dict, route_px: float = 3.0) -> dict:
    """Why traces miss: route off the human trace (> route_px mean distance over its length), onset (read no tube),
    front short, front long."""
    out = {"hit": 0, "route": 0, "onset": 0, "short": 0, "long": 0}
    for rr in rows.values():
        for r in rr:
            if r["hit"]:
                out["hit"] += 1
            elif r["cover"] is not None and r["cover"] > route_px:
                out["route"] += 1
            elif r["pred"] <= 0:
                out["onset"] += 1
            elif r["err"] < 0:
                out["short"] += 1
            else:
                out["long"] += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="ld")
    ap.add_argument("--apply", nargs="+", default=["ld", "m2", "m1"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--forced", action="store_true", help="also report with the human onset forced")
    ap.add_argument("--cap", default="fixed", choices=("fixed", "auto"), help="speed cap: fixed px/bin, or a factor "
                    "x each movie's own label-free growth speed")
    ap.add_argument("--arrival", action="store_true", help="read with ownership by arrival (ARRIVAL_GRID)")
    ap.add_argument("--root", help="kymograph folder (default: the oracle routes' runs/research/carry_front)")
    a = ap.parse_args(argv)
    if a.arrival:
        GRID.clear()
        GRID.update(ARRIVAL_GRID)
    tag = a.tag or f"tune_{a.tune}"
    od = OUT / "o1" / tag
    od.mkdir(parents=True, exist_ok=True)
    root = Path(a.root) if a.root else OUT
    data = {m: load(m, root) for m in set([a.tune] + a.apply)}
    summary = {}
    for forced in ([False, True] if a.forced else [False]):
        for v in a.variants:
            res = grid_search(data[a.tune][1], v, forced, a.cap)
            best = res[0]
            prm = best[3]
            name = f"{v}{'_forced' if forced else ''}"
            print(f"[{tag}] {name}: best on {a.tune}: hits {best[0]} both {best[1]} med|e| {-best[2]:.2f} {prm}", flush=True)
            print("   runner-ups: " + "; ".join(f"{r[0]}/{r[1]} {r[3]}" for r in res[1:4]), flush=True)
            for m in a.apply:
                mv, gs = data[m]
                pred, rows = predict(mv, gs, v, prm, forced, f"carry_front {name} ({tag})")
                pf = od / f"{m}_{name}.json"
                pf.write_text(json.dumps(pred))
                labels = mv.labels
                rep_new = score(labels, pred)
                rep_base = score(labels, baseline(m))
                pc = paired(per_grain(rep_base), per_grain(rep_new))
                Lf = rep_new["length_full"]
                rs = reasons(rows)
                summary[f"{m}/{name}"] = {
                    "params": prm, "vmax_used": pred.get("vmax_px_per_bin", prm["vmax"]), "lengths": Lf["within_tolerance"], "n": Lf["n"],
                    "both": rep_new["tips"]["length_and_tip"], "onsets": rep_new["onset"]["hits"],
                    "onset_n": rep_new["onset"]["n_timed"], "median_abs": Lf["median_abs_error"], "bias": Lf["bias"],
                    "base_lengths": rep_base["length_full"]["within_tolerance"],
                    "base_both": rep_base["tips"]["length_and_tip"], "paired": pc, "reasons": rs,
                    "rows": rows}
                print(f"   {m}: vmax {pred.get('vmax_px_per_bin', prm['vmax'])} lengths {Lf['within_tolerance']}/{Lf['n']} (0.8.8 {rep_base['length_full']['within_tolerance']}) "
                      f"d {pc['lengths'][0]:+d} [{pc['lengths'][1]:+.0f},{pc['lengths'][2]:+.0f}] | len&tip "
                      f"{rep_new['tips']['length_and_tip']} (0.8.8 {rep_base['tips']['length_and_tip']}) d {pc['both'][0]:+d} "
                      f"[{pc['both'][1]:+.0f},{pc['both'][2]:+.0f}] | onsets {rep_new['onset']['hits']}/{rep_new['onset']['n_timed']}"
                      f" | med|e| {Lf['median_abs_error']:.2f} bias {Lf['bias']:+.2f} | {rs}", flush=True)
    (od / "summary.json").write_text(json.dumps(summary, default=float))


if __name__ == "__main__":
    main()
