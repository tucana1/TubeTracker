"""Simulated reviews: a person traces each grain once or twice; the other traced lengths are filled in and scored.

Protocols, per scored grain (isolated, not excluded) with >= 2 FULL human traces:
  P1  the person traces the grain once, at its LATEST full trace;
  P2  once, at its EARLIEST full trace with a tube >= 8 px (grains where that is not also the latest);
  P3  twice, at both (the same grains as P2).
Targets: the grain's other FULL traces not in contact (as sparsetrack.evaluate.score); length within max(2 px, 10%);
length-and-tip also needs the tip within max(5 px, 10%) of the apex (reference coordinates).

Methods:
  M0  SparseTrack 0.8.8 alone (length and tip + drift);
  M1  the app's curve until 3 Oct 2026 (the model's rescaled to the person's lengths, frozen as
      prototypes.review_curve.curves.rescaled_curve); tip on the app's route
      (tubetracker.app.overlay.route_at: the first traced route at or after the bin, else the last one carried on
      along the model's route, cut to the length, moved with the analysis' drift);
  M2  the image fill (fill.read_lengths) on the carried route's kymograph, through the person's lengths;
  M3  M2 within D bins of a trace, a fallback (M1 or M0) farther away.
Onset (the review state, the same for every method): the model's, unless it comes after the person's first trace
or the model read no germination (the app then makes the person set it: they give theirs). ``--onset human``: the
person gives the onset for every grain.

M2's settings (and M3's D and fallback) are tuned on one movie and applied unchanged to the others. Paired
comparisons: summed per-grain differences in hits with a 95% bootstrap over grains (scripts/compare_predictions.py).

    python -m prototypes.review_fill.evaluate --tune ld --apply ld m2 m1 [--onset human] [--tag NAME]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from prototypes.review_fill.build import OUT, baseline, full_traces, grains, labels, series_index
from prototypes.review_fill.fill import arclen, dense, point_at, read_lengths
from sparsetrack.report import turned_path
from prototypes.review_curve.curves import rescaled_curve
from sparsetrack.review import lengths_by_bin
from tubetracker.app.overlay import continued, to_length

ROUTE = {"P1": "l", "P2": "f", "P3": "j"}  # P2: "f" = beyond the trace, 0.8.8's route per bin; "e" = carried
ANCHORS = {"P1": ("l",), "P2": ("e",), "P3": ("e", "l")}
GRID = {"w": (1.0, 2.0), "theta": (0.3, 0.4, 0.5, 0.6), "skip": (0.0, 3.0, 6.0), "c": (0.0, 1.0, 2.0, 3.0),
        "k_v": (1.5, 2.0, 3.0, 5.0), "v_floor": (1.0, 2.0), "onset": ("free", "review", "clip")}
M3_GRID = {"D": (20, 40, 60, 80, 100, 150), "fallback": ("M1", "M0"), "lam": (0.02, 0.05, 0.1, 0.2, 0.5),
           "tau": (0.5, 0.7, 0.8, 0.9, 1.0)}


def route_key(proto: str, prm: dict) -> str:
    return prm.get("p2", ROUTE["P2"]) if proto == "P2" else ROUTE[proto]


def tol(h: float) -> float:
    return max(2.0, 0.1 * h)


def tip_tol(h: float) -> float:
    return max(5.0, 0.1 * h)


class Grain:
    """One grain: its traces, 0.8.8's reading, and the built kymographs and routes."""

    def __init__(self, movie: str, lab: dict, pred: dict, gid: str):
        z = np.load(OUT / movie / f"{gid}.npz")
        self.meta = json.loads((OUT / movie / f"{gid}.json").read_text())
        self.gid, self.rs, self.nb = gid, int(self.meta["rs"]), int(self.meta["nb"])
        self.fpb = int(lab["frames_per_bin"])
        self.K = {k[2:]: z[k].astype(np.float32) for k in z.files if k.startswith("K_")}
        self.X = {k[2:]: z[k] for k in z.files if k.startswith("X_")}
        self.traced = list(self.meta["traced_bins"])
        self.traces = full_traces(lab, gid)
        self.pred = pred
        self.res = next((x for x in pred["grains"] if x["id"] == gid), None)
        self.on_h = (lab["labels"].get(gid) or {}).get("onset") or {}
        self.model_px = lengths_by_bin(self.res, self.fpb, self.nb) if self.res else np.zeros(self.nb)
        frames = (self.res or {}).get("length", {}).get("frames") or []
        self.drift = None
        d = ((self.res or {}).get("drift") or {}).get("xy")
        if d and frames:
            start = frames[0] // self.fpb
            self.drift = np.zeros((self.nb, 2))
            for b in range(self.nb):
                self.drift[b] = d[int(min(max(b - start, 0), len(d) - 1))]

    def row(self, b: int) -> int:
        return int(min(max(b, self.rs), self.nb - 1) - self.rs)

    def drift_at(self, b: int) -> np.ndarray:
        return self.drift[b] if self.drift is not None else np.zeros(2)

    def anchors(self, proto: str) -> list[tuple[int, float, dict]]:
        a = self.meta["anchors"]
        out = []
        for k in ANCHORS[proto]:
            b = a[k]["bin"]
            t = next(t for bb, t in self.traces if bb == b)
            out.append((b, float(t["length_px"]), t))
        return out

    def targets(self, proto: str) -> list[tuple[int, float, np.ndarray, dict]]:
        ab = {b for b, _, _ in self.anchors(proto)}
        return [(b, float(t["length_px"]), np.asarray(t["path_xy_ref"][-1], float), t) for b, t in self.traces
                if b not in ab and not t.get("contact")]

    def protocols(self) -> list[str]:
        return ["P1"] + (["P2", "P3"] if "e" in self.meta["anchors"] else [])

    # ---- the onset the review holds ---------------------------------------------------------
    def model_onset(self) -> int | None:
        r = self.res or {}
        if r.get("status") == "emerged_within" and r.get("onset_frame") is not None:
            return int(r["onset_frame"]) // self.fpb
        if r.get("status") == "emerged_at_start":
            return 0
        return None

    def human_onset(self) -> tuple[int | None, int | None]:
        v = self.on_h.get("verdict")
        if v == "emerged_within":
            return self.on_h.get("last_absent_bin"), self.on_h.get("first_visible_bin")
        if v == "emerged_at_start":
            return None, 0
        return None, None

    def review_onset(self, b1: int, mode: str) -> tuple[int | None, int, bool]:
        """(last absent bin, first visible bin, whether the person gave it) for a review whose first trace is at
        ``b1``."""
        la_h, fv_h = self.human_onset()
        fv_m = self.model_onset()
        if mode != "human" and fv_m is not None and fv_m <= b1:
            return (fv_m - 1 if fv_m > 0 else None), fv_m, False
        if fv_h is not None and fv_h <= b1:
            return la_h, fv_h, True
        return b1 - 1, b1, True

    # ---- the methods ----------------------------------------------------------------------------
    def m0(self, b: int, frame: int) -> tuple[float, np.ndarray | None]:
        r = self.res
        if r is None:
            return 0.0, None
        fr = np.asarray(r["length"]["frames"])
        i = int(np.argmin(np.abs(fr - frame)))
        L = float(r["length"]["px"][i])
        tip = (r.get("tip") or {}).get("xy")
        if not tip or L <= 0:
            return L, None
        t = np.asarray(tip[i], float)
        d = (r.get("drift") or {}).get("xy")
        if d:
            t = t + np.asarray(d[min(i, len(d) - 1)], float)
        return L, t

    def m1_curve(self, anchors: list, fv: int) -> np.ndarray:
        return rescaled_curve(self.model_px, fv, [(b, L) for b, L, _ in anchors])

    def lin_curve(self, anchors: list, fv: int) -> np.ndarray:
        """Reference (ML): straight lines from zero at the onset through the person's lengths, on at the last
        stretch's speed."""
        pts = [(fv - 1, 0.0)] + sorted((b, L) for b, L, _ in anchors)
        b = np.arange(self.nb, dtype=float)
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        out = np.interp(b, x, y)
        v = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1)
        out[b > x[-1]] = y[-1] + v * (b[b > x[-1]] - x[-1])
        out[b < fv] = 0.0
        return np.maximum.accumulate(np.maximum(out, 0.0))

    def m1_tip(self, b: int, L: float, anchors: list) -> np.ndarray | None:
        """The app's tip: overlay.route_at with the person's routes in the analysis' frame, cut to L, moved by the
        analysis' drift."""
        if L < 2.0:
            return None
        drawn = sorted(((ba, (np.asarray(t["path_xy_ref"], float) - self.drift_at(ba)).tolist()) for ba, _, t in anchors),
                       key=lambda x: x[0])
        later = [p for ba, p in drawn if ba >= b]
        if later:
            route = later[0]
        else:
            model = []
            if self.res and self.res.get("path"):
                model = turned_path(self.res, series_index(self.res, b, self.fpb), self.pred).tolist()
            route = continued(drawn[-1][1], model)
        if len(route) < 2:
            return None
        return np.asarray(to_length(route, L)[-1], float) + self.drift_at(b)

    def m2_curve(self, proto: str, prm: dict, mode: str, prior: np.ndarray | None = None,
                 lam: float = 0.0) -> np.ndarray:
        """M2's length at every row (bin rs + row); with ``prior`` (lengths by bin) and ``lam``, pulled towards it."""
        anchors = self.anchors(proto)
        la, fv, given = self.review_onset(min(b for b, _, _ in anchors), mode)
        K = self.K[f"{route_key(proto, prm)}_{prm['w']:g}"]
        rows = [(self.row(b), L) for b, L, _ in anchors]
        impose = prm["onset"] == "review" or given or mode == "human"
        onset = None
        if impose:
            onset = (None if la is None else la - self.rs, max(fv - self.rs, 0))
        pr = None if prior is None else np.asarray(prior, float)[self.rs:self.rs + K.shape[0]]
        L = read_lengths(K, rows, prm["theta"], prm["skip"], prm["c"], prm["k_v"], prm["v_floor"], onset=onset,
                         start=max(fv - self.rs, 0), prior=pr, lam=lam)
        if prm["onset"] == "clip":  # read freely, then nothing before the review's onset (as the app shows it)
            L[: max(fv - self.rs, 0)] = 0.0
        return L

    def m2_tip(self, proto: str, b: int, L: float, prm: dict) -> np.ndarray | None:
        if L <= 0:
            return None
        X = self.X[route_key(proto, prm)][self.traced.index(b)]
        return point_at(X, L)[0]

    def marked(self, proto: str, b: int, h: float, prm: dict) -> dict:
        """Along the protocol's route at bin b: the share of the human length (past ``skip``) the tube map marks
        (K >= theta), and the share of the 20 px beyond the apex it marks (material ahead: a crossing or older tube)."""
        K = self.K[f"{route_key(proto, prm)}_{prm['w']:g}"][self.row(b)]
        on = K >= prm["theta"]
        i0, i1 = int(prm["skip"]), max(int(round(h)), int(prm["skip"]) + 1)
        return {"marked": round(float(on[i0:i1].mean()), 2) if i1 <= len(on) else None,
                "ahead": round(float(on[i1 + 2:i1 + 22].mean()), 2) if i1 + 22 <= len(on) else None}

    def cover(self, proto: str, b: int, h: float, trace: dict, prm: dict) -> float:
        """Mean distance (px) of the protocol's route at bin b, over the human length, to the human trace."""
        X = self.X[route_key(proto, prm)][self.traced.index(b)]
        sel = arclen(X) <= h + 1e-6
        hum = dense(trace["path_xy_ref"], 0.5)
        d = np.hypot(X[sel][:, None, 0] - hum[None, :, 0], X[sel][:, None, 1] - hum[None, :, 1]).min(axis=1)
        return float(d.mean()) if len(d) else float("nan")


def load_movie(movie: str) -> list[Grain]:
    lab, pred = labels(movie), baseline(movie)
    return [Grain(movie, lab, pred, gid) for gid in grains(lab) if (OUT / movie / f"{gid}.npz").exists()]


def row_of(g: Grain, proto: str, b: int, h: float, apex: np.ndarray, trace: dict, L: float,
           tip: np.ndarray | None) -> dict:
    te = float(np.hypot(*(tip - apex))) if tip is not None and L > 0 else None
    hit = abs(L - h) <= tol(h)
    return {"gid": g.gid, "bin": b, "h": round(h, 2), "pred": round(float(L), 2), "err": round(float(L - h), 2),
            "hit": bool(hit), "both": bool(hit and te is not None and te <= tip_tol(h)),
            "tip_err": None if te is None else round(te, 2)}


def run(gs: list[Grain], prm: dict, m3: dict | None, mode: str, methods=("M0", "M1", "M2", "M3", "ML"),
        diag: bool = True) -> dict:
    """rows[proto][method] = per-target rows."""
    out = {p: {m: [] for m in methods} for p in ("P1", "P2", "P3")}
    for g in gs:
        for proto in g.protocols():
            targets = g.targets(proto)
            if not targets:
                continue
            anchors = g.anchors(proto)
            la, fv, given = g.review_onset(min(b for b, _, _ in anchors), mode)
            c1 = g.m1_curve(anchors, fv) if ("M1" in methods or "M3" in methods) else None
            cl = g.lin_curve(anchors, fv) if "ML" in methods else None
            c2 = g.m2_curve(proto, prm, mode) if ("M2" in methods or "M3" in methods) else None
            c3 = None
            if "M3" in methods and m3 is not None and m3.get("rule") == "prior":
                c3 = g.m2_curve(proto, prm, mode, prior=c1, lam=m3["lam"])
            for b, h, apex, t in targets:
                frame = t.get("source_frame") or b * g.fpb + g.fpb // 2
                vals = {}
                if "M0" in methods or "M3" in methods:
                    vals["M0"] = g.m0(b, frame)
                if c1 is not None:
                    L1 = float(c1[b]) if b < len(c1) else 0.0
                    vals["M1"] = (L1, g.m1_tip(b, L1, anchors))
                if cl is not None:
                    Ll = float(cl[b])
                    vals["ML"] = (Ll, g.m1_tip(b, Ll, anchors))
                if c2 is not None:
                    L2 = float(c2[g.row(b)])
                    vals["M2"] = (L2, g.m2_tip(proto, b, L2, prm))
                if "M3" in methods and m3 is not None:
                    if c3 is not None:
                        L3 = float(c3[g.row(b)])
                        vals["M3"] = (L3, g.m2_tip(proto, b, L3, prm))
                    elif m3.get("rule") == "conf":  # the image fill where the map marks what it read, else fallback
                        sup = g.marked(proto, b, vals["M2"][0], prm)["marked"] if vals["M2"][0] > prm["skip"] + 1 else 0.0
                        vals["M3"] = vals["M2"] if (sup or 0.0) >= m3["tau"] else vals[m3["fallback"]]
                    else:
                        dist = min(abs(b - ba) for ba, _, _ in anchors)
                        vals["M3"] = vals[m3["fallback"]] if dist > m3["D"] else vals["M2"]
                for m in methods:
                    if m in vals:
                        r = row_of(g, proto, b, h, apex, t, *vals[m])
                        if m == "M2" and diag:
                            r["cover"] = round(g.cover(proto, b, h, t, prm), 2)
                            r["onset_given"] = given
                            r.update(g.marked(proto, b, h, prm))
                        out[proto][m].append(r)
    return out


def counts(rows: list[dict]) -> dict:
    e = [abs(r["err"]) for r in rows]
    return {"n": len(rows), "hits": sum(r["hit"] for r in rows), "both": sum(r["both"] for r in rows),
            "med_abs": round(float(np.median(e)), 2) if e else None}


def per_grain(rows: list[dict]) -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = {}
    for r in rows:
        v = out.setdefault(r["gid"], [0, 0])
        v[0] += r["hit"]
        v[1] += r["both"]
    return {k: tuple(v) for k, v in out.items()}


def paired(base: list[dict], new: list[dict], n_boot: int = 4000, seed: int = 0) -> dict:
    b, n = per_grain(base), per_grain(new)
    g = sorted(set(b) & set(n))
    if not g:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    out = {}
    for k, name in enumerate(("lengths", "both")):
        d = np.array([n[x][k] - b[x][k] for x in g])
        lo, hi = np.percentile(d[idx].sum(axis=1), [2.5, 97.5])
        out[name] = (int(d.sum()), float(lo), float(hi))
    out["grains"] = len(g)
    return out


def objective(res: dict, method: str) -> tuple:
    rows = [r for p in res for r in res[p][method]]
    c = counts(rows)
    return (c["hits"], c["both"], -(c["med_abs"] or 0.0))


STAGES = (("w", "theta", "c", "skip"), ("k_v", "v_floor", "onset"), ("w", "theta", "c", "skip"))
START = {"w": 1.0, "theta": 0.3, "skip": 0.0, "c": 1.0, "k_v": 2.0, "v_floor": 1.0, "onset": "free"}


def tune(gs: list[Grain], mode: str, log=print) -> tuple[dict, dict, list]:
    """M2's settings (all protocols together; a grid over each stage's settings in turn, the others held at the best
    so far: ``STAGES``), then M3's rule."""
    best = dict(START)
    seen: dict = {}
    t0 = time.time()
    for stage in STAGES:
        for vals in itertools.product(*(GRID[k] for k in stage)):
            prm = {**best, **dict(zip(stage, vals))}
            key = json.dumps(prm, sort_keys=True)
            if key not in seen:
                seen[key] = (objective(run(gs, prm, None, mode, methods=("M2",), diag=False), "M2"), prm)
        best = max(seen.values(), key=lambda x: x[0])[1]
        log(f"  after {stage}: {max(seen.values(), key=lambda x: x[0])[0]} {best} ({time.time() - t0:.0f} s)")
    scored = sorted(seen.values(), key=lambda x: x[0], reverse=True)
    log(f"  grid of {len(scored)} settings in {time.time() - t0:.0f} s; best {scored[0][0]} {scored[0][1]}")
    for s, p in scored[1:6]:
        log(f"    runner-up {s} {p}")
    prm = scored[0][1]
    m3s = []
    cands = [{"rule": "dist", "D": D, "fallback": fb} for D, fb in itertools.product(M3_GRID["D"], M3_GRID["fallback"])]
    cands += [{"rule": "prior", "lam": lam} for lam in M3_GRID["lam"]]
    cands += [{"rule": "conf", "tau": tau, "fallback": fb} for tau, fb in itertools.product(M3_GRID["tau"], M3_GRID["fallback"])]
    for m3 in cands:
        res = run(gs, prm, m3, mode, diag=False)
        m3s.append((objective(res, "M3"), m3))
    m3s.sort(key=lambda x: (x[0], -x[1].get("D", 0)), reverse=True)
    log(f"  M3 best {m3s[0][0]} {m3s[0][1]}; others " + "; ".join(f"{s} {m}" for s, m in m3s[1:5]))
    return prm, m3s[0][1], scored[:20]


def report(name: str, res: dict, log=print) -> dict:
    out = {}
    for proto in ("P1", "P2", "P3"):
        R = res[proto]
        if not R.get("M0"):
            continue
        line = {m: counts(R[m]) for m in R}
        for m in ("M2", "M3"):
            for base in ("M1", "M0"):
                line[f"{m}-{base}"] = paired(R[base], R[m])
        out[proto] = line
        c = line
        log(f"  {name} {proto} (n={c['M0']['n']} traces, {c['M2-M1'].get('grains', 0)} grains): " + " | ".join(
            f"{m} {c[m]['hits']}/{c[m]['both']} med {c[m]['med_abs']}" for m in ("M0", "M1", "M2", "M3", "ML")))
        log("      " + " | ".join(f"{k} len {c[k]['lengths'][0]:+d} [{c[k]['lengths'][1]:+.0f},{c[k]['lengths'][2]:+.0f}]"
                                    f" l&t {c[k]['both'][0]:+d} [{c[k]['both'][1]:+.0f},{c[k]['both'][2]:+.0f}]"
                                    for k in ("M2-M1", "M2-M0", "M3-M1", "M3-M0") if c[k]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="ld")
    ap.add_argument("--apply", nargs="+", default=["ld", "m2", "m1"])
    ap.add_argument("--onset", default="review", choices=("review", "human"))
    ap.add_argument("--tag")
    ap.add_argument("--params", help="JSON of M2 settings (skip tuning)")
    ap.add_argument("--m3", help="JSON of M3 settings (skip tuning)")
    ap.add_argument("--settings-from", help="an earlier evaluation's tag: its tuned M2 and M3 settings, unchanged")
    a = ap.parse_args(argv)
    try:
        from prototypes.review_fill import fastdp
        fastdp.install()  # the same fronts, compiled (fastdp.check)
    except ImportError:  # no numba: the numpy DP (about 10x slower grids)
        pass
    tag = a.tag or f"tune_{a.tune}_{a.onset}"
    od = OUT / "eval" / tag
    od.mkdir(parents=True, exist_ok=True)
    data = {m: load_movie(m) for m in dict.fromkeys([a.tune] + a.apply)}
    print(f"[{tag}] grains: " + ", ".join(f"{m} {len(g)}" for m, g in data.items()), flush=True)
    top = []
    if a.settings_from:
        prev = json.loads((OUT / "eval" / a.settings_from / "summary.json").read_text())
        a.params, a.m3 = json.dumps(prev["params"]), json.dumps(prev["m3"])
    if a.params:
        prm, m3 = json.loads(a.params), json.loads(a.m3) if a.m3 else {"rule": "dist", "D": 10 ** 6, "fallback": "M1"}
    else:
        prm, m3, top = tune(data[a.tune], a.onset)
        if a.m3:
            m3 = json.loads(a.m3)
    print(f"[{tag}] M2 {prm}; M3 {m3}", flush=True)
    summary = {"params": prm, "m3": m3, "onset": a.onset, "tuned_on": a.tune,
               "top": [(s, p) for s, p in top], "movies": {}}
    for m in a.apply:
        res = run(data[m], prm, m3, a.onset)
        summary["movies"][m] = {"table": report(m, res), "rows": res}
    (od / "summary.json").write_text(json.dumps(summary, default=float))
    print(f"[{tag}] -> {od / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
