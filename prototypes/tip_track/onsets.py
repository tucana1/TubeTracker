"""Germination onsets from the detector's rim response, judged leave one movie out against 0.8.8 (paired over grains).

    python -m prototypes.tip_track.onsets

Rim response at bin b: the strongest detector peak within r - 2 .. r + 25 px of where the grain is then (rim.py).
Rule: onset = the first bin where it is >= thr and stays so in >= frac of the next W bins (optionally only peaks
within ang_tol degrees of that bin's peak angle counting), then walked back while >= thr_lo (at most `back` bins).
Settings are chosen on two movies (most onset hits, all human 'emerged within' grains counted, a grain called never
counting as a miss), applied to the third. Combined rules with the 0.8.8 reading's own onset:
  late_N: the detector's onset where the reading's is later by more than N bins or missing (N chosen on two movies);
  agree:  the earlier of the two where both see the tube leaving on the same side (exit angles within 45 deg);
  missing: the detector's only where the reading says no tube.
Onset hit = within 600 source frames (2 bins) of the human bracket, as sparsetrack.evaluate.score.
"""
from __future__ import annotations

import itertools
import json
import math
import sys

import numpy as np

from sparsetrack.evaluate import interval_distance

from .common import OUT, baseline, labels

TOL = 600.0
BAND = (-2.0, 25.0)
STUCK = {"m1": ["g005", "g065", "g033", "g047"]}  # 0.8.8's m1 floods that never start properly


def series_full(G: dict, lo: int, band=BAND, sector: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per bin (from lo): the best in-band peak's value and angle (radians, from where the grain is then); with
    ``sector`` = (angle, half-width) in radians, only peaks within it."""
    n = len(G["peaks"])
    R, A = np.zeros(n), np.full(n, np.nan)
    pos = np.asarray(G["pos"], float)
    for t, pk in enumerate(G["peaks"]):
        for v, X, Y, d in pk:
            if band[0] <= d <= band[1]:
                a = math.atan2(Y - pos[lo + t][1], X - pos[lo + t][0])
                if sector is not None and angdiff(a, sector[0]) > sector[1]:
                    continue
                R[t] = v
                A[t] = a
                break
    return R, A


def angdiff(a, b):
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - b))))


def onset(R: np.ndarray, A: np.ndarray, thr: float, W: int, frac: float, thr_lo: float | None = None,
          back: int = 6, ang_tol: float | None = None) -> int | None:
    """First index t with R[t] >= thr that stays up in >= frac of the W bins from t (see the module docstring)."""
    n = len(R)
    above = R >= thr
    need = frac * W
    for t in np.flatnonzero(above):
        if t + W > n:
            break
        ok = above[t:t + W]
        if ang_tol is not None:
            ok = ok & (angdiff(A[t:t + W], A[t]) <= math.radians(ang_tol))
        if ok.sum() >= need - 1e-9:
            s = t
            if thr_lo is not None:
                while s > 0 and t - s < back and R[s - 1] >= thr_lo and (
                        ang_tol is None or angdiff(A[s - 1], A[t]) <= math.radians(ang_tol)):
                    s -= 1
            return int(s)
    return None


class Movie:
    def __init__(self, name: str):
        self.name = name
        self.rim = json.loads((OUT / f"rim_{name}.json").read_text())
        self.lo, self.fpb = self.rim["lo"], self.rim["fpb"]
        self.L = labels(name)
        self.pred = {g["id"]: g for g in baseline(name)["grains"]}
        grains = {g: v for g, v in self.L["grains"].items() if not v.get("excluded") and v.get("isolated", True)}
        self.emerged = [g for g in sorted(grains) if (self.L["labels"].get(g, {}).get("onset") or {}).get("verdict")
                        == "emerged_within" and g in self.rim["grains"]]
        self.never = [g for g in sorted(grains) if (self.L["labels"].get(g, {}).get("onset") or {}).get("verdict")
                      == "no_emergence_by_end" and g in self.rim["grains"]]
        self.debris = [g for g, v in self.rim["grains"].items() if v["kind"] == "debris"]
        self.S = {g: series_full(v, self.lo) for g, v in self.rim["grains"].items()}
        self._sec: dict = {}

    def sector_series(self, g: str, width_deg: float):
        """The rim response only within width_deg of the 0.8.8 reading's exit (the whole band without one)."""
        key = (g, width_deg)
        if key not in self._sec:
            a = self.base_angle(g)
            self._sec[key] = self.S[g] if a is None else series_full(self.rim["grains"][g], self.lo,
                                                                     sector=(a, math.radians(width_deg)))
        return self._sec[key]

    def frame(self, t: int | None) -> int | None:
        return None if t is None else (self.lo + t) * self.fpb + self.fpb // 2

    def err(self, g: str, frame: int | None) -> float | None:
        if frame is None:
            return None
        on = self.L["labels"][g]["onset"]
        return interval_distance(frame, on.get("last_absent_frame"), on.get("first_visible_frame"))

    def base_frame(self, g: str) -> int | None:
        p = self.pred.get(g)
        return p.get("onset_frame") if p and p.get("status") == "emerged_within" else None

    def base_angle(self, g: str) -> float | None:
        p = self.pred.get(g)
        if not p or not p.get("exit_xy"):
            return None
        return math.atan2(p["exit_xy"][1] - p["y"], p["exit_xy"][0] - p["x"])

    def det(self, g: str, prm: dict) -> tuple[int | None, float | None]:
        prm = dict(prm)
        sec = prm.pop("sector", None)
        R, A = self.S[g] if not sec or g not in self.pred else self.sector_series(g, sec)
        t = onset(R, A, **prm)
        return (None, None) if t is None else (self.frame(t), float(A[t]) if np.isfinite(A[t]) else None)


def hits(mv: Movie, frames: dict) -> dict:
    """{grain: True/False} over the human 'emerged within' grains (no onset = miss)."""
    out = {}
    for g in mv.emerged:
        e = mv.err(g, frames.get(g))
        out[g] = e is not None and abs(e) <= TOL
    return out


def combine(mv: Movie, g: str, prm: dict, rule: str, N: int = 0) -> int | None:
    fb, (fd, ad) = mv.base_frame(g), mv.det(g, prm)
    if rule == "base":
        return fb
    if rule == "det":
        return fd
    if rule == "missing":
        return fd if fb is None else fb
    if rule == "late":
        if fd is None:
            return fb
        return fd if fb is None or fb - fd > N * mv.fpb else fb
    if rule == "agree":
        ab = mv.base_angle(g)
        if fd is None or fb is None:
            return fb
        if ab is not None and ad is not None and angdiff(ad, ab) <= math.radians(45):
            return min(fb, fd)
        return fb
    raise ValueError(rule)


GRID = [dict(thr=thr, W=W, frac=frac, thr_lo=lo, back=6, ang_tol=at, sector=sec)
        for thr in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
        for W, frac in ((3, 1.0), (5, 0.8), (5, 1.0), (8, 0.75), (12, 0.7), (20, 0.7))
        for lo in (None, "half")
        for at in (None, 35.0)
        for sec in (None, 40.0)]


def resolve(prm: dict) -> dict:
    p = dict(prm)
    if p["thr_lo"] == "half":
        p["thr_lo"] = 0.5 * p["thr"]
    return p


def boot(d: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    lo, hi = np.percentile(d[idx].sum(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def total(mvs, prm, rule="det", N=0):
    return sum(sum(hits(mv, {g: combine(mv, g, prm, rule, N) for g in mv.emerged}).values()) for mv in mvs)


def choose(train: list[Movie], rule: str = "det"):
    best, arg = -1, None
    Ns = (0, 2, 4, 8, 16, 32) if rule == "late" else (0,)
    for prm0, N in itertools.product(GRID, Ns):
        prm = resolve(prm0)
        s = total(train, prm, rule, N)
        # ties: the higher threshold (fewer false starts), then the shorter window
        key = (s, prm["thr"], -prm["W"])
        if best == -1 or key > best:
            best, arg = key, (prm, N)
    return arg, best[0]


def report(mv: Movie, prm: dict, rule: str, N: int, log=print) -> dict:
    frames = {g: combine(mv, g, prm, rule, N) for g in mv.emerged}
    base = {g: mv.base_frame(g) for g in mv.emerged}
    h_new, h_base = hits(mv, frames), hits(mv, base)
    d = np.array([int(h_new[g]) - int(h_base[g]) for g in mv.emerged])
    lo, hi = boot(d)
    early_new = sum(1 for g in mv.emerged if (e := mv.err(g, frames[g])) is not None and e < -TOL)
    early_base = sum(1 for g in mv.emerged if (e := mv.err(g, base[g])) is not None and e < -TOL)
    late_new = sum(1 for g in mv.emerged if (e := mv.err(g, frames[g])) is not None and e > TOL)
    late_base = sum(1 for g in mv.emerged if (e := mv.err(g, base[g])) is not None and e > TOL)
    none_new = sum(1 for g in mv.emerged if frames[g] is None)
    none_base = sum(1 for g in mv.emerged if base[g] is None)
    never = {g: combine(mv, g, prm, rule, N) is not None for g in mv.never}
    deb = {g: mv.det(g, prm)[0] is not None for g in mv.debris} if rule != "base" else {}
    timed_new = sum(1 for g in mv.emerged if frames[g] is not None)
    out = {"movie": mv.name, "rule": rule, "N": N, "prm": prm, "hits": int(sum(h_new.values())),
           "base_hits": int(sum(h_base.values())), "n": len(mv.emerged), "diff": int(d.sum()), "ci": [lo, hi],
           "timed": timed_new, "early": early_new, "early_base": early_base, "late": late_new, "late_base": late_base,
           "none": none_new, "none_base": none_base, "never_fired": never, "debris_fired": deb,
           "gained": [g for g in mv.emerged if h_new[g] and not h_base[g]],
           "lost": [g for g in mv.emerged if h_base[g] and not h_new[g]],
           "stuck": {g: (frames.get(g), mv.err(g, frames.get(g))) for g in STUCK.get(mv.name, []) if g in frames}}
    log(f"  {mv.name} {rule:7s} N={N:<2d}: {out['hits']}/{out['n']} vs 0.8.8 {out['base_hits']}/{out['n']} "
        f"({out['diff']:+d}, 95% CI {lo:+.0f} to {hi:+.0f}); timed {timed_new}; early {early_new} (0.8.8 {early_base}), "
        f"late {late_new} ({late_base}), none {none_new} ({none_base}); never-germinated fired "
        f"{sum(never.values())}/{len(never)}; debris fired {sum(deb.values())}/{len(deb)}; "
        f"gained {out['gained']} lost {out['lost']}")
    return out


def main(log=print):
    names = ["ld", "m2", "m1"]
    mvs = {m: Movie(m) for m in names}
    results = []
    for rule in ("det", "late", "agree", "missing"):
        log(f"\nrule {rule}")
        for test in names:
            train = [mvs[m] for m in names if m != test]
            (prm, N), s = choose(train, "det" if rule in ("agree", "missing") else rule)
            log(f" held out {test}: chosen on {[m.name for m in train]} ({s} hits there): {prm} N={N}")
            results.append(report(mvs[test], prm, rule, N, log))
    (OUT / "onsets_lomo.json").write_text(json.dumps(results, default=str))


if __name__ == "__main__":
    main()
