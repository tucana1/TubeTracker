"""Tune the global tip-trajectory reader's weights on one movie (random search, then local refinement), apply them
unchanged to the others; paired comparison with SparseTrack 0.8.8 over grains (95% bootstrap), by length class.

    python -m prototypes.tip_trajectory.tune --tune ld --apply m2 m1 --n 400 --tag t_ld
Output: OUT/tune_<tag>.json (best settings, all tried settings' tuning scores, per-movie results).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from . import dp
from .common import OUT, baseline, labels, length_class

SPACE = {  # name: (low, high) uniform, or a tuple of choices
    "c": (-2.5, 3.0), "theta": (-0.2, 1.2), "w_det": (0.0, 2.5), "w_sup": (0.0, 2.0), "w_gap": (0.0, 1.0),
    "w_ahead": (0.0, 1.0), "w_tan": (0.0, 2.0), "w_carry": (0.0, 0.3), "w_on": (0.0, 3.0), "l0": (5.0, 30.0),
    "vmax": (1.5, 8.0), "shrink": (0.5, 6.0), "w_shrink": (0.0, 1.0), "w_cons": (0.0, 1.0), "w_share": (0.0, 1.0),
    "cap_tip": (3.0, 15.0), "cap_share": (3.0, 15.0), "iso": (True, False), "w_ext": (0.0, 1.0), "c_end": (-3.0, 3.0), "w_l0": (0.0, 3.0),
    "vmax_factor": (0.0, 0.0, 1.0, 1.5, 2.0, 3.0), "w_hold": (0.0, 1.5), "w_reacq": (0.0, 3.0), "cap_reacq": (4.0, 20.0), "det_norm": (True, False)}


def sample(rng, around: dict | None = None, scale: float = 0.15) -> dict:
    p = dict(dp.DEFAULT)
    for k, v in SPACE.items():
        if isinstance(v[0], bool) or len(v) > 2:
            p[k] = (type(v[0])(rng.choice(v)) if around is None or rng.random() < 0.2 else around[k])
        elif around is None:
            p[k] = float(rng.uniform(*v))
        else:
            lo, hi = v
            p[k] = float(np.clip(around[k] + rng.normal(0, scale * (hi - lo)), lo, hi)) if rng.random() < 0.4 else around[k]
    return p


def objective(ev: dict) -> float:
    return ev["lengths"] + ev["lt"] + ev["onsets"]


def by_class(rep: dict) -> dict:
    out = {}
    for r in rep["rows"]:
        for f in r.get("full", []):
            c = length_class(f["human"])
            hit = abs(f["error"]) <= max(2.0, 0.1 * f["human"])
            n, h = out.get(c, (0, 0))
            out[c] = (n + 1, h + hit)
    return out


def report(movie: str, doc: dict, p: dict, only_flood: bool = False) -> dict:
    lab, base = labels(movie), baseline(movie)
    ev = dp.evaluate(movie, doc, p, only_flood)
    b_rep = dp.score(lab, base)
    pg_b, pg_n = dp.per_grain(b_rep), dp.per_grain(ev["rep"])
    pr = dp.paired({g: pg_b[g] for g in doc["grains"] if g in pg_b}, {g: pg_n[g] for g in doc["grains"] if g in pg_n})
    return {"movie": movie, "only_flood": only_flood, "onsets": ev["onsets"], "n_on": ev["n_on"], "lengths": ev["lengths"],
            "n": ev["n"], "lt": ev["lt"],
            "base": {"onsets": b_rep["onset"]["hits"], "n_on": b_rep["onset"]["n_timed"],
                     "lengths": b_rep["length_full"]["within_tolerance"], "lt": b_rep["tips"]["length_and_tip"]},
            "paired": {"onsets": pr[0], "lengths": pr[1], "lt": pr[2]},
            "class_new": by_class(ev["rep"]), "class_base": by_class(b_rep),
            "bias": ev["rep"]["length_full"]["bias"], "mae": ev["rep"]["length_full"]["median_abs_error"]}


def fmt(r: dict) -> str:
    pr = r["paired"]
    cl = " ".join(f"{c} {r['class_new'].get(c, (0, 0))[1]}/{r['class_new'].get(c, (0, 0))[0]} (0.8.8 {r['class_base'].get(c, (0, 0))[1]})"
                  for c in ("young", "mid", "long"))
    return (f"{r['movie']}{' (flood grains only)' if r['only_flood'] else ''}: onsets {r['onsets']}/{r['n_on']} "
            f"(0.8.8 {r['base']['onsets']}/{r['base']['n_on']}; {pr['onsets'][0]:+d}, {pr['onsets'][1]:+.0f}..{pr['onsets'][2]:+.0f}) "
            f"lengths {r['lengths']}/{r['n']} (0.8.8 {r['base']['lengths']}; {pr['lengths'][0]:+d}, {pr['lengths'][1]:+.0f}.."
            f"{pr['lengths'][2]:+.0f}) l&t {r['lt']} (0.8.8 {r['base']['lt']}; {pr['lt'][0]:+d}, {pr['lt'][1]:+.0f}..{pr['lt'][2]:+.0f})"
            f" | {cl} | bias {r['bias']:.2f} med|e| {r['mae']:.2f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", required=True)
    ap.add_argument("--apply", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--refine", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="t")
    ap.add_argument("--start", help="JSON of a setting to start the refinement from")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    doc = dp.load(a.tune)
    tried = []
    t0 = time.time()
    best, best_j = dict(dp.DEFAULT), -1.0
    if a.start:
        best = {**best, **json.loads(a.start)}
    starts = [best] + [sample(rng) for _ in range(a.n)]
    for i, p in enumerate(starts):
        ev = dp.evaluate(a.tune, doc, p)
        j = objective(ev)
        tried.append((j, ev["lengths"], ev["lt"], ev["onsets"], p))
        if j > best_j:
            best, best_j = p, j
            print(f"[{i}] {j:.0f}: lengths {ev['lengths']} l&t {ev['lt']} onsets {ev['onsets']} ({time.time() - t0:.0f} s)",
                  flush=True)
    for i in range(a.refine):
        p = sample(rng, best, 0.1 if i > a.refine // 2 else 0.2)
        ev = dp.evaluate(a.tune, doc, p)
        j = objective(ev)
        tried.append((j, ev["lengths"], ev["lt"], ev["onsets"], p))
        if j > best_j:
            best, best_j = p, j
            print(f"[r{i}] {j:.0f}: lengths {ev['lengths']} l&t {ev['lt']} onsets {ev['onsets']} ({time.time() - t0:.0f} s)",
                  flush=True)
    print("best", json.dumps(best), flush=True)
    results = []
    for m in [a.tune] + list(a.apply):
        d = doc if m == a.tune else dp.load(m)
        for of in (False, True):
            r = report(m, d, best, of)
            results.append(r)
            print(fmt(r), flush=True)
    tried.sort(key=lambda x: -x[0])
    (OUT / f"tune_{a.tag}.json").write_text(json.dumps({"best": best, "results": results,
                                                         "tried": [t[:4] + (t[4],) for t in tried[:50]]}, default=str))


if __name__ == "__main__":
    main()
