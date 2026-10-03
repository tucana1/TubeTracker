"""Helpers to score O3 (0.8.8's own routes, o3.py) and the soft recency term, paired against 0.8.8 over grains; the
entry point that produced the reported numbers is followup.py (this module's own main is an earlier, superseded run).

Settings: the O1 front settings tuned on ld for that variant (o1/tune_ld/summary.json "ld/b", "ld/c"), and for
sensitivity those tuned on m2 ("m2/b", "m2/c" of o1/tune_m2). Onsets: as the front gives them ("dp"), or 0.8.8's kept
("088": its status and onset, the front held at 0 before 0.8.8's onset bin and >= 1 px from it). Soft recency
(carry.dp_recency): its own settings (mu, delta, wait, and void for the variant with the void cost) tuned on ld with
the front settings fixed; m2-tuned for sensitivity.

    python -m prototypes.carry_front.o3eval            # writes runs/research/carry_front/o3eval.json + predictions
"""
from __future__ import annotations

import copy
import itertools
import json
import time
from pathlib import Path

import numpy as np

from prototypes.carry_front.carry import OUT, arrival_bins, dp_bounded, dp_recency
from prototypes.carry_front.o1 import (baseline, lengths_from_front, load, paired, per_grain, reasons, score,
                                       tip_xy, trace_eval)

SOFT_GRID = [dict(mu=mu, delta=d, wait=w, void=k) for mu, d, w, k in
             itertools.product((0.1, 0.25, 0.5), (10, 20), (10, 20), (0.0, 0.1, 0.3))]
PRED = OUT / "o3" / "pred"


def bounds_088(g, res: dict, fpb: int, S: np.ndarray, vmax: int):
    """0.8.8's onset imposed: front 0 before its onset bin, a tube (>= 1 px) from it; never germinated: 0
    throughout. jmin capped at vmax (else the step from the forced 0 is infeasible)."""
    n, N = g.nb - g.rs, len(S)
    lo, hi = np.zeros(n, int), np.full(n, N, int)
    jmin = min(N, int(np.searchsorted(S, 1.0)) + 1, max(1, vmax))
    st = res.get("status")
    if st == "emerged_within" and res.get("onset_frame") is not None:
        i = int(res["onset_frame"]) // fpb - g.rs
        hi[: max(0, i)] = 0
        lo[max(0, i):] = jmin
    elif st == "emerged_at_start":
        lo[:] = jmin
    else:
        hi[:] = 0
    return lo, hi


def read(g, v: str, prm: dict, soft: dict | None, bd=None) -> np.ndarray:
    K = g.K[f"{v}_{prm['w']:g}"]
    ev = K - prm["theta"]
    ns = int(np.searchsorted(g.S, prm["skip"], side="right")) if prm["skip"] > 0 else 0
    ev[:, :ns] = 0.0
    lo, hi = bd if bd is not None else (None, None)
    if soft and (soft["mu"] > 0 or soft["void"] > 0):
        f = dp_recency(ev, prm["vmax"], arrival_bins(K, prm["theta"]), soft["mu"], soft["delta"], soft["wait"],
                       gap=4, exempt=int(np.searchsorted(g.S, 3.0)), lo=lo, hi=hi, void=soft["void"],
                       marked=K >= prm["theta"], hole=3)
    else:
        f = dp_bounded(ev, prm["vmax"], lo, hi)
    return lengths_from_front(f, g.S, prm["c"], lo)


def run(mv, grains, v, prm, onset: str, soft: dict | None, method: str):
    pred = copy.deepcopy(baseline(mv.name))
    pred["method"] = method
    byid = {g["id"]: g for g in pred["grains"]}
    rows = {}
    for g in grains:
        res = byid.get(g.gid)
        if res is None:
            continue
        bd = bounds_088(g, res, mv.fpb, g.S, prm["vmax"]) if onset == "088" else None
        L = read(g, v, prm, soft, bd)
        rows[g.gid] = trace_eval(g, v, L)
        frames = list(res["length"]["frames"])
        idx = [g.idx((fr - mv.fpb // 2) // mv.fpb) for fr in frames]
        px = [round(float(L[i]), 2) for i in idx]
        res["length"] = {"frames": frames, "px": px}
        res["tip"] = {"frames": frames, "xy": [[round(float(x), 2) for x in tip_xy(g.route(v, i), g.S, float(L[i]))]
                                                for i in idx]}
        res["drift"] = {"frames": frames, "xy": [[0.0, 0.0]] * len(frames)}
        for key in ("path_by_bin", "bend"):
            res.pop(key, None)
        if onset == "dp":
            grown = [k for k, p in enumerate(px) if p > 0]
            if not grown:
                res.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None)
            elif grown[0] == 0:
                res.update(status="emerged_at_start", onset_frame=frames[0], onset_interval=None)
            else:
                k = grown[0]
                res.update(status="emerged_within", onset_frame=frames[k], onset_interval=[frames[k - 1], frames[k]])
    return pred, rows


def hits(grains, v, prm, soft) -> tuple[int, int]:
    h = b = 0
    for g in grains:
        for r in trace_eval(g, v, read(g, v, prm, soft)):
            h += r["hit"]
            b += r["both"]
    return h, b


def tune_soft(grains, v, prm) -> dict:
    best = None
    for s in SOFT_GRID:
        sc = hits(grains, v, prm, s)
        if best is None or sc > best[0]:
            best = (sc, s)
    return best[1]


def tune_soft_novoid(grains, v, prm) -> dict:
    best = None
    for s in SOFT_GRID:
        if s["void"] > 0:
            continue
        sc = hits(grains, v, prm, s)
        if best is None or sc > best[0]:
            best = (sc, s)
    return best[1]


def report(mv, pred, rows) -> dict:
    rep, base = score(mv.labels, pred), score(mv.labels, baseline(mv.name))
    pc = paired(per_grain(base), per_grain(rep))
    L = rep["length_full"]
    return {"lengths": L["within_tolerance"], "n": L["n"], "both": rep["tips"]["length_and_tip"],
            "onsets": rep["onset"]["hits"], "onset_n": rep["onset"]["n_timed"], "median_abs": L["median_abs_error"],
            "bias": L["bias"], "paired": pc, "reasons": reasons(rows)}


def fmt(name, m, r):
    p = r["paired"]
    return (f"{name:34s} {m}: {r['lengths']}/{r['n']} d {p['lengths'][0]:+d} [{p['lengths'][1]:+.0f},{p['lengths'][2]:+.0f}]"
            f" | l&t {r['both']} d {p['both'][0]:+d} [{p['both'][1]:+.0f},{p['both'][2]:+.0f}] | onsets "
            f"{r['onsets']}/{r['onset_n']} d {p['onsets'][0]:+d} | med|e| {r['median_abs']:.2f} bias {r['bias']:+.2f} | "
            f"{r['reasons']}")


def main():
    o1ld = json.loads((OUT / "o1" / "tune_ld" / "summary.json").read_text())
    o1m2 = json.loads((OUT / "o1" / "tune_m2" / "summary.json").read_text())
    sets = {"O1": OUT, "O3": OUT / "o3"}
    out = {}
    PRED.mkdir(parents=True, exist_ok=True)
    data = {}
    for s, v in (("O3", "b"), ("O3", "c"), ("O1", "b"), ("O1", "c")):
        if not any(k[0] == s for k in data):  # one set in memory at a time (the machine is short of RAM)
            data = {(s, m): load(m, sets[s], only={"b", "c"}, all_scored=(s == "O3")) for m in ("ld", "m2", "m1")}
        name = f"{s}{v}"
        ld_prm, m2_prm = o1ld[f"ld/{v}"]["params"], o1m2[f"m2/{v}"]["params"]
        t0 = time.time()
        soft_ld = tune_soft(data[(s, "ld")][1], v, ld_prm)
        soft_ld_nv = tune_soft_novoid(data[(s, "ld")][1], v, ld_prm)
        soft_m2 = tune_soft(data[(s, "m2")][1], v, ld_prm)
        print(f"{name}: soft tuned on ld {soft_ld} (no void: {soft_ld_nv}); on m2 {soft_m2} ({time.time() - t0:.0f}s)",
              flush=True)
        configs = [("ld settings, front onsets", ld_prm, "dp", None),
                   ("ld settings, 0.8.8 onsets", ld_prm, "088", None),
                   ("ld settings + soft recency (no void)", ld_prm, "dp", soft_ld_nv),
                   ("ld settings + soft recency + void", ld_prm, "dp", soft_ld),
                   ("ld + soft/void, 0.8.8 onsets", ld_prm, "088", soft_ld),
                   ("ld settings + soft/void tuned on m2", ld_prm, "dp", soft_m2),
                   ("m2 settings, front onsets", m2_prm, "dp", None)]
        if s == "O1":
            configs = [c for c in configs if "0.8.8" not in c[0]]
        for label, prm, onset, soft in configs:
            for m in ("ld", "m2", "m1"):
                mv, gs = data[(s, m)]
                pred, rows = run(mv, gs, v, prm, onset, soft, f"carry_front {name} {label}")
                r = report(mv, pred, rows)
                key = f"{name} | {label} | {m}"
                out[key] = {**r, "params": prm, "soft": soft, "onset": onset}
                print(fmt(f"{name} {label}", m, r), flush=True)
                if s == "O3":
                    fn = PRED / f"{name}_{label.replace(' ', '_').replace(',', '').replace('/', '-')}_{m}.json"
                    fn.write_text(json.dumps(pred))
        (OUT / "o3eval.json").write_text(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
