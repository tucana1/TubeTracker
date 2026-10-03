"""Simulated reviews (prototypes/review_fill's protocols) of candidate reviewed curves (``curves.py``).

Per scored grain with >= 2 FULL traces: P1 the person traces it at its latest full trace, P2 at its earliest full
trace >= 8 px, P3 both; the curve is read at the grain's other FULL traces not in contact (length within max(2 px,
10%); length-and-tip also needs the app's tip, ``overlay.route_at`` on the person's routes, within max(5 px, 10%)).
M0 = 0.8.8 alone, M1 = the app until 3 Oct 2026 (``curves.rescaled_curve``). Onset: the review's (0.8.8's unless
it comes after the person's first trace or 0.8.8 read none: then the person's), or with ``--onset human`` the
person's always. Candidates are chosen on ld (``--tune``) and checked unchanged on m2 and m1.

    python -m prototypes.review_curve.evaluate --tune ld --apply ld m2 m1 [--onset human] [--tag NAME]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from prototypes.review_curve.curves import curve
from prototypes.review_fill.build import OUT as FILL_OUT
from prototypes.review_fill.evaluate import counts, load_movie, paired, row_of

OUT = FILL_OUT.parent / "review_curve"

CANDIDATES = {"clip": dict(family="add", W=0, after="none")}
for W in (10, 20, 40, 80, None):
    tag = "all" if W is None else W
    CANDIDATES[f"add-carry-{tag}"] = dict(family="add", W=W, after="carry")
    CANDIDATES[f"add-carry-stall-{tag}"] = dict(family="add", W=W, after="carry", stall=True)
    CANDIDATES[f"add-none-stall-{tag}"] = dict(family="add", W=W, after="none", stall=True)
    CANDIDATES[f"add-decay-stall-{tag}"] = dict(family="add", W=W, after="decay", stall=True)
    CANDIDATES[f"stretch-{tag}"] = dict(family="stretch", W=W)
for k in list(CANDIDATES):
    if k != "clip" and k.startswith(("add-carry-stall", "add-none-stall", "add-decay-stall")):
        CANDIDATES[k + "-shift"] = dict(CANDIDATES[k], onset="shift")
# round 2 (after round 1 on ld: carrying a correction past its anchor and long windows cost; a stall ramp should
# need a real shortfall): the correction fades out after an anchor, the stall rule in three strengths
ROUND2 = {"clip": CANDIDATES["clip"]}
for W in (10, 20, 40):
    ROUND2[f"decay-{W}"] = dict(family="add", W=W, after="decay")
    for rule in ("any", "tol", "zero"):
        ROUND2[f"decay-stall{rule}-{W}"] = dict(family="add", W=W, after="decay", stall=rule)
for k in list(ROUND2):
    ROUND2[k + "-shift"] = dict(ROUND2[k], onset="shift")
# round 3: the chosen curve with the onset handled three ways when the review's onset differs from the model's
ROUND3 = {"chosen": dict(family="add", W=20, after="decay", stall="tol")}
ROUND3["chosen-down"] = dict(ROUND3["chosen"], onset="down")
ROUND3["chosen-shift"] = dict(ROUND3["chosen"], onset="shift")
ROUNDS = {"r1": CANDIDATES, "r2": ROUND2, "r3": ROUND3}


def evaluate(gs, mode: str, names: list[str], cands: dict | None = None) -> dict:
    """rows[proto][method] for M0, M1 and the candidates ``names``."""
    out = {p: {m: [] for m in ["M0", "M1", *names]} for p in ("P1", "P2", "P3")}
    for g in gs:
        for proto in g.protocols():
            targets = g.targets(proto)
            if not targets:
                continue
            anchors = g.anchors(proto)
            la, fv, given = g.review_onset(min(b for b, _, _ in anchors), mode)
            pts = [(b, L) for b, L, _ in anchors]
            curves = {"M1": g.m1_curve(anchors, fv)}
            curves.update({nm: curve(g.model_px, fv, pts, **(cands or CANDIDATES)[nm]) for nm in names})
            for b, h, apex, t in targets:
                frame = t.get("source_frame") or b * g.fpb + g.fpb // 2
                out[proto]["M0"].append(row_of(g, proto, b, h, apex, t, *g.m0(b, frame)))
                for nm, c in curves.items():
                    L = float(c[b]) if b < len(c) else 0.0
                    out[proto][nm].append(row_of(g, proto, b, h, apex, t, L, g.m1_tip(b, L, anchors)))
    return out


def total(res: dict, m: str) -> tuple:
    rows = [r for p in res for r in res[p][m]]
    c = counts(rows)
    return (c["hits"], c["both"], -(c["med_abs"] or 0.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="ld")
    ap.add_argument("--apply", nargs="+", default=["ld", "m2", "m1"])
    ap.add_argument("--onset", default="review", choices=("review", "human"))
    ap.add_argument("--tag")
    ap.add_argument("--round", default="r1", choices=tuple(ROUNDS))
    ap.add_argument("--pick", help="report this candidate (chosen on the tuning movie among ties) instead of the "
                                   "first-ranked")
    a = ap.parse_args(argv)
    tag = a.tag or f"tune_{a.tune}_{a.onset}_{a.round}"
    cands = ROUNDS[a.round]
    names = list(cands)
    data = {m: load_movie(m) for m in dict.fromkeys([a.tune] + a.apply)}
    res = {m: evaluate(gs, a.onset, names, cands) for m, gs in data.items()}
    ranked = sorted(names, key=lambda nm: total(res[a.tune], nm), reverse=True)
    print(f"[{tag}] onset {a.onset}; candidates ranked on {a.tune} (hits, len&tip, -median |e|) over P1-P3:")
    print(f"   M0 {total(res[a.tune], 'M0')}  M1 {total(res[a.tune], 'M1')}")
    for nm in ranked:
        cells = []
        for m in data:
            cells.append(f"{m} " + " ".join(f"{p}:{counts(res[m][p][nm])['hits']}" for p in res[m]))
        print(f"   {nm:28s} {a.tune} {total(res[a.tune], nm)} | " + " | ".join(cells))
    best = a.pick or ranked[0]
    if a.pick and total(res[a.tune], a.pick) != total(res[a.tune], ranked[0]):
        print(f"[{tag}] note: {a.pick} {total(res[a.tune], a.pick)} is not tied with the first-ranked on {a.tune}")
    print(f"[{tag}] chosen on {a.tune}: {best} {cands[best]}")
    summary = {"onset": a.onset, "tuned_on": a.tune, "round": a.round, "chosen": best, "params": cands[best],
               "ranked": [(nm, total(res[a.tune], nm)) for nm in ranked], "movies": {}}
    for m in a.apply:
        table = {}
        for p in ("P1", "P2", "P3"):
            R = res[m][p]
            if not R["M0"]:
                continue
            table[p] = {k: counts(R[k]) for k in ("M0", "M1", best)}
            table[p]["new-M1"] = paired(R["M1"], R[best])
            table[p]["new-M0"] = paired(R["M0"], R[best])
            table[p]["M0-M1"] = paired(R["M1"], R["M0"])
            c = table[p]
            print(f"  {m} {p} (n={c['M0']['n']}): M0 {c['M0']['hits']}/{c['M0']['both']} med {c['M0']['med_abs']} | "
                  f"M1 {c['M1']['hits']}/{c['M1']['both']} med {c['M1']['med_abs']} | {best} {c[best]['hits']}/"
                  f"{c[best]['both']} med {c[best]['med_abs']} || new-M1 len {c['new-M1']['lengths'][0]:+d} "
                  f"[{c['new-M1']['lengths'][1]:+.0f},{c['new-M1']['lengths'][2]:+.0f}] l&t {c['new-M1']['both'][0]:+d} "
                  f"[{c['new-M1']['both'][1]:+.0f},{c['new-M1']['both'][2]:+.0f}] | new-M0 len "
                  f"{c['new-M0']['lengths'][0]:+d} [{c['new-M0']['lengths'][1]:+.0f},{c['new-M0']['lengths'][2]:+.0f}]"
                  f" l&t {c['new-M0']['both'][0]:+d} [{c['new-M0']['both'][1]:+.0f},{c['new-M0']['both'][2]:+.0f}]",
                  flush=True)
        summary["movies"][m] = {"table": table, "rows": {p: {k: res[m][p][k] for k in ("M0", "M1", best)}
                                                         for p in res[m]}}
    od = OUT / tag
    od.mkdir(parents=True, exist_ok=True)
    (od / "summary.json").write_text(json.dumps(summary, default=float))
    print(f"[{tag}] -> {od / 'summary.json'}")


if __name__ == "__main__":
    main()
