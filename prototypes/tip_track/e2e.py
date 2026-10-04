"""The end-to-end benchmark (scripts/synth_bench.py with the tipdet options, each movie with its held-out detector)
against 0.8.8: paired bootstrap over grains for onsets (every human 'emerged within' grain; none = miss), lengths and
length + tip (all FULL traces and young ones, human < 15 px), the germination curve, and a check that readings the
detector did not touch are 0.8.8's.

    python -m prototypes.tip_track.e2e DIR      # DIR/{ld,m2,m1}_real_0/predictions.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sparsetrack.evaluate import interval_distance, score

from .common import baseline, labels
from .onsets import boot
from .young import per_grain


def onset_hits(L: dict, pred: dict) -> dict:
    P = {g["id"]: g for g in pred["grains"]}
    out = {}
    for gid, g in L["grains"].items():
        if g.get("excluded") or not g.get("isolated", True):
            continue
        on = (L["labels"].get(gid) or {}).get("onset") or {}
        if on.get("verdict") != "emerged_within" or gid not in P:
            continue
        p = P[gid]
        f = p.get("onset_frame") if p.get("status") == "emerged_within" else None
        out[gid] = f is not None and abs(interval_distance(f, on.get("last_absent_frame"),
                                                           on.get("first_visible_frame"))) <= 600
    return out


def main(folder: str, log=print):
    for m in ("ld", "m2", "m1"):
        f = Path(folder) / f"{m}_real_0" / "predictions.json"
        if not f.exists():
            log(f"{m}: no predictions yet")
            continue
        new, base, L = json.loads(f.read_text()), baseline(m), labels(m)
        rn, rb = score(L, new), score(L, base)
        hn, hb = onset_hits(L, new), onset_hits(L, base)
        g = sorted(hb)
        d = np.array([int(hn[k]) - int(hb[k]) for k in g])
        lo, hi = boot(d)
        log(f"\n{m}: onsets {sum(hn.values())}/{len(g)} vs 0.8.8 {sum(hb.values())} ({d.sum():+d}, 95% CI {lo:+.0f} to "
            f"{hi:+.0f}); scorer: {rn['onset']['hits']}/{rn['onset']['n_timed']} timed (0.8.8 {rb['onset']['hits']}/"
            f"{rb['onset']['n_timed']}); gained {[k for k in g if hn[k] and not hb[k]]} lost "
            f"{[k for k in g if hb[k] and not hn[k]]}")
        for name, yo in (("all FULL", False), ("young (<15 px)", True)):
            pb, pn = per_grain(rb, yo), per_grain(rn, yo)
            ks = sorted(pb)
            dl = np.array([pn[k][0] - pb[k][0] for k in ks])
            db = np.array([pn[k][1] - pb[k][1] for k in ks])
            n = sum(pb[k][2] for k in ks)
            (a, b), (c, e) = boot(dl), boot(db)
            log(f"  {name}: lengths {sum(pn[k][0] for k in ks)}/{n} vs {sum(pb[k][0] for k in ks)} ({dl.sum():+d}, CI "
                f"{a:+.0f} to {b:+.0f}); length+tip {sum(pn[k][1] for k in ks)} vs {sum(pb[k][1] for k in ks)} "
                f"({db.sum():+d}, CI {c:+.0f} to {e:+.0f})")
        Pn, Pb = rn.get("population") or {}, rb.get("population") or {}
        log(f"  germination curve: T50 human {Pb.get('t50_human')}, 0.8.8 {Pb.get('t50_model')}, new "
            f"{Pn.get('t50_model')} frames; germinated human {Pb.get('germinated_human')}, 0.8.8 "
            f"{Pb.get('germinated_model')}, new {Pn.get('germinated_model')}; max gap {Pb.get('max_gap')} -> "
            f"{Pn.get('max_gap')}")
        log(f"  length bias {rb['length_full']['bias']:+.2f} -> {rn['length_full']['bias']:+.2f} px, median |err| "
            f"{rb['length_full']['median_abs_error']:.2f} -> {rn['length_full']['median_abs_error']:.2f}")
        B = {r["id"]: r for r in base["grains"]}
        touched = same = differ = 0
        for r in new["grains"]:
            if r.get("tip_detector") and (r["tip_detector"].get("young_bins") or r["tip_detector"].get("moved_onset_bins")):
                touched += 1
                continue
            b = B.get(r["id"])
            if b is not None and b["length"]["px"] == r["length"]["px"] and b.get("onset_frame") == r.get("onset_frame"):
                same += 1
            else:
                differ += 1
        log(f"  readings changed by the detector: {touched}; untouched and identical to 0.8.8: {same}; untouched but "
            f"different: {differ}")


if __name__ == "__main__":
    main(sys.argv[1])
