"""The final table: per held-out movie, the onset network's hits against 0.8.8, 0.8.8 + the tip detector's `later`
onsets (end to end) and the tip detector's rim rules alone (from rim_series.py, rules of prototypes/tip_track:
fixed level thr 0.3 for 3 bins; relative to the grain's own plateau, 0.5 x its 98th percentile, floor 0.2; the level
rule chosen on the other two movies), paired over the same grains with bootstrap intervals; AUROC; T50.

    python -m prototypes.onset_net.compare --tags ld=v1 m2=v1 m1=v1 [--src drift]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .common import OUT, TOL, boot_ci, labels, onset_error

os.environ.setdefault("TT_TIPTRACK_OUT", str(OUT / "tiptrack"))


def rim_rules(names) -> dict:
    """{movie: {rule: {grain: onset frame or None}}} for the tip detector's rim rules (empty without rim files)."""
    if not all((OUT / "tiptrack" / f"rim_{m}.json").exists() for m in names):
        return {}
    from prototypes.tip_track import onsets as O
    from prototypes.tip_track.fixed import LEVEL
    from prototypes.tip_track.young import REL, rel_onset
    mvs = {m: O.Movie(m) for m in names}
    out = {}
    for test in names:
        mv = mvs[test]
        train = [mvs[m] for m in names if m != test]
        (prm, _), s = O.choose(train, "det")
        gr = [g for g in mv.rim["grains"] if mv.rim["grains"][g]["kind"] == "graded"]
        out[test] = {
            "level": {g: mv.det(g, LEVEL)[0] for g in gr},
            "relative": {g: mv.frame(rel_onset(mv.S[g][0], floor=0.2, **REL)) for g in gr},
            "level_lomo": {g: O.combine(mv, g, prm, "det") for g in gr},
            "_lomo_prm": prm,
            "_debris": {r: {g: (mv.det(g, LEVEL)[0] if r == "level" else
                                mv.frame(rel_onset(mv.S[g][0], floor=0.2, **REL)) if r == "relative" else
                                O.combine(mv, g, prm, "det")) is not None for g in mv.debris}
                        for r in ("level", "relative", "level_lomo")},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True, help="movie=tag")
    ap.add_argument("--src", default="drift")
    a = ap.parse_args()
    tags = dict(t.split("=") for t in a.tags)
    names = list(tags)
    rr = rim_rules(names) if len(names) == 3 else {}
    table = {}
    for m in names:
        ev = json.loads((OUT / f"eval_{m}_{tags[m]}.json").read_text())
        S = ev["scores"][a.src]
        L = labels(m)
        rows = {r["grain"]: r for r in S["rows"]}
        em = list(rows)
        fpb = ev["fpb"]
        h = {k: np.array([int(rows[g][k] is not None and abs(rows[g][k] * fpb) <= TOL) for g in em])
             for k in ("net", "088", "later")}
        ent = {"n": len(em), "net": int(h["net"].sum())}
        for k in ("088", "later"):
            d = h["net"] - h[k]
            ent[k] = int(h[k].sum())
            ent[f"d_{k}"] = int(d.sum())
            ent[f"ci_{k}"] = boot_ci(d)
        for rule, fr in (rr.get(m) or {}).items():
            if rule.startswith("_"):
                continue
            hr = np.array([int((e := onset_error(L["labels"][g], fr.get(g))) is not None and abs(e) <= TOL)
                           for g in em])
            d = h["net"] - hr
            ent[rule] = int(hr.sum())
            ent[f"d_{rule}"] = int(d.sum())
            ent[f"ci_{rule}"] = boot_ci(d)
        if m in rr:
            ent["rim_debris_fired"] = {r: int(sum(v.values())) for r, v in rr[m]["_debris"].items()}
            ent["rim_lomo_prm"] = rr[m]["_lomo_prm"]
        ent["auroc_pre5_vs_young"] = S["auroc_pre5_vs_young"]
        ent["auroc_12before_vs_6after"] = S["auroc_12before_vs_6after"]
        for k in ("net", "088", "later"):
            pop = S[f"scorer_{k}"]["population"] or {}
            ent[f"t50_{k}"] = pop.get("t50_model")
            ent[f"gap_{k}"] = pop.get("max_gap")
            ent[f"germ_{k}"] = pop.get("germinated_model")
            ent[f"scorer_{k}"] = f"{S[f'scorer_{k}']['hits']}/{S[f'scorer_{k}']['n_timed']}"
        ent["t50_human"] = (S["scorer_net"]["population"] or {}).get("t50_human")
        ent["early"], ent["late"], ent["never_or_start"] = S["early"], S["late"], S["never_or_start"]
        ent["early_088"], ent["late_088"] = S["early_088"], S["late_088"]
        ent["never_called"], ent["start_called"], ent["debris_onset"] = (S["never_called"], S["start_called"],
                                                                         S["debris_onset"])
        table[m] = ent
        line = (f"{m} ({tags[m]}, {a.src}): net {ent['net']}/{ent['n']}; vs 0.8.8 {ent['088']} ({ent['d_088']:+d}, "
                f"{ent['ci_088'][0]:+.0f}..{ent['ci_088'][1]:+.0f}); vs later {ent['later']} ({ent['d_later']:+d}, "
                f"{ent['ci_later'][0]:+.0f}..{ent['ci_later'][1]:+.0f})")
        for rule in ("level", "relative", "level_lomo"):
            if rule in ent:
                line += (f"; vs rim {rule} {ent[rule]} ({ent[f'd_{rule}']:+d}, {ent[f'ci_{rule}'][0]:+.0f}.."
                         f"{ent[f'ci_{rule}'][1]:+.0f})")
        line += (f"; AUROC {ent['auroc_pre5_vs_young']:.3f} / {ent['auroc_12before_vs_6after']:.3f}; T50 human "
                 f"{ent['t50_human']} net {ent['t50_net']} 0.8.8 {ent['t50_088']} later {ent['t50_later']}; gap "
                 f"{ent['gap_net']} / {ent['gap_088']} / {ent['gap_later']}; early {ent['early']} ({ent['early_088']}) "
                 f"late {ent['late']} ({ent['late_088']}) none {ent['never_or_start']}; never {ent['never_called']}; "
                 f"start {ent['start_called']}; debris {ent['debris_onset']}"
                 + (f"; rim debris fired {ent['rim_debris_fired']}" if "rim_debris_fired" in ent else ""))
        print(line)
    (OUT / f"compare_{'_'.join(f'{m}-{t}' for m, t in tags.items())}_{a.src}.json").write_text(
        json.dumps(table, default=str, indent=1))


if __name__ == "__main__":
    main()
