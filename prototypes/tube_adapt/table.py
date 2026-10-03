"""Tables of the adaptation experiment: per fold and pooled over folds (each grain judged once, by the network that did
not see its traces), against the shipped network on the same grains, paired over grains with 95% bootstrap intervals.

    python -m prototypes.tube_adapt.table k2f0 k2f1 --variants a b c [--markdown]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from prototypes.synth_v6.recall import summarise
from prototypes.tube_adapt.common import BASE_DUMP, OUT
from prototypes.tube_adapt.crops import fold_grains
from prototypes.tube_adapt.evaluate import paired
from prototypes.tube_net.pixels import extra
from prototypes.tube_net.pixpair import compare


def load(name):
    f = OUT / "eval" / name
    return json.loads(f.read_text()) if f.exists() else None


def pix_cells(rows: list[dict], rim: tuple[int, int] | None, te: list[dict]) -> dict:
    s = summarise(rows) | extra(rows)
    a, y, lg = s["all"], s["young<=8"], s["long>=50"]
    ext = np.array([r["extent"] for r in te]) if te else np.zeros(0)
    return {"traced": 100 * a["point_recall"], "long": 100 * lg["point_recall"], "stubs": a["stub_seen"],
            "tips": a["tip_seen"], "n": a["traces"], "young": y["stub_seen"], "young_tips": y["tip_seen"],
            "young_n": y["traces"], "beside": 100 * a["beside"],
            "rim": (100 * rim[0] / max(rim[1], 1)) if rim else float("nan"), "rim_counts": rim,
            "end_med": float(np.median(ext)) if len(ext) else float("nan"),
            "within2": int(np.sum(np.abs(ext) <= 2)) if len(ext) else 0, "te_n": len(ext)}


def fmt_pix(c: dict) -> str:
    return (f"{c['traced']:.0f}% | {c['long']:.0f}% | {c['stubs']}/{c['n']}, {c['tips']} | {c['young']}/{c['young_n']}, "
            f"{c['young_tips']} | {c['beside']:.1f}% | {c['rim']:.1f}% ({c['rim_counts'][0]}/{c['rim_counts'][1]}) | "
            f"{c['end_med']:+.1f}, {c['within2']}/{c['te_n']}")


def fmt_pp(c: dict) -> str:
    t, s, ti, y = c["traced"], c["stubs"], c["tips"], c["young_stubs"]
    return (f"{t[0]:+.1f} pts ({t[1]:+.1f} to {t[2]:+.1f}); stubs {s[0]:+.0f} ({s[1]:+.0f} to {s[2]:+.0f}); tips "
            f"{ti[0]:+.0f} ({ti[1]:+.0f} to {ti[2]:+.0f}); young stubs {y[0]:+.0f} ({y[1]:+.0f} to {y[2]:+.0f})")


def fmt_e2e(base: dict, new: dict, grains: list[str]) -> str:
    p = paired(base, new, grains)
    on, ln, bo = p["onset_hit"], p["len_hit"], p["both_hit"]
    n_on = sum(1 for g in grains if g in new and new[g]["onset_hit"] is not None)
    return (f"lengths {ln[4]}/{p['len_n']} (base {ln[3]}; {ln[0]:+d}, {ln[1]:+.0f} to {ln[2]:+.0f}) | length and tip "
            f"{bo[4]} (base {bo[3]}; {bo[0]:+d}, {bo[1]:+.0f} to {bo[2]:+.0f}) | onsets {on[4]}/{n_on} (base {on[3]}; "
            f"{on[0]:+d}, {on[1]:+.0f} to {on[2]:+.0f})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folds", nargs="+")
    ap.add_argument("--variants", nargs="+", default=["a", "b", "c"])
    a = ap.parse_args(argv)
    base_e2e = json.loads(BASE_DUMP.read_text())["m1"]["grains"]
    pooled = {v: {"rows": [], "te": [], "rim": [0, 0], "e2e": {}} for v in ["base"] + a.variants}
    for fold in a.folds:
        _, held = fold_grains(fold)
        print(f"\n== {fold}: held out {len(held)} grains ({' '.join(held)})")
        bp = load(f"pix_{fold}_base_m1.json")
        for v in ["base"] + a.variants:
            px = bp if v == "base" else load(f"pix_{fold}_{v}_m1.json")
            if px is None:
                print(f"  {v}: no pixel check yet")
                continue
            rim = px["summary"]["rim_before_onset"]
            rc = (rim["marked"], rim["grain_bins"])
            cells = pix_cells(px["rows"], rc, px["tip_extent"])
            line = f"  pix {v:5s}: {fmt_pix(cells)}"
            if v != "base" and bp is not None:
                line += f"\n             vs base: {fmt_pp(compare(bp['rows'], px['rows']))}"
            print(line)
            P = pooled[v]
            P["rows"] += px["rows"]; P["te"] += px["tip_extent"]; P["rim"][0] += rc[0]; P["rim"][1] += rc[1]
        for v in ["base"] + a.variants:
            if v == "base":
                e = load(f"e2e_{fold}_base.json")
                if e is not None:
                    same = all(e["grains"][g][k] == base_e2e[g][k] for g in held for k in ("onset_hit", "len_hit", "both_hit"))
                    print(f"  e2e base (harness) reproduces the 0.8.8 baseline on these grains: {same}")
                for g in held:
                    pooled["base"]["e2e"][g] = base_e2e[g]
                continue
            e = load(f"e2e_{fold}_{v}.json")
            if e is None:
                print(f"  e2e {v}: not yet")
                continue
            print(f"  e2e {v:5s}: {fmt_e2e(base_e2e, e['grains'], held)}")
            pooled[v]["e2e"].update(e["grains"])
    print("\n== pooled over folds")
    B = pooled["base"]
    print(f"  pix base : {fmt_pix(pix_cells(B['rows'], tuple(B['rim']), B['te']))}")
    for v in a.variants:
        P = pooled[v]
        if P["rows"]:
            print(f"  pix {v:5s}: {fmt_pix(pix_cells(P['rows'], tuple(P['rim']), P['te']))}\n             vs base: "
                  f"{fmt_pp(compare(B['rows'], P['rows']))}")
        if P["e2e"]:
            g = sorted(P["e2e"])
            print(f"  e2e {v:5s} ({len(g)} grains): {fmt_e2e(base_e2e, P['e2e'], g)}")
    print("\n== movie 2 (forgetting; the shipped network saw m2's traces)")
    b2 = load("pix_base_m2.json")
    if b2:
        r = b2["summary"]["rim_before_onset"]
        print(f"  base : {fmt_pix(pix_cells(b2['rows'], (r['marked'], r['grain_bins']), b2['tip_extent']))}")
    for fold in a.folds:
        for v in a.variants:
            m2 = load(f"pix_{fold}_{v}_m2.json")
            if m2 and b2:
                r = m2["summary"]["rim_before_onset"]
                print(f"  {fold}_{v}: {fmt_pix(pix_cells(m2['rows'], (r['marked'], r['grain_bins']), m2['tip_extent']))}"
                      f"\n        vs base: {fmt_pp(compare(b2['rows'], m2['rows']))}")


if __name__ == "__main__":
    main()


def detail(fold: str, variant: str) -> None:
    """Per held-out grain: the 0.8.8 baseline's and the variant's FULL-trace errors (model - human, px) and onsets."""
    base_pred = json.loads(BASE_DUMP.read_text())["m1"]
    e = load(f"e2e_{fold}_{variant}.json")
    _, held = fold_grains(fold)
    from sparsetrack.evaluate import load as load_pred, score
    from prototypes.tube_adapt.common import labels, subset
    rep = score(subset(labels("m1"), held), load_pred(base_pred["pred"]))
    brows = {r["grain"]: r for r in rep["rows"]}
    for g in held:
        b, n = brows.get(g, {}), e["grains"].get(g, {})
        bf = {f["frame"]: f for f in b.get("full", [])}
        nf = {f["frame"]: f for f in n.get("full", [])}
        cells = []
        for fr in sorted(set(bf) | set(nf)):
            x, y = bf.get(fr), nf.get(fr)
            h = (x or y)["human"]
            fe = lambda f: f"{f['error']:+.1f}" if f else "-"
            cells.append(f"b{fr // 300} {h:.0f}px {fe(x)} -> {fe(y)}")
        print(f"  {g}: onset err {b.get('onset_error')} -> {n.get('onset_error')} | " + "; ".join(cells))
