"""Tables from evaluate3.py's dumps: top-1 / 3 / 8 within 4 px of the apex by length class, the reader's candidates,
the onset AUROC.

    python -m prototypes.tip_detector.table3 eval3_m2_<tag>.json eval3_ld_<tag>.json eval3_m1_<tag>.json [--md]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .common import OUT
from .summary import auroc

CLASSES = ("young", "mid", "long")


def rows(res: dict):
    tr = res["traces"]
    cls = np.array([t["cls"] for t in tr])
    out = {}
    for m in tr[0]["methods"]:
        d = [np.array([p[3] for p in t["methods"][m]] + [np.inf] * 10)[:10] for t in tr]
        d = np.stack(d)
        hit = {k: (d[:, :k] <= 4).any(1) for k in (1, 3, 8)}
        rd = np.array([min([p[3] for p in t["reader"][m]] or [np.inf]) for t in tr])
        hit["reader"] = rd <= 4
        av = np.array([t["apex_val"][m] for t in tr])
        hit["av10"], hit["av20"] = av >= 0.1, av >= 0.2  # the map reaches this within 4 px of the apex (any rank)
        if "reader_all" in tr[0]:  # the apex among the top K peaks of the reader's whole region
            ra = np.stack([np.array(t["reader_all"][m] + [np.inf] * 30)[:30] for t in tr])
            for k in (6, 12, 20, 30):
                hit[f"rk{k}"] = (ra[:, :k] <= 4).any(1)
        pre = [p["methods"][m][0] for p in res["pre_onset"] if m in p["methods"]]
        yv = [t["methods"][m][0][2] for t in tr if t["cls"] == "young" and t["methods"][m]]
        thr = np.percentile(yv, 20) if yv else np.nan
        out[m] = {"n": len(tr), "hit": hit, "cls": cls, "auroc": auroc(yv, pre),
                  "fa80": float(np.mean(np.asarray(pre) > thr)) if pre else float("nan"),
                  "rank_long": float(np.median([t["rank"][m] for t in tr if t["cls"] == "long"] or [np.nan])),
                  "apexval_long": float(np.median([t["apex_val"][m] for t in tr if t["cls"] == "long"] or [np.nan]))}
    return out


def fmt(h, cls):
    return " | ".join(f"{int(h[cls == c].sum())}/{int((cls == c).sum())}" for c in CLASSES)


def main(paths, md=False):
    if md:
        print("| held out (traces: young, mid, long) | detector | top-1 within 4 px: all (young, mid, long) | top-3 | top-8 | "
              "reader's 9 candidates | top-20 of the reader's region | map >= 0.2 at the apex | onset AUROC (false "
              "alarms at 80% young) |")
        print("|---|---|---|---|---|---|---|---|---|")
    for p in paths:
        f = Path(p) if Path(p).exists() else OUT / p
        res = json.loads(f.read_text())
        for m, r in rows(res).items():
            h, cls, n = r["hit"], r["cls"], r["n"]
            if md:
                c = lambda k: f"{int(h[k].sum())} ({', '.join(str(int(h[k][cls == q].sum())) for q in CLASSES)})"
                nc = ", ".join(str(int((cls == q).sum())) for q in CLASSES)
                print(f"| {res['movie']} ({n}: {nc}) | {m} | {c(1)} | {c(3)} | {c(8)} | {c('reader')} | "
                      f"{c('rk20') if 'rk20' in h else '-'} | {c('av20')} | {r['auroc']:.2f} ({r['fa80']:.2f}) |")
            else:
                print(f"{res['movie']:3s} {m:10s} n={n:3d} top1 {int(h[1].sum()):3d} top3 {int(h[3].sum()):3d} "
                      f"top8 {int(h[8].sum()):3d} | top1 {fmt(h[1], cls)} | top3 {fmt(h[3], cls)} | "
                      f"top8 {fmt(h[8], cls)} | reader {int(h['reader'].sum()):3d} ({fmt(h['reader'], cls)}) | "
                      f"AUROC {r['auroc']:.2f} FA80 {r['fa80']:.2f} | long rank med {r['rank_long']:.0f} "
                      f"apexval med {r['apexval_long']:.2f}")
                if "rk6" in h:
                    print("      reader region top-K: " + "  ".join(
                        f"K={k}: {int(h[f'rk{k}'].sum())} ({fmt(h[f'rk{k}'], cls)})" for k in (6, 12, 20, 30)))
                print(f"      apex value >= 0.1: {int(h['av10'].sum())} ({fmt(h['av10'], cls)})  "
                      f">= 0.2: {int(h['av20'].sum())} ({fmt(h['av20'], cls)})")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--md"]
    main(a, md="--md" in sys.argv)
