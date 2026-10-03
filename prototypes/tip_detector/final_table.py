"""Markdown tables for the README from the three held-out evaluations.

    python -m prototypes.tip_detector.final_table TAG_m2 TAG_ld TAG_m1     # eval_<movie>_<tag>.json, held-out movie order
"""
from __future__ import annotations

import io
import json
import sys

import numpy as np

from .common import OUT
from .summary import table

SHOW = ["det", "absD_0", "absD_1", "absD_2", "absD_0_out", "absC_2", "tubenet"]


def main(tags):
    print("| held out | method | top-1 <= 4 px | young | mid | long | top-1 <= max(4, 10% L) | top-3 <= 4 px | "
          "apex rank (median) | route-given window | onset AUROC | false alarms at 80% young |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for movie, tag in zip(("m2", "ld", "m1"), tags):
        res = json.loads((OUT / f"eval_{movie}_{tag}.json").read_text())
        rows = table(res, [m for m in SHOW if m in res["traces"][0]["methods"]], file=io.StringIO())
        tr = res["traces"]
        cls = np.array([t["cls"] for t in tr])
        n = {c: int((cls == c).sum()) for c in ("young", "mid", "long")}
        for m, r in rows.items():
            if "top1" not in r:
                continue
            bc = r["by_cls"]
            rk = [t["rank"].get(m) for t in tr if m in t.get("rank", {})]
            print(f"| {movie} ({len(tr)}) | {m} | {r['top1'] * len(tr):.0f} ({100 * r['top1']:.0f}%) | "
                  f"{bc['young'][0]}/{n['young']} | {bc['mid'][0]}/{n['mid']} | {bc['long'][0]}/{n['long']} | "
                  f"{r['top1_tol'] * len(tr):.0f} | {r['top3'] * len(tr):.0f} | "
                  f"{np.median(rk) if rk else float('nan'):.0f} | "
                  f"{r.get('local', float('nan')) * len(tr):.0f} | "
                  f"{r['auroc']:.2f} | {r['fa80']:.2f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
