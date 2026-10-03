"""Tables from evaluate.py's dumps.

    python -m prototypes.tip_detector.summary runs/research/tip_detector/eval_m1_ldm2.json [...]
"""
from __future__ import annotations

import json
import sys

import numpy as np

CLASSES = ("young", "mid", "long")


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def table(res: dict, methods=None, show_out=True, file=sys.stdout):
    tr = res["traces"]
    allm = list(tr[0]["methods"]) if tr else []
    methods = methods or [m for m in allm if show_out or not m.endswith("_out")]
    n_cls = {c: sum(t["cls"] == c for t in tr) for c in CLASSES}
    print(f"\n{res['movie']}: {len(tr)} FULL traces (young {n_cls['young']}, mid {n_cls['mid']}, long {n_cls['long']}); "
          f"net {res['net']}", file=file)
    print(f"{'method':12s} | top1<=4px  young  mid  long | top1<=max(4,10%L) | top3<=4px  young  mid  long | "
          f"err med (IQR) | onset AUROC  FA@80%young  rim-share", file=file)
    rows = {}
    for m in methods:
        d1 = np.array([t["methods"][m][0][3] if t["methods"][m] else np.inf for t in tr])
        tol = np.array([max(4.0, 0.1 * t["L"]) for t in tr])
        d3 = np.array([min([p[3] for p in t["methods"][m][:3]] or [np.inf]) for t in tr])
        cls = np.array([t["cls"] for t in tr])
        h1, h1t, h3 = d1 <= 4, d1 <= tol, d3 <= 4
        f = lambda h, c: f"{int(h[cls == c].sum()):2d}/{int((cls == c).sum()):<2d}"
        q = np.percentile(d1[np.isfinite(d1)], [25, 50, 75]) if np.isfinite(d1).any() else [np.nan] * 3
        # onset: best value in the disc at young FULL traces vs before the onset
        base = m
        pre = [p["methods"][base][0] for p in res["pre_onset"] if base in p["methods"]]
        rim = [abs(p["methods"][base][1]) <= 5 for p in res["pre_onset"] if base in p["methods"]]
        yv = [t["methods"][m][0][2] for t in tr if t["cls"] == "young" and t["methods"][m]]
        au = auroc(yv, pre) if pre else float("nan")
        thr = np.percentile(yv, 20) if yv else np.nan
        fa = float(np.mean(np.asarray(pre) > thr)) if pre else float("nan")
        rows[m] = dict(top1=float(h1.mean()), top1_tol=float(h1t.mean()), top3=float(h3.mean()),
                       by_cls={c: [int(h1[cls == c].sum()), int(h3[cls == c].sum()), int((cls == c).sum())]
                               for c in CLASSES},
                       err_q=[float(v) for v in q], auroc=au, fa80=fa,
                       rim_share=float(np.mean(rim)) if rim else float("nan"))
        print(f"{m:12s} | {int(h1.sum()):3d}/{len(tr):<3d} {f(h1, 'young')} {f(h1, 'mid')} {f(h1, 'long')} | "
              f"{int(h1t.sum()):3d}/{len(tr):<3d}         | {int(h3.sum()):3d}/{len(tr):<3d} {f(h3, 'young')} "
              f"{f(h3, 'mid')} {f(h3, 'long')} | {q[1]:5.1f} ({q[0]:.1f}-{q[2]:.1f}) | "
              f"{au:5.2f}       {fa:5.2f}        {rows[m]['rim_share']:.2f}" if pre else
              f"{m:12s} | {int(h1.sum()):3d}/{len(tr):<3d} {f(h1, 'young')} {f(h1, 'mid')} {f(h1, 'long')} | "
              f"{int(h1t.sum()):3d}/{len(tr):<3d}         | {int(h3.sum()):3d}/{len(tr):<3d} {f(h3, 'young')} "
              f"{f(h3, 'mid')} {f(h3, 'long')} | {q[1]:5.1f} ({q[0]:.1f}-{q[2]:.1f}) |", file=file)
    if tr and "rank" in tr[0]:
        print(f"{'method':12s} | apex rank: median  top1  <=3  <=10  (young / mid / long medians) | "
              f"route-given window: top1<=4px  young  mid  long", file=file)
        for m in methods:
            if m not in tr[0]["rank"] and m.replace("_out", "") not in tr[0]["local"]:
                continue
            cls = np.array([t["cls"] for t in tr])
            rk = np.array([t["rank"].get(m, np.nan) for t in tr], float)
            loc = np.array([t["local"].get(m, np.nan) for t in tr], float) if m in tr[0]["local"] else None
            med = lambda c: np.nanmedian(rk[cls == c]) if (cls == c).any() else np.nan
            s1 = (f"{m:12s} | {np.nanmedian(rk):5.0f}  {int((rk <= 1).sum()):3d}  {int((rk <= 3).sum()):3d}  "
                  f"{int((rk <= 10).sum()):3d}   ({med('young'):.0f} / {med('mid'):.0f} / {med('long'):.0f})"
                  if np.isfinite(rk).any() else f"{m:12s} |  -")
            if loc is not None:
                h = loc <= 4
                s1 += (f" | {int(h.sum()):3d}/{len(tr):<3d} " +
                       " ".join(f"{int(h[cls == c].sum()):2d}/{int((cls == c).sum()):<2d}" for c in CLASSES))
                rows.setdefault(m, {})["local"] = float(h.mean())
            print(s1, file=file)
    return rows


if __name__ == "__main__":
    for p in sys.argv[1:]:
        table(json.loads(open(p).read()))
