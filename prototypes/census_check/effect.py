"""What leaving the census check's flagged grains out does to each movie's germination share and T50.

    python -m prototypes.census_check.effect [--variant SMALL/global] [--json runs/census_check/effect.json]

Flags: each movie's leave-one-movie-out flags from loo.py (the classifier fitted on the other two movies); model
readings: germ.py predict (SparseTrack 0.8.0 defaults, the movie's whole census read as grains). Per movie, over its
judged census-isolated grains away from the edge: the annotator's germinated share and T50 over the grains they kept,
the model's over all of them (now), without the flagged ones, and over the annotator's kept grains (a perfect census
check). The change in the absolute error against the annotator (flagged out minus all) gets a 95% bootstrap interval
over grains (grains resampled with their human label, if kept, and the model's reading).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from prototypes.census_check.germ import OUT, curve, human_interval, judged_population

REPO = Path(__file__).resolve().parents[2]


def stats(movie: str, flags: set[str], n_boot: int = 2000, seed: int = 0) -> dict:
    from sparsetrack.report import onset_intervals
    ids, kept, L = judged_population(movie)
    fpb, nb = int(L["frames_per_bin"]), int(L["n_bins"])
    last = (nb - 1) * fpb + fpb // 2
    pred = {g["id"]: g for g in json.loads((OUT / f"pred_{movie}" / "predictions.json").read_text())["grains"]}
    hum = {g: human_interval((L["labels"].get(g) or {}).get("onset") or {}, fpb, nb) for g in kept}
    mod = {g: (onset_intervals({"grains": [pred[g]]}) or [None])[0] for g in ids if g in pred}

    def pops(sample):
        h = [hum[g] for g in sample if g in kept and hum[g] is not None]
        m_all = [mod[g] for g in sample if mod.get(g) is not None]
        m_fl = [mod[g] for g in sample if mod.get(g) is not None and g not in flags]
        m_kept = [mod[g] for g in sample if g in kept and mod.get(g) is not None]
        return curve(h, last), curve(m_all, last), curve(m_fl, last), curve(m_kept, last)

    a, m, f, k = pops(ids)

    def err(x, y, key):
        if x[key] is None or y[key] is None:
            return np.nan
        return abs(x[key] - y[key]) / (fpb if key == "t50" else 0.01)  # T50 in bins, share in percentage points

    out = {"n": len(ids), "flagged": sorted(flags & set(ids)), "flagged_excluded": sorted(flags & (set(ids) - kept)),
           "annotator": a, "model_all": m, "model_flagged_out": f, "model_kept": k}
    rng = np.random.default_rng(seed)
    for key in ("share", "t50"):
        d0 = err(f, a, key) - err(m, a, key)
        ds = []
        for _ in range(n_boot):
            s = list(rng.choice(ids, len(ids)))
            a_, m_, f_, _ = pops(s)
            ds.append(err(f_, a_, key) - err(m_, a_, key))
        lo, hi = np.nanpercentile(ds, [2.5, 97.5])
        out[f"d_abs_err_{key}"] = [float(d0), float(lo), float(hi)]
        out[f"abs_err_{key}_all"], out[f"abs_err_{key}_flagged_out"] = err(m, a, key), err(f, a, key)
        out[f"abs_err_{key}_kept"] = err(k, a, key)
    return out


def fmt(c: dict, fpb: int) -> str:
    t = "-" if c["t50"] is None else f"{c['t50'] / fpb:.1f}"
    return f"{100 * c['share']:.0f}%, T50 {t}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="*", default=None)
    ap.add_argument("--json", default=str(REPO / "runs/census_check/effect.json"))
    a = ap.parse_args(argv)
    loo = json.loads((REPO / "runs/census_check/loo.json").read_text())
    res = {}
    for v in (a.variant or list(loo)):
        res[v] = {}
        for mv, r in loo[v].items():
            if not (OUT / f"pred_{mv}" / "predictions.json").exists():
                continue
            s = stats(mv, set(r["flagged_ids"]))
            res[v][mv] = s
            fpb = 300
            print(f"{v:13s} {mv}: annotator {fmt(s['annotator'], fpb)} | model all {fmt(s['model_all'], fpb)} | "
                  f"flagged out ({len(s['flagged'])}: {' '.join(s['flagged'])}) {fmt(s['model_flagged_out'], fpb)} | "
                  f"annotator's grains {fmt(s['model_kept'], fpb)} || |error| change: share "
                  f"{s['d_abs_err_share'][0]:+.1f} pts ({s['d_abs_err_share'][1]:+.1f} to {s['d_abs_err_share'][2]:+.1f}), "
                  f"T50 {s['d_abs_err_t50'][0]:+.2f} bins ({s['d_abs_err_t50'][1]:+.2f} to {s['d_abs_err_t50'][2]:+.2f})")
    Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
