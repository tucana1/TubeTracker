"""End-to-end hits on YOUNG traces (human length <= 8 px), model vs baseline, paired over grains.

    python -m prototypes.synth_v6.young_e2e --labels benchmark/labels/m2_v1.json \
        --base BASE_predictions.json --new NEW_predictions.json

Rescores both prediction files with ``sparsetrack.evaluate.score`` (the scorer ``synth_bench --real`` uses) and
counts, for FULL traces no longer than ``--max-len``: length within max(2 px, 10%) and length-and-tip (tip within
max(5 px, 10%) of the human apex). The 95% interval is a bootstrap over grains of the summed per-grain change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sparsetrack.evaluate import load, score


def per_grain(labels: str, pred: str, max_len: float = 8.0) -> dict[str, list[tuple]]:
    rep = score(load(labels), json.loads(Path(pred).read_text()))
    out = {}
    for r in rep["rows"]:
        for f in r.get("full", []):
            h = f["human"]
            if h > max_len:
                continue
            hit = abs(f["error"]) <= max(2.0, 0.1 * h)
            both = hit and f.get("tip_error", 1e9) <= max(5.0, 0.1 * h)
            out.setdefault(r["grain"], []).append((f["frame"], hit, both, f["error"]))
    return out


def compare(labels: str, base: str, new: str, max_len: float = 8.0, n_boot: int = 4000, seed: int = 0) -> dict:
    b, n = per_grain(labels, base, max_len), per_grain(labels, new, max_len)
    gids = sorted(set(b) | set(n))
    tot = lambda d, i: sum(int(x[i]) for v in d.values() for x in v)
    res = {"traces_base": sum(len(v) for v in b.values()), "traces_new": sum(len(v) for v in n.values()),
           "len_base": tot(b, 1), "len_new": tot(n, 1), "both_base": tot(b, 2), "both_new": tot(n, 2), "grains": len(gids)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gids), (n_boot, len(gids)))
    for key, i in (("len", 1), ("both", 2)):
        d = np.array([sum(int(x[i]) for x in n.get(g, [])) - sum(int(x[i]) for x in b.get(g, [])) for g in gids])
        lo, hi = np.percentile(d[idx].sum(axis=1), [2.5, 97.5])
        res[f"{key}_diff"], res[f"{key}_ci"] = int(d.sum()), (float(lo), float(hi))
    res["changed"] = {g: {"base": [(fr, int(h), round(e, 1)) for fr, h, _, e in b.get(g, [])],
                          "new": [(fr, int(h), round(e, 1)) for fr, h, _, e in n.get(g, [])]}
                      for g in gids if [x[1] for x in b.get(g, [])] != [x[1] for x in n.get(g, [])]}
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--max-len", type=float, default=8.0)
    a = ap.parse_args()
    r = compare(a.labels, a.base, a.new, a.max_len)
    print(f"young traces (<= {a.max_len:g} px): {r['traces_new']} | lengths {r['len_base']} -> {r['len_new']} "
          f"({r['len_diff']:+d}, 95% CI {r['len_ci'][0]:+.0f} to {r['len_ci'][1]:+.0f}) | length and tip "
          f"{r['both_base']} -> {r['both_new']} ({r['both_diff']:+d}, 95% CI {r['both_ci'][0]:+.0f} to "
          f"{r['both_ci'][1]:+.0f}) over {r['grains']} grains")
    for g, v in r["changed"].items():
        print(f"  {g}: base {v['base']}  new {v['new']}")
