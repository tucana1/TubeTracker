"""Score several prediction files on one labels file, side by side, with paired comparisons against one of them.

    python scripts/compare_predictions.py --labels benchmark/labels/m1_v1.json \
        --pred 0.6.0=runs/sparsetrack/m1_frozen_0.6.0/predictions.json \
        --pred 0.7.0=runs/sparsetrack/m1_frozen_0.7.0/predictions.json --baseline 0.6.0 \
        --out benchmark/reports/m1_v1_frozen.md

Per prediction: onsets within 2 bins, FULL-trace lengths within max(2 px, 10%), length and tip, the germination
curves' T50 and growth-rate agreement (``sparsetrack.evaluate``). Against the baseline: the change in onset, length
and length-and-tip hits with a 95% bootstrap interval over grains (the unit that was sampled).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sparsetrack.evaluate import load, score  # noqa: E402


def per_grain(rep: dict, len_abs: float = 2.0, len_rel: float = 0.10, tip_abs: float = 5.0,
              tip_rel: float = 0.10) -> dict[str, tuple[int, int, int]]:
    """grain -> (onset hit, length hits, length-and-tip hits)."""
    out = {}
    for r in rep["rows"]:
        on = int(abs(r["onset_error"]) <= rep["onset"]["tolerance_frames"]) if "onset_error" in r else 0
        full = r.get("full", [])
        ln = sum(abs(f["error"]) <= max(len_abs, len_rel * f["human"]) for f in full)
        both = sum(abs(f["error"]) <= max(len_abs, len_rel * f["human"])
                   and f.get("tip_error", 1e9) <= max(tip_abs, tip_rel * f["human"]) for f in full)
        out[r["grain"]] = (on, ln, both)
    return out


def paired(base: dict, new: dict, n_boot: int = 4000, seed: int = 0) -> list[str]:
    g = sorted(set(base) & set(new))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    out = []
    for k, name in enumerate(("onsets", "lengths", "length and tip")):
        d = np.array([new[x][k] - base[x][k] for x in g])
        lo, hi = np.percentile(d[idx].sum(axis=1), [2.5, 97.5])
        out.append(f"{name} {d.sum():+d} (95% CI {lo:+.0f} to {hi:+.0f})")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pred", action="append", required=True, help="NAME=predictions.json (repeat)")
    ap.add_argument("--baseline", help="NAME of the prediction to compare the others with (default: the first)")
    ap.add_argument("--out", help="markdown report (default: print only)")
    a = ap.parse_args(argv)
    labels = load(a.labels)
    preds = dict(p.split("=", 1) for p in a.pred)
    base = a.baseline or next(iter(preds))
    reps = {n: score(labels, json.loads(Path(f).read_text())) for n, f in preds.items()}
    grains = {n: per_grain(r) for n, r in reps.items()}
    lines = [f"# {Path(a.labels).name}: {', '.join(preds)} (paired against {base})", "",
             "| predictions | onsets | lengths | length and tip | T50 (model vs annotator, frames) | growth rates within tolerance |",
             "|---|---|---|---|---|---|"]
    for n, r in reps.items():
        o, L, P, G = r["onset"], r["length_full"], r.get("population") or {}, r.get("growth") or {}
        t50 = (f"{P['t50_model']:.0f} vs {P['t50_human']:.0f}" if P.get("t50_model") is not None
               and P.get("t50_human") is not None else "-")
        gr = (f"{G['rate_within']}/{G['grains_with_rate']} (r {G['rate_pearson']:.2f})"
              if G.get("grains_with_rate") and G.get("rate_pearson") is not None else "-")
        lines.append(f"| {n} | {o['hits']}/{o['n_timed']} | {L['within_tolerance']}/{L['n']} | "
                     f"{r['tips']['length_and_tip']} | {t50} | {gr} |")
    lines.append("")
    for n in reps:
        if n != base:
            lines.append(f"- {n} vs {base}: " + "; ".join(paired(grains[base], grains[n])) +
                         f" over {len(set(grains[base]) & set(grains[n]))} grains")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
