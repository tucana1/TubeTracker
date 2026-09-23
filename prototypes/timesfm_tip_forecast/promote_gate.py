"""Probe-gated checkpoint promotion (H184/H185): val loss is blind to faint
recall, so no checkpoint promotes on val alone.  Scores a candidate
--eval-csv (eval_tip_cnn.py output on /tmp/rerun_probes.csv) against the
frozen v1 baseline: any CONFIRMED->miss flip (dist_nearest crossing 20
up) on a clean/faint probe, or any silent->fire flip (heat_at_truth
crossing 0.10 up) on a ghost/background probe, FAILS promotion.
Corner-vs-apex heat is reported, not gated.  Exit 0 PASS, 1 FAIL.
"""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd

CONFIRM_PX = 20.0
FIRE_HEAT = 0.10
MUST_CONFIRM = ["P58-clean", "P3-faint", "P13-clean", "P22-clean", "P11-faint",
                "P8-apex"]
MUST_SILENT = ["P22-ghost", "P55-background"]
# H232: pre-emergence silence — no tube exists, so no peak may sit
# within confirm range of the tip (v4valid fired 15-18px grain-rim
# snaps here that v1 correctly refuses at 187-218px).
PRE_SILENCE = ["P22-pre47", "P22-pre58"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--probes-csv", default=None,
                    help="versioned probe manifest; every tag listed must "
                    "appear in the candidate CSV with finite metrics")
    args = ap.parse_args()

    cand = pd.read_csv(args.eval_csv).set_index("tag")
    base = pd.read_csv(args.baseline_csv).set_index("tag")
    fails: list[str] = []
    # P0 hardening: the full expected tag set must be present with
    # finite metrics. Missing rows and NaN comparisons used to escape
    # as unrecorded non-failures; now they FAIL loudly.
    if args.probes_csv is not None:
        import csv
        with open(args.probes_csv) as f:
            manifest_tags = [r["tag"] for r in csv.DictReader(f)
                             if r.get("tag")]
        for tag in manifest_tags:
            if tag not in cand.index:
                fails.append(f"{tag}: absent from candidate eval")
                print(f"{tag}: ABSENT from candidate eval [FAIL]")
    for tag in list(cand.index):
        for col in ("dist_nearest", "heat_at_truth"):
            if col in cand.columns:
                try:
                    v = float(cand.loc[tag, col])
                except (TypeError, ValueError):
                    v = float("nan")
                if not math.isfinite(v):
                    fails.append(f"{tag}: non-finite {col} (no-detection "
                                 "must be recorded, never NaN)")
                    print(f"{tag}: non-finite {col} [FAIL]")

    def _finite(value) -> float:
        v = float(value)
        return v if math.isfinite(v) else float("inf")

    for tag in MUST_CONFIRM:
        if tag not in base.index:
            d1 = float(cand.loc[tag, "dist_nearest"]) if tag in cand.index else float("nan")
            ok = tag in cand.index and d1 <= CONFIRM_PX
            print(f"{tag}: new coverage, cand {d1:.1f} [{'ok' if ok else 'FAIL'}]")
            if not ok:
                fails.append(f"{tag}: new-coverage MISS {d1:.1f}")
            continue
        d0 = float(base.loc[tag, "dist_nearest"])
        if tag not in cand.index:
            print(f"{tag}: base {d0:.1f} -> cand MISSING [FAIL]")
            fails.append(f"{tag}: missing from candidate")
            continue
        d1 = _finite(cand.loc[tag, "dist_nearest"])
        status = "ok" if d1 <= CONFIRM_PX else "MISS"
        if d0 <= CONFIRM_PX and d1 > CONFIRM_PX:
            fails.append(f"{tag}: CONFIRMED->{d1:.1f}")
            status += " FLIP"
        print(f"{tag}: base {d0:.1f} -> cand {d1:.1f} [{status}]")
    for tag in MUST_SILENT:
        if tag not in cand.index or tag not in base.index:
            fails.append(f"{tag}: missing from candidate or baseline")
            print(f"{tag}: MISSING [FAIL]")
            continue
        h0, h1 = _finite(base.loc[tag, "heat_at_truth"]), _finite(cand.loc[tag, "heat_at_truth"])
        status = "ok" if h1 < FIRE_HEAT else "FIRING"
        if h0 < FIRE_HEAT and h1 >= FIRE_HEAT:
            fails.append(f"{tag}: silent->{h1:.3f}")
            status += " FLIP"
        print(f"{tag}: base {h0:.3f} -> cand {h1:.3f} [{status}]")
    for tag in PRE_SILENCE:
        if tag not in cand.index:
            if args.probes_csv is not None:
                continue  # already recorded ABSENT above; don't double-count
            fails.append(f"{tag}: missing pre-emergence probe")
            print(f"{tag}: MISSING [FAIL]")
            continue
        d1 = _finite(cand.loc[tag, "dist_nearest"])
        status = "ok" if d1 > CONFIRM_PX else "SNAP"
        if d1 <= CONFIRM_PX:
            fails.append(f"{tag}: pre-emergence snap->{d1:.1f}")
            status += " FLIP"
        print(f"{tag}: cand {d1:.1f} [{status}]")
    if fails:
        print("PROMOTION: FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PROMOTION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
