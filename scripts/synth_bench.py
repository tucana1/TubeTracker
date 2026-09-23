"""Score SparseTrack parameter variants on the synthetic seeds and the legacy real labels.

    .venv/bin/python scripts/synth_bench.py                      # default variants
    .venv/bin/python scripts/synth_bench.py --seeds 0 1 2 --set onset_source=matched mf_z=3

Synthetic movies come from `sparsetrack synth` + `prepare --frames-per-bin 25`
(runs/sparsetrack/synth/s{seed}_cache with synth_s{seed}_truth.json); the onset
tolerance there is 50 synthetic frames (= 600 source frames).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sparsetrack.analyze import Params, analyze  # noqa: E402
from sparsetrack.evaluate import load, score  # noqa: E402

SYN = REPO / "runs/sparsetrack/synth"
LEGACY_IDS = ["g025", "g014", "g037", "g034", "g013", "g030", "g029"]


def parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def run(params: Params, seeds: list[int], legacy: bool = True) -> dict:
    agg = {"on_hit": 0, "on_n": 0, "on_truth": 0, "early": 0, "late": 0, "len_hit": 0, "len_n": 0,
           "abs_ok": 0, "abs_n": 0, "ctrl_fp": 0, "ctrl_n": 0, "missed": 0, "errs": []}
    for s in seeds:
        cache, truth = SYN / f"s{s}_cache", SYN / f"synth_s{s}_truth.json"
        if not cache.exists():
            continue
        pred = analyze(cache, f"/tmp/tt_bench/s{s}", params=params, log=lambda *a: None)
        rep = score(load(truth), pred, onset_tol=50)
        o, L, a = rep["onset"], rep["length_full"], rep["absences"]
        agg["on_hit"] += o["hits"]; agg["on_n"] += o["n_timed"]; agg["on_truth"] += o["n_human_emerged_within"]
        agg["early"] += o["early"]; agg["late"] += o["late"]
        agg["len_hit"] += L["within_tolerance"]; agg["len_n"] += L["n"]
        agg["abs_ok"] += a["correct"]; agg["abs_n"] += a["n"]
        conf = rep["germination_confusion"]
        agg["ctrl_fp"] += sum(v for k, v in conf.get("no_emergence_by_end", {}).items() if k != "no_emergence_by_end")
        agg["ctrl_n"] += sum(conf.get("no_emergence_by_end", {}).values())
        agg["missed"] += sum(v for k, v in conf.get("emerged_within", {}).items() if k != "emerged_within")
        agg["errs"] += [r["onset_error"] for r in rep["rows"] if "onset_error" in r]
    out = {"synthetic": agg}
    if legacy:
        pred = analyze(REPO / "runs/sparsetrack/ld", "/tmp/tt_bench/ld", params=params, only=LEGACY_IDS,
                       log=lambda *a: None)
        rep = score(load(REPO / "benchmark/labels/legacy_v0.json"), pred)
        out["legacy"] = {"on_hit": rep["onset"]["hits"], "on_n": rep["onset"]["n_timed"],
                         "len_hit": rep["length_full"]["within_tolerance"], "len_n": rep["length_full"]["n"],
                         "len_med": rep["length_full"]["median_abs_error"]}
    return out


def line(name: str, r: dict) -> str:
    import numpy as np
    s = r["synthetic"]
    med = float(np.median(np.abs(s["errs"]))) if s["errs"] else float("nan")
    txt = (f"{name:34s} synth onset {s['on_hit']:3d}/{s['on_truth']:3d} (med|e| {med:5.0f}, early {s['early']}, "
           f"late {s['late']}, missed {s['missed']}) | len {s['len_hit']}/{s['len_n']} | abs {s['abs_ok']}/{s['abs_n']} | "
           f"ctrl FP {s['ctrl_fp']}/{s['ctrl_n']}")
    if "legacy" in r:
        g = r["legacy"]
        txt += f" || legacy onset {g['on_hit']}/{g['on_n']}, len {g['len_hit']}/{g['len_n']} (med {g['len_med']:.2f})"
    return txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--set", nargs="*", default=None, help="key=value overrides for one variant")
    args = ap.parse_args()
    variants = ([("custom " + " ".join(args.set), Params(**{k: parse_value(v) for k, v in
                                                            (kv.split("=", 1) for kv in args.set)}))]
                if args.set else [("wedge_fixed (v1 default)", Params()),
                                  ("matched z3", Params(onset_source="matched", mf_z=3.0)),
                                  ("matched z4", Params(onset_source="matched", mf_z=4.0))])
    for name, prm in variants:
        print(line(name, run(prm, args.seeds)), flush=True)
