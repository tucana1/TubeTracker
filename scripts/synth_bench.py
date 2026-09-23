"""Score SparseTrack parameter variants on the synthetic seeds and the legacy real labels.

    .venv/bin/python scripts/synth_bench.py                      # default variants, suite v1
    .venv/bin/python scripts/synth_bench.py --suite v2 --breakdown --set onset_source=matched mf_z=3

Synthetic movies come from `sparsetrack synth [--preset v2]`; their caches (`prepare
--frames-per-bin 25 --ref-start 0`, 440 MB each) are rebuilt from the movies when missing and
deleted after scoring unless --keep-caches (suite v1: runs/sparsetrack/synth/s{seed}_cache + synth_s{seed}_truth.json;
suite v2/v3: v{2,3}s{seed}_cache + synthv{2,3}_s{seed}_truth.json). The onset tolerance
there is 50 synthetic frames (= 600 source frames). Seeds 3-4 of v2, v3 and v4 are the held-out
synthetic test: report them only for a frozen variant (--seeds 3 4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sparsetrack.analyze import Params, analyze  # noqa: E402
from sparsetrack.evaluate import load, score  # noqa: E402

SYN = REPO / "runs/sparsetrack/synth"
LEGACY_IDS = ["g025", "g014", "g037", "g034", "g013", "g030", "g029"]
SUITES = {"v1": ("s{}_cache", "synth_s{}_truth.json"), "v2": ("v2s{}_cache", "synthv2_s{}_truth.json"),
          "v3": ("v3s{}_cache", "synthv3_s{}_truth.json"), "v4": ("v4s{}_cache", "synthv4_s{}_truth.json")}


def ensure_cache(cache: Path, truth_path: Path) -> bool | None:
    """Build a missing cache from its movie (~20 s; 440 MB). True if built now, None if no movie.

    Caches are not kept by default: each is rebuilt when needed and deleted after scoring.
    """
    if (cache / "grains.json").exists():
        return False
    movie = truth_path.with_name(truth_path.name.replace("_truth.json", ".mp4"))
    if not movie.exists():
        return None
    import contextlib
    import io
    from sparsetrack import stack
    from sparsetrack.cli import write_census
    with contextlib.redirect_stdout(io.StringIO()):
        stack.prepare(movie, cache, frames_per_bin=25, ref_bins=3, ref_start=0, log=lambda *a: None)
        write_census(cache, 3, False)
    return True


def parse_value(v: str):
    if "," in v or v.startswith("("):  # a tuple: a,b or (a,b)
        return tuple(parse_value(x) for x in v.strip("()").split(",") if x)
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def _within(err: float, truth_len: float) -> bool:
    return abs(err) <= max(2.0, 0.1 * truth_len)


def run(params: Params, seeds: list[int], suite: str = "v1", legacy: bool = True, keep_caches: bool = False) -> dict:
    agg = {"on_hit": 0, "on_n": 0, "on_truth": 0, "early": 0, "late": 0, "len_hit": 0, "len_n": 0,
           "abs_ok": 0, "abs_n": 0, "ctrl_fp": 0, "ctrl_n": 0, "missed": 0, "errs": [], "rows": []}
    cache_fmt, truth_fmt = SUITES[suite]
    for s in seeds:
        cache, truth_path = SYN / cache_fmt.format(s), SYN / truth_fmt.format(s)
        built = ensure_cache(cache, truth_path)
        if built is None:
            print(f"  (no movie for {suite} seed {s}: skipped)", flush=True)
            continue
        pred = analyze(cache, f"/tmp/tt_bench/{suite}_s{s}", params=params, log=lambda *a: None)
        if built and not keep_caches:
            import shutil
            shutil.rmtree(cache)
        truth = load(truth_path)
        rep = score(truth, pred, onset_tol=50)
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
        for r in rep["rows"]:
            t = truth["labels"].get(r["grain"], {}).get("truth", {})
            agg["rows"].append({**r, "seed": s, "truth": t})
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
    s = r["synthetic"]
    med = float(np.median(np.abs(s["errs"]))) if s["errs"] else float("nan")
    txt = (f"{name:34s} synth onset {s['on_hit']:3d}/{s['on_truth']:3d} (med|e| {med:5.0f}, early {s['early']}, "
           f"late {s['late']}, missed {s['missed']}) | len {s['len_hit']}/{s['len_n']} | abs {s['abs_ok']}/{s['abs_n']} | "
           f"ctrl FP {s['ctrl_fp']}/{s['ctrl_n']}")
    if "legacy" in r:
        g = r["legacy"]
        txt += f" || legacy onset {g['on_hit']}/{g['on_n']}, len {g['len_hit']}/{g['len_n']} (med {g['len_med']:.2f})"
    return txt


def breakdown(r: dict) -> str:
    """Onset and length hit rates split by the truth attributes of each synthetic grain."""
    rows = r["synthetic"]["rows"]
    groups = {
        "all": lambda t: True,
        "dark line": lambda t: t.get("bright_core") is False,
        "bright core": lambda t: t.get("bright_core") is True,
        "rotates": lambda t: t.get("rotates") is True,
        "drifts": lambda t: t.get("max_drift_px", 0) > 0,
        "curls": lambda t: t.get("curl_rad_per_px", 0) != 0,
        "free (touch/cross)": lambda t: t.get("free") is True,
        "docked particle": lambda t: t.get("dock_frame") is not None,
        "slow (<0.4 px/bin)": lambda t: t.get("rate_px_per_bin", 9) < 0.4,
        "faint (amp<1.3)": lambda t: t.get("amplitude", 9) < 1.3,
        "maturing (tau>5)": lambda t: t.get("tau_bins", 0) > 5,
        "look evolves": lambda t: t.get("evolves") is True,
        "stuck to substrate": lambda t: t.get("anchored") is True,
        "landing grain": lambda t: t.get("arriving") is True,
        "fat stub": lambda t: t.get("stub") is True,
        "sways": lambda t: t.get("sways_px", 0) > 0,
    }
    out = [f"{'group':22s} {'onset in tol':>14s} {'len in tol':>12s} {'med |len err|':>14s} {'controls ok':>12s}"]
    for name, sel in groups.items():
        sub = [x for x in rows if sel(x["truth"])]
        timed = [x for x in sub if x.get("human") == "emerged_within" and "onset_error" in x]
        n_germ = sum(1 for x in sub if x.get("human") == "emerged_within")
        hit = sum(1 for x in timed if abs(x["onset_error"]) <= 50)
        full = [f for x in sub for f in x.get("full", [])]
        lh = sum(_within(f["error"], f["human"]) for f in full)
        med = float(np.median([abs(f["error"]) for f in full])) if full else float("nan")
        ctrl = [x for x in sub if x.get("human") == "no_emergence_by_end"]
        cok = sum(1 for x in ctrl if x.get("pred") == "no_emergence_by_end")
        out.append(f"{name:22s} {hit:>6d}/{n_germ:<7d} {lh:>5d}/{len(full):<6d} {med:>14.2f} {cok:>5d}/{len(ctrl):<6d}")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--suite", choices=tuple(SUITES), default="v1")
    ap.add_argument("--no-legacy", action="store_true")
    ap.add_argument("--breakdown", action="store_true", help="hit rates by synthetic truth attribute")
    ap.add_argument("--dump", help="write per-grain rows (JSON) here")
    ap.add_argument("--keep-caches", action="store_true", help="keep caches rebuilt from the movies (440 MB each)")
    ap.add_argument("--set", nargs="*", default=None, help="key=value overrides for one variant")
    args = ap.parse_args()
    variants = ([("custom " + " ".join(args.set), Params(**{k: parse_value(v) for k, v in
                                                            (kv.split("=", 1) for kv in args.set)}))]
                if args.set is not None else [("default", Params()),
                                              ("wedge_fixed (v1 onset)", Params(onset_source="wedge_fixed"))])
    for name, prm in variants:
        res = run(prm, args.seeds, args.suite, legacy=not args.no_legacy, keep_caches=args.keep_caches)
        print(line(name, res), flush=True)
        if args.breakdown:
            print(breakdown(res), flush=True)
        if args.dump:
            Path(args.dump).write_text(json.dumps(res["synthetic"]["rows"], default=float))
