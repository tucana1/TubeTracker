"""End to end on synthetic movies with moving grains (``synth_moves``): SparseTrack with each grain_track, scored
against the truth (onsets within 50 synthetic frames, lengths within max(2 px, 10%)), per grain and by how the
grain moves.

    python -m prototypes.grain_tracking.synth_end2end runs/grain_tracking/synth/moves_m2_s31 --set grain_track=follow
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sparsetrack.analyze import Params, analyze
from sparsetrack.evaluate import load, score


def _within(err: float, truth_len: float) -> bool:
    return abs(err) <= max(2.0, 0.1 * truth_len)


def chosen_grains(stem: Path, n_static: int = 12, seed: int = 0) -> list[str]:
    """The movie's census grains standing for its moving truth grains, and ``n_static`` still ones."""
    cache, truth = Path(f"{stem}_cache"), load(Path(f"{stem}_truth.json"))
    census = json.loads((cache / "grains.json").read_text())["grains"]
    tracks = truth["grain_tracks"]["tracks"]
    moving, still = [], []
    for tid, tg in truth["grains"].items():
        if not tg.get("isolated", True):
            continue
        d = [np.hypot(g["x"] - tg["x"], g["y"] - tg["y"]) for g in census]
        k = int(np.argmin(d))
        if d[k] > 6:
            continue
        tr = tracks.get(tg["census_id"])
        (moving if tr and tr["kind"] in ("drift", "push") else still).append(census[k]["id"])
    rng = np.random.default_rng(seed)
    return moving + list(rng.choice(still, min(n_static, len(still)), replace=False))


def run(stem: Path, params: Params, out: Path, only: list[str] | None = None) -> dict:
    cache, truth_path = Path(f"{stem}_cache"), Path(f"{stem}_truth.json")
    truth = load(truth_path)
    pred = analyze(cache, out, params=params, only=only, log=lambda *a: None)
    rep = score(truth, pred, onset_tol=50)
    tracks = truth["grain_tracks"]["tracks"]
    rows = {}
    for r in rep["rows"]:
        if not r["matched"]:
            continue
        tg = truth["grains"][r["grain"]]
        tr = tracks.get(tg.get("census_id"))
        kind = tr["kind"] if tr else "static"
        if tr and tr.get("vanish_bin") is not None:
            kind += "+vanish"
        full = r.get("full", [])
        rows[r["grain"]] = {"kind": kind, "onset_hit": (abs(r["onset_error"]) <= 50) if "onset_error" in r else None,
                            "len_hit": sum(_within(f["error"], f["human"]) for f in full), "len_n": len(full),
                            "abs_ok": sum(a["pred"] < 2.0 for a in r.get("absences", [])),
                            "abs_n": len(r.get("absences", [])), "call": [r.get("human"), r.get("pred")]}
    return {"onset": [rep["onset"]["hits"], rep["onset"]["n_timed"], rep["onset"]["n_human_emerged_within"]],
            "length": [rep["length_full"]["within_tolerance"], rep["length_full"]["n"]], "rows": rows,
            "pred": str(Path(out) / "predictions.json")}


def compare(a: dict, b: dict, n_boot: int = 4000) -> str:
    """b against a, per kind of grain, with a paired bootstrap over grains for lengths."""
    out = []
    kinds = sorted({r["kind"] for r in a["rows"].values()})
    for kind in kinds + ["all"]:
        g = [k for k in a["rows"] if k in b["rows"] and (kind == "all" or a["rows"][k]["kind"] == kind)]
        la = sum(a["rows"][k]["len_hit"] for k in g)
        lb = sum(b["rows"][k]["len_hit"] for k in g)
        n = sum(a["rows"][k]["len_n"] for k in g)
        oa = sum(bool(a["rows"][k]["onset_hit"]) for k in g)
        ob = sum(bool(b["rows"][k]["onset_hit"]) for k in g)
        d = np.array([b["rows"][k]["len_hit"] - a["rows"][k]["len_hit"] for k in g])
        ci = ""
        if len(g) > 2:
            idx = np.random.default_rng(0).integers(0, len(g), (n_boot, len(g)))
            lo, hi = np.percentile(d[idx].sum(axis=1), [2.5, 97.5])
            ci = f" (95% CI {lo:+.0f} to {hi:+.0f})"
        out.append(f"{kind:14s} grains {len(g):3d}: lengths {la} -> {lb} of {n}{ci}; onsets {oa} -> {ob}")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+", help="movie stems: STEM_cache and STEM_truth.json")
    ap.add_argument("--set", nargs="*", default=["grain_track=follow"], help="variants: key=value[,key=value...]")
    ap.add_argument("--work", default="runs/grain_tracking/synth_e2e")
    ap.add_argument("--all-grains", action="store_true", help="every census grain, not the moving ones + 12 still")
    args = ap.parse_args()
    variants = []
    for spec in args.set:
        over = {}
        for kv in spec.split(","):
            k, v = kv.split("=", 1)
            cur = getattr(Params(), k)
            over[k] = type(cur)(v) if not isinstance(cur, bool) else v.lower() == "true"
        variants.append((spec, replace(Params(), **over)))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    for stem in args.stems:
        stem = Path(stem)
        only = None if args.all_grains else chosen_grains(stem)
        base = run(stem, Params(), work / f"{stem.name}_phase", only)
        out = {"phase": base}
        print(f"{stem.name}: {len(base['rows'])} grains; phase onset {base['onset']}, lengths {base['length']}",
              flush=True)
        for spec, prm in variants:
            new = run(stem, prm, work / f"{stem.name}_{spec.replace(',', '_')}", only)
            out[spec] = new
            print(f"  {spec}: onset {new['onset']}, lengths {new['length']}")
            print(compare(base, new), flush=True)
        (work / f"{stem.name}_e2e.json").write_text(json.dumps(out))
