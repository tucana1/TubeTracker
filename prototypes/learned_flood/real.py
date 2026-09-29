"""Learned evidence on the human benchmarks (ld_v1, m2_v1): SparseTrack on tube-probability movies.

    python -m prototypes.learned_flood.real --model runs/learned_flood/unet_v5_a.pt \
        [--movies ld m2] [--set key=value ...] [--dump-real OUT.json] [--baseline BASE.json]

Each real movie is read by the network bin by bin (the same registered bin, "before" and "after"
images SparseTrack uses) and written as a cache of P(tube) x SCALE_P next to the model
(runs/learned_flood/prob_<model>_<movie>, reused when present). The unchanged SparseTrack
pipeline then runs on it, with each grain's local registration measured on the real images, and
is scored like ``scripts/synth_bench.py --real`` (same per-grain dump, same paired comparison).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import sparsetrack.analyze as A
from sparsetrack import stack
from sparsetrack.evaluate import load, score

from .data import normalise
from .evaluate import SCALE_P, image_registration, registered_frame
from .model import load as load_model, predict

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from synth_bench import REAL, _within, paired, parse_value  # noqa: E402


def prob_cache(image_cache: Path, net, out_cache: Path, log=print) -> Path:
    """P(tube) x SCALE_P for every bin, in reference coordinates; whole frames at once."""
    if (out_cache / "meta.json").exists():
        return out_cache
    out_cache.mkdir(parents=True, exist_ok=True)
    bins, meta = stack.load(image_cache)
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    early = np.mean([registered_frame(bins, shifts, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([registered_frame(bins, shifts, b) for b in range(nb - 4, nb - 1)], axis=0)
    out = np.lib.format.open_memmap(out_cache / "bins.npy", mode="w+", dtype=np.float16, shape=bins.shape)
    started = time.time()
    for b in range(nb):
        out[b] = (predict(net, normalise(registered_frame(bins, shifts, b), early, late), tile=4096)[0]
                  * SCALE_P).astype(np.float16)
    out.flush()
    del out
    m = {**meta, "shifts": [[0.0, 0.0]] * nb, "raw_shifts": [[0.0, 0.0]] * nb,
         "evidence": f"learned tube probability x {SCALE_P} (from {image_cache})"}
    (out_cache / "meta.json").write_text(json.dumps(m, indent=1))
    shutil.copy(image_cache / "grains.json", out_cache / "grains.json")
    log(f"probability cache {out_cache.name}: {nb} bins in {time.time() - started:.0f} s")
    return out_cache


def score_movie(name: str, pcache: Path, params: A.Params, out_dir: Path) -> dict:
    image_cache, labels = (REPO / p for p in REAL[name])
    with image_registration(image_cache), contextlib.redirect_stdout(io.StringIO()):
        pred = A.analyze(pcache, out_dir, grains_path=labels, params=params, log=lambda *a: None)
    rep = score(load(labels), pred)
    grains = {}
    for r in rep["rows"]:
        full = r.get("full", [])
        grains[r["grain"]] = {"onset_hit": (abs(r["onset_error"]) <= 600) if "onset_error" in r else None,
                              "len_hit": sum(_within(f["error"], f["human"]) for f in full), "len_n": len(full)}
    return {"on_hit": rep["onset"]["hits"], "on_n": rep["onset"]["n_timed"],
            "len_hit": rep["length_full"]["within_tolerance"], "len_n": rep["length_full"]["n"],
            "len_med": rep["length_full"]["median_abs_error"], "len_bias": rep["length_full"]["bias"],
            "errs": [(f["error"], f["human"], r["grain"], f["frame"]) for r in rep["rows"] for f in r.get("full", [])],
            "grains": grains, "pred": str(out_dir / "predictions.json")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--movies", nargs="+", choices=tuple(REAL), default=list(REAL))
    ap.add_argument("--set", nargs="*", default=[], help="SparseTrack Params overrides, key=value")
    ap.add_argument("--dump-real")
    ap.add_argument("--baseline")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args(argv)
    model = Path(args.model)
    net = load_model(str(model)).to(args.device)
    # probability maps have no grain rims, so no settling check; drift is checked on the real images
    params = replace(A.Params(settle=False, drift_check=True),
                     **{k: parse_value(v) for k, v in (kv.split("=", 1) for kv in args.set)})
    res = {}
    for name in args.movies:
        pcache = prob_cache(REPO / REAL[name][0], net, model.parent / f"prob_{model.stem}_{name}")
        res[name] = score_movie(name, pcache, params, Path(f"/tmp/tt_bench/learned_{model.stem}_{name}"))
        g = res[name]
        print(f"{name}: onset {g['on_hit']}/{g['on_n']}, len {g['len_hit']}/{g['len_n']} "
              f"(med {g['len_med']:.2f}, bias {g['len_bias']:+.2f})", flush=True)
        worst = sorted(g["errs"], key=lambda e: -abs(e[0]))[:8]
        print(f"  worst {name} length errors: " + ", ".join(f"{gid}@{fr} {e:+.1f}/{h:.0f}" for e, h, gid, fr in worst))
    if args.baseline:
        print(paired(json.loads(Path(args.baseline).read_text()), res))
    if args.dump_real:
        Path(args.dump_real).write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
