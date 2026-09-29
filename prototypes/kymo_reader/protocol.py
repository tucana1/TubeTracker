"""The honest-validation protocol for one trained model.

    python -m prototypes.kymo_reader.protocol --model runs/kymo_reader/models/A.pt --tag A --train-movie none
    python -m prototypes.kymo_reader.protocol --model runs/kymo_reader/models/B.pt --tag B --train-movie ld
    python -m prototypes.kymo_reader.protocol --model runs/kymo_reader/models/C.pt --tag C --train-movie m2

- A (synthetic only): onset_px and min_len chosen on the synthetic validation movies; the length
  start on SparseTrack's paths is fixed in advance ("edge": the annotators start a trace where the tube
  leaves the grain's visible edge); scored on ld and m2.
- B / C (synthetic + one movie's labels): start, onset_px and min_len chosen on the training movie's
  labels; scored on the other movie only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import evaluate as E
from . import store
from .model import load

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/kymo_reader"
VAL = [OUT / "data/synth_synthv5_s3.npz", OUT / "data/synth_synthv5m2_s20.npz"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--train-movie", choices=("none", "ld", "m2"), required=True)
    ap.add_argument("--start", default=None, help="override the length start (census, rest, edge)")
    ap.add_argument("--vmax", type=float, default=4.0)
    a = ap.parse_args(argv)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load(a.model, dev)
    if a.train_movie == "none":
        val = [sm for p in VAL if p.exists() for sm in store.load(p)]
        start = a.start or "edge"
        ch = E.choose_on_synth(net, val, dev, a.vmax, start)
        ch["start"] = start
        ch["onset_source"] = "kymo"  # fixed in advance: SparseTrack's synthetic onsets were read with its front
        tests = ["ld", "m2"]
    else:
        ch = E.choose_on_movie(net, a.train_movie, dev, a.vmax)
        if a.start:
            ch["start"] = a.start
        tests = [m for m in ("ld", "m2") if m != a.train_movie]
    print(f"{a.tag}: chosen without the test movie: {ch}", flush=True)
    res = {"choice": ch}
    for mv in tests:
        out = E.run_movie(net, mv, dev, a.tag, ch["start"], a.vmax, ch["onset_px"], ch["min_len"], ch["onset_source"])
        res[mv] = out
        print(f"== {mv} ({a.tag}; start={ch['start']}, onset from {ch['onset_source']}, onset_px={ch['onset_px']}, "
              f"min_len={ch['min_len']})")
        print(E.fmt_row("st053", out["st053"]))
        for k in (f"{a.tag}_st", f"{a.tag}_route", "st053_route"):
            print(E.fmt_row(k.replace(a.tag + "_", "kymo_"), out[k]))
        v = out[f"{a.tag}_route"]["vs_st053_route"]
        print("  same human routes, kymo vs SparseTrack: " + ", ".join(
            f"{k} {v[k]['delta']:+d} [{v[k]['ci'][0]:+.0f},{v[k]['ci'][1]:+.0f}]" for k in ("onset_hit", "len_hit", "both_hit")))
        print(f"  grains read: {out['n_read']}", flush=True)
        if a.train_movie == "none":  # secondary row, fixed in advance: the reader's lengths under st053's own calls
            t2 = f"{a.tag}_stonset"
            o2 = E.run_movie(net, mv, dev, t2, ch["start"], a.vmax, ch["onset_px"], ch["min_len"], "st053")
            res[f"{mv}_stonset"] = o2
            print(E.fmt_row("kymo_st+st053on", o2[f"{t2}_st"]), flush=True)
    (OUT / "results" / f"protocol_{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
