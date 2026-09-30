"""Leave-one-movie-out: fine-tune a recipe on one labelled movie's traces, judge it on the other.

    python -m prototypes.tube_net.loo NAME --init BASE.pt --norm batch --real real3 [--synth v5] [--steps 3000] ...

Trains NAME_ld (traces of the dev movie ld; judged on movie 2) and NAME_m2 (movie 2's traces; judged on ld) one
after the other into runs/tube_net/, then runs the pixel check of each on the movie it did not see
(runs/tube_net/pix_NAME_<trained on>.json), with inputs relative to the local background (``--bg``). The
end-to-end benches (``bench.py``) are run separately, one at a time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/tube_net"
SYNTH = {
    # tubes_synth_v1's own training shards (v5: dev field seeds 0-2, movie 2's field seeds 10-12)
    "v5": "runs/learned_flood/shards/train_v5s[0-2].npz,runs/learned_flood/shards/train_v5m2s1[0-2].npz",
    # v6 shards (bulbs, rounded tips, the grain's own change; prototypes/synth_v6)
    "v6": "runs/synth_v6/shards/train_v6*.npz",
}
REAL = {"real": "runs/learned_flood/shards/real_{}.npz", "real2": "runs/tube_net/shards/real2_{}.npz",
        "real3": "runs/tube_net/shards/real3_{}.npz"}
OTHER = {"ld": "m2", "m2": "ld"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--init", required=True)
    ap.add_argument("--norm", choices=("group", "batch"), default="batch")
    ap.add_argument("--real", choices=tuple(REAL), default="real3")
    ap.add_argument("--synth", nargs="+", default=["v5"], help="synthetic sets, each an equal part of the synthetic "
                                                               "share (keys of SYNTH or globs)")
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--movies", nargs="+", default=["ld", "m2"], help="whose traces to train on (one model each)")
    ap.add_argument("--bg", type=int, default=96, help="pixel check with inputs relative to the local background")
    a = ap.parse_args(argv)
    for mv in a.movies:
        model = OUT / f"{a.name}_{mv}.pt"
        synth = [SYNTH.get(s, s) for s in a.synth]
        share = (1.0 - a.real_frac) / len(synth)
        data = [f"{s}={share}" for s in synth] + [f"{REAL[a.real].format(mv)}={a.real_frac}"]
        if not model.exists():
            cmd = [sys.executable, "-u", "-m", "prototypes.tube_net.train", "--out", str(model), "--init", a.init,
                   "--norm", a.norm, "--steps", str(a.steps), "--lr", str(a.lr), "--crop", str(a.crop), "--data", *data]
            with open(OUT / "logs" / f"train_{model.stem}.log", "w") as fh:
                subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)
        judged = OTHER[mv]
        pix = OUT / f"pix_{model.stem}{f'_bg{a.bg}' if a.bg else ''}.json"
        cmd = [sys.executable, "-u", "-m", "prototypes.tube_net.pixels", str(model), judged, "--json", str(pix),
               "--bg", str(a.bg), "--grains"]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(r.stdout.strip().splitlines()[-1] if r.returncode == 0 else r.stderr[-2000:], flush=True)


if __name__ == "__main__":
    main()
