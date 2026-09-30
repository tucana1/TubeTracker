"""Leave one movie out over the three labelled movies: the tube network recipe trained on two movies' traces, judged on
the third (30 Sep 2026).

    python -m prototypes.tube_net.loo3 NAME --held m2 [--held ld ...] [--init runs/tube_net/tn_bn_syn.pt]
    python -m prototypes.tube_net.loo3 NAME --train ld m2 m1 --no-check        # the final network, all three movies

The recipe is 0.8.0's (``prototypes/learned_flood/models/tubes_bn_real_ld_m2.recipe.sh``): BatchNorm base
``tn_bn_syn``, 3000 steps at lr 1e-3, each batch of 32 a quarter v5 shards, a quarter v6 shards and half trace crops
(v3, flat caps) of the training movies - their crops pooled, so each movie counts by its number of crops - and
``bg_px`` 96 stored in the checkpoint. Movie 1's crops are ``realdata.py m1 ... --flat-cap --over-grain 0.6 --follow``.
Trains ``runs/tube_net/NAME_<movies>.pt`` (e.g. tn3_ldm1 = ld + m1 traces, judged on m2) unless it exists, then runs
the pixel check on the held-out movie (``pixels.py``, in memory) into ``runs/tube_net/pix_NAME_<movies>.json``.
The fold that trains on ld + m2 is 0.8.0's own network (the same recipe and crops): judge movie 1 with it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/tube_net"
V5 = "runs/learned_flood/shards/train_v5s[0-2].npz,runs/learned_flood/shards/train_v5m2s1[0-2].npz"
V6 = "runs/synth_v6/shards/train_v6*.npz"
REAL = "runs/tube_net/shards/real3_{}.npz"
MOVIES = ("ld", "m2", "m1")


def fold_name(name: str, movies) -> str:
    return f"{name}_{''.join(m for m in MOVIES if m in movies)}"


def train(name: str, movies, init: str, steps: int = 3000, lr: float = 1e-3, real_frac: float = 0.5,
          seed: int = 0, out: Path | None = None) -> Path:
    model = out or OUT / f"{fold_name(name, movies)}.pt"
    if model.exists():
        return model
    syn = (1.0 - real_frac) / 2
    data = [f"{V5}={syn}", f"{V6}={syn}", ",".join(REAL.format(m) for m in movies) + f"={real_frac}"]
    cmd = [sys.executable, "-u", "-m", "prototypes.tube_net.train", "--out", str(model), "--init", init,
           "--norm", "batch", "--steps", str(steps), "--lr", str(lr), "--bg-px", "96", "--seed", str(seed),
           "--data", *data]
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    with open(OUT / "logs" / f"train_{model.stem}.log", "w") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)
    return model


def check(model: Path, held: str) -> str:
    pix = OUT / f"pix_{model.stem}_on_{held}.json"
    cmd = [sys.executable, "-u", "-m", "prototypes.tube_net.pixels", str(model), held, "--json", str(pix), "--grains"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip().splitlines()[-1] if r.returncode == 0 else r.stderr[-2000:]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--held", nargs="*", default=[], choices=MOVIES, help="one fold per held-out movie")
    ap.add_argument("--train", nargs="*", default=None, choices=MOVIES, help="train on exactly these movies")
    ap.add_argument("--init", default="runs/tube_net/tn_bn_syn.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--real-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="checkpoint path (with --train)")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args(argv)
    folds = [([m for m in MOVIES if m != h], [h]) for h in a.held]
    if a.train is not None:
        folds.append((a.train, [m for m in MOVIES if m not in a.train]))
    for movies, held in folds:
        model = train(a.name, movies, a.init, a.steps, a.lr, a.real_frac, a.seed,
                      Path(a.out) if a.out and a.train is not None else None)
        print(f"{model.name}: trained on {'+'.join(movies)}", flush=True)
        if not a.no_check:
            for h in held:
                print(f"  {check(model, h)}", flush=True)


if __name__ == "__main__":
    main()
