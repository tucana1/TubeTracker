"""Run the adaptation experiment for folds and variants: train (one process at a time), then judge on the fold's held-out
movie-1 grains (pixel check) and on movie 2 (forgetting); end to end in a second queue.

    python -m prototypes.tube_adapt.run train k2f0 --variants a b c      # train + pixel checks, in order
    python -m prototypes.tube_adapt.run e2e k2f0 --variants base a b c    # end to end on the held-out grains
    python -m prototypes.tube_adapt.run e2e k2f0 --variants a --wait      # wait for each model to exist first

Models: runs/research/tube_adapt/models/<fold>_<variant>.pt; results: runs/research/tube_adapt/eval/
{pix_<fold>_<variant>_m1.json, pix_<fold>_<variant>_m2.json, e2e_<fold>_<variant>.json}. ``base`` = the shipped
network (no movie-1 labels).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from prototypes.tube_adapt.common import BASE_NET, OUT, REPO
from prototypes.tube_adapt.crops import fold_grains


def model_path(fold: str, v: str):
    return BASE_NET if v == "base" else OUT / "models" / f"{fold}_{v}.pt"


def sh(cmd: list[str], log) -> None:
    with open(log, "a") as fh:
        fh.write(" ".join(map(str, cmd)) + "\n")
        fh.flush()
        subprocess.run(list(map(str, cmd)), cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("train", "e2e", "pix"))
    ap.add_argument("folds", nargs="+")
    ap.add_argument("--variants", nargs="+", default=["a", "b", "c"])
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--no-m2", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="extra finetune.py arguments")
    a = ap.parse_args(argv)
    py = [sys.executable, "-u", "-m"]
    for fold in a.folds:
        train, held = fold_grains(fold)
        for v in a.variants:
            m = model_path(fold, v)
            log = OUT / "logs" / f"{fold}_{v}.log"
            if a.what in ("train", "pix"):
                if a.what == "train" and v != "base" and not m.exists():
                    sh(py + ["prototypes.tube_adapt.finetune", "--name", f"{fold}_{v}", "--variant", v, "--fold", fold]
                       + a.extra, log)
                out = OUT / "eval" / f"pix_{fold}_{v}_m1.json"
                if not out.exists():
                    sh(py + ["prototypes.tube_adapt.evaluate", "pix", m, "--movie", "m1", "--grains", *held, "--out", out],
                       log)
                out2 = OUT / "eval" / f"pix_{fold}_{v}_m2.json"
                if not a.no_m2 and v != "base" and not out2.exists():
                    sh(py + ["prototypes.tube_adapt.evaluate", "pix", m, "--movie", "m2", "--out", out2], log)
            else:
                while a.wait and not m.exists():
                    time.sleep(30)
                time.sleep(5 if a.wait else 0)
                out = OUT / "eval" / f"e2e_{fold}_{v}.json"
                if not out.exists():
                    sh(py + ["prototypes.tube_adapt.evaluate", "e2e", m, "--grains", *held, "--out", out,
                             "--tag", f"{fold}_{v}"], log)


if __name__ == "__main__":
    main()
