"""Fine-tune a tube network on a movie's pseudo-label crops, with ``prototypes/tube_adapt/finetune.py``'s recipe
(1500 steps, batch 32, AdamW lr 5e-4 one-cycle, 64 px crops, 0.8.0's losses and augmentation, BatchNorm statistics
frozen) and half of every batch replay of the starting network's own training data.

    python -m prototypes.self_train.train --name m1_r1_sb --pseudo m1_r1 --variant sb [--init NET.pt]
        [--replay ldm2|ldm1]

Variants: ``sa`` = pseudo-traced crops 50% (tube_adapt's variant a); ``sb`` = pseudo-traced 25% + propagated
(between, after) 25% (variant b without negatives, as the pseudo-labels have none). Replay ``ldm2`` (the shipped
network's: ld + m2 trace crops v3 25%, one v5 and one v6 synthetic shard 12.5% each) or ``ldm1`` (tn3_ldm1's: ld + m1
trace crops instead; for adapting it to movie 2 without ever showing it movie 2's traces).
Models: runs/research/self_train/models/<name>.pt.
"""

from __future__ import annotations

import argparse
import sys

from prototypes.self_train.select import OUT
from prototypes.tube_adapt import finetune
from prototypes.tube_adapt.common import BASE_NET, V5, V6

REPLAY = {"ldm2": "runs/tube_net/shards/real3_ld.npz,runs/tube_net/shards/real3_m2.npz",
          "ldm1": "runs/tube_net/shards/real3_ld.npz,runs/tube_net/shards/real3_m1.npz"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pseudo", required=True, help="pseudo-label name (shards trace_/prop_<name>.npz)")
    ap.add_argument("--variant", choices=("sa", "sb"), default="sb")
    ap.add_argument("--init", default=str(BASE_NET))
    ap.add_argument("--replay", choices=tuple(REPLAY), default="ldm2")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a, rest = ap.parse_known_args(argv)
    tr = str(OUT / "shards" / f"trace_{a.pseudo}.npz")
    pr = str(OUT / "shards" / f"prop_{a.pseudo}.npz")

    def sets_for(variant, fold, trace_name, grains):
        replay = [(f"real {a.replay}", finetune.load(REPLAY[a.replay].split(",")), 0.25),
                  ("v5", finetune.load(V5.split(",")), 0.125), ("v6", finetune.load(V6.split(",")), 0.125)]
        if a.variant == "sa":
            return [("pseudo traced", finetune.load([tr]), 0.5)] + replay
        return [("pseudo traced", finetune.load([tr]), 0.25),
                ("pseudo propagated", finetune.load([pr], {"between", "after"}), 0.25)] + replay

    finetune.OUT = OUT
    finetune.sets_for = sets_for
    finetune.main(["--name", a.name, "--variant", "a", "--init", a.init, "--steps", str(a.steps),
                   "--seed", str(a.seed)] + rest)


if __name__ == "__main__":
    main(sys.argv[1:])
