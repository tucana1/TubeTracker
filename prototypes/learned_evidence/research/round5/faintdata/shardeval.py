"""Losses (train.py's) and pixel precision/recall at P > 0.5 of models on the first N samples of each shard, eval mode,
whole 96 px samples, no augmentation. A sanity check of training: v2 against a new model on v2's and my shards.

    python shardeval.py N MODEL.pt [MODEL.pt ...]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/user/TubeTracker")
from prototypes.learned_evidence.model import load  # noqa: E402
from prototypes.learned_evidence.train import losses  # noqa: E402

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata")
SHARDS = sorted(glob.glob(str(ME.parent.parent / "le/shards/train_*.npz"))) + sorted(glob.glob(str(ME / "shards/train_*.npz")))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    n = int(sys.argv[1])
    nets = {Path(m).stem: load(m, "cpu") for m in sys.argv[2:]}
    print(f"{'shard':22s} " + " | ".join(f"{k[:18]:>18s}: dice  prec  rec " for k in nets))
    for f in SHARDS:
        d = np.load(f)
        x = torch.from_numpy(d["x"][:n].astype(np.float32))
        body = torch.from_numpy(d["body"][:n].astype(np.float32))
        tip = torch.from_numpy(d["tip"][:n].astype(np.float32))
        cells = []
        for net in nets.values():
            logits = net(x)
            L = losses(logits, body, tip)
            p = torch.sigmoid(logits[:, 0]) > 0.5
            tp = float((p & (body > 0.5)).sum())
            prec, rec = tp / max(float(p.sum()), 1.0), tp / max(float((body > 0.5).sum()), 1.0)
            cells.append(f"{float(L['dice']):.3f} {prec:.3f} {rec:.3f}")
        print(f"{Path(f).stem:22s} " + " | ".join(f"{c:>34s}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
