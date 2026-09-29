"""Validation losses (and body recall / false marks at P >= 0.5) of models on shards.

    python -m prototypes.synth_v6.valloss MODEL.pt [MODEL2.pt ...] --shards a.npz b.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from prototypes.learned_flood.model import load
from prototypes.learned_flood.train import losses


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=480)
    a = ap.parse_args(argv)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for sh in a.shards:
        z = np.load(sh)
        x = torch.from_numpy(z["x"][:a.n].astype(np.float32))
        body = torch.from_numpy(z["body"][:a.n].astype(np.float32))
        tip = torch.from_numpy(z["tip"][:a.n].astype(np.float32))
        for m in a.models:
            net = load(m).to(dev)
            with torch.no_grad():
                out = torch.cat([net(x[i:i + 64].to(dev)).cpu() for i in range(0, len(x), 64)])
            L = losses(out, body, tip)
            p = torch.sigmoid(out[:, 0]) >= 0.5
            rec = float((p & (body > 0)).sum() / max(float(body.sum()), 1))
            fp = float((p & (body == 0)).sum() / max(float((body == 0).sum()), 1))
            print(f"{sh.split('/')[-1]:24s} {m.split('/')[-1]:26s} " + " ".join(f"{k} {float(v):.4f}" for k, v in L.items())
                  + f" | body recall {100 * rec:.1f}% false marks {100 * fp:.2f}%", flush=True)


if __name__ == "__main__":
    main()
