"""Train or fine-tune the tube network on several shard sets, each at a fixed share of every batch.

    python -m prototypes.tube_net.train --out runs/tube_net/X.pt --norm batch --steps 8000 --lr 2e-3 \
        --data "runs/learned_flood/shards/train_v5*.npz=1"
    python -m prototypes.tube_net.train --out runs/tube_net/Y.pt --init runs/tube_net/X.pt --steps 3000 --lr 1e-3 \
        --data "runs/learned_flood/shards/train_v5s[0-2].npz,runs/learned_flood/shards/train_v5m2s1[0-2].npz=0.5" \
               "runs/learned_flood/shards/real_ld.npz=0.5"

The network is sparsetrack.learned's (``--norm group`` = every earlier model, ``batch`` = fixed statistics at
inference); augmentation and losses are prototypes.learned_flood.train's. Shards stay float16 in memory. A set is
``GLOB[,GLOB...]=SHARE``; partially labelled shards (human traces) carry a weight map ``w``.
"""

from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import torch

from prototypes.learned_flood.train import augment, losses
from sparsetrack.learned import _unet

VAL = ["runs/learned_flood/shards/val_v5s3.npz", "runs/synth_v6/shards/val_v6ld_s33.npz"]


def load(patterns: list[str]) -> dict:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no shards match {patterns}")
    parts = [np.load(f) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("x", "body", "tip")}
    out["x"] = out["x"].astype(np.float16, copy=False)
    out["w"] = np.concatenate([p["w"] if "w" in p.files else np.ones(p["body"].shape, np.uint8) for p in parts])
    out["files"] = files
    return out


def batch(d: dict, idx: np.ndarray):
    return tuple(torch.from_numpy(d[k][idx].astype(np.float32)) for k in ("x", "body", "tip", "w"))


def parse_sets(items: list[str]) -> list[tuple[list[str], float]]:
    sets = []
    for it in items:
        globs, share = it.rsplit("=", 1)
        sets.append((globs.split(","), float(share)))
    tot = sum(s for _, s in sets)
    return [(g, s / tot) for g, s in sets]


@torch.no_grad()
def validate(net, dev, files: list[str], n: int = 480) -> str:
    net.eval()
    out = []
    for f in files:
        d = load([f])
        vx, vb, vt, _ = (t.to(dev) for t in batch(d, np.arange(min(n, len(d["x"])))))
        logit = net(vx)
        L = losses(logit, vb, vt)
        p = torch.sigmoid(logit[:, 0]) >= 0.5
        rec = float((p & (vb > 0.5)).sum() / max(float(vb.sum()), 1.0))
        fm = float((p & (vb < 0.5)).sum() / max(float((vb < 0.5).sum()), 1.0))
        out.append(f"{f.split('/')[-1].replace('.npz', '')}: loss {float(L['total']):.4f} recall {100 * rec:.1f}% "
                   f"false {100 * fm:.2f}%")
    net.train()
    return " | ".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="GLOB[,GLOB...]=SHARE, one per shard set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", help="fine-tune from this checkpoint (same normalisation)")
    ap.add_argument("--norm", choices=("group", "batch"), default="group")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--crop", type=int, default=64, help="random training sub-crop (0 = the whole 96 px sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", nargs="*", default=VAL)
    ap.add_argument("--bg-px", type=int, default=0, help="store this inference option in the checkpoint: inputs "
                                                         "relative to the local background over this many px")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    sets = [(load(globs), share) for globs, share in parse_sets(a.data)]
    for d, share in sets:
        print(f"{100 * share:3.0f}% of each batch from {len(d['x'])} samples in {len(d['files'])} shards "
              f"({', '.join(f.split('/')[-1] for f in d['files'])}); tube {100 * d['body'].mean():.1f}%, "
              f"scored {100 * d['w'].mean():.0f}%", flush=True)
    counts = [int(round(a.batch * s)) for _, s in sets]
    counts[-1] = a.batch - sum(counts[:-1])
    dev = torch.device(a.device)
    widths = (16, 32, 64, 128)
    net = _unet(widths, a.norm)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        if ck.get("norm", "group") != a.norm:
            raise SystemExit(f"{a.init} uses {ck.get('norm', 'group')} normalisation, not {a.norm}")
        net.load_state_dict(ck["state"])
    net = net.to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    started, run = time.time(), {}
    for step in range(1, a.steps + 1):
        parts = [batch(d, rng.integers(len(d["x"]), size=n)) for (d, _), n in zip(sets, counts) if n > 0]
        xb, bb, tb, wb = (torch.cat(t) for t in zip(*parts))
        x, body, tip, w = (t.to(dev) for t in augment(xb, bb, tb, g, a.crop, wb))
        L = losses(net(x), body, tip, w)
        opt.zero_grad()
        L["total"].backward()
        opt.step()
        sched.step()
        for k, v in L.items():
            run[k] = run.get(k, 0.0) + float(v.detach())
        if step % 250 == 0 or step == a.steps:
            msg = " ".join(f"{k} {v / 250:.4f}" for k, v in run.items())
            run = {}
            if a.val and (step % 1000 == 0 or step == a.steps):
                msg += " || " + validate(net, dev, a.val)
            print(f"step {step:5d} {time.time() - started:6.0f}s  {msg}", flush=True)
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "widths": widths, "norm": a.norm,
                **({"bg_px": a.bg_px} if a.bg_px else {}), "args": vars(a)}, a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
