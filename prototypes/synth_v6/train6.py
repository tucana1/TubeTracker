"""``prototypes.learned_flood.train`` with shards kept as float16 in memory (batches are converted on the fly)
and optional mixing of a second shard set (e.g. the v5 shards tubes_synth_v1 was trained on) at a fixed share
of each batch. Same network, augmentation, losses and schedule.

    python -m prototypes.synth_v6.train6 --shards "runs/synth_v6/shards/train_*.npz" \
        --val runs/synth_v6/shards/val_v6ld_s33.npz --out runs/synth_v6/tubes_synth_v6.pt --steps 8000
"""

from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import torch

from prototypes.learned_flood.model import UNet
from prototypes.learned_flood.train import augment, losses


def load(patterns: list[str]) -> dict:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no shards match {patterns}")
    parts = [np.load(f) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("x", "body", "tip")}
    out["w"] = np.concatenate([p["w"] if "w" in p.files else np.ones(p["body"].shape, np.uint8) for p in parts])
    out["files"] = files
    return out


def batch(d: dict, idx: np.ndarray):
    return tuple(torch.from_numpy(d[k][idx].astype(np.float32)) for k in ("x", "body", "tip", "w"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--mix", nargs="*", default=[], help="a second shard set drawn at --mix-frac of each batch")
    ap.add_argument("--mix-frac", type=float, default=0.5)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--init", help="fine-tune from this checkpoint")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = load(a.shards)
    mix = load(a.mix) if a.mix else None
    val = load(a.val) if a.val else None
    print(f"train samples {len(data['x'])} from {len(data['files'])} shards; tube pixels "
          f"{100 * data['body'].mean():.1f}%" + (f"; mixed in {len(mix['x'])} from {len(mix['files'])} shards at "
                                                  f"{100 * a.mix_frac:.0f}%" if mix else ""), flush=True)
    dev = torch.device(a.device)
    net = UNet()
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["state"])
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    started, run = time.time(), {}
    for step in range(1, a.steps + 1):
        n_mix = int(round(a.batch * a.mix_frac)) if mix is not None else 0
        xb, bb, tb, wb = batch(data, rng.integers(len(data["x"]), size=a.batch - n_mix))
        if n_mix:
            parts = batch(mix, rng.integers(len(mix["x"]), size=n_mix))
            xb, bb, tb, wb = (torch.cat([u, v]) for u, v in zip((xb, bb, tb, wb), parts))
        x, body, tip, w = (t.to(dev) for t in augment(xb, bb, tb, g, a.crop, wb))
        net.train()
        L = losses(net(x), body, tip, w)
        opt.zero_grad()
        L["total"].backward()
        opt.step()
        sched.step()
        for k, v in L.items():
            run[k] = run.get(k, 0.0) + float(v.detach())
        if step % 100 == 0 or step == a.steps:
            msg = " ".join(f"{k} {v / 100:.4f}" for k, v in run.items())
            run = {}
            if val is not None and (step % 500 == 0 or step == a.steps):
                net.eval()
                with torch.no_grad():
                    vx, vb, vt, _ = (t.to(dev) for t in batch(val, np.arange(min(512, len(val["x"])))))
                    VL = losses(net(vx), vb, vt)
                msg += " | val " + " ".join(f"{k} {float(v):.4f}" for k, v in VL.items())
            print(f"step {step:5d} {time.time() - started:6.0f}s  {msg}", flush=True)
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "widths": (16, 32, 64, 128),
                "args": vars(a)}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
