"""The repository's training recipe (prototypes/learned_evidence/train.py: its UNet, augment, losses, AdamW, OneCycle,
batch sampling and seeds, imported, not copied), with three additions only:
  - a checkpoint every --ckpt-every steps with every random stream, resumed after a container restart;
  - --init: start from another model's weights (continuing from v2) instead of train.py's random init;
  - at the end, the losses of the final model on the first 256 samples of each shard (eval mode), per group.
Shards are kept as stored (float16/uint8) and a batch is converted when drawn (identical values; half the memory).

    python train5.py --shards A.npz ... --out model.pt [--steps 6000] [--lr 2e-3] [--threads 2] [--init v2.pt]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/user/TubeTracker")
from prototypes.learned_evidence.model import UNet  # noqa: E402
from prototypes.learned_evidence.train import augment, losses  # noqa: E402


def load_shards(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no shards match {patterns}")
    parts = [np.load(f) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("x", "body", "tip")}
    return out | {"files": files, "sizes": [len(p["x"]) for p in parts]}


@torch.no_grad()
def per_shard(net, data, n=256):
    net.eval()
    rows, start = {}, 0
    for f, size in zip(data["files"], data["sizes"]):
        sl = slice(start, start + min(n, size))
        start += size
        L = losses(net(torch.from_numpy(data["x"][sl].astype(np.float32))),
                   torch.from_numpy(data["body"][sl].astype(np.float32)),
                   torch.from_numpy(data["tip"][sl].astype(np.float32)))
        rows[Path(f).stem] = {k: round(float(v), 4) for k, v in L.items()}
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pct-start", type=float, default=0.1)
    ap.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--init", default=None, help="start from this model's weights (same widths)")
    ap.add_argument("--ckpt-every", type=int, default=250)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    data = load_shards(args.shards)
    x_all, body_all, tip_all = (torch.from_numpy(data[k]) for k in ("x", "body", "tip"))
    print(f"train samples {len(x_all)} from {len(data['files'])} shards; tube pixels "
          f"{100 * float(np.mean(data['body'])):.1f}%", flush=True)
    for f in data["files"]:
        print("  shard", f, flush=True)
    net = UNet(widths=args.widths)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if tuple(ck.get("widths", (16, 32, 64, 128))) != tuple(args.widths):
            raise SystemExit("--init model has other widths")
        net.load_state_dict(ck["state"])
        print(f"initialised from {args.init}", flush=True)
    print(f"parameters {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=args.pct_start)
    started, run, first = time.time(), {}, 1
    ckpt = f"{args.out}.ckpt"
    if os.path.exists(ckpt):  # resume: weights, optimiser, schedule and every random stream
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        g.set_state(st["g"])
        torch.set_rng_state(st["torch_rng"])
        first = st["step"] + 1
        print(f"resumed from {ckpt} at step {st['step']}", flush=True)
    for step in range(first, args.steps + 1):
        idx = torch.randint(len(x_all), (args.batch,), generator=g)
        x, body, tip = augment(x_all[idx].float(), body_all[idx].float(), tip_all[idx].float(), g, args.crop)
        net.train()
        L = losses(net(x), body, tip)
        opt.zero_grad()
        L["total"].backward()
        opt.step()
        sched.step()
        for k, v in L.items():
            run[k] = run.get(k, 0.0) + float(v.detach())
        if step % 100 == 0 or step == args.steps:
            n = 100 if step % 100 == 0 else step % 100
            print(f"step {step:5d} {time.time() - started:6.0f}s  " + " ".join(f"{k} {v / n:.4f}" for k, v in run.items()),
                  flush=True)
            run = {}
        if step % args.ckpt_every == 0 and step < args.steps:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "g": g.get_state(), "torch_rng": torch.get_rng_state(), "step": step}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    tmp = f"{args.out}.tmp"  # written whole, then renamed
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "widths": tuple(args.widths),
                "args": vars(args) | {"shard_files": data["files"], "recipe": "prototypes/learned_evidence/train.py "
                                      "augment + losses, AdamW wd 1e-4, OneCycleLR"}}, tmp)
    os.replace(tmp, args.out)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"saved {args.out} ({time.time() - started:.0f} s)", flush=True)
    rows = per_shard(net, data)
    for k, v in rows.items():
        print(f"final (eval mode, first 256 samples) {k:22s} " + " ".join(f"{a} {b:.4f}" for a, b in v.items()), flush=True)
    Path(f"{args.out}.pershard.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
