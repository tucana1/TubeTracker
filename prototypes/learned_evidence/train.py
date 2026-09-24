"""Train the evidence U-Net on synthetic shards (CPU is enough for this size).

    python -m prototypes.learned_evidence.train --shards runs/learned_evidence/shards/train_*.npz \
        --val runs/learned_evidence/shards/val.npz --out runs/learned_evidence/unet.pt
"""

from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import UNet


def load_shards(patterns: list[str]) -> dict[str, np.ndarray]:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no shards match {patterns}")
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("x", "body", "tip")} | {"files": files}


def augment(x: torch.Tensor, body: torch.Tensor, tip: torch.Tensor, g: torch.Generator, crop: int = 0):
    if crop and crop < x.shape[-1]:
        oy, ox = (int(v) for v in torch.randint(x.shape[-1] - crop + 1, (2,), generator=g))
        x, body, tip = (t[..., oy:oy + crop, ox:ox + crop] for t in (x, body, tip))
    k = int(torch.randint(4, (1,), generator=g))
    flip = bool(torch.randint(2, (1,), generator=g))
    def geo(t):
        t = torch.rot90(t, k, dims=(-2, -1))
        return torch.flip(t, dims=(-1,)) if flip else t
    x, body, tip = geo(x), geo(body), geo(tip)
    n = x.shape[0]
    gain = 0.7 + 0.7 * torch.rand(n, 1, 1, 1, generator=g)
    offset = 0.4 * (torch.rand(n, 1, 1, 1, generator=g) - 0.5)
    x = x * gain + offset
    # single-bin noise differs between movies (bin size, codec, camera): noisier current bin than references
    sig = torch.rand(n, 1, 1, 1, generator=g) * torch.tensor([0.25, 0.08, 0.08]).view(1, 3, 1, 1)
    x = x + sig * torch.randn(x.shape, generator=g)
    return x, body, tip


def losses(logits: torch.Tensor, body: torch.Tensor, tip: torch.Tensor) -> dict[str, torch.Tensor]:
    lb, lt = logits[:, 0], logits[:, 1]
    bce = F.binary_cross_entropy_with_logits(lb, body, pos_weight=torch.tensor(2.0))
    p = torch.sigmoid(lb)
    dice = 1.0 - (2 * (p * body).sum() + 1.0) / (p.sum() + body.sum() + 1.0)
    tip_l = (F.binary_cross_entropy_with_logits(lt, tip, reduction="none") * (1.0 + 10.0 * tip)).mean()
    return {"bce": bce, "dice": dice, "tip": tip_l, "total": bce + dice + 0.5 * tip_l}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--crop", type=int, default=64, help="random training sub-crop (0 = whole sample)")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    data = load_shards(args.shards)
    x_all = torch.from_numpy(data["x"].astype(np.float32))
    body_all = torch.from_numpy(data["body"].astype(np.float32))
    tip_all = torch.from_numpy(data["tip"].astype(np.float32))
    val = load_shards(args.val) if args.val else None
    print(f"train samples {len(x_all)} from {len(data['files'])} shards; tube pixels {100 * body_all.mean():.1f}%")
    net = UNet(widths=args.widths)
    print(f"parameters {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    started, run = time.time(), {}
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(x_all), (args.batch,), generator=g)
        x, body, tip = augment(x_all[idx], body_all[idx], tip_all[idx], g, args.crop)
        net.train()
        L = losses(net(x), body, tip)
        opt.zero_grad()
        L["total"].backward()
        opt.step()
        sched.step()
        for k, v in L.items():
            run[k] = run.get(k, 0.0) + float(v.detach())
        if step % 100 == 0 or step == args.steps:
            msg = " ".join(f"{k} {v / 100:.4f}" for k, v in run.items())
            run = {}
            if val is not None and (step % 500 == 0 or step == args.steps):
                net.eval()
                with torch.no_grad():
                    vx = torch.from_numpy(val["x"][:512].astype(np.float32))
                    vb = torch.from_numpy(val["body"][:512].astype(np.float32))
                    vt = torch.from_numpy(val["tip"][:512].astype(np.float32))
                    VL = losses(net(vx), vb, vt)
                msg += " | val " + " ".join(f"{k} {float(v):.4f}" for k, v in VL.items())
            print(f"step {step:5d} {time.time() - started:6.0f}s  {msg}", flush=True)
    torch.save({"state": net.state_dict(), "widths": tuple(args.widths), "args": vars(args)}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
