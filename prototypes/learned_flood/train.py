"""Train the evidence U-Net on synthetic shards (CPU is enough for this size).

    python -m prototypes.learned_flood.train --shards runs/learned_flood/shards/train_*.npz \
        --val runs/learned_flood/shards/val.npz --out runs/learned_flood/unet.pt
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
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("x", "body", "tip")} | {"files": files}
    # partially labelled shards (human traces) carry a weight map; synthetic ones are fully scored
    out["w"] = np.concatenate([p["w"] if "w" in p.files else np.ones_like(p["body"]) for p in parts])
    return out


def augment(x: torch.Tensor, body: torch.Tensor, tip: torch.Tensor, g: torch.Generator, crop: int = 0,
            w: torch.Tensor | None = None):
    w = torch.ones_like(body) if w is None else w
    if crop and crop < x.shape[-1]:
        oy, ox = (int(v) for v in torch.randint(x.shape[-1] - crop + 1, (2,), generator=g))
        x, body, tip, w = (t[..., oy:oy + crop, ox:ox + crop] for t in (x, body, tip, w))
    k = int(torch.randint(4, (1,), generator=g))
    flip = bool(torch.randint(2, (1,), generator=g))
    def geo(t):
        t = torch.rot90(t, k, dims=(-2, -1))
        return torch.flip(t, dims=(-1,)) if flip else t
    x, body, tip, w = geo(x), geo(body), geo(tip), geo(w)
    n = x.shape[0]
    gain = 0.7 + 0.7 * torch.rand(n, 1, 1, 1, generator=g)
    offset = 0.4 * (torch.rand(n, 1, 1, 1, generator=g) - 0.5)
    x = x * gain + offset
    # single-bin noise differs between movies (bin size, codec, camera): noisier current bin than references
    sig = torch.rand(n, 1, 1, 1, generator=g) * torch.tensor([0.25, 0.08, 0.08]).view(1, 3, 1, 1)
    x = x + sig * torch.randn(x.shape, generator=g)
    return x, body, tip, w


def losses(logits: torch.Tensor, body: torch.Tensor, tip: torch.Tensor,
           w: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    lb, lt = logits[:, 0], logits[:, 1]
    w = torch.ones_like(body) if w is None else w
    n = w.sum() + 1.0
    bce = (F.binary_cross_entropy_with_logits(lb, body, pos_weight=torch.tensor(2.0, device=lb.device),
                                              reduction="none") * w).sum() / n
    p = torch.sigmoid(lb) * w
    dice = 1.0 - (2 * (p * body).sum() + 1.0) / (p.sum() + (body * w).sum() + 1.0)
    tip_l = (F.binary_cross_entropy_with_logits(lt, tip, reduction="none") * (1.0 + 10.0 * tip) * w).sum() / n
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
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--real", nargs="*", default=[], help="partially labelled real shards (realdata.py)")
    ap.add_argument("--real-frac", type=float, default=0.5, help="share of each batch drawn from --real")
    ap.add_argument("--init", help="start from this checkpoint (fine-tuning)")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    data = load_shards(args.shards)
    x_all = torch.from_numpy(data["x"].astype(np.float32))
    body_all = torch.from_numpy(data["body"].astype(np.float32))
    tip_all = torch.from_numpy(data["tip"].astype(np.float32))
    w_all = torch.from_numpy(data["w"].astype(np.float32))
    real = load_shards(args.real) if args.real else None
    if real is not None:
        rx, rb, rt, rw = (torch.from_numpy(real[k].astype(np.float32)) for k in ("x", "body", "tip", "w"))
        print(f"real samples {len(rx)} from {len(real['files'])} shards, {100 * args.real_frac:.0f}% of each batch")
    val = load_shards(args.val) if args.val else None
    print(f"train samples {len(x_all)} from {len(data['files'])} shards; tube pixels {100 * body_all.mean():.1f}%")
    dev = torch.device(args.device)
    net = UNet(widths=args.widths)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["state"])
    net = net.to(dev)
    print(f"parameters {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    started, run = time.time(), {}
    for step in range(1, args.steps + 1):
        n_real = int(round(args.batch * args.real_frac)) if real is not None else 0
        idx = torch.randint(len(x_all), (args.batch - n_real,), generator=g)
        xb, bb, tb, wb = x_all[idx], body_all[idx], tip_all[idx], w_all[idx]
        if n_real:
            j = torch.randint(len(rx), (n_real,), generator=g)
            xb, bb, tb, wb = (torch.cat([a, c[j]]) for a, c in ((xb, rx), (bb, rb), (tb, rt), (wb, rw)))
        x, body, tip, w = (t.to(dev) for t in augment(xb, bb, tb, g, args.crop, wb))
        net.train()
        L = losses(net(x), body, tip, w)
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
                    vx = torch.from_numpy(val["x"][:512].astype(np.float32)).to(dev)
                    vb = torch.from_numpy(val["body"][:512].astype(np.float32)).to(dev)
                    vt = torch.from_numpy(val["tip"][:512].astype(np.float32)).to(dev)
                    VL = losses(net(vx), vb, vt)
                msg += " | val " + " ".join(f"{k} {float(v):.4f}" for k, v in VL.items())
            print(f"step {step:5d} {time.time() - started:6.0f}s  {msg}", flush=True)
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "widths": tuple(args.widths),
                "args": vars(args)}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
