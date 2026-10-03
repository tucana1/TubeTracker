"""Train the tip detector on the human traces of some movies (leave-one-movie-out: the third is never seen).

    python -m prototypes.tip_detector.train --train ld m2 --name tip_ldm2

Each step: 16 extractions, half from each training movie, 70% FULL-trace extractions; each rotated by any angle,
maybe flipped, cut to 128 px, gain/offset/noise/blur per channel. Loss: CenterNet focal loss on a Gaussian (sigma 2 px)
at each apex, plus 1.0 x weighted BCE of the body head, both only inside the scored region (build.py).
A fixed schedule (no early stopping: nothing of the held-out movie is looked at).

Version 2 (3 Oct, after the first m1 test): also scored as background, pixels with no structure within 8 px (|A| below
4 grey levels and |D| below 3 everywhere within 8 px of them; label-free), away from the labels (not within 8 px of
a traced tube or 10 px of an apex, not in a blind zone). Version 1 scored only round the labelled grains and their
traces, and its maps rose to 0.15-0.3 on empty background far from any structure (never seen by the loss), as high
as at real tips. The tip head's bias starts at -2.19 (P = 0.1), as in CenterNet.
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from .build import EXT, KINDS
from .common import OUT, unet

T = 128
SIG = 2.0
S = 2 * EXT


class Data:
    def __init__(self, movies, channels, rng, empty_bg=True):
        self.rng = rng
        self.sets = []
        for m in movies:
            z = np.load(OUT / f"samples_{m}.npz")
            info = z["info"]
            kind = info[:, 2].astype(int)
            full = np.nonzero(kind <= 2)[0]
            other = np.nonzero(kind > 2)[0]
            sc = z["scored"]
            if empty_bg:
                sc = self.add_empty(z["x"], sc, z["body"], z["blind"], z["apex"])
            self.sets.append({
                "x": z["x"][:, channels], "sc": sc, "bo": z["body"], "ap": z["apex"],
                "full": full, "other": other})
            print(f"{m}: {len(full)} FULL-trace extractions, {len(other)} others "
                  f"({ {k: int((kind == i).sum()) for i, k in enumerate(KINDS)} })", flush=True)
        self.jj, self.ii = np.meshgrid(np.arange(T, dtype=np.float32), np.arange(T, dtype=np.float32))

    @staticmethod
    def add_empty(x, sc, body, blind, apex):
        """Scored masks plus the empty background (no structure within 8 px, away from the labels)."""
        k8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        jj, ii = np.meshgrid(np.arange(S), np.arange(S))
        out, added = [], []
        for i in range(len(x)):
            a, d = x[i, 0].astype(np.float32) * 20.0, x[i, 1].astype(np.float32) * 8.0  # grey levels
            struct = ((np.abs(a) > 4) | (np.abs(d) > 3)).astype(np.uint8)
            prot = cv2.dilate(struct, k8) | cv2.dilate(np.unpackbits(body[i], axis=-1)[:, :S], k8) | \
                np.unpackbits(blind[i], axis=-1)[:, :S]
            for q in apex[i]:
                if np.isfinite(q[0]):
                    prot |= (np.hypot(jj - q[0], ii - q[1]) <= 10).astype(np.uint8)
            s0 = np.unpackbits(sc[i], axis=-1)[:, :S]
            s1 = s0 | (1 - prot)
            added.append(s1.mean() - s0.mean())
            out.append(np.packbits(s1, axis=-1))
        print(f"  empty background: +{100 * np.mean(added):.0f}% of pixels scored", flush=True)
        return np.stack(out)

    def sample(self):
        rng = self.rng
        d = self.sets[rng.integers(len(self.sets))]
        pool = d["full"] if rng.random() < 0.7 or not len(d["other"]) else d["other"]
        i = int(pool[rng.integers(len(pool))])
        x = d["x"][i].astype(np.float32)
        sc = np.unpackbits(d["sc"][i], axis=-1)[:, :S]
        bo = np.unpackbits(d["bo"][i], axis=-1)[:, :S]
        ap = d["ap"][i]
        return self.augment(x, sc, bo, ap)

    def augment(self, x, sc, bo, ap):
        rng = self.rng
        th = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(th), np.sin(th)
        Rm = np.array([[c, -s], [s, c]])
        if rng.random() < 0.5:
            Rm = Rm @ np.diag([-1.0, 1.0])
        side = S / (abs(c) + abs(s))
        slack = max(0.0, (side - T) / 2 - 1)
        t = rng.uniform(-slack, slack, 2)
        cs, cd = (S - 1) / 2, (T - 1) / 2
        M = np.zeros((2, 3))
        M[:, :2] = Rm
        M[:, 2] = cd + t - Rm @ [cs, cs]
        xw = cv2.warpAffine(np.ascontiguousarray(x.transpose(1, 2, 0)), M, (T, T), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
        if xw.ndim == 2:
            xw = xw[:, :, None]
        xw = xw.transpose(2, 0, 1).copy()
        scw = cv2.warpAffine(sc, M, (T, T), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bow = cv2.warpAffine(bo, M, (T, T), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        hm = np.zeros((T, T), np.float32)
        pos = np.zeros((T, T), np.float32)
        for a in ap:
            if not np.isfinite(a[0]):
                continue
            q = M[:, :2] @ a + M[:, 2]
            hm = np.maximum(hm, np.exp(-((self.jj - q[0]) ** 2 + (self.ii - q[1]) ** 2) / (2 * SIG ** 2)))
            qi, qj = int(round(q[1])), int(round(q[0]))
            if 0 <= qi < T and 0 <= qj < T:
                pos[qi, qj] = 1.0
                hm[qi, qj] = 1.0
        # intensity
        for k in range(len(xw)):
            xw[k] = xw[k] * np.exp(rng.uniform(np.log(0.6), np.log(1.6))) + rng.uniform(-0.1, 0.1)
            if rng.random() < 0.5:
                xw[k] += rng.normal(0, rng.uniform(0, 0.15), (T, T)).astype(np.float32)
        if rng.random() < 0.3:
            sg = rng.uniform(0.3, 1.0)
            for k in range(len(xw)):
                xw[k] = cv2.GaussianBlur(xw[k], (0, 0), sg)
        return xw, hm, pos, scw.astype(np.float32), bow.astype(np.float32)

    def batch(self, n):
        xs, hs, ps, ws, bs = zip(*(self.sample() for _ in range(n)))
        f = lambda v: np.ascontiguousarray(np.stack(v), np.float32)
        return f(xs), f(hs)[:, None], f(ps)[:, None], f(ws)[:, None], f(bs)[:, None]


def main(argv=None):
    import torch
    import torch.nn.functional as F
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--channels", default="ADC", help="subset of A, D, C")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--body-w", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-empty", action="store_true", help="version 1: no empty-background negatives")
    args = ap.parse_args(argv)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    chans = ["ADC".index(c) for c in args.channels]
    data = Data(args.train, chans, rng, empty_bg=not args.no_empty)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = unet(len(chans), 2)
    with torch.no_grad():
        net.head.bias[0] = -2.19
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    t0, run = time.time(), np.zeros(3)
    for it in range(1, args.iters + 1):
        x, hm, pos, w, body = (torch.from_numpy(v).to(dev) for v in data.batch(args.batch))
        out = net(x)
        lt, lb = out[:, :1], out[:, 1:2]
        logp, log1p = F.logsigmoid(lt), F.logsigmoid(-lt)
        p = torch.sigmoid(lt)
        l_pos = -((1 - p) ** 2 * logp * pos).sum()
        l_neg = -(((1 - hm) ** 4) * p ** 2 * log1p * (1 - pos) * w).sum()
        l_tip = (l_pos + l_neg) / pos.sum().clamp(min=1.0)
        bce = F.binary_cross_entropy_with_logits(lb, body, pos_weight=torch.tensor(3.0, device=dev), reduction="none")
        l_body = (bce * w).sum() / w.sum().clamp(min=1.0)
        loss = l_tip + args.body_w * l_body
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += [loss.item(), l_tip.item(), l_body.item()]
        if it % 250 == 0:
            print(f"it {it}: loss {run[0] / 250:.3f} tip {run[1] / 250:.3f} body {run[2] / 250:.3f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            run[:] = 0
    path = OUT / f"{args.name}.pt"
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "channels": args.channels,
                "out_ch": 2, "widths": (16, 32, 64, 128), "train": args.train, "args": vars(args)}, path)
    (OUT / f"{args.name}.json").write_text(json.dumps({"args": vars(args), "seconds": time.time() - t0}))
    print(f"saved {path} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
