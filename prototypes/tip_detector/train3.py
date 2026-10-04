"""Train a version 3 tip detector on the human traces of some movies (leave-one-movie-out: the third is never seen).

    python -m prototypes.tip_detector.train3 --train ld m2 --name tip3_ldm2 [--channels A,D6,D12,D24,C]
        [--shift] [--long-w 2] [--protect] [--widths 16,32,64,128,128] [--iters 2500]

As train.py (version 2: CenterNet focal loss + body BCE inside the scored region, empty background scored, any-angle
rotation, flips, gain/offset/noise/blur, 128 px crops, batch 16 with half from each training movie, AdamW + one-cycle,
a fixed schedule), on build3.py's samples (int8, memory-mapped), with these options:
  --channels   any of A, D6, D12, D24, C (A,D6,C = version 2's inputs)
  --shift      also the shifted-bin apex samples (weak supervision; target sigma per apex)
  --long-w     sampling weight of extractions holding the apex of a long trace (>= 60 px) of a crowded movie
  --protect    in movies where under 60% of the grains are labelled (m2, m1), leave unscored the pixels with structure
               (|A| > 4 or |D6| > 3 grey levels within 2 px) that are not within 8 px of a labelled trace or r + 8 px of
               a labelled grain: there the tips of unlabelled tubes would otherwise be taught as background
  --widths     U-Net widths (5 levels: twice the reach, ~90 px)
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from .build import EXT
from .build3 import KINDS3
from .common import OUT, unet
from .common3 import CHANNELS3, Q, SCALE3

S = 2 * EXT


class Data:
    def __init__(self, movies, channels, rng, T=128, shift=False, long_w=1.0, protect=False, full_p=0.7):
        self.rng, self.T, self.full_p = rng, T, full_p
        self.ci = [CHANNELS3.index(c) for c in channels]
        self.sets = []
        ia, idd = CHANNELS3.index("A"), CHANNELS3.index("D6")
        for m in movies:
            x = np.load(OUT / f"samples3_{m}.npy", mmap_mode="r")
            z = np.load(OUT / f"samples3_{m}.npz")
            info = z["info"]
            kind = info[:, 2].astype(int)
            use = np.ones(len(kind), bool) if shift else kind != KINDS3.index("shift_apex")
            full = np.nonzero(use & np.isin(kind, [0, 1, 2, 6]))[0]
            other = np.nonzero(use & ~np.isin(kind, [0, 1, 2, 6]))[0]
            crowded = float(z["labelled_fraction"]) < 0.6
            sc = self.masks(x, z["scored"], z["body"], z["blind"], z["near"], z["apex"], ia, idd,
                            protect and crowded)
            long_ = (np.nan_to_num(z["apex"][:, :, 3], nan=0.0) >= 60).any(1)
            w = np.where(long_[full] & crowded, long_w, 1.0)
            self.sets.append({"x": x, "sc": sc, "bo": z["body"], "ap": z["apex"], "full": full, "other": other,
                              "wfull": w / w.sum()})
            print(f"{m}: {len(full)} FULL-trace extractions (long apex {int(long_[full].sum())}, weight "
                  f"{long_w if crowded else 1.0}), {len(other)} others "
                  f"({ {k: int((kind[use] == i).sum()) for i, k in enumerate(KINDS3)} }); protect "
                  f"{protect and crowded}", flush=True)
        self.jj, self.ii = np.meshgrid(np.arange(T, dtype=np.float32), np.arange(T, dtype=np.float32))

    @staticmethod
    def masks(x, sc, body, blind, near, apex, ia, idd, protect):
        """Scored masks plus the empty background (version 2), less unlabelled structure if ``protect``."""
        k8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        jj, ii = np.meshgrid(np.arange(S), np.arange(S))
        out, added, removed = [], [], []
        for i in range(len(x)):
            a = x[i, ia].astype(np.float32) / Q * SCALE3["A"]
            d = x[i, idd].astype(np.float32) / Q * SCALE3["D6"]
            struct = ((np.abs(a) > 4) | (np.abs(d) > 3)).astype(np.uint8)
            prot = cv2.dilate(struct, k8) | cv2.dilate(np.unpackbits(body[i], axis=-1)[:, :S], k8) | \
                np.unpackbits(blind[i], axis=-1)[:, :S]
            for q in apex[i]:
                if np.isfinite(q[0]):
                    prot |= (np.hypot(jj - q[0], ii - q[1]) <= 10).astype(np.uint8)
            s0 = np.unpackbits(sc[i], axis=-1)[:, :S]
            s1 = s0 | (1 - prot)
            if protect:
                unl = cv2.dilate(struct, k2) & (1 - np.unpackbits(near[i], axis=-1)[:, :S])
                removed.append((s1 & unl).mean())
                s1 = s1 & (1 - unl)
            added.append(s1.mean() - s0.mean())
            out.append(np.packbits(s1, axis=-1))
        print(f"  scored: {100 * np.mean(added):+.0f}% of pixels (empty background"
              f"{f', less {100 * np.mean(removed):.0f}% unlabelled structure' if protect else ''})", flush=True)
        return np.stack(out)

    def sample(self):
        rng = self.rng
        d = self.sets[rng.integers(len(self.sets))]
        if rng.random() < self.full_p or not len(d["other"]):
            i = int(d["full"][rng.choice(len(d["full"]), p=d["wfull"])])
        else:
            i = int(d["other"][rng.integers(len(d["other"]))])
        x = d["x"][i, self.ci].astype(np.float32) / Q
        sc = np.unpackbits(d["sc"][i], axis=-1)[:, :S]
        bo = np.unpackbits(d["bo"][i], axis=-1)[:, :S]
        return self.augment(x, sc, bo, d["ap"][i])

    def augment(self, x, sc, bo, ap):
        rng, T = self.rng, self.T
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
        xw = np.stack([cv2.warpAffine(np.ascontiguousarray(ch), M, (T, T), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT) for ch in x])
        scw = cv2.warpAffine(sc, M, (T, T), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bow = cv2.warpAffine(bo, M, (T, T), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        hm = np.zeros((T, T), np.float32)
        pos = np.zeros((T, T), np.float32)
        for a in ap:
            if not np.isfinite(a[0]):
                continue
            q = M[:, :2] @ a[:2] + M[:, 2]
            sig = float(a[2]) if np.isfinite(a[2]) else 2.0
            hm = np.maximum(hm, np.exp(-((self.jj - q[0]) ** 2 + (self.ii - q[1]) ** 2) / (2 * sig ** 2)))
            qi, qj = int(round(q[1])), int(round(q[0]))
            if 0 <= qi < T and 0 <= qj < T:
                pos[qi, qj] = 1.0
                hm[qi, qj] = 1.0
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
    ap.add_argument("--channels", default="A,D6,D12,D24,C")
    ap.add_argument("--widths", default="16,32,64,128")
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--body-w", type=float, default=1.0)
    ap.add_argument("--shift", action="store_true")
    ap.add_argument("--long-w", type=float, default=1.0)
    ap.add_argument("--protect", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    chans = [c.strip() for c in args.channels.split(",")]
    widths = tuple(int(w) for w in args.widths.split(","))
    data = Data(args.train, chans, rng, T=args.crop, shift=args.shift, long_w=args.long_w, protect=args.protect)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = unet(len(chans), 2, widths)
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
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "channels": chans, "version": 3,
                "out_ch": 2, "widths": widths, "train": args.train, "args": vars(args)}, path)
    (OUT / f"{args.name}.json").write_text(json.dumps({"args": vars(args), "seconds": time.time() - t0}))
    print(f"saved {path} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
