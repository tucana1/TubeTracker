"""Train the grain-aware onset network on two movies (leave one movie out: the third is never seen).

    python -m prototypes.onset_net.train --train ld m1 --name on_ldm1 --iters 3000

Per-bin classifier: is this grain's own tube visible at bin b? Input (4, 72, 72): A, D, C at bin b from the grain-
centred crops (common.py) and the grain's disc. Labels: 0 at bins <= last_absent_bin, 1 at bins >= first_visible_bin
(unlabelled between), never-germinated grains and census debris 0 everywhere, emerged-at-start grains 1 everywhere.

Each step: 64 samples, half from each training movie, half positive; a grain drawn uniformly, then one of its bins,
bins within 15 of the onset 3x as likely. Augmentation before A/D/C are formed (on the raw grain-centred crops):
a focus step (with p 0.3: Gaussian blur sigma 0.6-2 of every bin before, or from, a random bin up to b+1, the
reference bins included when they lie on that side) and a sub-pixel motion step (p 0.3: bins from a random bin
shifted by N(0, 0.5 px)); after: centre jitter +/-3 px (the disc stays at the crop centre), the 8 rotations/flips,
gain 0.6-1.6 (x 0.85-1.15 per channel), noise, blur, disc radius x 0.9-1.1. BCE, AdamW + one-cycle, fixed schedule.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
import time

import cv2
import numpy as np

from .common import NET_HALF, OUT, adc, centre_cut, dihedral, disc, make_net, window

NEAR = 15


class Data:
    def __init__(self, movies, rng, focus_p=0.3, shift_p=0.3, near_w=3.0, debris=True):
        self.rng, self.focus_p, self.shift_p = rng, focus_p, shift_p
        self.sets = []
        for m in movies:
            idx = json.loads((OUT / f"index_{m}.json").read_text())
            crops = np.load(OUT / f"crops_{m}.npy")  # in RAM (float16; ~70-150 MB per movie)
            rs, nb = idx["rs"], idx["nb"]
            pools = {0: [], 1: []}
            for i, it in enumerate(idx["items"]):
                if it["kind"] == "debris" and not debris:
                    continue
                y = np.asarray(it["y"])
                b = np.arange(nb)
                ok = b >= rs + 1
                for cls in (0, 1):
                    bb = b[ok & (y == cls)]
                    if not len(bb):
                        continue
                    if it["fv"] is not None and it["verdict"] == "emerged_within":
                        w = np.where(np.abs(bb - it["fv"]) <= NEAR, near_w, 1.0)
                    else:
                        w = np.ones(len(bb))
                    pools[cls].append((i, bb, w / w.sum()))
            self.sets.append({"m": m, "idx": idx, "crops": crops, "rs": rs, "nb": nb, "pools": pools})
            print(f"{m}: {len(idx['items'])} items; grains with negatives {len(pools[0])}, positives {len(pools[1])}; "
                  f"bins neg {sum(len(p[1]) for p in pools[0])}, pos {sum(len(p[1]) for p in pools[1])}", flush=True)

    def one(self, s: dict, i: int, b: int) -> np.ndarray:
        rng = self.rng
        crops, rs = s["crops"], s["rs"]
        it = s["idx"]["items"][i]
        mb, pb = window(s["nb"], b, rs)
        eb = list(range(rs, rs + 3))
        need = sorted(set(mb) | set(pb) | set(eb))
        raw = {k: crops[i, k].astype(np.float32) for k in need}
        if rng.random() < self.focus_p:
            st = int(rng.integers(rs, b + 2))
            sig = float(rng.uniform(0.6, 2.0))
            before = rng.random() < 0.5
            for k in need:
                if (k < st) == before:
                    raw[k] = cv2.GaussianBlur(raw[k], (0, 0), sig, borderType=cv2.BORDER_REFLECT)
        if rng.random() < self.shift_p:
            st = int(rng.integers(rs, b + 2))
            dx, dy = rng.normal(0, 0.5, 2)
            Mx = np.float32([[1, 0, dx], [0, 1, dy]])
            for k in need:
                if k >= st:
                    raw[k] = cv2.warpAffine(raw[k], Mx, raw[k].shape[::-1], flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_REFLECT)
        M = np.mean([raw[k] for k in mb], axis=0)
        P = np.mean([raw[k] for k in pb], axis=0)
        E = np.mean([raw[k] for k in eb], axis=0)
        x = adc(M, P, E)
        jx, jy = (int(v) for v in rng.integers(-3, 4, 2))
        x = centre_cut(x, NET_HALF, jx, jy)
        x = x * float(rng.uniform(0.6, 1.6)) * rng.uniform(0.85, 1.15, (3, 1, 1)).astype(np.float32)
        if rng.random() < 0.2:
            sb = float(rng.uniform(0.5, 1.0))
            x = np.stack([cv2.GaussianBlur(c, (0, 0), sb) for c in x])
        x = x + rng.normal(0, 1, x.shape).astype(np.float32) * rng.uniform(0, 0.08, (3, 1, 1)).astype(np.float32)
        g = disc(it["r"] * float(rng.uniform(0.9, 1.1)), NET_HALF)
        x = np.concatenate([x, g[None]]).astype(np.float32)
        return np.ascontiguousarray(dihedral(x, int(rng.integers(0, 8))))

    def batch(self, n: int):
        rng = self.rng
        xs, ys = [], []
        per = n // len(self.sets)
        for s in self.sets:
            for j in range(per):
                cls = int(j % 2)
                pool = s["pools"][cls]
                i, bb, w = pool[int(rng.integers(len(pool)))]
                b = int(rng.choice(bb, p=w))
                xs.append(self.one(s, i, b))
                ys.append(cls)
        return np.stack(xs), np.asarray(ys, np.float32)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--focus_p", type=float, default=0.3)
    ap.add_argument("--shift_p", type=float, default=0.3)
    ap.add_argument("--no_debris", action="store_true")
    ap.add_argument("--widths", default="16,32,64,64")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = Data(a.train, rng, a.focus_p, a.shift_p, debris=not a.no_debris)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    widths = tuple(int(v) for v in a.widths.split(","))
    net = make_net(4, widths).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.15)
    lossf = torch.nn.BCEWithLogitsLoss()
    q: queue.Queue = queue.Queue(maxsize=6)
    stop = threading.Event()

    def producer():
        while not stop.is_set():
            q.put(data.batch(a.batch))

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    t0 = time.time()
    run_loss, run_acc = [], []
    log = open(OUT / f"train_{a.name}.log", "w")
    for step in range(a.iters):
        x, y = q.get()
        xt, yt = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
        net.train()
        out = net(xt)
        loss = lossf(out, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run_loss.append(float(loss.item()))
        run_acc.append(float(((out > 0).float() == yt).float().mean().item()))
        if (step + 1) % 100 == 0:
            msg = (f"step {step + 1}: loss {np.mean(run_loss):.4f} acc {np.mean(run_acc):.3f} "
                   f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f} s")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            run_loss, run_acc = [], []
    stop.set()
    torch.save({"state": net.state_dict(), "in_ch": 4, "widths": list(widths), "drop": 0.3, "train": a.train,
                "args": vars(a)}, OUT / f"{a.name}.pt")
    (OUT / f"{a.name}.json").write_text(json.dumps({"train": a.train, "args": vars(a),
                                                    "seconds": round(time.time() - t0)}))
    print(f"saved {a.name}.pt in {time.time() - t0:.0f} s", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
