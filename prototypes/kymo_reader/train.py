"""Train KymoNet: synthetic kymographs (exact targets) plus, optionally, one real movie's partial labels.

    python -m prototypes.kymo_reader.train --out runs/kymo_reader/models/A.pt \
        --synth runs/kymo_reader/data/synth_synthv5_s{0,1,2}.npz ... \
        --val runs/kymo_reader/data/synth_synthv5_s3.npz ... [--real ld] [--steps 4000]

Masked binary cross-entropy per sample (mean over its labelled cells), batches of random crops
(``--t-crop`` bins x ``--s-crop`` px; the arc crop starts at the exit half of the time), lateral
mirroring and a contrast gain as augmentation. Every ``--val-every`` steps the synthetic validation
set is read end to end (DP front) and the checkpoint with the most lengths in tolerance is kept.
Real labels of the movie being tested are never loaded here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import crf
from . import evaluate as E
from . import features, store
from .model import KymoNet, save

REPO = Path(__file__).resolve().parents[2]


def load_sets(paths, variants=None) -> list[dict]:
    out = []
    for p in paths:
        for sm in store.load(p):
            if variants and sm["info"].get("variant") not in variants:
                continue
            if "lab_m" in sm and int(sm["lab_m"].sum()) == 0:
                continue
            out.append(sm)
    return out


def make_batch(samples, idx, rng, t_crop: int, s_crop: int, augment: bool = True, fixed: bool = True):
    """A batch of random crops. ``fixed``: always padded to (t_crop, s_crop) - the GPU (MPS) recompiles
    for every new shape, which made variable shapes several times slower."""
    items = []
    for i in idx:
        sm = samples[i]
        T, S = sm["dyn"].shape[:2]
        tc, sc = min(T, t_crop), min(S, s_crop)
        t0 = int(rng.integers(0, T - tc + 1))
        s0 = 0 if rng.random() < 0.5 else int(rng.integers(0, S - sc + 1))
        flip = augment and rng.random() < 0.5
        gain = float(np.exp(rng.normal(0.0, 0.2))) if augment else 1.0
        dyn, stat = features.inputs(sm, t0, t0 + tc, s0, s0 + sc, gain=gain, flip=flip)
        y, m = features.target(sm, t0, t0 + tc, s0, s0 + sc)
        items.append((dyn, stat, y, m))
    Tm = t_crop if fixed else max(x[0].shape[1] for x in items)
    Sm = s_crop if fixed else max(x[0].shape[2] for x in items)
    B = len(items)
    D = np.zeros((B, features.C_DYN, Tm, Sm), np.float32)
    St = np.zeros((B, features.C_STAT, Sm), np.float32)
    Y = np.zeros((B, Tm, Sm), np.float32)
    Mk = np.zeros((B, Tm, Sm), np.float32)
    lo = np.zeros((B, Tm), np.int64)
    hi = np.full((B, Tm), Sm, np.int64)
    tv = np.zeros((B, Tm), bool)
    nc = np.zeros(B, np.int64)
    for b, (dyn, stat, y, m) in enumerate(items):
        D[b, :, :dyn.shape[1], :dyn.shape[2]] = dyn
        St[b, :, :stat.shape[1]] = stat
        Y[b, :y.shape[0], :y.shape[1]] = y
        Mk[b, :m.shape[0], :m.shape[1]] = m
        l, h = crf.bounds(y, m)
        lo[b, :len(l)], hi[b, :len(h)] = l, h
        tv[b, :len(l)] = True
        nc[b] = y.shape[1]
    return D, St, Y, Mk, lo, hi, tv, nc


class Prefetch:
    """Batches assembled on a background thread (numpy releases the GIL for the heavy parts)."""

    def __init__(self, fn, depth: int = 4):
        import queue
        import threading
        self.q = queue.Queue(maxsize=depth)
        self.fn = fn
        self.stop = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self.stop:
            self.q.put(self.fn())

    def next(self):
        return self.q.get()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--synth", nargs="*", default=[])
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--real", nargs="*", default=[], help="real movies whose labels are trained on (never the test movie)")
    ap.add_argument("--real-variants", nargs="*", default=["route", "st"])
    ap.add_argument("--real-grains", nargs="*", default=None, help="only these grains' real labels (cross-validation)")
    ap.add_argument("--p-real", type=float, default=0.35)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--t-crop", type=int, default=160)
    ap.add_argument("--s-crop", type=int, default=96)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--crf", type=float, default=0.0, help="weight of the monotone-front CRF loss (0 = BCE only)")
    ap.add_argument("--crf-vmax", type=int, default=4)
    ap.add_argument("--front", type=float, default=0.0, help="weight of the per-bin front-position loss")
    ap.add_argument("--dil-t", default=None, help="comma-separated time dilations (ablation), e.g. 1,1,1,1,1,1,1,1,1,1")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="stop after this step (the LR schedule still spans --steps): reproduces a run cut short")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    syn = load_sets(a.synth, variants=None)
    real = load_sets([REPO / f"runs/kymo_reader/data/real_{m}.npz" for m in a.real], variants=a.real_variants)
    if a.real_grains is not None:
        real = [sm for sm in real if sm["info"]["grain"] in set(a.real_grains)]
    val = load_sets(a.val)
    print(f"data: {len(syn)} synthetic, {len(real)} real ({','.join(a.real) or '-'}), {len(val)} validation "
          f"samples ({time.time() - t0:.0f} s); device {dev}", flush=True)
    kw = {} if not a.dil_t else {"dil_t": tuple(int(x) for x in a.dil_t.split(","))}
    net = KymoNet(width=a.width, **kw).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state"])
    print(f"parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.3f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(a), "val": []}
    best = -1.0
    n_real = int(round(a.p_real * a.batch)) if real else 0
    run_loss = []
    brng = np.random.default_rng(a.seed + 1)

    pool = syn + real

    def next_batches():  # one fixed-shape batch: synthetic items first, then real ones
        idx = list(brng.integers(0, len(syn), a.batch - n_real)) if syn else []
        if n_real:
            idx += list(len(syn) + brng.integers(0, len(real), n_real))
        return [make_batch(pool, idx, brng, a.t_crop, a.s_crop)]

    feed = Prefetch(next_batches)
    last_step = min(a.steps, a.stop_after or a.steps)
    for step in range(1, last_step + 1):
        net.train()
        batches = feed.next()
        loss = 0.0
        for D, St, Y, Mk, lo, hi, tv, nc in batches:
            D, St, Y, Mk, lo, hi, tv, nc = (torch.from_numpy(x).to(dev) for x in (D, St, Y, Mk, lo, hi, tv, nc))
            logit = net(D, St)
            bce = F.binary_cross_entropy_with_logits(logit, Y, reduction="none") * Mk
            per = bce.sum(dim=(1, 2)) / Mk.sum(dim=(1, 2)).clamp(min=1.0)
            loss = loss + per.sum() / a.batch
            if a.crf > 0:
                loss = loss + a.crf * crf.loss(logit, lo, hi, tv, nc, a.crf_vmax) * (len(D) / a.batch)
            if a.front > 0:
                loss = loss + a.front * crf.front_loss(logit, lo, hi, tv, nc) * (len(D) / a.batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss.append(float(loss.detach()))
        if step % 100 == 0:
            print(f"step {step}: loss {np.mean(run_loss[-100:]):.4f} ({time.time() - t0:.0f} s)", flush=True)
        if val and (step % a.val_every == 0 or step == last_step):
            rep = E.synth_score(net, val, dev)
            rep["step"] = step
            log["val"].append(rep)
            print(f"  val step {step}: lengths {rep['len_hit']}/{rep['len_n']} (SparseTrack on the same routes "
                  f"{rep['st_len_hit']}), absences {rep['abs_ok']}/{rep['abs_n']}, onsets {rep['on_hit']}/{rep['on_n']} "
                  f"(SparseTrack {rep['st_on_hit']})", flush=True)
            if rep["len_hit"] + 0.5 * rep["on_hit"] > best:
                best = rep["len_hit"] + 0.5 * rep["on_hit"]
                save(net, out, args=vars(a), step=step, val=rep)
    save(net, out.with_name(out.stem + "_last.pt"), args=vars(a), step=last_step)
    if not val:
        save(net, out, args=vars(a), step=a.steps)
    out.with_suffix(".json").write_text(json.dumps(log, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
