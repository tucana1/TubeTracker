"""Fine-tune the tube network from the shipped checkpoint on a fold's movie-1 crops plus replay of its own training data.

    python -m prototypes.tube_adapt.finetune --name k2f0_b --variant b --fold k2f0 [--grains g009 g011 ...]

Variants (each batch of 32; replay = 0.8.0's own training sets: ld + m2 trace crops v3 and a subset of the v5/v6
synthetic shards, ``common.REAL_LDM2``/``V5``/``V6``):

- ``a``: movie 1's traced crops (``trace_<fold>``: trace +/- 2 bins, exits, negatives) 50%, replay 50%;
- ``b``: traced crops 20%, propagated tube crops (between and after traces) 20%, propagated negatives before onset 10%,
  replay 50%;
- ``c``: as b, plus skeleton recall (arXiv 2404.03010): 1 - sum(P * S) / sum(S) over the tubed skeleton S of the
  labelled body (centreline, dilated 1 px, inside the body), weight ``--skel-weight``;
- ``ctrl``: replay only (no movie-1 labels): the same schedule, to see what fine-tuning alone does.

Losses are 0.8.0's (BCE with pos_weight 2 + Dice + 0.5 tip, unscored pixels weight 0). BatchNorm statistics are
frozen at the checkpoint's (``--bn frozen``, default): full-frame inference keeps the shipped normalisation and only
the weights adapt. Inputs and checkpoint options (BatchNorm, ``bg_px`` 96) are the checkpoint's.
``--grains`` restricts the movie-1 crops to those grains (learning curve; the traced shard must be built for them:
``crops.py --grains ... --name NAME``, then ``--trace-name NAME``).
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prototypes.learned_flood.train import augment, losses
from prototypes.tube_adapt.common import BASE_NET, OUT, REAL_LDM2, REPO, V5, V6
from sparsetrack.learned import _unet


def skeleton(body: np.ndarray) -> np.ndarray:
    """Tubed skeleton of each body mask: its skeleton dilated by 1 px, inside the body."""
    from skimage.morphology import skeletonize
    import cv2
    out = np.zeros_like(body, dtype=np.uint8)
    k = np.ones((3, 3), np.uint8)
    for i in range(len(body)):
        if body[i].any():
            out[i] = cv2.dilate(skeletonize(body[i] > 0).astype(np.uint8), k) & (body[i] > 0)
    return out


def load(patterns: list[str], kinds: set | None = None, grains: set | None = None) -> dict:
    files = sorted({f for p in patterns for f in glob.glob(str(REPO / p) if not p.startswith("/") else p)})
    if not files:
        raise SystemExit(f"no shards match {patterns}")
    parts = []
    for f in files:
        z = np.load(f)
        keep = np.ones(len(z["body"]), bool)
        if kinds is not None and "kind" in z.files:
            keep &= np.isin(z["kind"], list(kinds))
        if grains is not None and "grain" in z.files:
            keep &= np.isin(z["grain"], list(grains))
        d = {k: z[k][keep] for k in ("x", "body", "tip")}
        d["w"] = z["w"][keep] if "w" in z.files else np.ones(d["body"].shape, np.uint8)
        d["skel"] = z["skel"][keep] if "skel" in z.files else None
        parts.append(d)
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("x", "body", "tip", "w")}
    out["x"] = out["x"].astype(np.float16, copy=False)
    out["tip"] = out["tip"].astype(np.float16, copy=False)
    sk = [p["skel"] if p["skel"] is not None else skeleton(p["body"]) for p in parts]
    out["skel"] = np.concatenate(sk)
    out["files"] = files
    return out


def batch(d: dict, idx: np.ndarray):
    return tuple(torch.from_numpy(d[k][idx].astype(np.float32)) for k in ("x", "body", "tip", "w", "skel"))


def skel_recall(logits: torch.Tensor, skel: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits[:, 0])
    s = skel * (w > 0)
    return 1.0 - (p * s).sum() / (s.sum() + 1.0)


def sets_for(variant: str, fold: str, trace_name: str | None, grains: set | None) -> list[tuple[str, dict, float]]:
    tr = str(OUT / "shards" / f"trace_{trace_name or fold}.npz")
    pr = str(OUT / "shards" / f"prop_{fold}.npz")
    replay = [("real ld+m2", load(REAL_LDM2.split(",")), 0.25), ("v5", load(V5.split(",")), 0.125),
              ("v6", load(V6.split(",")), 0.125)]
    if variant == "ctrl":
        return [(n, d, 2 * s) for n, d, s in replay]
    if variant == "a":
        return [("m1 traced", load([tr]), 0.5)] + replay
    if variant in ("b", "c"):
        return [("m1 traced", load([tr]), 0.2),
                ("m1 propagated", load([pr], {"between", "after"}, grains), 0.2),
                ("m1 negatives", load([pr], {"negative"}, grains), 0.1)] + replay
    raise SystemExit(f"unknown variant {variant}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", choices=("a", "b", "c", "ctrl"), required=True)
    ap.add_argument("--fold", default="k2f0")
    ap.add_argument("--trace-name", help="traced shard name if not the fold's (learning curve)")
    ap.add_argument("--grains", nargs="*", help="only these grains' propagated crops (learning curve)")
    ap.add_argument("--init", default=str(BASE_NET))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--skel-weight", type=float, default=1.0)
    ap.add_argument("--bn", choices=("frozen", "train"), default="frozen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    sets = sets_for(a.variant, a.fold, a.trace_name, set(a.grains) if a.grains else None)
    tot = sum(s for _, _, s in sets)
    sets = [(n, d, s / tot) for n, d, s in sets]
    for n, d, s in sets:
        print(f"{100 * s:4.1f}% of each batch: {n}, {len(d['x'])} samples ({', '.join(Path(f).name for f in d['files'])});"
              f" tube {100 * d['body'].mean():.1f}%, scored {100 * d['w'].mean():.0f}%", flush=True)
    counts = [int(round(a.batch * s)) for _, _, s in sets]
    counts[-1] = a.batch - sum(counts[:-1])
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    net = _unet(ck.get("widths", (16, 32, 64, 128)), ck.get("norm", "group"))
    net.load_state_dict(ck["state"])
    dev = torch.device(a.device)
    net = net.to(dev).train()

    def bn_mode():
        if a.bn == "frozen":
            for m in net.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
    bn_mode()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    use_skel = a.variant == "c"
    started, run = time.time(), {}
    for step in range(1, a.steps + 1):
        parts = [batch(d, rng.integers(len(d["x"]), size=n)) for (_, d, _), n in zip(sets, counts) if n > 0]
        xb, bb, tb, wb, sb = (torch.cat(t) for t in zip(*parts))
        x, b2, tip, w = augment(xb, torch.stack([bb, sb], 1), tb, g, a.crop, wb)
        x, body, skel, tip, w = (t.to(dev) for t in (x, b2[:, 0], b2[:, 1], tip, w))
        logits = net(x)
        L = losses(logits, body, tip, w)
        L["skel"] = skel_recall(logits, skel, w)
        total = L["total"] + (a.skel_weight * L["skel"] if use_skel else 0.0)
        opt.zero_grad()
        total.backward()
        opt.step()
        sched.step()
        for k, v in L.items():
            run[k] = run.get(k, 0.0) + float(v.detach())
        if step % 250 == 0 or step == a.steps:
            print(f"step {step:5d} {time.time() - started:6.0f}s  " + " ".join(f"{k} {v / 250:.4f}" for k, v in run.items()),
                  flush=True)
            run = {}
    out = OUT / "models" / f"{a.name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "widths": ck.get("widths", (16, 32, 64, 128)),
                "norm": ck.get("norm", "group"), "bg_px": ck.get("bg_px", 0), "args": vars(a)}, out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
