"""Fine-tune the evidence network on a movie's human traces, and check whether that helps.

Human traces are sparse: a few polylines per grain, at a few bins. So only pixels the labels speak for
are supervised (``real_samples``):
- tube along a trace, and background in a band just past the tube's walls, whose extent is measured
  on the image (real tubes are wider than synthetic ones); nothing further out, where untraced tubes
  may lie, and nothing near another traced tube;
- since tubes only grow, also bins between and before the traces: before onset the grain's future
  path is background, and between two traces the tube reaches at least the earlier one and at most
  the later one;
- a ring round each grain is background away from its own tube, and a "no tube" answer makes the
  whole ring background;
- the tip target is the apex of full traces.
Everywhere else the network is held to what it predicted before (distillation). Synthetic shards,
when given, are mixed into every batch.

Scoring a tuned model on the traces it was tuned on is in-sample, and so is scoring it at the
frames it was tuned on: tuned on traces from a few frames, a network reads other grains better in
exactly those frames and worse elsewhere. So grains and traced bins are both split into folds; each
fold's grains are read at that fold's bins by a model tuned without those grains and without any
label within ``GUARD_BINS`` of those bins, and the starting model is scored on the same traces.
Then a final model is tuned on every label. It is written as ``unet_ft.pt``, the name the launcher
looks for, only if the tuned models read more lengths or onsets right and neither fewer. That is the
model to freeze and score once on movie 2; it belongs to the imaging conditions it was tuned on.

    python -m prototypes.learned_evidence.finetune --field runs/sparsetrack/ld \
        --labels benchmark/labels/ld_v1.json --work runs/learned_evidence/ld_ft
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import TIP_SIGMA, CacheView, tip_heatmap
from .model import UNet, best_device
from .train import augment, load_shards, losses

# tube within POS_PX of a trace; background in a NEG_BAND-wide band starting past the tube's walls, measured on
# the image (at least NEG_MIN px out); in between (walls, blurred tip) nothing is supervised
POS_PX, NEG_MIN, NEG_BAND = 2.0, 7.0, 7.0
RING_PX = 15.0  # the ring round a grain that holds nothing but its own tube, from its rim + 2 px out to rim + this
GUARD_BINS = 3  # no label within this many bins of a held-out bin is used for tuning
SHIPPED = Path(__file__).parent / "models" / "unet_v2_sample_field.pt"


def assign_folds(labels: dict, folds: int) -> tuple[dict[str, int], dict[int, int]]:
    """Grain folds and bin folds for the cross-validated score. Labelled grains are dealt
    round-robin in id order; traced bins go, most traced first, to the fold with the fewest traces
    so far. Fold k's grains are scored only at fold k's bins, by a model tuned without those
    grains and without any label within ``GUARD_BINS`` of those bins. Both are needed: tuned on
    traces from a few bins, a network reads other grains better in exactly those bins (it learns
    the bins themselves) and not elsewhere, so a split by grain alone flatters it."""
    grain = {gid: i % folds for i, gid in enumerate(sorted(labels.get("labels", {})))}
    counts: dict[int, int] = {}
    for lab in labels.get("labels", {}).values():
        for key, tr in (lab.get("traces") or {}).items():
            if tr.get("state") in ("full", "partial"):
                counts[int(tr.get("bin", key))] = counts.get(int(tr.get("bin", key)), 0) + 1
    load, bins = [0] * folds, {}
    for b, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        k = int(np.argmin(load))
        bins[b], load[k] = k, load[k] + n
    return grain, bins


def fold_labels(labels: dict, grain: dict[str, int], bins: dict[int, int]) -> dict:
    """The labels as the cross-validated score reads them: each grain's onset, and its traces at
    the bins of its own fold only."""
    out = {**labels, "labels": {}}
    for gid, lab in labels.get("labels", {}).items():
        k = grain.get(gid)
        out["labels"][gid] = {**lab, "traces": {key: tr for key, tr in (lab.get("traces") or {}).items()
                                                 if bins.get(int(tr.get("bin", key))) == k}}
    return out


def _traces(labels: dict, grains: set[str]):
    for gid in sorted(grains):
        for key, tr in sorted((labels["labels"].get(gid, {}).get("traces") or {}).items(), key=lambda kv: int(kv[0])):
            yield gid, int(tr.get("bin", key)), tr


def _dist(paths: list[np.ndarray], ox: float, oy: float, size: int) -> np.ndarray:
    """Distance (px) from every crop pixel to the nearest of ``paths`` (reference-frame points).
    The labelling tool's reference x is ``Renderer.crop``'s: crop pixel j covers [ox + j, ox + j + 1)."""
    line = np.zeros((size, size), np.uint8)
    for pts in paths:
        q = np.round((np.asarray(pts, np.float64) - [ox + 0.5, oy + 0.5]) * 4).astype(np.int64)
        if np.abs(q).max() > 2 ** 28:
            continue
        q = q.astype(np.int32)
        if len(q) == 1:
            cv2.circle(line, (int(q[0, 0]), int(q[0, 1])), 0, 1, -1, shift=2)
        else:
            cv2.polylines(line, [q.reshape(-1, 1, 2)], False, 1, 1, shift=2)
    if not line.any():
        return np.full((size, size), np.inf, np.float32)
    return cv2.distanceTransform(1 - line, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def _near(marks: list[tuple[np.ndarray, float]], ox: float, oy: float, size: int) -> np.ndarray:
    """Crop pixels within each mark's margin of its points (other traced tubes and their walls)."""
    out = np.zeros((size, size), np.uint8)
    for pts, margin in marks:
        q = np.round((np.asarray(pts, np.float64) - [ox + 0.5, oy + 0.5]) * 4)
        pad = 4 * (margin + 2)
        if np.abs(q).max() > 2 ** 28 or (q.max(axis=0) < -pad).any() or (q.min(axis=0) > 4 * size + pad).any():
            continue
        q = q.astype(np.int32)
        if len(q) == 1:
            cv2.circle(out, (int(q[0, 0]), int(q[0, 1])), int(4 * margin), 1, -1, shift=2)
        else:
            cv2.polylines(out, [q.reshape(-1, 1, 2)], False, 1, int(2 * margin + 1), shift=2)
    return out.astype(bool)


def neg_start(x: np.ndarray, d: np.ndarray, mask: np.ndarray) -> float:
    """Distance from a trace where background starts: 2 px past the outermost 1 px shell round the
    trace whose median after - before change is still above 20% of the peak (over the level far
    out), at least ``NEG_MIN``. The outermost, not the first below: across a bright-cored tube the
    change passes through zero between the core and the dark walls. ``x`` (n, 3, h, w) crops, ``d``
    their distances to the trace, ``mask`` the pixels to read (beside it, off grains and other tubes)."""
    change = np.abs(x[:, 2].astype(np.float32) - x[:, 1].astype(np.float32))
    shell = np.floor(d).astype(int)
    prof = {}
    for k in range(30):
        v = change[mask & (shell == k)]
        if len(v) >= 8:
            prof[k] = float(np.median(v))
    far = [prof[k] for k in range(22, 30) if k in prof]
    near = [prof[k] for k in range(20) if k in prof]
    if not far or not near or max(near) - np.median(far) < 0.1:  # tube not seen: keep the minimum
        return NEG_MIN
    level = float(np.median(far)) + 0.2 * (max(near) - float(np.median(far)))
    outer = max(k for k in range(20) if prof.get(k, -1.0) > level)
    return float(np.clip(outer + 1 + 2.0, NEG_MIN, 22.0))


def _beyond(pts: np.ndarray, ox: float, oy: float, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Crop pixels past either end of a trace (before its exit point, past its apex)."""
    out = np.zeros(xx.shape, bool)
    for seq in (pts[::-1], pts):  # apex end, then exit end
        end = seq[-1]
        far = [q for q in seq[:-1] if np.hypot(*(end - q)) >= 2.0]
        t = end - (far[-1] if far else seq[0])
        t = t / max(float(np.hypot(*t)), 1e-9)
        out |= (xx - (end[0] - ox - 0.5)) * t[0] + (yy - (end[1] - oy - 0.5)) * t[1] > 0
    return out


def _centres(pts: np.ndarray, step: float, jitter: float, rng) -> list[tuple[float, float]]:
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    at = np.arange(step / 2, max(s[-1], step / 2 + 1e-6), step) if s[-1] > step else [s[-1] / 2]
    out = []
    for a in at:
        x, y = np.interp(a, s, pts[:, 0]), np.interp(a, s, pts[:, 1])
        out.append((x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter)))
    return out


def _length(pts: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0


def _cut(pts: np.ndarray, s0: float) -> np.ndarray | None:
    """The part of a polyline from arclength ``s0`` on (None if it is not that long)."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    if s0 >= s[-1]:
        return None
    i = max(1, int(np.searchsorted(s, s0, side="right")))
    a = (s0 - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
    return np.vstack([pts[i - 1] + a * (pts[i] - pts[i - 1]), pts[i:]])


def last_absent_bin(labels: dict, gid: str, fpb: int) -> int | None:
    on = labels["labels"].get(gid, {}).get("onset") or {}
    if on.get("verdict") != "emerged_within":
        return None
    if on.get("last_absent_bin") is not None:
        return int(on["last_absent_bin"])
    return int(on["last_absent_frame"]) // fpb if on.get("last_absent_frame") is not None else None


def real_samples(field: str | Path, labels: dict, grains: set[str] | None = None, half: int = 48,
                 step: float = 24.0, jitter: float = 12.0, seed: int = 0, between: int = 8,
                 exclude_bins=()) -> dict[str, np.ndarray]:
    """Training crops from the labels of ``grains`` (default: every labelled grain): input ``x``,
    targets ``body`` and ``tip``, and where each is supervised (``w``, ``wt``). Only these grains'
    traces are read, also for keeping background away from other traced tubes. No crop is taken in
    ``exclude_bins``; traces there still tell where a tube runs and bound the bins around them.

    A trace says more than where the tube is in its own bin, since tubes only grow:
    - in its bin, tube along it and background beside it, and past it along the grain's longest
      traced path (where the tube will grow, but has not yet);
    - every ``between`` bins between two traces, tube where both traces agree up to the earlier
      one's length, and background beside the later one and along the path past its length;
    - before the grain's onset (the last bin marked absent, or a "no tube" answer), background
      along the whole path;
    - in every one of these bins, background in a ring just outside the grain, except near its own
      traces (a grain grows one tube): trained on faint tubes alone, the network starts calling faint
      structures on grain rims tubes, and germinations come out at the start of the movie.
    So supervision is spread over the movie, not only the traced bins: trained on those alone, the
    network learns them in particular and reads other bins worse."""
    view = CacheView(field)
    census = labels["grains"]
    use = {g for g in (grains if grains is not None else labels.get("labels", {}))
           if g in census and not census[g].get("excluded")}
    exclude = set(exclude_bins)
    rng = np.random.default_rng(seed)
    size = 2 * half
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    lo_x, hi_x, lo_y, hi_y = half + 4, view.r.width - half - 4, half + 4, view.r.height - half - 4
    traced = [(gid, b, tr, np.asarray(tr.get("path_xy_ref") or [], np.float64)) for gid, b, tr in _traces(labels, use)]

    def crop(gid, b, cx, cy, others):
        g = census[gid]
        cx, cy = float(np.clip(round(cx), lo_x, hi_x)), float(np.clip(round(cy), lo_y, hi_y))
        x = view.sample(b, cx, cy, half)
        own = np.hypot(xx - (g["x"] - cx + half - 0.5), yy - (g["y"] - cy + half - 0.5)) < g["r"] + 1.0
        return (x, cx, cy, own, _near(others, cx - half, cy - half, size)) if np.isfinite(x).all() else None

    # how far out background starts beside each trace, from the image at the trace's middle
    start = {}
    seen = [(gid, b, pts) for gid, b, tr, pts in traced if tr.get("state") in ("full", "partial", "unsure") and len(pts)]
    rough = [(gid, pts, NEG_MIN) for gid, _, pts in seen]
    for gid, b, tr, pts in traced:
        if tr.get("state") in ("full", "partial") and len(pts) >= 2:
            c = crop(gid, b, *_centres(pts, 1e9, 0.0, rng)[0], [(p, m) for o, p, m in rough if o != gid])
            if c is None:
                start[(gid, b)] = NEG_MIN
                continue
            x, cx, cy, own, near = c
            d = _dist([pts], cx - half, cy - half, size)
            start[(gid, b)] = neg_start(x[None], d[None], (~_beyond(pts, cx - half, cy - half, xx, yy) & ~own & ~near)[None])
    marks = [(gid, pts, start.get((gid, b), NEG_MIN)) for gid, b, pts in seen]
    out = {k: [] for k in ("x", "body", "tip", "w", "wt")}
    info, widths = [], []

    def keep(x, body, tip, w, wt, where):
        out["x"].append(x.astype(np.float16))
        out["body"].append(body.astype(np.uint8))
        out["tip"].append(tip.astype(np.float16))
        out["w"].append(w.astype(np.uint8))
        out["wt"].append(wt.astype(np.uint8))
        info.append(where)

    zero = np.zeros((size, size), np.float32)
    for gid in sorted(use):
        g = census[gid]
        mine = [(b, tr, pts) for o, b, tr, pts in traced if o == gid]
        others = [(p, m) for o, p, m in marks if o != gid]
        for b, tr, _ in mine:  # a "no tube" answer: a ring round the grain is background
            if tr.get("state") != "no_tube" or b in exclude:
                continue
            c = crop(gid, b, g["x"] + rng.uniform(-jitter, jitter), g["y"] + rng.uniform(-jitter, jitter), others)
            if c is not None:
                x, cx, cy, own, near = c
                ring = np.hypot(xx - (g["x"] - cx + half - 0.5), yy - (g["y"] - cy + half - 0.5))
                w = (ring >= g["r"] + 2.0) & (ring <= g["r"] + RING_PX) & ~near & ~own
                keep(x, zero, zero, w, w, (gid, b, cx, cy))
        burst = min([b for b, tr, _ in mine if tr.get("state") == "burst"], default=None)
        tubes = [(b, tr, pts) for b, tr, pts in mine if tr.get("state") in ("full", "partial") and len(pts) >= 2
                 and (burst is None or b < burst)]
        if not tubes:
            continue
        widths += [start[(gid, b)] for b, _, _ in tubes]
        path = max((pts for _, _, pts in tubes), key=_length)  # where the tube goes
        plan = [(b, i, i) for i, (b, _, _) in enumerate(tubes)]
        for i in range(len(tubes) - 1):
            if tubes[i + 1][1].get("state") == "full":  # a partial trace is no upper bound
                plan += [(b, i, i + 1) for b in range(tubes[i][0] + between, tubes[i + 1][0] - between // 2, between)]
        pre = {b for b, tr, _ in mine if tr.get("state") == "no_tube" and b < tubes[0][0]}
        absent = last_absent_bin(labels, gid, view.fpb)
        if absent is not None and view.rs + 3 <= min(absent, tubes[0][0] - 1):
            top = min(absent, tubes[0][0] - 1)
            pre |= {top, (view.rs + 3 + top) // 2}
        plan += [(b, None, None) for b in sorted(pre)]
        for b, lo, hi in sorted(p for p in plan if p[0] not in exclude and (burst is None or p[0] < burst)):
            here = lo is not None and lo == hi
            p_lo, p_hi = (tubes[lo][2] if lo is not None else None), (tubes[hi][2] if hi is not None else None)
            n_lo = start[(gid, tubes[lo][0])] if lo is not None else 0.0
            n_hi = start[(gid, tubes[hi][0])] if hi is not None else 0.0
            partial = here and tubes[lo][1].get("state") == "partial"
            # the known path past where the tube can have reached by now (none past a partial trace)
            future = None if partial else _cut(path, _length(p_hi) + n_hi if p_hi is not None else 3.0)
            for centre in _centres(path, step, jitter, rng):
                c = crop(gid, b, *centre, others)
                if c is None:
                    continue
                x, cx, cy, own, near = c
                ox, oy = cx - half, cy - half
                d_lo = _dist([p_lo], ox, oy, size) if p_lo is not None else None
                d_hi = d_lo if here else _dist([p_hi], ox, oy, size) if p_hi is not None else None
                body = np.zeros((size, size), bool)
                neg = np.zeros((size, size), bool)
                if p_lo is not None:
                    # tube beside the trace, not in the round caps past its ends: past the apex the front is
                    # blurred, and labelling it tube teaches the network to read every tube ~1.5 px long
                    body = (d_lo <= POS_PX) & ~_beyond(p_lo, ox, oy, xx, yy)
                    if not here:  # between two traces: only where the later one agrees (the tube can sway)
                        body &= d_hi <= POS_PX + 1.5
                if p_hi is not None:
                    neg = (d_hi >= n_hi) & (d_hi <= n_hi + NEG_BAND)
                    if partial:  # the tube may go on past a partial trace's last point
                        neg &= np.hypot(xx - (p_hi[-1, 0] - ox - 0.5), yy - (p_hi[-1, 1] - oy - 0.5)) > n_hi + NEG_BAND
                if future is not None:
                    neg |= _dist([future], ox, oy, size) <= POS_PX + 1.0
                rim = np.hypot(xx - (g["x"] - ox - 0.5), yy - (g["y"] - oy - 0.5))
                neg |= (rim >= g["r"] + 2.0) & (rim <= g["r"] + RING_PX)
                if d_hi is not None:
                    neg &= d_hi >= n_hi
                if d_lo is not None and not here:
                    neg &= d_lo >= n_lo
                neg &= ~near & ~own
                body &= ~own
                w = body | neg
                if here and tubes[lo][1].get("state") == "full":
                    tip = tip_heatmap([(p_lo[-1, 0] - 0.5, p_lo[-1, 1] - 0.5)], cx, cy, half)
                    wt = (w | (np.hypot(xx - (p_lo[-1, 0] - ox - 0.5), yy - (p_lo[-1, 1] - oy - 0.5)) <= 3 * TIP_SIGMA)) & ~own
                else:  # no tip where the apex may be (at or past the earlier trace's end)
                    tip = zero
                    wt = neg | (body & (np.hypot(xx - (p_lo[-1, 0] - ox - 0.5), yy - (p_lo[-1, 1] - oy - 0.5)) > 6.0)
                                if p_lo is not None else neg)
                keep(x, body, tip, w, wt, (gid, b, cx, cy))
    if not info:
        raise SystemExit("no usable traces in the labels")
    return {**{k: np.stack(v) for k, v in out.items()}, "info": info, "neg_start": widths}


def _wbce(logit, target, w, pos_weight=None):
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=logit.device)
    loss = F.binary_cross_entropy_with_logits(logit, target, reduction="none", pos_weight=pw)
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def real_loss(logits, body, tip, w, wt, teacher=None, distill: float = 0.5):
    """Supervised where the traces say, as ``train.losses`` weighs synthetic pixels (tube pixels x2,
    tips x(1 + 10 tip)); elsewhere, when a ``teacher`` output is given, held to it."""
    lb, lt = logits[:, 0], logits[:, 1]
    loss = _wbce(lb, body, w, pos_weight=2.0) + 0.5 * _wbce(lt, tip, wt * (1.0 + 10.0 * tip))
    if teacher is not None and distill > 0:
        tb, tt = torch.sigmoid(teacher[:, 0]), torch.sigmoid(teacher[:, 1])
        loss = loss + distill * (_wbce(lb, tb, 1.0 - w) + 0.5 * _wbce(lt, tt, 1.0 - wt))
    return loss


def _batch(real, i, g, crop):
    t = {k: torch.from_numpy(real[k][i].astype(np.float32)) for k in ("x", "body", "tip", "w", "wt")}
    # the same crop and turns for the input, the targets and where they are supervised
    x, bw, tw = augment(t["x"], torch.stack([t["body"], t["w"]], 1), torch.stack([t["tip"], t["wt"]], 1), g, crop=crop)
    return x, bw[:, 0], tw[:, 0], bw[:, 1], tw[:, 1]


@torch.no_grad()
def check(net, real, device, n: int = 512) -> str:
    """Supervised pixels of held-out crops read right (P > 0.5 on tube, < 0.5 on background)."""
    net.eval()
    tp = tn = npos = nneg = 0
    for s in range(0, min(n, len(real["x"])), 64):
        x = torch.from_numpy(real["x"][s:s + 64].astype(np.float32)).to(device)
        p = torch.sigmoid(net(x)[:, 0]).cpu().numpy()
        body, w = real["body"][s:s + 64].astype(bool), real["w"][s:s + 64].astype(bool)
        tp += int(((p > 0.5) & body & w).sum())
        npos += int((body & w).sum())
        tn += int(((p <= 0.5) & ~body & w).sum())
        nneg += int((~body & w).sum())
    net.train()
    return f"held-out traces: tube {100 * tp / max(npos, 1):.1f}%, background {100 * tn / max(nneg, 1):.1f}%"


def tune(start: str | Path, real: dict, syn: dict | None = None, steps: int = 1500, batch: int = 16,
         lr: float = 3e-4, distill: float = 0.5, device=None, seed: int = 0, val: dict | None = None,
         log=print) -> UNet:
    device = torch.device(device or best_device())
    ck = torch.load(start, map_location="cpu", weights_only=False)
    net = UNet(widths=tuple(ck.get("widths") or (16, 32, 64, 128)))
    net.load_state_dict(ck["state"])
    net.to(device)
    teacher = copy.deepcopy(net).eval() if distill > 0 else None
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    if val is not None:
        log(f"  before: {check(net, val, device)}")
    net.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        x, body, tip, w, wt = (t.to(device) for t in _batch(real, rng.integers(len(real["x"]), size=batch), g, 64))
        n = len(x)
        if syn is not None:  # one forward pass over both, so batch norm sees one mixed batch
            j = rng.integers(len(syn["x"]), size=batch)
            xs, bs, ts = augment(*(torch.from_numpy(syn[k][j].astype(np.float32)) for k in ("x", "body", "tip")),
                                 g, crop=64)
            x = torch.cat([x, xs.to(device)])
        logits = net(x)
        with torch.no_grad():
            t_out = teacher(x[:n]) if teacher is not None else None
        loss = real_loss(logits[:n], body, tip, w, wt, t_out, distill)
        if syn is not None:
            loss = loss + losses(logits[n:], bs.to(device), ts.to(device))["total"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0 or step == steps:
            msg = f"  step {step:5d} {time.time() - t0:5.0f}s loss {loss.item():.4f}"
            log(msg + (f"; {check(net, val, device)}" if val is not None else ""))
    return net.eval()


def save(net: UNet, path: Path, **info) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": {k: v.cpu() for k, v in net.state_dict().items()}, "widths": net.widths,
                "args": info}, path)
    return path


def speed_cap(pcache: Path, field: Path, grains_path: Path) -> float:
    """SparseTrack's automatic front speed cap on this evidence, as the pipeline's learned run sets it
    (the per-bin decoder takes it from there)."""
    from sparsetrack import analyze as A
    from sparsetrack import stack
    from sparsetrack.render import Renderer

    from .evaluate import adaptive_crop, image_registration

    p = A.Params(settle=False, onset_source="front")
    bins, meta = stack.load(pcache)
    doc = json.loads(Path(grains_path).read_text())
    census = list(doc["grains"].values()) if isinstance(doc["grains"], dict) else doc["grains"]
    grains = [g for g in census if not g.get("excluded")]
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    with adaptive_crop(), image_registration(field):  # the pipeline's order
        scale = A.growth_scale(Renderer(bins, meta), meta, grains, p, lambda *a: None, physical)
    return float(np.clip(p.vmax_factor * scale, p.vmax_px, p.vmax_cap)) if scale is not None else p.vmax_px


def read_perbin(net, field: Path, labels_path: Path, work: Path, tag: str, keep_cache: bool = False,
                log=print, **decoder) -> dict:
    """The pipeline's per-bin run (learned evidence, grown crop, burst-aware fit) with this network;
    ``decoder`` holds settings from ``calibrate.py``."""
    from . import evaluate, reach

    pcache = evaluate.prob_cache(field, net, work / f"prob_{tag}", log=log)
    try:
        vmax = speed_cap(pcache, field, labels_path)
        return reach.analyze(pcache, field, grains_path=labels_path, log=lambda *a: None, big=300, burst=True,
                             vmax=vmax, **decoder)
    finally:
        if not keep_cache:
            shutil.rmtree(pcache, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", required=True, help="prepared cache of the labelled movie")
    ap.add_argument("--labels", required=True, help="that movie's human labels (never movie 2)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default=None,
                    help="starting model (default: the dev movie's runs/learned_evidence/ld/unet.pt if it exists, "
                         "else the shipped one)")
    ap.add_argument("--synthetic", nargs="*", default=[], help="synthetic shard globs mixed into every batch")
    ap.add_argument("--folds", type=int, default=3,
                    help="folds of grains and of traced bins for the cross-validated score (0: skip it, and tune "
                         "and keep the final model)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16, help="real crops per batch (as many synthetic ones again)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--distill", type=float, default=0.5, help="weight holding unsupervised pixels to the start")
    ap.add_argument("--decoder", default=None,
                    help="per-bin decoder settings from calibrate.py for the check's readings (default: the adopted "
                         "runs/learned_evidence/ld_cal/decoder.json if it exists, as the launcher uses it; 'none' for "
                         "the default offset)")
    ap.add_argument("--no-final", action="store_true", help="only the cross-validated score")
    ap.add_argument("--keep-caches", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    field, work, labels_path = Path(args.field), Path(args.work), Path(args.labels)
    if "m2" in labels_path.name:
        raise SystemExit("movie 2 is the held-out benchmark: never fine-tune on it")
    labels = json.loads(labels_path.read_text())
    start = Path(args.model) if args.model else next(
        p for p in (Path("runs/learned_evidence/ld/unet.pt"), SHIPPED) if p.exists())
    work.mkdir(parents=True, exist_ok=True)
    log_file = (work / "log.txt").open("a")

    def log(*a):
        print(*a, flush=True)
        print(*a, file=log_file, flush=True)

    from .calibrate import ADOPTED, decoder_settings
    if args.decoder is None and ADOPTED.exists():  # check the model with the decoder it will be used with
        args.decoder = str(ADOPTED)
    args.decoder = None if args.decoder == "none" else args.decoder
    dec = decoder_settings(args.decoder, start, log=log)
    if dec:
        log(f"per-bin decoder settings from {args.decoder}: {dec}")
    syn = load_shards(args.synthetic) if args.synthetic else None
    log(f"{labels_path.name}: starting from {start}; synthetic samples {len(syn['x']) if syn else 0}")
    kw = dict(syn=syn, steps=args.steps, batch=args.batch, lr=args.lr, distill=args.distill, device=args.device,
              seed=args.seed, log=log)
    info = dict(labels=str(labels_path), field=str(field), started_from=str(start),
                **{k: getattr(args, k) for k in ("steps", "batch", "lr", "distill", "seed", "synthetic", "decoder")})
    if args.folds > 1:
        from sparsetrack.evaluate import score

        from . import evaluate
        from .model import load as load_model

        grain, bins = assign_folds(labels, args.folds)
        scored = fold_labels(labels, grain, bins)
        all_bins = set(range(int(labels.get("n_bins") or 10 ** 4)))
        cv = {}
        for k in range(args.folds):
            held = {gid for gid, f in grain.items() if f == k}
            held_bins = {b for b, f in bins.items() if f == k}
            guard = {b + o for b in held_bins for o in range(-GUARD_BINS, GUARD_BINS + 1)}
            train = real_samples(field, labels, set(grain) - held, seed=args.seed + k, exclude_bins=guard)
            val = real_samples(field, labels, held, jitter=0.0, seed=args.seed + k,
                               exclude_bins=all_bins - held_bins)
            log(f"fold {k + 1}/{args.folds}: {len(held)} grains held out, scored at bins "
                f"{', '.join(map(str, sorted(held_bins)))}; tuning on {len(train['x'])} crops from the other "
                f"{len(set(grain) - held)} grains, none within {GUARD_BINS} bins of those")
            net = tune(start, train, val=val, **kw)
            save(net, work / f"fold{k}" / "unet.pt", **info, fold=k, folds=args.folds, held_out=sorted(held),
                 held_out_bins=sorted(held_bins))
            pred = read_perbin(net, field, labels_path, work, f"fold{k}", args.keep_caches, log, **dec)
            cv.update({r["id"]: r for r in pred["grains"] if r["id"] in held})
        base = read_perbin(load_model(str(start)), field, labels_path, work, "start", args.keep_caches, log, **dec)
        tuned = {**base, "grains": [cv.get(r["id"], r) for r in base["grains"]],
                 "method": f"per-bin decoder, fine-tuned ({args.folds}-fold cross-validated)"}
        (work / "perbin_start.json").write_text(json.dumps(base))
        (work / "perbin_cv.json").write_text(json.dumps(tuned))
        (work / "labels_cv.json").write_text(json.dumps(scored))
        rs, rt = score(scored, base), score(scored, tuned)
        pb = evaluate.paired_bootstrap(rt, rs, rs["onset"]["tolerance_frames"])
        # better on lengths or onsets, and worse on neither: onsets make the germination curve
        adopt = min(pb["length_diff"], pb["onset_diff"]) >= 0 and max(pb["length_diff"], pb["onset_diff"]) > 0
        lines = [f"{labels_path.name}: {rs['grains_scored']} grains scored, each at its fold's traced bins by a model "
                 f"tuned without that grain and without any label within {GUARD_BINS} bins of those bins",
                 evaluate.e2e_summary("per-bin, start", rs), evaluate.e2e_summary("per-bin, tuned", rt),
                 f"tuned - start over {pb['grains']} grains: onset {pb['onset_diff']:+.0f} "
                 f"[{pb['onset_ci'][0]:+.0f}, {pb['onset_ci'][1]:+.0f}], lengths {pb['length_diff']:+.0f} "
                 f"[{pb['length_ci'][0]:+.0f}, {pb['length_ci'][1]:+.0f}] (95% paired bootstrap over grains)",
                 "adopted: the tuned model reads more lengths or onsets right than the start, and neither fewer"
                 if adopt else "not adopted: the tuned model does not read more lengths or onsets right than the start "
                 "without reading fewer of the other"]
        log("\n".join(lines))
        (work / "report.txt").write_text("\n".join(lines) + "\n")
        (work / "scores.json").write_text(json.dumps({"start": rs, "tuned_cv": rt, "paired": pb, "adopted": adopt},
                                                     default=str, indent=1))
    else:
        adopt = True
    if not args.no_final:
        real = real_samples(field, labels, seed=args.seed)
        log(f"final model: tuning on {len(real['x'])} crops from every labelled grain")
        # only an adopted model takes the name the launcher looks for
        out = save(tune(start, real, **kw), work / ("unet_ft.pt" if adopt else "unet_ft_not_adopted.pt"), **info)
        log(f"saved {out}" + (f": freeze it, then score it once on movie 2 (pipeline --model {out} --heldout-once)"
                              if adopt else ": the cross-validated score says keep the start model"))
    log_file.close()


if __name__ == "__main__":
    main()
