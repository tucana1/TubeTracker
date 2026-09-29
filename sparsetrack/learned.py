"""Learned tube evidence and the arrival-flood reader, for crowded fields.

A small U-Net reads each registered bin with the movie's "before" and "after" images and
outputs P(tube body) per pixel. The shipped model (``models/tubes_synth_v1.pt``) was trained
only on codec-exact synthetic movies built on the real fields of the dev movie and movie 2
(no human labels), so both human benchmarks stay held out for it.

The flood reads one grain's tube from those maps: every pixel gets an arrival bin (the first
bin from which it stays tube), and the tube grows in arrival order from the rim. A newly
arrived piece joins only if it touches the tube's most recently joined pixels (its tip), so
material that was there first (a foreign tube, a crossing) or that appears beside old tube
(sway) is never claimed. Length at a bin = geodesic distance from the rim to the farthest
pixel claimed by then.

On the human benchmarks (27 Sep 2026) the flood beat the change reader on crowded movie 2
(22 vs 9 of 54 FULL traces in tolerance, paired +13, 95% CI +4 to +23) and lost on the sparse
dev movie (38 vs 61 of 104). ``Params.reader = "hybrid"`` uses it only for grains whose change
region touches a neighbour or whose background change lifts the map threshold above its
floor (keeping the change reader's onset for the latter): dev movie 61/104 (unchanged),
movie 2 23/54 (paired +14, 95% CI +5 to +23), onsets 13/26 and 6/19.

Needs torch (``pip install .[cnn]``) to build a probability movie; reading one does not.
"""

from __future__ import annotations

import heapq
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from . import stack
from .render import Renderer

MODEL = Path(__file__).parent / "models" / "tubes_synth_v1.pt"
P_SCALE = 250.0       # probability movies are stored as uint8 P x P_SCALE
IN_SCALE = 20.0       # network input: grey levels per unit, relative to the before image's median
HALO = 3.0            # the rim's own change (focus, swelling) reaches this far out


# ----------------------------------------------------------------------------- network
def _unet(widths=(16, 32, 64, 128)):
    import torch
    from torch import nn
    import torch.nn.functional as F

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.widths = tuple(widths)
            self.enc = nn.ModuleList()
            c = 3
            for w in widths:
                self.enc.append(block(c, w))
                c = w
            self.dec = nn.ModuleList(block(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(len(widths) - 1)))
            self.head = nn.Conv2d(widths[0], 2, 1)

        def forward(self, x):
            skips = []
            for i, blk in enumerate(self.enc):
                x = blk(x if i == 0 else F.max_pool2d(x, 2))
                skips.append(x)
            x = skips.pop()
            for blk in self.dec:
                s = skips.pop()
                x = blk(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
            return self.head(x)

    return UNet()


def load_model(path: str | Path = MODEL, device: str | None = None):
    import torch
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    net = _unet(ck.get("widths", (16, 32, 64, 128)))
    net.load_state_dict(ck["state"])
    return net.eval().to(device)


def tube_probability(net, img: np.ndarray, early: np.ndarray, late: np.ndarray) -> np.ndarray:
    """P(tube body) for one registered frame, given the movie's before and after images."""
    import torch
    m = float(np.nanmedian(early))
    x = np.nan_to_num((np.stack([img, early, late]).astype(np.float32) - m) / IN_SCALE, nan=0.0)
    h, w = img.shape
    k = 2 ** (len(net.widths) - 1)
    x = np.pad(x, ((0, 0), (0, (-h) % k), (0, (-w) % k)), mode="reflect")
    dev = next(net.parameters()).device
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(x)[None].to(dev)))[0, 0, :h, :w].cpu().numpy()


def _registered(bins: np.ndarray, shifts: np.ndarray, b: int) -> np.ndarray:
    dx, dy = shifts[b]
    img = np.asarray(bins[b], np.float32)
    return cv2.warpAffine(img, np.float32([[1, 0, -dx], [0, 1, -dy]]), (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def prob_cache(cache_dir: str | Path, model: str | Path = MODEL, log=print) -> Path:
    """The movie's tube-probability cache (built once, next to the image cache): a SparseTrack
    cache in reference coordinates whose bins are uint8 P(tube) x P_SCALE."""
    import hashlib
    cache_dir = Path(cache_dir)
    out = cache_dir / f"prob_{Path(model).stem}"
    sha1 = hashlib.sha1(Path(model).read_bytes()).hexdigest()
    if (out / "meta.json").exists():
        built_by = json.loads((out / "meta.json").read_text()).get("model_sha1")
        if built_by in (None, sha1):  # None: built before fingerprints were kept (the shipped model's)
            return out
        log(f"{out.name} was built by another version of {Path(model).name}: building it again")
        (out / "meta.json").unlink()  # a stopped rebuild must not look finished
    net = load_model(model)
    bins, meta = stack.load(cache_dir)
    shifts = np.asarray(meta["shifts"], np.float64)
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    early = np.mean([_registered(bins, shifts, b) for b in range(rs, rs + 3)], axis=0)
    late = np.mean([_registered(bins, shifts, b) for b in range(nb - 4, nb - 1)], axis=0)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out / "bins.npy", mode="w+", dtype=np.uint8, shape=bins.shape)
    started = time.time()
    for b in range(nb):
        arr[b] = np.round(tube_probability(net, _registered(bins, shifts, b), early, late) * P_SCALE).astype(np.uint8)
    arr.flush()
    del arr
    m = {**meta, "shifts": [[0.0, 0.0]] * nb, "raw_shifts": [[0.0, 0.0]] * nb,
         "evidence": f"P(tube) x {P_SCALE} from {Path(model).name}", "model_sha1": sha1}
    shutil.copy(cache_dir / "grains.json", out / "grains.json")
    (out / "meta.json").write_text(json.dumps(m, indent=1))  # last: a stopped build does not look finished
    log(f"tube probabilities ({Path(model).name}): {nb} bins in {time.time() - started:.0f} s -> {out}")
    return out


# ----------------------------------------------------------------------------- flood
def arrivals(present: np.ndarray, blocked: np.ndarray, persist: int = 10, frac: float = 0.7) -> np.ndarray:
    """Per pixel, the first bin from which it is present in >= ``frac`` of the next ``persist``
    bins (len(present) where never)."""
    ex = present & ~blocked[None]
    n = len(ex)
    cs = np.concatenate([np.zeros((1,) + ex.shape[1:], np.int16), np.cumsum(ex, axis=0, dtype=np.int16)])
    idx = np.arange(n)
    hi = np.minimum(idx + persist, n)
    ok = ex & ((cs[hi] - cs[idx]) / (hi - idx)[:, None, None] >= frac)
    return np.where(ok.any(axis=0), ok.argmax(axis=0), n)


def _extend_dist(dist: np.ndarray, comp: np.ndarray, sources: np.ndarray, bridge: int) -> None:
    """Rim distances for a joining component: a straight hop of <= ``bridge`` px from the tube's
    pixels (with their distances), then geodesically inside the component."""
    ys, xs = np.nonzero(comp)
    y0, y1 = max(ys.min() - bridge - 1, 0), min(ys.max() + bridge + 2, dist.shape[0])
    x0, x1 = max(xs.min() - bridge - 1, 0), min(xs.max() + bridge + 2, dist.shape[1])
    sub_d, sub_c = dist[y0:y1, x0:x1], comp[y0:y1, x0:x1]
    sy, sx = np.nonzero(np.isfinite(sub_d) & sources[y0:y1, x0:x1])
    h, w = sub_c.shape
    out = np.full((h, w), np.inf)
    pq = []
    for y, x in zip(*np.nonzero(sub_c)):
        dd = np.hypot(sy - y, sx - x)
        near = dd <= bridge + 0.5
        if near.any():
            out[y, x] = float(np.min(sub_d[sy[near], sx[near]] + dd[near]))
            pq.append((out[y, x], int(y), int(x)))
    heapq.heapify(pq)
    while pq:
        d, y, x = heapq.heappop(pq)
        if d > out[y, x]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if (dy or dx) and 0 <= yy < h and 0 <= xx < w and sub_c[yy, xx]:
                    nd = d + math.hypot(dy, dx)
                    if nd < out[yy, xx]:
                        out[yy, xx] = nd
                        heapq.heappush(pq, (nd, yy, xx))
    upd = sub_c & (out < sub_d)
    sub_d[upd] = out[upd]


def centreline(tube: np.ndarray, dist: np.ndarray, tip: tuple[int, int], centre: float, gr: float,
               reach: float = 4.5) -> list[tuple[float, float]]:
    """Rim-to-tip polyline (y, x) through a flooded tube: from the tip, back along the geodesic that
    gave its rim distance (each step to the tube pixel within ``reach`` px, the bridge, whose distance
    plus the step is the current one, taking the longest such step), then out to the rim on the ray
    through the last pixel. Its length is about the tip's rim distance, the length reported."""
    h, w = dist.shape
    y, x = tip
    out = [(float(y), float(x))]
    r = int(math.ceil(reach))
    while True:
        y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, h), max(x - r, 0), min(x + r + 1, w)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        step = np.hypot(yy - y, xx - x)
        ok = tube[y0:y1, x0:x1] & np.isfinite(dist[y0:y1, x0:x1]) & (step <= reach) & (step > 0)
        ok &= dist[y0:y1, x0:x1] + step <= dist[y, x] + 1e-3  # on a shortest path to (y, x)
        if not ok.any():
            break
        k = np.flatnonzero(ok.ravel())[int(np.argmin(dist[y0:y1, x0:x1].ravel()[ok.ravel()]))]
        y, x = int(yy.ravel()[k]), int(xx.ravel()[k])
        out.append((float(y), float(x)))
    a = math.atan2(out[-1][0] - centre, out[-1][1] - centre)
    rim = (centre + gr * math.sin(a), centre + gr * math.cos(a))
    return [rim] + out[::-1]


def from_exit(line: list[tuple[float, float]], centre: float, gr: float, zone: float) -> tuple[list, float]:
    """A flood centreline (rim point first, as ``centreline`` returns it) cut at the tube's exit: its last point,
    walking out from the rim, before it first leaves ``zone`` px of the grain centre (a tube still wholly within
    the zone: its tip). The flood may start beside the exit and hug the rim before the tube turns out; an
    annotator measures from where the tube leaves the grain, so that detour must not count (a tube curling back
    to its grain later is still measured from where it first left). Returns the cut line (a rim point on the
    exit's ray first) and its length."""
    pts = np.asarray(line[1:], float)
    rad = np.hypot(pts[:, 0] - centre, pts[:, 1] - centre)
    out = rad > zone
    k = len(pts) - 1 if not out.any() else max(int(np.argmax(out)) - 1, 0)
    a = math.atan2(pts[k][0] - centre, pts[k][1] - centre)
    cut = [(centre + gr * math.sin(a), centre + gr * math.cos(a))] + [tuple(q) for q in pts[k:]]
    arr = np.asarray(cut)
    return cut, float(np.sum(np.hypot(*np.diff(arr, axis=0).T))) if len(arr) > 1 else 0.0


def flood(arr: np.ndarray, rg: np.ndarray, ang: np.ndarray, blocked: np.ndarray, gr: float, recent: int = 12,
          bridge: int = 4, start_band: float = 4.0, min_len: float = 8.0, give_up: int = 40) -> dict:
    """Grow one grain's tube through an arrival map (see the module docstring).

    Returns the tube mask, each tube pixel's arrival bin and rim distance, the emergence bin
    and the length per bin.
    """
    n = int(arr.max())
    blocked = blocked | (rg < gr + HALO)
    start = (rg >= gr + HALO) & (rg <= gr + HALO + start_band)  # the map may miss the px next to the rim
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge + 1, 2 * bridge + 1))
    tube = np.zeros(arr.shape, bool)
    t_in = np.full(arr.shape, -1, np.int32)
    dist = np.full(arr.shape, np.inf)
    length = np.zeros(n)
    emerge = None
    grew = None  # last bin at which the tube claimed anything
    for b in range(n):
        if emerge is not None and b - grew > give_up and not (tube & (rg > gr + min_len)).any():
            tube[:], t_in[:], dist[:], emerge = False, -1, np.inf, None  # stopped short: rim noise, start again
            length[:b] = 0.0                                               # ...and forget its lengths
        new = (arr == b) & ~blocked
        if new.any():
            nl, lab = cv2.connectedComponents(new.astype(np.uint8), connectivity=8)
            if not tube.any():
                _, lab_all = cv2.connectedComponents(((arr <= b) & ~blocked).astype(np.uint8), connectivity=8)
                old_far = (arr < b - 3) & (rg > gr + 10.0)
                for l in range(1, nl):
                    c = lab == l
                    if not (c & start).any():
                        continue
                    a = np.angle(np.exp(1j * (ang[c] - np.angle(np.mean(np.exp(1j * ang[c]))))))
                    if np.ptp(a) > np.deg2rad(60):          # an arc round the rim, not a stub leaving it
                        continue
                    if (np.isin(lab_all, np.unique(lab_all[c])) & old_far).any():
                        continue                             # the leading end of a structure already there
                    tube |= c
                    t_in[c] = b
                    dist[c & start] = rg[c & start] - gr    # lengths count from the rim, gap included
                    _extend_dist(dist, c, c & start, 1)
                if tube.any():
                    emerge = grew = b
            else:
                tip = tube & (t_in >= t_in.max() - recent)
                seeds = cv2.dilate(tip.astype(np.uint8), ker).astype(bool)
                for l in range(1, nl):
                    c = lab == l
                    if (c & seeds).any():
                        _extend_dist(dist, c, tip, bridge)
                        tube |= c
                        t_in[c] = b
                        grew = b
        fin = tube & np.isfinite(dist)
        length[b] = float(dist[fin].max()) if fin.any() else 0.0
    return {"tube": tube, "t_in": t_in, "dist": dist, "emerge": emerge, "length": np.maximum.accumulate(length)}


def flood_compete(arr: np.ndarray, rg: np.ndarray, ang: np.ndarray, blocked: np.ndarray, gr: float,
                  rivals: list[tuple[float, float, float]], recent: int = 12, bridge: int = 4, start_band: float = 4.0,
                  min_len: float = 8.0, give_up: int = 40, orphan_px: int = 15) -> dict:
    """``flood`` with every tube in view growing at once, so that tubes compete for new material.

    ``rivals``: the other grains in the crop, (cy, cx, r) in crop pixels (their discs are in ``blocked``).
    Each bin's new pieces go to one owner: the target's tube, a rival grain's tube, or an "orphan" (material
    growing from no grain in view, e.g. a tube from a grain outside the crop). A piece touching the recent tips
    of several tubes goes to the one that grew most recently, then to the tip it is nearest; a tube's start
    at its rim loses to a tube actively growing there. So a stopped tube does not take over another tube
    passing its tip, and a tube passing a grain's rim does not start that grain. Orphans compete once they
    are credible (``orphan_px`` pixels grown over two or more bins), so that specks of noise near a tube
    do not steal its growth. Returns what ``flood`` returns, for the target."""
    n = int(arr.max())
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grains = [(None, None, gr, rg, ang)] + [(cy, cx, r, np.hypot(xx - cx, yy - cy), np.arctan2(yy - cy, xx - cx))
                                             for cy, cx, r in rivals]
    blocked = blocked.copy()
    for _, _, r, rgg, _ in grains:
        blocked |= rgg < r + HALO
    starts = [(rgg >= r + HALO) & (rgg <= r + HALO + start_band) for _, _, r, rgg, _ in grains]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bridge + 1, 2 * bridge + 1))
    owner = np.zeros((h, w), np.int32)  # 0 none; 1 + k grain k's tube (k = 0 the target); >= 1 + len(grains) orphans
    t_in = np.full((h, w), -1, np.int32)
    last, size, bins_grown = {}, {}, {}
    started = [False] * len(grains)
    next_orphan = 1 + len(grains)
    dist = np.full((h, w), np.inf)
    length = np.zeros(n)
    emerge = grew = None
    lab_all = None

    def credible(o):
        return o <= len(grains) or (size[o] >= orphan_px and bins_grown[o] >= 2)

    for b in range(n):
        if emerge is not None and b - grew > give_up and not ((owner == 1) & (rg > gr + min_len)).any():
            t_in[owner == 1], dist[:] = -1, np.inf  # stopped short: rim noise, start again
            owner[owner == 1], emerge, started[0] = 0, None, False
            length[:b] = 0.0
            last.pop(1, None)
        new = (arr == b) & ~blocked
        if new.any():
            nl, lab, stats, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8), connectivity=8)
            lab_all = None
            for l in range(1, nl):
                x0, y0, cw_, ch_ = (int(v) for v in stats[l, :4])
                y0, y1 = max(y0 - bridge - 1, 0), min(y0 + ch_ + bridge + 1, h)
                x0, x1 = max(x0 - bridge - 1, 0), min(x0 + cw_ + bridge + 1, w)
                c = lab[y0:y1, x0:x1] == l
                near = cv2.dilate(c.astype(np.uint8), ker).astype(bool)
                ow, tw = owner[y0:y1, x0:x1], t_in[y0:y1, x0:x1]
                tips = {}
                for o in np.unique(ow[near & (ow > 0)]):
                    o = int(o)
                    sel = near & (ow == o) & (tw >= last[o] - recent)
                    if sel.any():  # distance from the piece to that tube's very tip
                        ty, tx = np.nonzero((ow == o) & (tw >= last[o] - 2))
                        cy_, cx_ = np.nonzero(c)
                        d = (float(np.min(np.hypot(ty[:, None] - cy_[None], tx[:, None] - cx_[None])))
                             if len(ty) else float(bridge + 1))
                        tips[o] = d
                cand_start = []
                for k, (_, _, r, rgg, angg) in enumerate(grains):
                    if started[k] or not (c & starts[k][y0:y1, x0:x1]).any():
                        continue
                    a = angg[y0:y1, x0:x1][c]
                    if np.ptp(np.angle(np.exp(1j * (a - np.angle(np.mean(np.exp(1j * a))))))) > np.deg2rad(60):
                        continue  # an arc round the rim, not a stub leaving it
                    if lab_all is None:
                        _, lab_all = cv2.connectedComponents(((arr <= b) & ~blocked).astype(np.uint8), connectivity=8)
                    old_far = (arr < b - 3) & (rgg > r + 10.0)
                    if (np.isin(lab_all, np.unique(lab_all[y0:y1, x0:x1][c])) & old_far).any():
                        continue  # the leading end of a structure already there
                    cand_start.append(k)
                active = [o for o in tips if credible(o) and last[o] >= b - recent]
                if active:  # the tube growing most recently, then the nearest tip
                    win = min(active, key=lambda o: (-last[o], tips[o]))
                elif cand_start:
                    win = 1 + cand_start[0]
                elif tips:
                    win = min(tips, key=lambda o: (not credible(o), -last[o], tips[o]))
                else:
                    win = next_orphan
                    next_orphan += 1
                sub_owner, sub_t = owner[y0:y1, x0:x1], t_in[y0:y1, x0:x1]
                if win == 1:  # the target: rim distances as ``flood`` keeps them
                    full = np.zeros((h, w), bool)
                    full[y0:y1, x0:x1] = c
                    if not started[0]:
                        st = full & starts[0]
                        dist[st] = rg[st] - gr  # lengths count from the rim, gap included
                        _extend_dist(dist, full, st, 1)
                        started[0], emerge = True, b
                    else:
                        _extend_dist(dist, full, (owner == 1) & (t_in >= last[1] - recent), bridge)
                    grew = b
                elif win <= len(grains):
                    started[win - 1] = True
                sub_owner[c], sub_t[c] = win, b
                size[win] = size.get(win, 0) + int(c.sum())
                bins_grown[win] = bins_grown.get(win, 0) + (last.get(win) != b)
                last[win] = b
        fin = (owner == 1) & np.isfinite(dist)
        length[b] = float(dist[fin].max()) if fin.any() else 0.0
    tube = owner == 1
    return {"tube": tube, "t_in": np.where(tube, t_in, -1), "dist": np.where(tube, dist, np.inf), "emerge": emerge,
            "length": np.maximum.accumulate(length)}


def read_grain(renderer: Renderer, prob: Renderer, meta: dict, grain: dict, others: list[dict], p,
               _drift: np.ndarray | None = None) -> dict:
    """One grain read by the flood, as a result dict of the same shape as ``analyze_grain``'s."""
    from .analyze import checked_drift, followed_drift, hold_nan, local_shifts, read_lost, reads_in_grain_frame
    fpb, rs = int(meta["frames_per_bin"]), int(meta.get("ref_start", 0))
    n_bins = int(meta["n_bins"]) - rs
    gx, gy, gr = grain["x"], grain["y"], grain["r"]
    half = p.flood_half
    centre = half - 0.5
    flags = ["reader:flood"]
    if getattr(p, "grain_track", "phase") in ("follow", "auto") and _drift is None:
        fd = followed_drift(renderer, meta, grain, others, p)
        if fd["lost_from"] is not None and p.lost_policy == "hold":
            return read_lost(renderer, meta, grain, p, fd, half=half, flags=tuple(flags), read=lambda m, d: read_grain(
                renderer, prob, m, grain, others, p, _drift=d))
        _drift = hold_nan(fd["drift"])
    followed = reads_in_grain_frame(_drift, gr, p)
    if followed:
        # a grain that has moved far is cropped at its whole-pixel place, the rest of its drift registered by warping
        # (nearer, the whole drift is warped, as the phase track's is)
        off = np.round(_drift) if np.abs(_drift).max() > p.track_recentre_px else np.zeros_like(_drift)
        off_abs = np.zeros((int(meta["n_bins"]), 2))
        off_abs[rs:rs + n_bins] = off
        ls, resid = _drift, _drift - off
        late = np.nan_to_num(np.mean([renderer.crop(b, gx, gy, half, off_abs) for b in range(rs + n_bins - 4,
                                                                                         rs + n_bins - 1)], axis=0))
    else:
        off_abs = None
        crops = np.stack([renderer.crop(b, gx, gy, half) for b in range(rs, rs + n_bins)])
        ls = local_shifts(np.nan_to_num(crops, nan=float(np.nanmedian(crops))), centre, gr, p.reg_pad, p.ref_bins)
        ls, drift_flag = checked_drift(ls, p)
        flags += [drift_flag] if drift_flag else []
        late = np.nan_to_num(crops[-4:-1].mean(axis=0), nan=0.0)
        del crops
        resid = ls
    warp = lambda c, s: cv2.warpAffine(np.nan_to_num(c).astype(np.float32), np.float32([[1, 0, -s[0]], [0, 1, -s[1]]]),
                                       (2 * half, 2 * half), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    pstack = np.stack([np.clip(warp(prob.crop(b, gx, gy, half, off_abs), s), 0, 255).astype(np.uint8)
                       for b, s in zip(range(rs, rs + n_bins), resid)])
    present = pstack >= 0.5 * P_SCALE
    yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float32)
    rg, ang = np.hypot(xx - centre, yy - centre), np.arctan2(yy - centre, xx - centre)
    blocked = rg < gr - 1.0
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        if -o["r"] - 5 < ox < 2 * half + o["r"] + 5 and -o["r"] - 5 < oy < 2 * half + o["r"] + 5:
            blocked |= np.hypot(xx - ox, yy - oy) < o["r"] + p.other_block_px
    arr = arrivals(present, blocked)
    if getattr(p, "flood_compete", False):
        rivals = [(o["y"] - gy + centre, o["x"] - gx + centre, o["r"]) for o in others
                  if -o["r"] < o["x"] - gx + centre < 2 * half + o["r"] and -o["r"] < o["y"] - gy + centre < 2 * half + o["r"]]
        fl = flood_compete(arr, rg, ang, blocked, gr, rivals, p.flood_recent, p.flood_bridge, p.flood_start_band)
    else:
        fl = flood(arr, rg, ang, blocked, gr, p.flood_recent, p.flood_bridge, p.flood_start_band)
    length = np.maximum(fl["length"] - p.flood_tip_px, 0.0) * (fl["length"] > 0)
    frames = [b * fpb + fpb // 2 for b in range(rs, rs + n_bins)]
    to_ref = lambda y, x: [round(float(x - centre + gx), 2), round(float(y - centre + gy), 2)]
    res = {"id": grain["id"], "x": gx, "y": gy, "r": gr, "flags": flags, "map_threshold": 1.0,
           "local_shift_max_px": round(float(np.hypot(*ls.T).max()), 2)}
    if followed:  # paths and tips are in the grain's frame: its place in the field is census + drift
        res["drift"] = {"frames": frames, "xy": np.round(ls, 2).tolist()}
    b = fl["emerge"]
    if b is None or length[-1] < p.min_tube_px:
        res.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None,
                   length={"frames": frames, "px": [0.0] * n_bins}, path=[])
        res["_diag"] = (late, np.where(arr < n_bins, 3.0 * (n_bins - arr) / n_bins, 0).astype(np.float32), fl["tube"], None, None, None, centre)
        return res
    tube, t_in, dist = fl["tube"], fl["t_in"], fl["dist"]
    tips = []
    zone = gr + HALO + p.flood_start_band
    exit_len = np.zeros(n_bins)
    for t in range(n_bins):  # tip = the farthest tube pixel claimed by then
        sel = tube & (t_in <= t) & np.isfinite(dist)
        y, x = np.unravel_index(int(np.argmax(np.where(sel, dist, -1.0))), dist.shape) if sel.any() else (centre, centre)
        tips.append(to_ref(y, x))
        if p.flood_from_exit and sel.any() and length[t] > 0:
            exit_len[t] = from_exit(centreline(sel, dist, (int(y), int(x)), centre, gr, p.flood_bridge + 0.5),
                                    centre, gr, zone)[1]
    if p.flood_from_exit:  # measured along the tube from where it leaves the grain, never longer than before
        length = np.maximum.accumulate(np.where(length > 0, np.minimum(length, exit_len), 0.0))
    # the tube's pixels on the way from the rim to the final tip, nearest the rim first (the look-back's exit)
    ty, tx = np.unravel_index(int(np.argmax(np.where(tube & np.isfinite(dist), dist, -1.0))), dist.shape)
    order = sorted(zip(*np.nonzero(tube & np.isfinite(dist))), key=lambda q: dist[q])
    route = [q for q in order if math.hypot(q[0] - ty, q[1] - tx) <= dist[ty, tx] - dist[q] + 3.0][::3]
    line = centreline(tube, dist, (int(ty), int(tx)), centre, gr, p.flood_bridge + 0.5)  # what is drawn and reviewed
    if p.flood_from_exit:
        line = from_exit(line, centre, gr, zone)[0]
    if p.flood_lookback > 0 and route:
        # the flood starts once the map is sure; a young tube often shows at its own exit earlier,
        # below that certainty: walk back while the exit sector stays above the lower threshold
        a0 = math.atan2(route[0][0] - centre, route[0][1] - centre)
        sector = (rg >= gr + 1.0) & (rg <= gr + 8.0) & (np.abs(np.angle(np.exp(1j * (ang - a0)))) <= np.deg2rad(25))
        sig = pstack[:, sector].max(axis=1) / P_SCALE if sector.any() else np.zeros(n_bins)
        t = b
        while t > 0 and np.sum(sig[max(0, t - 3):t] >= p.flood_lookback) >= 2:
            t -= 1
        if t < b:  # lengths ramp from the earlier onset to the flood's first reading
            length[t:b] = np.linspace(0.0, length[b], b - t, endpoint=False)
            flags.append(f"onset_lookback:{b - t}")
            b = t
    status = "emerged_at_start" if b == 0 else "emerged_within"
    res.update(status=status, onset_frame=frames[b], onset_interval=None if b == 0 else [frames[b - 1], frames[b]],
               length={"frames": frames, "px": [round(float(v), 2) for v in length]},
               tip={"frames": frames, "xy": tips}, path=[to_ref(y, x) for y, x in line],
               exit_xy=to_ref(*line[0]),
               final_length_px=round(float(length[-1]), 2), path_length_px=round(float(dist[ty, tx]), 2))
    pts = np.array([[x, y] for y, x in line], float)
    res["_diag"] = (late, np.where(arr < n_bins, 3.0 * (n_bins - arr) / n_bins, 0).astype(np.float32), tube, pts, None, None, centre)
    return res


def with_onset(flood_res: dict, change_res: dict) -> dict:
    """The flood's lengths under the change reader's germination call and onset."""
    out = dict(flood_res)
    out.update(status=change_res["status"], onset_frame=change_res.get("onset_frame"),
               onset_interval=change_res.get("onset_interval"), flags=flood_res["flags"] + ["onset:change"])
    frames, px = flood_res["length"]["frames"], list(flood_res["length"]["px"])
    if change_res["status"] == "no_emergence_by_end":
        px = [0.0] * len(px)
    elif change_res.get("onset_frame") is not None:
        px = [v if f >= change_res["onset_frame"] else 0.0 for f, v in zip(frames, px)]
    out["length"] = {"frames": frames, "px": px}
    out["final_length_px"] = round(float(px[-1]), 2) if px else 0.0
    return out
