"""Learned tube evidence and the arrival-flood reader, for crowded fields.

A small U-Net reads each registered bin with the movie's "before" and "after" images and
outputs P(tube body) per pixel. The shipped model (0.8.0 on, ``models/tubes_bn_real_ld_m2.pt``;
recipe ``prototypes/learned_flood/models/tubes_bn_real_ld_m2.recipe.sh``) uses BatchNorm and inputs
relative to the local background (so a pixel's map depends on its surroundings, not on the whole
frame, and uneven illumination is taken out as in the training crops), trained on codec-exact
synthetic movies (v5 and v6) and fine-tuned on both labelled movies' traces. Judged on the movie
whose traces it did not see, the recipe marks far more of the traced tubes than the synthetic-only
``models/tubes_synth_v1.pt`` (0.5-0.7) and gives per-grain growth rates that follow the annotator's
(movie 2: correlation 0.72 vs 0.19; prototypes/tube_net/README.md). Both labelled movies are
therefore development sets for it; movie 1 is its test.

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
import warnings
from pathlib import Path

import cv2
import numpy as np

from . import stack
from .render import Renderer

MODEL = Path(__file__).parent / "models" / "tubes_bn_real_ld_m2.pt"  # 0.5.0-0.7.0: tubes_synth_v1.pt
P_SCALE = 250.0       # probability movies are stored as uint8 P x P_SCALE
IN_SCALE = 20.0       # network input: grey levels per unit, relative to the before image's median
HALO = 3.0            # the rim's own change (focus, swelling) reaches this far out


# ----------------------------------------------------------------------------- network
def _unet(widths=(16, 32, 64, 128), norm: str = "group"):
    """The network. ``norm="group"`` (every model up to 29 Sep 2026) normalises each feature over the whole input, so
    a pixel's output depends on the input's size and content; ``"batch"`` (fixed statistics at inference) makes it
    depend only on the pixel's surroundings, so full-frame maps equal the maps of the crops it was trained on."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    if norm not in ("group", "batch"):
        raise ValueError(f"unknown normalisation {norm!r}: group or batch")

    def nrm(c):
        return nn.GroupNorm(8, c) if norm == "group" else nn.BatchNorm2d(c)

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nrm(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nrm(cout), nn.ReLU(inplace=True))

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
    net = _unet(ck.get("widths", (16, 32, 64, 128)), ck.get("norm", "group"))
    net.load_state_dict(ck["state"])
    net.bg_px = int(ck.get("bg_px", 0) or 0)  # > 0: inputs relative to the local background (tube_probability)
    return net.eval().to(device)


def local_background(early: np.ndarray, px: int) -> np.ndarray:
    """The "before" image's median over about ``px`` px round every pixel (a smooth map): the level a training crop
    of that size was normalised by. Movie 2's illumination falls off by up to ~75 grey levels across the frame, so
    one median for the whole frame leaves most of it far from the level the network saw in training."""
    h, w = early.shape
    f = 8
    small = cv2.resize(np.nan_to_num(early, nan=float(np.nanmedian(early))).astype(np.float32),
                       (max(w // f, 1), max(h // f, 1)), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(px / f)) | 1)
    med = cv2.medianBlur(np.clip(np.round(small), 0, 255).astype(np.uint8), k).astype(np.float32)
    return cv2.resize(med, (w, h), interpolation=cv2.INTER_LINEAR)


def tube_probability(net, img: np.ndarray, early: np.ndarray, late: np.ndarray) -> np.ndarray:
    """P(tube body) for one registered frame, given the movie's before and after images: inputs relative to the before
    image's median over the whole frame, or - for a network whose checkpoint names ``bg_px`` - to its local median
    over that many px (``local_background``), as each training crop was normalised by its own median."""
    import torch
    m = local_background(early, net.bg_px) if getattr(net, "bg_px", 0) else float(np.nanmedian(early))
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


def prob_cache(cache_dir: str | Path, model: str | Path = MODEL, log=print, progress=None) -> Path:
    """The movie's tube-probability cache (built once, next to the image cache): a SparseTrack
    cache in reference coordinates whose bins are uint8 P(tube) x P_SCALE. ``progress(done, total)``, if given,
    is called after every bin built."""
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
        if progress is not None:
            progress(b + 1, nb)
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
          bridge: int = 4, start_band: float = 4.0, min_len: float = 8.0, give_up: int = 40, halo: float = HALO,
          arc_deg: float = 60.0, old_far_px: float = 10.0, old_far_bins: int = 3) -> dict:
    """Grow one grain's tube through an arrival map (see the module docstring).

    Start rules: nothing within ``halo`` px of the rim is claimed; a tube starts on a piece reaching the
    ``start_band`` px beyond that, spanning at most ``arc_deg`` degrees round the grain, and not joined (through
    what has arrived so far) to material that arrived more than ``old_far_bins`` bins earlier more than
    ``old_far_px`` px beyond the rim. A tube that stops for ``give_up`` bins before getting ``min_len`` px beyond
    the rim was rim noise: it is forgotten and the flood starts again.

    Returns the tube mask, each tube pixel's arrival bin and rim distance, the emergence bin
    and the length per bin.
    """
    n = int(arr.max())
    blocked = blocked | (rg < gr + halo)
    start = (rg >= gr + halo) & (rg <= gr + halo + start_band)  # the map may miss the px next to the rim
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
                old_far = (arr < b - old_far_bins) & (rg > gr + old_far_px)
                for l in range(1, nl):
                    c = lab == l
                    if not (c & start).any():
                        continue
                    a = np.angle(np.exp(1j * (ang[c] - np.angle(np.mean(np.exp(1j * ang[c]))))))
                    if np.ptp(a) > np.deg2rad(arc_deg):     # an arc round the rim, not a stub leaving it
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
    present = pstack >= p.flood_p * P_SCALE
    # beyond the movie frame the crops repeat its edge row or column (registration and cropping both do), so a tube
    # that reaches the edge would run on along that streak (m2 g089: 260 px past the edge): nothing grows there
    outside = np.stack([warp(renderer.outside(b, gx, gy, half, off_abs).astype(np.float32), s) > 0.5
                        for b, s in zip(range(rs, rs + n_bins), resid)])
    present &= ~outside
    yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float32)
    rg, ang = np.hypot(xx - centre, yy - centre), np.arctan2(yy - centre, xx - centre)
    blocked = rg < gr - 1.0
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        if -o["r"] - 5 < ox < 2 * half + o["r"] + 5 and -o["r"] - 5 < oy < 2 * half + o["r"] + 5:
            blocked |= np.hypot(xx - ox, yy - oy) < o["r"] + p.other_block_px
    arr = arrivals(present, blocked, p.flood_persist, p.flood_frac)
    if getattr(p, "flood_compete", False):
        rivals = [(o["y"] - gy + centre, o["x"] - gx + centre, o["r"]) for o in others
                  if -o["r"] < o["x"] - gx + centre < 2 * half + o["r"] and -o["r"] < o["y"] - gy + centre < 2 * half + o["r"]]
        fl = flood_compete(arr, rg, ang, blocked, gr, rivals, p.flood_recent, p.flood_bridge, p.flood_start_band)
    else:
        fl = flood(arr, rg, ang, blocked, gr, p.flood_recent, p.flood_bridge, p.flood_start_band, p.flood_min_len,
                   p.flood_give_up, p.flood_halo, p.flood_arc_deg, p.flood_old_far_px, p.flood_old_far_bins)
    length = np.maximum(fl["length"] - p.flood_tip_px, 0.0) * (fl["length"] > 0)
    frames = [b * fpb + fpb // 2 for b in range(rs, rs + n_bins)]
    to_ref = lambda y, x: [round(float(x - centre + gx), 2), round(float(y - centre + gy), 2)]
    res = {"id": grain["id"], "x": gx, "y": gy, "r": gr, "flags": flags, "map_threshold": 1.0,
           "local_shift_max_px": round(float(np.hypot(*ls.T).max()), 2)}
    if followed or np.any(ls):  # paths and tips are in the frame the grain was read in: in the field, census + drift
        res["drift"] = {"frames": frames, "xy": np.round(ls, 2).tolist()}
    b = fl["emerge"]
    if b is None or length[-1] < p.min_tube_px:
        res.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None,
                   length={"frames": frames, "px": [0.0] * n_bins}, path=[])
        res["_diag"] = (late, np.where(arr < n_bins, 3.0 * (n_bins - arr) / n_bins, 0).astype(np.float32), fl["tube"], None, None, None, centre)
        return res
    tube, t_in, dist = fl["tube"], fl["t_in"], fl["dist"]
    if outside.any() and (tube & cv2.dilate(outside.any(axis=0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).any():
        flags.append("tube_at_frame_edge")  # its length is known only up to the edge
    tips, tip_yx = [], []
    zone = gr + p.flood_halo + p.flood_start_band
    exit_len = np.zeros(n_bins)
    for t in range(n_bins):  # tip = the farthest tube pixel claimed by then
        sel = tube & (t_in <= t) & np.isfinite(dist)
        y, x = np.unravel_index(int(np.argmax(np.where(sel, dist, -1.0))), dist.shape) if sel.any() else (centre, centre)
        if p.flood_from_exit and sel.any() and length[t] > 0:
            exit_len[t] = from_exit(centreline(sel, dist, (int(y), int(x)), centre, gr, p.flood_bridge + 0.5),
                                    centre, gr, zone)[1]
            if p.flood_tip == "radial":
                # a young tube's blob widens along the rim, where pieces joining late get the greatest rim distance:
                # the farthest pixel from the grain is then the tip, if its length from the exit is the longer
                yr, xr = np.unravel_index(int(np.argmax(np.where(sel, rg, -1.0))), dist.shape)
                if (yr, xr) != (y, x):
                    er = from_exit(centreline(sel, dist, (int(yr), int(xr)), centre, gr, p.flood_bridge + 0.5),
                                   centre, gr, zone)[1]
                    if er > exit_len[t]:
                        exit_len[t], y, x = er, yr, xr
        tips.append(to_ref(y, x))
        tip_yx.append((int(y), int(x)))
    if p.flood_from_exit:  # measured along the tube from where it leaves the grain, never longer than before
        length = np.maximum.accumulate(np.where(length > 0, np.minimum(length, exit_len), 0.0))
    # the final tip: the pixel of greatest rim distance, or with lengths from the exit, the tip of the latest bin
    # whose length from the exit is the longest (the reported final length). The greatest rim distance can lie by the
    # grain, reached round a detour: ld g008's route was 3 px for a 61 px tube, m2 g012's 3 px for 55 px.
    if p.flood_from_exit and exit_len.max() > 0:
        ty, tx = tip_yx[int(np.flatnonzero(exit_len >= exit_len.max() - 1e-6)[-1])]
    else:
        ty, tx = np.unravel_index(int(np.argmax(np.where(tube & np.isfinite(dist), dist, -1.0))), dist.shape)
    # the tube's pixels on the way from the rim to the final tip, nearest the rim first (the look-back's exit)
    order = sorted(zip(*np.nonzero(tube & np.isfinite(dist))), key=lambda q: dist[q])
    route = [q for q in order if math.hypot(q[0] - ty, q[1] - tx) <= dist[ty, tx] - dist[q] + 3.0][::3]
    line = centreline(tube, dist, (int(ty), int(tx)), centre, gr, p.flood_bridge + 0.5)  # what is drawn and reviewed
    if p.flood_from_exit:
        line = from_exit(line, centre, gr, zone)[0]
    if p.flood_exit_edge:
        # measured from where the tube leaves the grain's visible edge along its exit, not the census circle (as the
        # change reader does, analyze.exit_edge): the flood counts from the census rim
        from .analyze import exit_edge
        early = np.mean([cv2.warpAffine(np.nan_to_num(renderer.crop(bb, gx, gy, half, off_abs)).astype(np.float32),
                                        np.float32([[1, 0, -s[0]], [0, 1, -s[1]]]), (2 * half, 2 * half),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                         for bb, s in zip(range(rs, rs + p.ref_bins), resid[:p.ref_bins])], axis=0)
        e = float(np.clip(exit_edge(early, centre, gr, math.atan2(line[0][0] - centre, line[0][1] - centre)),
                          -(gr - 1.0), 5.0))
        length = np.where(length > 0, np.maximum(length - e, 0.0), 0.0)
        res["exit_edge_px"] = round(e, 2)
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


def _band_centres(prof: np.ndarray, offs: np.ndarray, min_p: float, max_width: float) -> np.ndarray:
    """Per row of ``prof`` (a map sampled across a tube at offsets ``offs``), the middle of the band the route at
    offset 0 is on: uphill from offset 0 to the band's own peak (the steeper side first; a stronger neighbour does
    not set the level), then the midpoint of the run above half that peak, its ends interpolated (NaN: a peak below
    ``min_p``, or a run that leaves the window or is wider than ``max_width``)."""
    prof = np.asarray(prof, np.float64)
    n, m = prof.shape
    out = np.full(n, np.nan)
    if n == 0 or m < 3:
        return out
    j0, rows, idx = int(np.argmin(np.abs(offs))), np.arange(n), np.arange(m)[None, :]
    here = prof[:, j0]
    left = prof[:, j0 - 1] if j0 > 0 else np.full(n, -np.inf)
    right = prof[:, j0 + 1] if j0 < m - 1 else np.full(n, -np.inf)
    d = np.diff(prof, axis=1)  # d[:, k] = prof[k + 1] - prof[k]
    peak = np.full(n, j0)
    if j0 < m - 1:  # climbing right: up to the first k from j0 with prof[k + 1] <= prof[k]
        up = d[:, j0:] > 0
        peak = np.where((right > here) & (right >= left), j0 + np.where(up.all(axis=1), m - 1 - j0, np.argmin(up, axis=1)),
                        peak)
    if j0 > 0:  # climbing left: down to the first k from j0 with prof[k - 1] <= prof[k]
        dn = d[:, :j0][:, ::-1] < 0
        peak = np.where((left > here) & (left > right), j0 - np.where(dn.all(axis=1), j0, np.argmin(dn, axis=1)), peak)
    pv = prof[rows, peak]
    half = 0.5 * pv
    below = prof < half[:, None]
    lb = np.where(below & (idx < peak[:, None]), idx, -1).max(axis=1)  # the run holding the peak: (lb, rb) exclusive
    rb = np.where(below & (idx > peak[:, None]), idx, m).min(axis=1)
    ok = (pv >= min_p) & (lb >= 0) & (rb <= m - 1)
    a, b = np.clip(lb + 1, 1, m - 1), np.clip(rb - 1, 0, m - 2)
    lo = offs[a - 1] + (half - prof[rows, a - 1]) / np.maximum(prof[rows, a] - prof[rows, a - 1], 1e-9) * (offs[a] - offs[a - 1])
    hi = offs[b] + (prof[rows, b] - half) / np.maximum(prof[rows, b] - prof[rows, b + 1], 1e-9) * (offs[b + 1] - offs[b])
    ok &= (hi - lo) <= max_width
    out[ok] = 0.5 * (lo[ok] + hi[ok])
    return out


def _running_median(a: np.ndarray, h: int, axis: int = 0) -> np.ndarray:
    """Median over a window of ``h`` either side along ``axis`` (shorter at the ends; NaN ignored)."""
    a = np.asarray(a, float)
    pad = [(0, 0)] * a.ndim
    pad[axis] = (h, h)
    win = np.lib.stride_tricks.sliding_window_view(np.pad(a, pad, constant_values=np.nan), 2 * h + 1, axis=axis)
    with warnings.catch_warnings():  # an all-NaN window gives NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=-1)


def _across(prob: Renderer, layouts: list[tuple[int, np.ndarray, np.ndarray]], offs: np.ndarray) -> np.ndarray:
    """The network's map (0..1) across a route, averaged over ``layouts``: per bin, (bin, the route's points where
    the tube lay then, their normals), sampled at points + ``offs`` x normal."""
    prof = None
    for b, pts, nrm in layouts:
        q = (pts[:, None, :] + offs[None, :, None] * nrm[:, None, :] + np.asarray(prob.shifts[b], float)
             - 0.5).astype(np.float32)  # continuous coordinates -> pixel centres
        v = cv2.remap(np.asarray(prob.bins[b]), q[..., 0], q[..., 1], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
        prof = v if prof is None else prof + v
    return prof / (len(layouts) * P_SCALE)


def _centres_along(prof: np.ndarray, offs: np.ndarray, p) -> np.ndarray | None:
    """Per route point, how far its tube's middle is along its normal: the band's middle, gaps taken from the
    neighbours, median-filtered along the route and clamped; None where fewer than two points find a band."""
    c = _band_centres(prof, offs, p.centre_min_p, p.centre_max_width)
    ok = np.isfinite(c)
    if ok.sum() < 2:
        return None
    n = np.arange(len(c))
    c = np.interp(n, n[ok], c[ok])
    return np.clip(_running_median(c, p.centre_smooth // 2), -p.centre_max_shift, p.centre_max_shift)


def centre_route(res: dict, prob: Renderer, meta: dict, p) -> None:
    """Move a reading's route (and its tips) onto the middle of its tube, in place; with ``p.centre_per_bin`` also
    store how the route lay at each bin (``res["bend"]``, ``routes.bent``).

    A tube in these movies is two dark walls about 7 px apart with a clear middle, one wall darker than the other.
    The change reader's cheapest path and the flood's geodesic centreline both run along the darker wall: 3.5 px
    (median) from the middle, where annotators trace (ld, m2 and m1; 1 Oct 2026). The tube network marks the whole
    width, so the middle of its band across the tube is the tube's middle. Each route point (every px) moves along
    its normal to the midpoint of the half-maximum run of the network's map that holds it, the map averaged over the
    last ``p.centre_late`` bins the grain was seen, where the grain was then (route + drift); shifts are
    median-filtered along the route, points with no clear band take their neighbours'. Tubes bend and are pushed as
    they grow (the route centred on the end lies 2-3 px off traces made 100 bins earlier): per bin, the same on that
    bin's map (and its neighbours'), smoothed over 5 bins, kept every ``p.centre_knot_px`` px as offsets from the
    centred route. A route that turns with its grain (``rotation_deg``) is turned for each bin first (an offset
    along the route's own normal survives the turn). The change reader's tips (points of its route) move with their
    route point; the flood's (its farthest tube pixel) stay. Lengths, onsets and the exit are not changed."""
    from . import routes
    path = np.asarray(res.get("path") or [], float)
    if len(path) < 2 or routes.arc(path)[-1] < 2.0:
        return
    frames = np.asarray(res["length"]["frames"])
    fpb = int(meta["frames_per_bin"])
    until = res.get("observed_until_frame")
    last = int(np.flatnonzero(frames <= until)[-1]) if until is not None and (frames <= until).any() else len(frames) - 1
    idx = list(range(max(0, last - p.centre_late), last)) or [last]
    drift = np.asarray(res["drift"]["xy"], float) if res.get("drift") else np.zeros((len(frames), 2))
    rot = np.asarray(res.get("rotation_deg") or [], float)
    rot = rot if len(rot) == len(frames) else np.zeros(len(frames))
    pivot = np.asarray(res["exit_xy"] if p.rot_pivot == "exit" and res.get("exit_xy") else [res["x"], res["y"]], float)

    def lay(i, q, n):  # (bin, points, normals) as the route lay at frame index i: turned, then moved by the drift
        th = math.radians(float(rot[i]))
        turn = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        return int(frames[i]) // fpb, (q - pivot) @ turn.T + pivot + drift[i], n @ turn.T
    pts, _ = routes.resample(path, 1.0)
    if len(pts) < 3:
        return
    nrm = routes.normals(pts)
    offs = np.arange(-p.centre_reach, p.centre_reach + 1e-6, 0.25)
    c = _centres_along(_across(prob, [lay(i, pts, nrm) for i in idx], offs), offs, p)
    if c is None:
        return
    shift = np.asarray(c)[:, None] * nrm
    centred = np.round(pts + shift, 2)
    res["path"] = centred.tolist()
    res["path_length_px"] = round(float(routes.arc(centred)[-1]), 2)
    def moved(i, t, q, vec):  # tip t (frame index i, where the tube lay then) moved by its route point's vector
        _, qi, vi = lay(i, q, vec)
        k = int(np.argmin(np.hypot(qi[:, 0] - drift[i][0] - t[0], qi[:, 1] - drift[i][1] - t[1])))
        return k, vi[k]

    # the change reader's tips are points of its route and move with it; the flood's are its farthest tube pixel,
    # already on the tube (m2: 1-3 px from the annotator's apex, 5-7 px when moved with the route)
    tips = (res.get("tip") or {}).get("xy") if "reader:flood" not in res.get("flags", []) else None
    if tips and len(tips) == len(frames):
        res["tip"]["xy"] = [t if t is None or t[0] is None else
                            np.round(np.asarray(t, float) + moved(i, t, pts, shift)[1], 2).tolist()
                            for i, t in enumerate(tips)]
        tips = res["tip"]["xy"]
    res["route_centred_px"] = round(float(np.median(np.abs(c))), 2)
    if p.centre_lengths:  # lengths along the middle: the same point of the tube, its arc length on the centred route
        s0, s1 = routes.arc(pts), routes.arc(centred)
        px = [0.0 if v <= 0 else round(float(np.interp(v, s0, s1) if v <= s0[-1] else s1[-1] + v - s0[-1]), 2)
              for v in res["length"]["px"]]
        res["length"]["px"] = px
        res["final_length_px"] = px[-1] if px else 0.0
    if not p.centre_per_bin:
        return
    s, nrm_c = routes.arc(centred), routes.normals(centred)
    step = float(p.centre_knot_px)
    knots = np.arange(0.0, s[-1] + 1e-6, step)
    L = np.asarray(res["length"]["px"], float)
    rows = np.full((len(frames), len(knots)), np.nan)
    for i in range(last + 1):
        if L[i] <= 0.5:
            continue
        vis = s <= L[i] + step
        if vis.sum() < 3:
            continue
        near_bins = [j for j in (i - 1, i, i + 1) if 0 <= j <= last]
        ci = _centres_along(_across(prob, [lay(j, centred[vis], nrm_c[vis]) for j in near_bins], offs), offs, p)
        if ci is not None:
            k = knots <= s[vis][-1]
            rows[i, k] = np.interp(knots[k], s[vis], ci)
    smooth = np.where(np.isfinite(rows), _running_median(rows, 2, axis=0), np.nan)  # no further than it reached then
    px10 = [[int(round(10 * v)) for v in r[np.isfinite(r)]] if np.isfinite(r).any() else [] for r in smooth]
    if not any(px10):
        return
    res["bend"] = {"step_px": step, "px10": px10}
    if tips and len(tips) == len(frames):
        for i, t in enumerate(tips):
            if t is None or t[0] is None or i >= len(px10) or not px10[i]:
                continue
            k, n = moved(i, t, centred, nrm_c)
            tips[i] = np.round(np.asarray(t, float) + float(routes.offsets_along(s[k:k + 1], px10[i], step)[0]) * n,
                               2).tolist()


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
