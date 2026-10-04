"""Label-free per-movie self-training of the tube network (research step, not part of the default analysis).

    python -m sparsetrack selftrain CACHE --out NET.pt [--predictions PRED.json] [--start NET0.pt]
        [--replay SHARD.npz ...] [--steps 1500] [--seed 0]

The tube network was trained on other movies; a new movie differs in illumination, focus, debris and how far its
grains move. This step adapts the network to one movie with no labels, on the tracker's own confident readings of
that movie (pseudo-labels), once (study: prototypes/self_train/README.md, made movie-agnostic here):

1. **readings**: SparseTrack's defaults with the starting network and without the tip-trajectory reader, as the
   step was measured (``Params(model=start, tiptraj="off")``), on every grain of the cache's own census
   (``grains.json``: no labels file, no human exclusions), or ``predictions`` made that way;
2. **pseudo-traces** (``select``): grains with no flag saying the reading is unsafe (``UNSAFE``; a grain lost partway
   only up to ``lost_margin`` bins before it was lost); a bin is a confident reading where the length is >=
   ``min_len`` px, ``review.trace_confidence`` >= ``min_conf`` (the tube grew within the last few bins and is long
   enough), the drawn tube (as the app draws it) lies on the starting network's map (P >= 0.5) for >= ``min_share``
   of its length beyond the rim, and no point of it is within ``edge_px`` of the frame edge. Every ``spacing``-th
   confident bin, at most ``cap`` per grain, is a pseudo-trace: the drawn route moved onto the middle of its band on
   the map (``learned.centre_route``'s band-middle rule), cut ``apex_cut`` px short of the reading's apex, nothing
   within ``tip_blind`` px of the apex scored, no background band where more than ``band_max`` of it is marked.
   Grains read as never germinated whose surroundings stay clean on the map are background (``clean_*``). No
   negatives before the tracker's onset: on movie 1 half of them had a tube (the tracker's onsets were late);
3. **crops** (``traced_crops``, ``propagated_crops``): as the human-trace crops (``prototypes/tube_net/realdata.py``
   version 3): ``along`` crops along each pseudo-trace, ``exits`` at its exit, the same route at bins b +/- 1 where
   the grain did not move; with ``propagate`` (variant "sb") also the tube carried by DIS optical flow between two
   pseudo-traces across the bins that were not confident readings, and after the last one while the flow carries
   it reliably (``prototypes/tube_adapt/propagate.py``), with the tracker's own drift as the grain's place;
4. **fine-tuning** (``fine_tune``): from the starting network, ``steps`` steps of batch 32 (AdamW, lr 5e-4
   one-cycle, 64 px crops, 0.8.0's losses and augmentation, BatchNorm statistics frozen), each batch a quarter
   pseudo-traced crops, a quarter propagated crops, a quarter real trace crops and a quarter synthetic crops of the
   starting network's own training data (``replay``: shards with a weight map are trace crops, pooled; shards
   without one are synthetic, an equal part each).

The adapted network is then the movie's ``Params.model``: the hybrid's flood, the route centring, the drawn-tube
check and the tip-trajectory reader all read its maps. One round only (a second round starts to confirm its own
marks); training noise alone moves end-to-end lengths by about 3 of 50 on movie 1.

Settings were fixed on the labelled movies before any blind movie (prototypes/self_train/README.md, "The step"):
variant "sb" (with propagation: its maps mark more of the traced tubes on movies 1 and 2), one run. Judged with the
tip-trajectory reader (the 0.9.0 candidate's settings, held-out detector folds), only the maps changed: movie 1 28/50
lengths with seeds 0 and 1 (27/50 on the shipped maps; onsets 15-17/28 against 12/28); movie 2, from a network that
never saw it, 24, 31 and 31/54 with seeds 0-2 (28/54 on that network's own maps) - the seed moves movie 2 by up to 7.
Averaging three runs (``runs=3``) read 27/50 and 28/54: no better than one run.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import __version__, routes, stack
from .render import Renderer
from .review import lengths_by_bin, trace_confidence

SCHEMA = "sparsetrack.selftrain.v1"
UNSAFE = ("touches:", "drift_rejected", "shared_change_split", "rotates:", "onset_at_focus_change")
# the shipped network's own training data (prototypes/learned_flood/models/tubes_bn_real_ld_m2.recipe.sh): its ld and
# movie 2 trace crops and one shard of each synthetic family; paths in the main checkout (data not in git)
DEFAULT_REPLAY = ("runs/tube_net/shards/real3_ld.npz", "runs/tube_net/shards/real3_m2.npz",
                  "runs/learned_flood/shards/train_v5m2s10.npz", "runs/synth_v6/shards/train_v6m1_s50.npz")
IN_SCALE = 20.0          # training inputs: grey levels per unit, relative to the crop's "before" median
BODY, GAP, BAND = 3.0, 6.0, 14.0  # traced crops: tube within BODY px of the route, background GAP..BAND px off it


@dataclass
class Settings:
    # pseudo-traces (prototypes/self_train/select.py's rules)
    min_len: float = 8.0
    min_conf: float = 0.7
    min_share: float = 0.8
    lost_margin: int = 10
    edge_px: float = 10.0
    spacing: int = 3
    cap: int = 40
    apex_cut: float = 3.0
    tip_blind: float = 14.0
    band_max: float = 0.05
    clean_ring: float = 20.0
    clean_max_px: int = 5
    clean_frac: float = 0.95
    clean_every: int = 8
    # traced crops (prototypes/self_train/pcrops.py)
    half: int = 48
    along: int = 4
    exits: int = 1
    neighbours: tuple = (-1, 1)
    over_grain: float = 0.6
    max_move: float = 1.5
    # propagated crops (prototypes/tube_adapt/propagate.py's defaults)
    propagate: bool = True
    prop_every: int = 3
    prop_body_px: float = 2.0
    prop_gap_px: float = 8.0
    prop_band_px: float = 16.0
    prop_tip_blind: float = 10.0
    prop_partial_blind: float = 18.0
    prop_ext_margin: float = 8.0
    prop_max_dev: float = 3.0
    prop_fb_max: float = 1.5
    prop_one_sided: int = 30
    prop_after_cap: int = 35
    flow_step: int = 10
    # fine-tuning (prototypes/tube_adapt/finetune.py's recipe)
    batch: int = 32
    lr: float = 5e-4
    crop: int = 64
    share_traced: float = 0.25
    share_propagated: float = 0.25
    share_real: float = 0.25
    share_synthetic: float = 0.25


def sha1(path: str | Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------- pseudo-traces
class Maps:
    """A probability movie (uint8 P x ``learned.P_SCALE``, reference coordinates; ``learned.prob_cache``)."""

    def __init__(self, arr: np.ndarray, shifts):
        self.bins = arr  # named as Renderer's, for learned._across
        self.shifts = np.asarray(shifts, float)
        self.h, self.w = arr.shape[1:]

    @classmethod
    def load(cls, prob_dir: str | Path) -> "Maps":
        arr, meta = stack.load(prob_dir)
        return cls(arr, meta["shifts"])

    def at(self, b: int, pts: np.ndarray) -> np.ndarray:
        from .learned import P_SCALE
        q = (np.asarray(pts, float) + self.shifts[b] - 0.5).astype(np.float32)
        v = cv2.remap(np.asarray(self.bins[b]), q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return v.ravel() / P_SCALE

    def window(self, b: int, lo: np.ndarray, hi: np.ndarray):
        """(P window, its reference origin) over [lo, hi) (reference px, clipped to the frame)."""
        from .learned import P_SCALE
        sx, sy = self.shifts[b]
        x0, y0 = int(max(0, math.floor(lo[0] + sx))), int(max(0, math.floor(lo[1] + sy)))
        x1, y1 = int(min(self.w, math.ceil(hi[0] + sx))), int(min(self.h, math.ceil(hi[1] + sy)))
        return np.asarray(self.bins[b, y0:y1, x0:x1], np.float32) / P_SCALE, np.array([x0 - sx, y0 - sy])


def series(res: dict, fpb: int, nb: int):
    """Per bin from bin 0: length, drift (held before and after the reading's series; gaps interpolated), rotation
    (degrees) and the bin the series starts at."""
    frames = np.asarray((res.get("length") or {}).get("frames") or [], int)
    L = lengths_by_bin(res, fpb, nb)
    start = int(frames[0]) // fpb if len(frames) else 0
    drift = np.zeros((nb, 2))
    if res.get("drift") and res["drift"].get("xy"):
        d = np.asarray([[np.nan, np.nan] if v is None or v[0] is None else v for v in res["drift"]["xy"]], float)
        fin = np.isfinite(d).all(axis=1)
        if fin.any():
            idx = np.arange(len(d))
            d = np.stack([np.interp(idx, idx[fin], d[fin, k]) for k in (0, 1)], axis=1)
            n = min(len(d), nb - start)
            drift[start:start + n] = d[:n]
            drift[start + n:] = d[n - 1]
            drift[:start] = d[0]
    rot = np.zeros(nb)
    r = np.asarray(res.get("rotation_deg") or [], float)
    if len(r) == len(frames) and len(r):
        n = min(len(r), nb - start)
        rot[start:start + n] = r[:n]
    return L, drift, rot, start


def drawn(res: dict, i: int, length: float, drift: np.ndarray, rot: float, pivot: np.ndarray) -> np.ndarray:
    """The tube the app draws at series index ``i`` in reference coordinates (``learned.drawn_check``)."""
    th = math.radians(float(rot))
    turn = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return (routes.cut(routes.route_at(res, i), length) - pivot) @ turn.T + pivot + drift


def on_map(maps: Maps, b: int, route: np.ndarray, centre: np.ndarray, r: float) -> float:
    """Share of the route beyond the rim (and short of its last 2 px) on P >= 0.5 (NaN: too little of it)."""
    pts, s = routes.resample(route, 1.0)
    keep = (np.hypot(*(pts - centre).T) > r + 3.0) & (s < s[-1] - 2.0)
    if keep.sum() < 4:
        return float("nan")
    return float(np.mean(maps.at(b, pts[keep]) >= 0.5))


def band_marked(maps: Maps, b: int, route: np.ndarray, apex: np.ndarray, centre: np.ndarray, r: float,
                gap: float = GAP, band: float = BAND, blind: float = 10.0) -> float:
    """Share of the background band (``gap``..``band`` px off the route; not within ``blind`` of the apex, not on
    the grain) marked on the map."""
    lo, hi = route.min(axis=0) - band - 3, route.max(axis=0) + band + 3
    P, org = maps.window(b, lo, hi)
    if P.size == 0:
        return 0.0
    q = route - org - 0.5
    line = np.zeros(P.shape, np.uint8)
    cv2.polylines(line, [np.round(q * 4).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_8, 2)
    d = cv2.distanceTransform(1 - line, cv2.DIST_L2, 5)
    jj, ii = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[0]))
    a, c = apex - org - 0.5, centre - org - 0.5
    ring = (d >= gap) & (d <= band) & (np.hypot(jj - a[0], ii - a[1]) > blind) & (np.hypot(jj - c[0], ii - c[1]) > r + 2)
    return float(np.mean(P[ring] >= 0.5)) if ring.any() else 0.0


def centred(maps: Maps, b: int, route: np.ndarray, nb: int) -> tuple[np.ndarray, float]:
    """The route moved along its normals onto the middle of its band on the map (averaged over bins b-1..b+1), as
    ``learned.centre_route`` centres a reading's route (Params' centring defaults); (route every px, median shift)."""
    from .analyze import Params
    from .learned import _across, _centres_along
    p = Params()
    pts, _ = routes.resample(route, 1.0)
    if len(pts) < 3:
        return pts, 0.0
    nrm = routes.normals(pts)
    offs = np.arange(-10.0, 10.25, 0.5)
    prof = _across(maps, [(x, pts, nrm) for x in (b - 1, b, b + 1) if 0 <= x < nb], offs)
    ci = _centres_along(prof, offs, p)
    if ci is None:
        return pts, 0.0
    return pts + ci[:, None] * nrm, float(np.median(np.abs(ci)))


def ring_marks(maps: Maps, b: int, centre: np.ndarray, r: float, out: float) -> int:
    """Pixels of P >= 0.5 within r + 1 .. r + ``out`` px of the grain's centre."""
    P, org = maps.window(b, centre - r - out - 2, centre + r + out + 2)
    if P.size == 0:
        return 0
    jj, ii = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[0]))
    d = np.hypot(jj + org[0] + 0.5 - centre[0], ii + org[1] + 0.5 - centre[1])
    return int(np.sum((P >= 0.5) & (d > r + 1) & (d <= r + out)))


def select(pred: dict, census: dict, maps: Maps, nb: int, s: Settings, width: int, height: int, log=print):
    """Pseudo-labels from ``pred``'s readings of the grains in ``census`` (id -> census grain); (pseudo, stats).
    pseudo[gid] = {"kind": "germinated" | "clean", "drift": (nb, 2), "traces": {bin: trace}, "clean_bins": [...]},
    a trace as the labelling tool's (``path_xy_ref``, ``apex_xy_ref``, ``view_offset``, ``contact``, ``state``)."""
    fpb = int(pred.get("frames_per_bin", 300))
    exit_pivot = (pred.get("params") or {}).get("rot_pivot", "exit") == "exit"
    firsts = [int(r["length"]["frames"][0]) // fpb for r in pred["grains"] if (r.get("length") or {}).get("frames")]
    rs = min(firsts) if firsts else 0
    lo, hi = rs + 3, nb - 2
    pseudo, stats = {}, {}
    for res in pred["grains"]:
        gid = res["id"]
        g = census.get(gid)
        if g is None:
            continue
        st = {"status": res.get("status")}
        stats[gid] = st
        unsafe = [f for f in res.get("flags", []) if f.startswith(UNSAFE)]
        if unsafe:
            st["skipped"] = "unsafe flags " + ",".join(unsafe)
            continue
        L, drift, rot, start = series(res, fpb, nb)
        centre0 = np.array([res["x"], res["y"]], float)
        status = res.get("status") or ""
        if status.startswith("emerged"):
            if len(res.get("path") or []) < 2:
                st["skipped"] = "no route"
                continue
            pivot = np.asarray(res["exit_xy"] if exit_pivot and res.get("exit_xy") else centre0, float)
            onset_bin = int(res["onset_frame"]) // fpb if res.get("onset_frame") is not None else rs
            until = res.get("observed_until_frame")
            until_bin = int(until) // fpb - s.lost_margin if until is not None else nb
            elig, why = [], {"short": 0, "lost": 0, "conf": 0, "edge": 0, "share": 0}
            for b in range(max(lo, onset_bin), hi + 1):
                if L[b] < s.min_len:
                    why["short"] += 1
                    continue
                if b > until_bin:
                    why["lost"] += 1
                    continue
                conf = trace_confidence(L, b)
                if conf < s.min_conf:
                    why["conf"] += 1
                    continue
                route = drawn(res, b - start, L[b], drift[b], rot[b], pivot)
                if (route[:, 0].min() < s.edge_px or route[:, 1].min() < s.edge_px
                        or route[:, 0].max() > width - s.edge_px or route[:, 1].max() > height - s.edge_px):
                    why["edge"] += 1
                    continue
                share = on_map(maps, b, route, centre0 + drift[b], res["r"])
                if not share >= s.min_share:
                    why["share"] += 1
                    continue
                elig.append((b, conf, share, route))
            st.update(confident_bins=len(elig), rejected_bins=why)
            if not elig:
                st["skipped"] = "no confident bin"
                continue
            chosen = elig[:: s.spacing]
            if len(chosen) > s.cap:
                chosen = [chosen[k] for k in np.round(np.linspace(0, len(chosen) - 1, s.cap)).astype(int)]
            traces, shifts = {}, []
            for b, conf, share, route in chosen:
                route, sh = centred(maps, b, route, nb)
                shifts.append(sh)
                apex = route[-1]
                cutr = routes.cut(route, max(routes.arc(route)[-1] - s.apex_cut, 2.0))
                bm = band_marked(maps, b, route, apex, centre0 + drift[b], res["r"])
                traces[b] = {"bin": b, "state": "partial", "path_xy_ref": np.round(cutr, 2).tolist(),
                             "apex_xy_ref": np.round(apex, 2).tolist(), "view_offset": np.round(drift[b], 2).tolist(),
                             "contact": bool(bm > s.band_max), "length_px": round(float(L[b]), 2),
                             "model_confidence": round(conf, 3), "on_map": round(share, 3), "band_marked": round(bm, 3)}
            pseudo[gid] = {"kind": "germinated", "drift": drift, "traces": traces, "clean_bins": []}
            st.update(pseudo_traces=len(traces), first_confident=elig[0][0], onset_bin=onset_bin,
                      contact=sum(t["contact"] for t in traces.values()),
                      centring_shift_median=round(float(np.median(shifts)), 2))
        elif status == "no_emergence_by_end":
            checked = list(range(lo, hi + 1, 10))
            marks = [ring_marks(maps, b, centre0 + drift[b], res["r"], s.clean_ring) for b in checked]
            clean = float(np.mean([m <= s.clean_max_px for m in marks])) if marks else 0.0
            st.update(clean_share=round(clean, 3), ring_marks_max=int(max(marks) if marks else 0))
            if clean < s.clean_frac:
                st["skipped"] = "surroundings marked"
                continue
            pseudo[gid] = {"kind": "clean", "drift": drift, "traces": {},
                           "clean_bins": list(range(lo, hi + 1, s.clean_every))}
            st["clean_bins"] = len(pseudo[gid]["clean_bins"])
        else:
            st["skipped"] = f"status {status}"
    totals = {"grains_read": len(stats),
              "germinated_used": sum(v["kind"] == "germinated" for v in pseudo.values()),
              "clean_used": sum(v["kind"] == "clean" for v in pseudo.values()),
              "confident_bins": sum(v.get("confident_bins", 0) for v in stats.values()),
              "pseudo_traces": sum(len(v["traces"]) for v in pseudo.values()),
              "clean_bins": sum(len(v["clean_bins"]) for v in pseudo.values()),
              "skipped": {k: v["skipped"] for k, v in stats.items() if "skipped" in v}}
    log(f"pseudo-labels: {totals['grains_read']} grains read; {totals['germinated_used']} germinated + "
        f"{totals['clean_used']} clean used; {totals['confident_bins']} confident bins -> {totals['pseudo_traces']} "
        f"pseudo-traces")
    return pseudo, {"grains": stats, "totals": totals}


# -------------------------------------------------------------------------------------------------------- crops
class View:
    """Training inputs from a cache: the bin, the "before" and the "after" crops, relative to the before crop's
    median (``prototypes/learned_flood/data.py``'s ``CacheView`` and ``normalise``)."""

    def __init__(self, cache_dir: str | Path, ref_bins: int = 3, late_bins: int = 3):
        bins, meta = stack.load(cache_dir)
        self.r = Renderer(bins, meta)
        self.rs = int(meta.get("ref_start", 0))
        self.n_bins = int(meta["n_bins"])
        self.early_bins = list(range(self.rs, self.rs + ref_bins))
        self.late_bins = list(range(self.n_bins - 1 - late_bins, self.n_bins - 1))  # the last bin is often partial

    def sample(self, b: int, cx: float, cy: float, half: int) -> np.ndarray:
        img = self.r.crop(b, cx, cy, half)
        early = np.mean([self.r.crop(e, cx, cy, half) for e in self.early_bins], axis=0)
        late = np.mean([self.r.crop(e, cx, cy, half) for e in self.late_bins], axis=0)
        m = float(np.nanmedian(early))
        x = np.stack([img, early, late]).astype(np.float32)
        return np.nan_to_num((x - m) / IN_SCALE, nan=0.0)


def _line_dist(q: np.ndarray, shape) -> np.ndarray:
    """Distance (px) of every pixel centre from the polyline ``q`` (crop pixel coordinates)."""
    line = np.zeros(shape, np.uint8)
    cv2.polylines(line, [np.round(np.asarray(q) * 4).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1, cv2.LINE_8, 2)
    return cv2.distanceTransform(1 - line, cv2.DIST_L2, 5)


def trace_targets(path: np.ndarray, cx: float, cy: float, half: int, gxy, gr: float, band: bool, blind,
                  inside_unscored: bool = False):
    """(body, w) of a traced crop (``prototypes/tube_net/realdata._targets``, no flat cap): tube within BODY px of
    the route, scored background GAP..BAND px off it (if ``band``), unscored discs (x, y, r), the grain's inside
    background (or unscored over a pore facing the camera)."""
    size = 2 * half
    q = np.asarray(path, float) - [cx - half, cy - half] - 0.5
    d = _line_dist(q, (size, size))
    body = d <= BODY
    w = (d <= BODY) | (band & (d >= GAP) & (d <= BAND))
    jj, ii = np.meshgrid(np.arange(size), np.arange(size))
    for bx, by, rad in blind:
        w &= np.hypot(jj - (bx - cx + half - 0.5), ii - (by - cy + half - 0.5)) > rad
    inside = np.hypot(jj - (gxy[0] - cx + half - 0.5), ii - (gxy[1] - cy + half - 0.5)) <= gr - 1
    if inside_unscored:
        w &= ~inside
    else:
        w |= inside
    body &= ~inside
    return body.astype(np.uint8), w.astype(np.uint8)


class _Crops:
    def __init__(self, view: View, half: int):
        self.view, self.half = view, half
        self.x, self.body, self.w, self.kind, self.grain, self.info = [], [], [], [], [], []
        self.n: dict[str, int] = {"nan": 0}

    def add(self, b: int, cx: float, cy: float, targets, kind: str, gid: str) -> None:
        h, R = self.half, self.view.r
        cx = float(np.clip(round(cx), h + 4, R.width - h - 4))
        cy = float(np.clip(round(cy), h + 4, R.height - h - 4))
        x = self.view.sample(b, cx, cy, h)
        if not np.isfinite(x).all():
            self.n["nan"] += 1
            return
        body, w = targets(cx, cy)
        self.x.append(x.astype(np.float16)); self.body.append(body); self.w.append(w)
        self.kind.append(kind); self.grain.append(gid); self.info.append((b, cx, cy))
        self.n[kind] = self.n.get(kind, 0) + 1

    def arrays(self) -> dict | None:
        if not self.x:
            return None
        return {"x": np.stack(self.x), "body": np.stack(self.body), "w": np.stack(self.w), "tip": None,
                "kind": np.array(self.kind), "grain": np.array(self.grain), "info": np.array(self.info, np.float32)}


def traced_crops(view: View, pseudo: dict, census: dict, s: Settings, seed: int = 0, log=print) -> dict | None:
    """Crops at the pseudo-traces (and clean grains): ``prototypes/self_train/pcrops.build``."""
    half = s.half
    rng = np.random.default_rng(seed)
    lo, hi = view.rs + 3, view.n_bins - 2
    C = _Crops(view, half)
    skipped = 0
    jit = lambda: rng.uniform(-half / 3, half / 3)
    for gid, ps in pseudo.items():
        g = census[gid]
        drift = ps["drift"]
        for b, t in sorted(ps["traces"].items()):
            path = np.asarray(t["path_xy_ref"], float)
            apex = np.asarray(t["apex_xy_ref"], float)
            off = t["view_offset"]
            gxy = (g["x"] + off[0], g["y"] + off[1])
            band = not t["contact"]
            over = bool(s.over_grain) and np.hypot(path[0][0] - gxy[0], path[0][1] - gxy[1]) < s.over_grain * g["r"]
            sa = routes.arc(path)

            def tw(blind_r, path=path, gxy=gxy, band=band, apex=apex, over=over):
                return lambda cx, cy: trace_targets(path, cx, cy, half, gxy, g["r"], band, [(apex[0], apex[1], blind_r)],
                                                    over)
            for _ in range(s.along):
                u = rng.uniform(0, sa[-1])
                C.add(b, np.interp(u, sa, path[:, 0]) + jit(), np.interp(u, sa, path[:, 1]) + jit(), tw(s.tip_blind),
                      "trace", gid)
            for _ in range(s.exits):
                C.add(b, path[0][0] + jit(), path[0][1] + jit(), tw(s.tip_blind), "exit", gid)
            for k in s.neighbours:
                if not lo <= b + k <= hi:
                    continue
                if np.hypot(*(drift[b + k] - drift[b])) > s.max_move:
                    skipped += 1
                    continue
                u = rng.uniform(0, sa[-1])
                C.add(b + k, np.interp(u, sa, path[:, 0]) + jit(), np.interp(u, sa, path[:, 1]) + jit(),
                      tw(s.tip_blind + 1.5 * abs(k)), "neighbour", gid)
        for b in ps["clean_bins"]:
            c = np.array([g["x"], g["y"]]) + drift[b]

            def ring(cx, cy, c=c, r=g["r"]):
                jj, ii = np.meshgrid(np.arange(2 * half), np.arange(2 * half))
                d = np.hypot(jj - (c[0] - cx + half - 0.5), ii - (c[1] - cy + half - 0.5))
                return np.zeros((2 * half, 2 * half), np.uint8), (d <= r + 12.0).astype(np.uint8)
            C.add(b, c[0] + jit(), c[1] + jit(), ring, "clean", gid)
    out = C.arrays()
    if out is not None:
        log(f"traced crops: {len(out['x'])} {C.n}, neighbours skipped (grain moved) {skipped}; scored "
            f"{100 * out['w'].mean():.1f}%, tube among them {100 * out['body'].sum() / max(out['w'].sum(), 1):.1f}%")
    return out


# propagation (prototypes/tube_adapt/propagate.py)
def _dense(path, step: float = 1.0) -> np.ndarray:
    p = np.asarray(path, float)
    sa = routes.arc(p)
    ss = np.append(np.arange(0, sa[-1], step), sa[-1])
    return np.stack([np.interp(ss, sa, p[:, 0]), np.interp(ss, sa, p[:, 1])], 1)


def _resample_n(p: np.ndarray, n: int) -> np.ndarray:
    sa = routes.arc(p)
    t = np.linspace(0, sa[-1], n)
    return np.stack([np.interp(t, sa, p[:, 0]), np.interp(t, sa, p[:, 1])], 1)


def _closest(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric median closest-point distance between two point sets."""
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    return float(max(np.median(d.min(axis=1)), np.median(d.min(axis=0))))


class Region:
    """3-bin-mean registered frames of one region of the movie as uint8 for DIS optical flow, cached."""

    def __init__(self, R: Renderer, cx: float, cy: float, half: int, window_bins: list[int]):
        self.R, self.cx, self.cy, self.half = R, float(cx), float(cy), int(half)
        self._f: dict[int, np.ndarray] = {}
        vals = np.concatenate([self.mean3(b).ravel() for b in window_bins])
        self.lo, self.hi = (float(v) for v in np.nanpercentile(vals, [1, 99]))
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def mean3(self, b: int) -> np.ndarray:
        if b not in self._f:
            bs = [x for x in (b - 1, b, b + 1) if 0 <= x < self.R.n_bins]
            self._f[b] = np.mean([self.R.crop(x, self.cx, self.cy, self.half) for x in bs], axis=0)
        return self._f[b]

    def u8(self, b: int) -> np.ndarray:
        img = np.nan_to_num(self.mean3(b), nan=self.lo)
        return np.clip((img - self.lo) / max(self.hi - self.lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    def to_crop(self, p: np.ndarray) -> np.ndarray:
        return p - [self.cx - self.half, self.cy - self.half] - 0.5

    def to_ref(self, q: np.ndarray) -> np.ndarray:
        return q + [self.cx - self.half, self.cy - self.half] + 0.5

    def flow(self, a: int, c: int) -> np.ndarray:
        return self.dis.calc(self.u8(a), self.u8(c), None)


def _sample_flow(flow: np.ndarray, q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32)
    return np.stack([cv2.remap(flow[..., k], q[:, 0][None], q[:, 1][None], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)[0] for k in (0, 1)], 1)


def carry(reg: Region, pts: np.ndarray, b0: int, b1: int, step: int = 10) -> tuple[dict, dict]:
    """Points (reference coordinates at b0) carried to b1 in ``step``-bin steps: {bin: points} at every step bin and
    {bin: median forward-backward error of the step ending there (px)}."""
    sgn = 1 if b1 > b0 else -1
    seq = list(range(b0, b1, sgn * step)) + [b1]
    q = reg.to_crop(np.asarray(pts, float))
    out, fb = {b0: np.asarray(pts, float).copy()}, {}
    for a, c in zip(seq, seq[1:]):
        fwd, bwd = reg.flow(a, c), reg.flow(c, a)
        q2 = q + _sample_flow(fwd, q)
        back = q2 + _sample_flow(bwd, q2)
        fb[c] = float(np.median(np.hypot(*(back - q).T)))
        q = q2
        out[c] = reg.to_ref(q)
    return out, fb


def _at(carried: dict, b: int) -> np.ndarray:
    """Carried points at bin ``b``, linear in time between the step bins either side."""
    ks = sorted(carried)
    if b <= ks[0]:
        return carried[ks[0]]
    if b >= ks[-1]:
        return carried[ks[-1]]
    j = int(np.searchsorted(ks, b))
    a, c = ks[j - 1], ks[j]
    w = (b - a) / (c - a)
    return (1 - w) * carried[a] + w * carried[c]


def _valid_until(fb: dict, b0: int, fb_max: float) -> int:
    """The last step bin a chain started at ``b0`` reaches before its first step with a forward-backward error above
    ``fb_max`` px (``b0`` itself if the first step is already bad)."""
    last = b0
    for c in sorted(fb, key=lambda c: abs(c - b0)):
        if fb[c] > fb_max:
            break
        last = c
    return last


def prop_targets(route: np.ndarray, cx: float, cy: float, half: int, gxy, gr: float, s: Settings, band: bool,
                 blind_discs=(), blind_lines=(), inside_unscored: bool = False):
    """(body, w) of a propagated crop (``propagate.targets``, never negative): body within ``prop_body_px`` of the
    route, scored background ``prop_gap_px``..``prop_band_px`` from it, unscored discs and polylines (within
    ``prop_ext_margin``), the grain's inside background (or unscored)."""
    size = 2 * half
    off = np.array([cx - half, cy - half]) + 0.5
    d = _line_dist(route - off, (size, size))
    body = d <= s.prop_body_px
    w = body | (band & (d >= s.prop_gap_px) & (d <= s.prop_band_px))
    jj, ii = np.meshgrid(np.arange(size), np.arange(size))
    for bx, by, rad in blind_discs:
        w &= np.hypot(jj - (bx - off[0]), ii - (by - off[1])) > rad
    for ln in blind_lines:
        if len(ln) >= 2:
            w &= _line_dist(np.asarray(ln) - off, (size, size)) > s.prop_ext_margin
    inside = np.hypot(jj - (gxy[0] - off[0]), ii - (gxy[1] - off[1])) <= gr - 1
    if inside_unscored:
        w &= ~inside
    else:
        w |= inside
    body &= ~inside
    return body.astype(np.uint8), w.astype(np.uint8)


def propagated_crops(view: View, pseudo: dict, census: dict, s: Settings, seed: int = 0, log=print) -> dict | None:
    """The pseudo-traced tube carried by optical flow between consecutive pseudo-traces (``between``) and after the
    last one (``after``): ``prototypes/tube_adapt/propagate.build`` on the pseudo-labels, the tracker's drift as the
    grain's place, no negatives."""
    half, every = s.half, s.prop_every
    R = view.r
    rng = np.random.default_rng(seed)
    lo, hi = view.rs + 3, view.n_bins - 2
    C = _Crops(view, half)
    st = {"bins_both": 0, "bins_fwd_only": 0, "bins_bwd_only": 0, "skipped_dev": 0, "skipped_unreliable": 0,
          "skipped_shrank": 0, "after_stopped_fb": 0, "intervals": 0}
    jit = lambda: rng.uniform(-half / 3, half / 3)

    def along(route):
        sa = routes.arc(route)
        u = rng.uniform(0, sa[-1])
        return np.interp(u, sa, route[:, 0]), np.interp(u, sa, route[:, 1])

    for gid, ps in pseudo.items():
        g = census[gid]
        tr = sorted((b, t) for b, t in ps["traces"].items() if len(t["path_xy_ref"]) >= 2)
        if not tr:
            continue
        answered = [b for b, _ in tr]
        centre = lambda t: np.array([[g["x"] + t["view_offset"][0], g["y"] + t["view_offset"][1]]])
        allp = np.concatenate([np.asarray(t["path_xy_ref"], float) for _, t in tr])
        cx, cy = (allp.min(axis=0) + allp.max(axis=0)) / 2
        hw = int(min(400, max(96, np.max(np.abs(allp - [cx, cy])) + 64)))
        reg = Region(R, cx, cy, hw, answered)
        over = lambda t: bool(s.over_grain) and np.hypot(*(np.asarray(t["path_xy_ref"][0]) - centre(t)[0])) < \
            s.over_grain * g["r"]

        def emit(b, route, gc, kind, band, ov, ext=None, start=0, state="partial"):
            blind = [(route[-1][0], route[-1][1], (s.prop_tip_blind if kind == "between" else
                                                   max(s.prop_tip_blind, s.prop_band_px + 4.0))
                      if state == "full" else s.prop_partial_blind)]
            lines = [ext] if ext is not None and len(ext) >= 2 else []
            tg = lambda cx_, cy_: prop_targets(route, cx_, cy_, half, gc, g["r"], s, band, blind, lines, ov)
            C.add(b, *(np.array(along(route)) + [jit(), jit()]), tg, kind, gid)
            if (b - start) // every % 2 == 0:
                C.add(b, route[0][0] + jit(), route[0][1] + jit(), tg, kind, gid)

        # between consecutive pseudo-traces
        for (b1, t1), (b2, t2) in zip(tr, tr[1:]):
            r1, r2 = _dense(t1["path_xy_ref"]), _dense(t2["path_xy_ref"])
            l1, l2 = routes.arc(r1)[-1], routes.arc(r2)[-1]
            if l2 < l1 - max(5.0, 0.15 * l1):
                st["skipped_shrank"] += 1
                continue
            st["intervals"] += 1
            fwd, fb_f = carry(reg, np.vstack([r1, centre(t1)]), b1, b2, s.flow_step)
            bwd, fb_b = carry(reg, np.vstack([r2, centre(t2)]), b2, b1, s.flow_step)
            vf, vb = _valid_until(fb_f, b1, s.prop_fb_max), _valid_until(fb_b, b2, s.prop_fb_max)
            n1 = int(np.searchsorted(routes.arc(r2), l1, side="right"))  # r2's first L(b1) px
            band = not t1["contact"] and not t2["contact"]
            ov = over(t1)
            for b in range(b1 + 3, b2 - 2, every):
                if not lo <= b <= hi:
                    continue
                okf, okb = b <= vf, b >= vb
                f, k = _at(fwd, b), _at(bwd, b)
                fr, fc = f[:-1], f[-1]
                kr, kc = k[:-1], k[-1]
                kp, ext = kr[:max(n1, 2)], kr[max(n1, 2) - 1:]
                n = max(int(round(l1)) + 1, 2)
                if okf and okb:
                    if _closest(fr, kp) > s.prop_max_dev:
                        st["skipped_dev"] += 1
                        continue
                    w = (b - b1) / (b2 - b1)
                    route, gc = (1 - w) * _resample_n(fr, n) + w * _resample_n(kp, n), (1 - w) * fc + w * kc
                    st["bins_both"] += 1
                elif okf and b - b1 <= s.prop_one_sided:
                    route, gc, ext = _resample_n(fr, n), fc, (ext if okb else None)
                    st["bins_fwd_only"] += 1
                elif okb and b2 - b <= s.prop_one_sided:
                    route, gc = _resample_n(kp, n), kc
                    st["bins_bwd_only"] += 1
                else:
                    st["skipped_unreliable"] += 1
                    continue
                emit(b, route, gc, "between", band, ov, ext, b1)
        # after the last pseudo-trace
        bl, tl = tr[-1]
        end = hi
        if end >= bl + 3:
            rl = _dense(tl["path_xy_ref"])
            fwd, fb_f = carry(reg, np.vstack([rl, centre(tl)]), bl, end, s.flow_step)
            stop = _valid_until(fb_f, bl, s.prop_fb_max)
            st["after_stopped_fb"] += int(stop < end)
            for b in range(bl + 3, stop + 1, every):
                f = _at(fwd, b)
                emit(b, f[:-1], f[-1], "after", not tl["contact"], over(tl), None, bl)
    out = C.arrays()
    if out is not None:
        log(f"propagated crops: {len(out['x'])} {C.n} {st}; scored {100 * out['w'].mean():.1f}%, tube among them "
            f"{100 * out['body'].sum() / max(out['w'].sum(), 1):.1f}%")
    return out


# --------------------------------------------------------------------------------------------------- training
def load_shards(paths) -> dict:
    """Training crops of shards (``x`` float16, ``body`` uint8, ``tip`` float16, ``w`` uint8: ones where the shard
    has no weight map)."""
    parts = []
    for f in paths:
        z = np.load(f)
        d = {"x": z["x"].astype(np.float16, copy=False), "body": z["body"],
             "tip": z["tip"].astype(np.float16, copy=False) if "tip" in z.files else None,
             "w": z["w"] if "w" in z.files else np.ones(z["body"].shape, np.uint8)}
        if d["tip"] is None:
            d["tip"] = np.zeros(d["body"].shape, np.float16)
        parts.append(d)
    return {k: np.concatenate([p[k] for p in parts]) for k in ("x", "body", "tip", "w")}


def replay_sets(paths, s: Settings) -> list[tuple[str, dict, float]]:
    """Replay of the starting network's training data: shards with a weight map are trace crops (pooled, a
    ``share_real`` of each batch), shards without one synthetic (an equal part each of ``share_synthetic``)."""
    real, synth = [], []
    for f in paths:
        with np.load(f) as z:
            (real if "w" in z.files else synth).append(f)
    out = []
    if real:
        out.append(("replay traces", load_shards(real), s.share_real))
    for f in synth:
        out.append((f"replay {Path(f).stem}", load_shards([f]), s.share_synthetic / len(synth)))
    return out


def augment(x, body, tip, g, crop: int = 0, w=None):
    """0.8.0's augmentation (``prototypes/learned_flood/train.augment``): a random crop, rotation and flip for the
    batch, a gain and offset per sample, noisier current bin than references."""
    import torch
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
    sig = torch.rand(n, 1, 1, 1, generator=g) * torch.tensor([0.25, 0.08, 0.08]).view(1, 3, 1, 1)
    x = x + sig * torch.randn(x.shape, generator=g)
    return x, body, tip, w


def losses(logits, body, tip, w=None) -> dict:
    """0.8.0's losses: BCE (pos_weight 2) + Dice + 0.5 tip, unscored pixels weight 0."""
    import torch
    import torch.nn.functional as F
    lb, lt = logits[:, 0], logits[:, 1]
    w = torch.ones_like(body) if w is None else w
    n = w.sum() + 1.0
    bce = (F.binary_cross_entropy_with_logits(lb, body, pos_weight=torch.tensor(2.0, device=lb.device),
                                              reduction="none") * w).sum() / n
    p = torch.sigmoid(lb) * w
    dice = 1.0 - (2 * (p * body).sum() + 1.0) / (p.sum() + (body * w).sum() + 1.0)
    tip_l = (F.binary_cross_entropy_with_logits(lt, tip, reduction="none") * (1.0 + 10.0 * tip) * w).sum() / n
    return {"bce": bce, "dice": dice, "tip": tip_l, "total": bce + dice + 0.5 * tip_l}


def fine_tune(start: str | Path, sets: list[tuple[str, dict, float]], steps: int, seed: int, s: Settings,
              device: str | None = None, log=print) -> dict:
    """``steps`` steps from ``start`` on batches drawn from ``sets`` ((name, crops, share)); BatchNorm statistics
    frozen. Returns the checkpoint dict (the starting checkpoint's options kept)."""
    import torch
    from .learned import _unet
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    tot = sum(sh for _, _, sh in sets)
    sets = [(n, d, sh / tot) for n, d, sh in sets]
    for n, d, sh in sets:
        log(f"{100 * sh:4.1f}% of each batch: {n}, {len(d['x'])} crops; tube {100 * d['body'].mean():.1f}%, "
            f"scored {100 * d['w'].mean():.0f}%")
    counts = [int(round(s.batch * sh)) for _, _, sh in sets]
    counts[-1] = s.batch - sum(counts[:-1])
    ck = torch.load(str(start), map_location="cpu", weights_only=False)
    widths, norm = ck.get("widths", (16, 32, 64, 128)), ck.get("norm", "group")
    net = _unet(widths, norm)
    net.load_state_dict(ck["state"])
    dev = torch.device(device)
    net = net.to(dev).train()
    for m in net.modules():  # BatchNorm statistics frozen: full-frame inference keeps the starting normalisation
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
    opt = torch.optim.AdamW(net.parameters(), lr=s.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=s.lr, total_steps=steps, pct_start=0.1)

    def batch(d, idx):
        tip = d["tip"][idx] if d.get("tip") is not None else np.zeros(d["body"][idx].shape, np.float32)
        return tuple(torch.from_numpy(np.asarray(a).astype(np.float32)) for a in (d["x"][idx], d["body"][idx], tip,
                                                                                    d["w"][idx]))
    started, run = time.time(), {}
    for step in range(1, steps + 1):
        parts = [batch(d, rng.integers(len(d["x"]), size=n)) for (_, d, _), n in zip(sets, counts) if n > 0]
        xb, bb, tb, wb = (torch.cat(t) for t in zip(*parts))
        x, body, tip, w = augment(xb, bb, tb, g, s.crop, wb)
        x, body, tip, w = (t.to(dev) for t in (x, body, tip, w))
        L = losses(net(x), body, tip, w)
        opt.zero_grad()
        L["total"].backward()
        opt.step()
        sched.step()
        for k, v in L.items():
            run[k] = run.get(k, 0.0) + float(v.detach())
        if step % 250 == 0 or step == steps:
            k = 250 if step % 250 == 0 else step % 250
            log(f"step {step:5d} {time.time() - started:6.0f}s  " + " ".join(f"{a} {v / k:.4f}" for a, v in run.items()))
            run = {}
    return {"state": {k: v.cpu() for k, v in net.state_dict().items()}, "widths": widths, "norm": norm,
            "bg_px": ck.get("bg_px", 0)}


def soup(cks: list[dict]) -> dict:
    """Uniform weight average of fine-tunes of one starting network (a "model soup", Wortsman et al. 2022): floating
    tensors averaged (the frozen BatchNorm statistics are the same in all), integer buffers from the first."""
    import torch
    state = {}
    for k, v in cks[0]["state"].items():
        if torch.is_floating_point(v):
            state[k] = (sum(c["state"][k].double() for c in cks) / len(cks)).to(v.dtype)
        else:
            state[k] = v.clone()
    return {**{k: v for k, v in cks[0].items() if k != "state"}, "state": state}


# ------------------------------------------------------------------------------------------------------- driver
def selftrain(cache_dir: str | Path, out: str | Path, predictions: str | Path | None = None,
              start: str | Path | None = None, replay=None, steps: int = 1500, seed: int = 0, runs: int = 1,
              work: str | Path | None = None, settings: Settings | None = None, device: str | None = None,
              log=print) -> dict:
    """Adapt the tube network to the movie in ``cache_dir`` on its own confident readings; write it to ``out`` and
    its record (settings, inputs' sha1s, pseudo-labels, crops, timings) to ``out`` with suffix ``.json``.
    ``predictions``: SparseTrack's default readings of the census grains with ``start`` (else read now, into
    ``work``/readings). ``start``: the starting network (default the shipped one). ``replay``: shards of the starting
    network's own training data (default ``DEFAULT_REPLAY``, only for the shipped network). ``runs`` > 1: that many
    fine-tunes (seeds ``seed`` .. ``seed + runs - 1``, each on its own crops) averaged into one network (``soup``)."""
    from . import learned
    t0 = time.time()
    s = settings or Settings()
    cache_dir, out = Path(cache_dir), Path(out)
    start = Path(start) if start else learned.MODEL
    start_sha1 = sha1(start)
    if replay is None:
        if start_sha1 != sha1(learned.MODEL):
            raise ValueError("replay: give the shards of the starting network's own training data")
        replay = list(DEFAULT_REPLAY)
    replay = [Path(p) for p in replay]
    missing = [str(p) for p in replay if not p.exists()]
    if missing:
        raise FileNotFoundError(f"replay shards not found: {', '.join(missing)}")
    work = Path(work) if work else out.parent / f"{out.stem}_selftrain"
    census_doc = json.loads((cache_dir / "grains.json").read_text())
    census_list = list(census_doc["grains"].values()) if isinstance(census_doc["grains"], dict) else census_doc["grains"]
    census = {g["id"]: g for g in census_list}
    timing = {}
    # 1. readings of every census grain with the starting network
    if predictions is None:
        from .analyze import Params, analyze
        log(f"reading {len(census)} census grains with {start.name} (SparseTrack's defaults, no tip-trajectory reader)")
        pred = analyze(cache_dir, work / "readings", params=Params(model=str(start), tiptraj="off"), log=log)
        pred_path = work / "readings" / "predictions.json"
    else:
        pred_path = Path(predictions)
        pred = json.loads(pred_path.read_text())
        made_by = (pred.get("params") or {}).get("model")
        if made_by and Path(made_by).name != start.name:
            log(f"WARNING: the predictions were read with {Path(made_by).name}, not {start.name}")
    timing["readings_s"] = round(time.time() - t0, 1)
    # 2. pseudo-labels on the starting network's maps
    t1 = time.time()
    maps = Maps.load(learned.prob_cache(cache_dir, start, log))
    bins, meta = stack.load(cache_dir)
    nb, (h, w) = int(meta["n_bins"]), bins.shape[1:]
    pseudo, stats = select(pred, census, maps, nb, s, w, h, log)
    del maps
    timing["select_s"] = round(time.time() - t1, 1)
    record = {"schema": SCHEMA, "sparsetrack_version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "cache": str(cache_dir), "census_grains": len(census), "out": str(out),
              "start": {"path": str(start), "sha1": start_sha1},
              "replay": [{"path": str(p), "sha1": sha1(p)} for p in replay],
              "predictions": {"path": str(pred_path), "sha1": sha1(pred_path) if pred_path.exists() else None,
                              "grains_source": pred.get("grains_source"), "method": pred.get("method")},
              "steps": steps, "seed": seed, "runs": runs, "settings": asdict(s), "pseudo": stats["totals"]}
    # 3-4. crops and fine-tuning: ``runs`` fine-tunes (seeds seed .. seed + runs - 1, each on its own crops), averaged
    view = View(cache_dir)
    cks, crops, replayed, t_crops, t_train = [], [], None, 0.0, 0.0
    for k in range(runs if stats["totals"]["pseudo_traces"] else 0):
        t1 = time.time()
        traced = traced_crops(view, pseudo, census, s, seed + k, log)
        prop = propagated_crops(view, pseudo, census, s, seed + k, log) if s.propagate and traced is not None else None
        t_crops += time.time() - t1
        crops.append({"seed": seed + k, "traced": 0 if traced is None else len(traced["x"]),
                      "propagated": 0 if prop is None else len(prop["x"])})
        if traced is None:
            break
        t1 = time.time()
        replayed = replayed if replayed is not None else replay_sets(replay, s)
        sets = [("pseudo traced", traced, s.share_traced + (0.0 if prop is not None else s.share_propagated))]
        if prop is not None:
            sets.append(("pseudo propagated", prop, s.share_propagated))
        cks.append(fine_tune(start, sets + replayed, steps, seed + k, s, device, log))
        t_train += time.time() - t1
        del traced, prop, sets
    record["crops"] = crops[0] if crops else {"traced": 0, "propagated": 0}
    if len(crops) > 1:
        record["crops_by_run"] = crops
    timing.update(crops_s=round(t_crops, 1), train_s=round(t_train, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cks:
        # nothing confident to learn from: the starting network, unchanged
        shutil.copy(start, out)
        record.update(unchanged=True, reason="no confident reading to learn from")
        log("no pseudo-traces: the starting network is kept unchanged")
    else:
        ck = soup(cks) if len(cks) > 1 else cks[0]
        record["unchanged"] = False
        ck["selftrain"] = record
        import torch
        torch.save(ck, out)
    timing["total_s"] = round(time.time() - t0, 1)
    record["timing"] = timing
    record["out_sha1"] = sha1(out)
    out.with_suffix(".json").write_text(json.dumps({**record, "pseudo_grains": stats["grains"]}, indent=1, default=float))
    log(f"adapted network -> {out} ({timing['total_s']:.0f} s)")
    return record
