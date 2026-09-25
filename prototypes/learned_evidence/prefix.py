"""Prefix decoder ("end-state anchoring"): every earlier tube is a prefix of the final tube.

A pollen tube grows at its tip, so once a tube is there its earlier states are the same tube, shorter, apart from
deformation (it turns with its grain, sways, or is held by the substrate while the grain drifts). Per grain the
whole movie is explained at once by

  * one end-state path, traced where the tube is longest and clearest: the grain's last well-visible state (the
    last bins, or earlier when its tube bursts, fades or leaves: ``ref="auto"``, ``last_good_bin``),
  * a monotone growth curve L(t) along it (0.5 px levels, <= ``vmax_px`` per bin), and
  * a smooth per-bin deformation of that path (rigid rotation about the grain centre, sideways sway growing along
    the tube, or a tube held by the substrate while its grain drifts),

found jointly by exact dynamic programming over time x (deformation, length).  The evidence is the learned tube
probability read along the deformed path (log-odds), plus an image term near the exit that can add evidence for a
young tube the network misses but never veto one it sees.  Candidate end-state paths and deformation models are
compared by their whole-movie score; a candidate that leaves the rim tangentially, or whose far part was there before
its base, is another grain's tube.

Anchored mode (``anchors``): the end state is given as a trace in the labelling tool's form, a polyline in
reference coordinates at a bin, clicked from the tube's exit to its apex (a human's trace, or a machine proposal a
human accepted).  Path search and the ownership tests are skipped, since the trace says which tube is the grain's.
The path is the trace itself, each point moved sideways onto the evidence within ``anchor_snap_px``.  Lengths are
reported in the trace's own terms: the part clicked inside the rim is added and the trace's bin reads the traced
length; germination is called on the decoder's own scale, where the onset calibration applies.

Output is in SparseTrack's prediction schema (as ``reach.analyze``), so ``sparsetrack.evaluate.score`` applies.

On five held-out synthetic movies (186 grains; each version frozen before its one look) it put 84.6% of lengths
within tolerance, against the per-bin decoder's 71.0%. Scored as the labelling tool records brackets and traces, it
put 85% and 84% within tolerance, against 71% and 70% (annotators at 2 and 4 px). Anchored on one simulated human
trace per grain it put 89.3% of the lengths before the anchor within tolerance; anchored on the per-bin decoder's own
trace, fewer than the per-bin decoder alone. Onsets are called early against one-bin human brackets, like the
per-bin decoder's; the onset calibration (``calibrate.py``) addresses that.

Designed on synthetic movies of thin tubes. On the sample movie, which is real but has thick double-walled tubes and
grains moving tens of px, it is not yet reliable: it called 8 of 37 grains tubeless, where the per-bin decoder found
a tube on every one. A synthetic thick-tube movie showed the same gap. The image term used to veto young tubes that
did not yet look mature there, hence "support".
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

import sparsetrack.analyze as A
from sparsetrack import stack
from sparsetrack.evaluate import PRED_SCHEMA
from sparsetrack.render import Renderer

from .track import track_grain


@dataclass
class Params:
    half: int = 150              # crop half-size (px); grains whose path reaches the edge are read again with big
    big: int = 300
    scale: float = 16.0          # probability cache scale
    thr: float = 0.5             # end-state region threshold on the mean probability of the last bins
    end_bins: int = 5            # last full bins averaged for the end state
    rim_band: float = 5.0        # own ring: r-1 .. r+rim_band
    step: float = 0.5            # arclength sampling (px)
    lat: float = 1.0             # evidence = max over lateral offsets {-lat, 0, +lat}
    clip: float = 4.0            # evidence = clip(logit P, -clip, clip) / clip
    skip_px: float = 0.0         # evidence this close to the rim is ignored
    vmax_px: float = 4.0         # growth cap per bin
    extend_px: float = 6.0       # the end-state path is extended this far past its tip (faint tips)
    # deformation models
    rot_max: float = 45.0
    rot_step: float = 1.5
    rot_dmax: int = 2            # states per bin
    rot_pen: float = 0.02        # score per state step
    sway_max: float = 9.0
    sway_step: float = 1.0
    sway_s0: float = 25.0        # sway reaches full amplitude this far along
    sway_dmax: int = 1
    sway_pen: float = 0.02
    bend_px: float = 14.0        # anchored tube: only this much of the base follows the grain
    move_min_px: float = 1.5     # the anchored model is tried only for grains that moved this much
    models: tuple = ("rot", "sway", "anchor")
    # candidate paths
    n_tips: int = 6
    tip_nms_px: float = 8.0
    min_path_px: float = 4.0
    seed_band: float = 2.5       # rim contacts are taken on r-1 .. r+seed_band
    bridge_px: float = 8.0       # a region this close to the rim is bridged to it when none touches it
    exit_max_deg: float = 60.0   # a candidate leaving the rim more tangentially than this is a tube passing by
    birth_margin: float = 4.0    # ...and one whose far part was there this many bins before its base is foreign
    beyond: float = 0.5          # candidate score = claimed evidence - beyond * positive evidence past the front
    # end state: "end" = the last bins; "auto" = the last well-visible state (see GrainStack.last_good_bin)
    ref: str = "auto"
    ref_frac: float = 0.7
    # anchored mode
    anchor_snap_px: float = 3.0  # the trace is snapped to the evidence within this distance (0: used as drawn)
    anchor_pin: bool = True      # the length at the trace's bin is the trace's
    max_half: int = 450          # a trace reaching farther than big is read with a crop this big at most
    # outputs
    onset_px: float = 2.0
    min_tube_px: float = 6.0
    tip_px: float = 0.0          # added to every non-zero length
    young: float = 1.5           # young-tube correction (see finalise)
    young_px: float = 3.0
    cap_frac: float = 1.0        # stopped-tube correction (see finalise)
    stop_bins: int = 20
    cap_max_px: float = 3.0
    stop_tol: float = 0.5
    tracker: str = "template"    # grain tracking: "template" (track.py) or "local_shifts" (SparseTrack)
    # frame edge: "bin" takes a pixel whose source is outside the frame as no evidence (P = 0.5) in that bin only, so a
    # grain drifting to an edge keeps its tube where it is visible; "movie" blocks it for the whole movie once it
    # leaves the frame in any bin (the earlier behaviour)
    edge_mask: str = "bin"
    edge_margin: float = 1.0
    image_term: bool = True      # image evidence near the exit (see image_evidence)
    img_weight: float = 3.0
    img_s_max: float = 6.0
    img_half: float = 5.0        # cross-section half-width sampled (px)
    img_pre_gap: int = 4         # the pre-emergence reference ends this many bins before the first front >= 1 px
    img_ref_bins: int = 6
    img_ref_start: int = 8       # ...and starts no later than this bin (grains still landing before)
    img_t0: float = 4.0          # image evidence 0 at this many noise sigmas...
    img_tk: float = 2.0          # ...and +/-1 this many sigmas either side
    img_sig_floor_gl: float = 0.3
    img_fuse: str = "support"    # "support" (E_P + img_weight * max(E_img, 0)), "sum" (the image may also veto) or "max"
    img_ctrl_deg: tuple = (60.0, 120.0, 180.0, 240.0, 300.0)
    img_state_band: int = 3      # image pass: deformation states within this of the learned-evidence track
    img_min_amp: float = 3.0     # grey levels: fainter templates are not used


# ----------------------------------------------------------------------------- sampling helpers
def sample(img: np.ndarray, x: np.ndarray, y: np.ndarray, border: float = 0.0) -> np.ndarray:
    """Bilinear samples at (x, y); OpenCV's remap takes maps narrower than 32,767 columns, so long maps are split."""
    shp = x.shape
    xf = np.ascontiguousarray(x.reshape(-1), np.float32)
    yf = np.ascontiguousarray(y.reshape(-1), np.float32)
    out = np.empty(xf.size, np.float32)
    for i in range(0, xf.size, 32000):
        out[i:i + 32000] = cv2.remap(img, xf[None, i:i + 32000], yf[None, i:i + 32000], cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=border)[0]
    return out.reshape(shp)


def resample(path: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.hypot(*np.diff(path, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(ss, s, path[:, 0]), np.interp(ss, s, path[:, 1])], axis=1), ss


def normals(pts: np.ndarray) -> np.ndarray:
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    return np.stack([-tang[:, 1], tang[:, 0]], axis=1)


# ----------------------------------------------------------------------------- dynamic programming
def dp_joint(E: np.ndarray, vmax: int, dmax: int, pen: float, pin: int | None,
             allowed: np.ndarray | None = None, pin_level: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Joint MAP over time of (deformation state k, front level l): the front at level l claims path
    points [0, l); l is non-decreasing with steps <= vmax; k changes by <= dmax per bin at pen per step;
    with ``pin`` the last bin's state is fixed (no deformation at the end state), with ``pin_level`` its level.
    E: (T, K, n) evidence. Returns (levels (T,), states (T,), score)."""
    T, K, n = E.shape
    C = np.concatenate([np.zeros((T, K, 1), np.float32), np.cumsum(E, axis=2, dtype=np.float32)], axis=2)
    N1 = n + 1
    if allowed is not None:  # (T, K) states the track may use
        C = np.where(allowed[:, :, None], C, np.float32(-1e9))
    V = C[0].astype(np.float64)
    bl = np.zeros((T, K, N1), np.int16)
    bk = np.zeros((T, K, N1), np.int8)
    idx = np.arange(N1)
    neg = np.full((K, vmax), -np.inf)
    for t in range(1, T):
        pad = np.concatenate([neg, V], axis=1)
        win = np.lib.stride_tricks.sliding_window_view(pad, vmax + 1, axis=1)  # win[k, l, j] = V[k, l - vmax + j]
        j = np.argmax(win, axis=2)
        Vl = np.take_along_axis(win, j[..., None], axis=2)[..., 0]
        Ll = (idx[None, :] - vmax + j).astype(np.int16)
        if dmax == 0 or K == 1:
            best, bd, bls = Vl, np.zeros((K, N1), np.int8), Ll
        else:
            best = Vl.copy()
            bd = np.zeros((K, N1), np.int8)
            bls = Ll.copy()
            for d in range(-dmax, dmax + 1):
                if d == 0:
                    continue
                src = np.full((K, N1), -np.inf)
                srcL = np.zeros((K, N1), np.int16)
                if d > 0:
                    src[d:], srcL[d:] = Vl[:K - d], Ll[:K - d]
                else:
                    src[:K + d], srcL[:K + d] = Vl[-d:], Ll[-d:]
                val = src - pen * abs(d)
                better = val > best
                best = np.where(better, val, best)
                bd = np.where(better, d, bd).astype(np.int8)
                bls = np.where(better, srcL, bls).astype(np.int16)
        V = C[t] + best
        bl[t], bk[t] = bls, bd
    if pin_level is not None:
        pin_level = int(min(max(pin_level, 0), n))
    if pin is not None:
        kf = pin
        lf = int(np.argmax(V[pin])) if pin_level is None else pin_level
    elif pin_level is not None:
        kf, lf = int(np.argmax(V[:, pin_level])), pin_level
    else:
        kf, lf = np.unravel_index(int(np.argmax(V)), V.shape)
    score = float(V[kf, lf])
    lev = np.zeros(T, np.int64)
    st = np.zeros(T, np.int64)
    lev[-1], st[-1] = lf, kf
    for t in range(T - 1, 0, -1):
        lv, k = lev[t], st[t]
        lev[t - 1] = bl[t, k, lv]
        st[t - 1] = k - bk[t, k, lv]
    return lev, st, score


# ----------------------------------------------------------------------------- per-grain stacks
class GrainStack:
    """Grain-frame crops of the probability and image caches (per-grain registration on the image), up to the end
    state: the anchor's bin when one is given, else the grain's last well-visible state (``ref="auto"``) or the
    last bin."""

    def __init__(self, RP: Renderer, R_img: Renderer, meta: dict, grain: dict, others: list[dict], p: Params,
                 half: int, anchor_bin: int | None = None):
        self.rs, self.nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
        self.fpb = int(meta["frames_per_bin"])
        gx, gy, gr = float(grain["x"]), float(grain["y"]), float(grain["r"])
        self.gx, self.gy, self.gr, self.half = gx, gy, gr, half
        self.centre = half - 0.5
        T = self.nb - self.rs
        img = np.stack([R_img.crop(b, gx, gy, half) for b in range(self.rs, self.nb)])
        if np.isnan(img).any():
            img = np.nan_to_num(img, nan=float(np.nanmedian(img)))
        if p.tracker == "template":
            ls, self.ncc, _ = track_grain(img, self.centre, gr)
        else:
            ls = A.local_shifts(img, self.centre, gr, 12.0, 3)
        self.ls = ls
        size = (2 * half, 2 * half)
        self.P = np.empty((T, 2 * half, 2 * half), np.float32)
        self.img = np.empty_like(self.P)
        shifts = R_img.shifts[self.rs:self.nb] + ls
        j = np.arange(2 * half, dtype=np.float64) + 0.5
        for i, b in enumerate(range(self.rs, self.nb)):
            m = np.float32([[1, 0, -ls[i, 0]], [0, 1, -ls[i, 1]]])
            pb = np.nan_to_num(RP.crop(b, gx, gy, half) / p.scale).astype(np.float32)
            self.P[i] = cv2.warpAffine(pb, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            self.img[i] = cv2.warpAffine(img[i].astype(np.float32), m, size, flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
            if p.edge_mask == "bin":  # outside the frame in this bin: no evidence either way
                xs, ys = gx - half + j + shifts[i, 0], gy - half + j + shifts[i, 1]
                self.P[i][((ys < p.edge_margin) | (ys >= R_img.height - p.edge_margin))[:, None] |
                          ((xs < p.edge_margin) | (xs >= R_img.width - p.edge_margin))[None, :]] = 0.5
        yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float64)
        self.rg = np.hypot(xx - self.centre, yy - self.centre)
        blocked = self.rg < gr - 1.0
        ref_x, ref_y = gx - half + xx + 0.5, gy - half + yy + 0.5
        if p.edge_mask == "movie":
            blocked |= ~((ref_x + shifts[:, 0].min() >= 1) & (ref_x + shifts[:, 0].max() < R_img.width - 1) &
                         (ref_y + shifts[:, 1].min() >= 1) & (ref_y + shifts[:, 1].max() < R_img.height - 1))
        self.rings = []
        self.other_discs = np.zeros_like(blocked)
        for o in others:
            ox, oy = o["x"] - gx + self.centre, o["y"] - gy + self.centre
            if -o["r"] - 10 < ox < 2 * half + o["r"] + 10 and -o["r"] - 10 < oy < 2 * half + o["r"] + 10:
                d = np.hypot(xx - ox, yy - oy)
                self.other_discs |= d < o["r"] + 1.0
                self.rings.append((d >= o["r"] + 1.0) & (d <= o["r"] + p.rim_band))
        self.blocked = blocked | self.other_discs
        self.own_ring = (self.rg >= gr - 1.0) & (self.rg <= gr + p.rim_band)
        self.T = T
        self.T_all = T
        if anchor_bin is not None:
            te = min(max(int(anchor_bin) - self.rs + 1, 2), T)
        else:
            te = self.last_good_bin(p) if p.ref == "auto" else None
            if te is not None:
                te = max(8, te)
        if te is not None and te < T:
            # the end state is earlier than the last bin (the tube bursts, fades or leaves, or the anchor's trace
            # is there): later bins are not decoded, their lengths are held
            self.P, self.img, self.ls, self.T = self.P[:te], self.img[:te], self.ls[:te], te
        # the bin whose geometry the path describes: an anchor's own bin, else a bin inside the end-state mean
        self.iref = self.T - 1 if anchor_bin is not None else self.T - 2

    def last_good_bin(self, p: Params) -> int | None:
        """Number of bins up to the grain's last well-visible state: the last bin whose evidence region attached
        to the grain (median over 5 bins) is still at least ref_frac of its largest; None = the whole movie."""
        T = self.P.shape[0]
        reach_ok = (self.rg <= self.gr + p.bridge_px + 250) & ~self.blocked
        near = (self.rg <= self.gr + max(p.rim_band, p.bridge_px)) & ~self.blocked
        size = np.zeros(T)
        for t in range(T):
            m = (self.P[t] > p.thr) & reach_ok
            if not (m & near).any():
                continue
            _, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
            ids = np.unique(lab[m & near])
            size[t] = float(np.isin(lab, ids[ids > 0]).sum())
        sm = np.array([np.median(size[max(0, t - 2):t + 3]) for t in range(T)])
        if sm.max() <= 0:
            return None
        good = np.nonzero(sm >= p.ref_frac * sm.max())[0]
        last = int(good[-1]) + 1
        return None if last >= T - 3 else last


# ----------------------------------------------------------------------------- end state and candidate paths
def end_state_paths(G: GrainStack, p: Params) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Candidate centrelines (crop x, y from the rim outwards) of the grain's tube at the end state."""
    T = G.T
    Pend = G.P[max(0, T - 1 - p.end_bins):T - 1].mean(axis=0)
    m = (Pend > p.thr) & ~G.blocked
    _, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if (m & G.own_ring).any():
        ids = np.unique(lab[m & G.own_ring])
        comp = np.isin(lab, ids[ids > 0])
        # rim contacts: pieces of the region on a thin band at the rim (a curl lying in the wider ring still gets
        # tips); a faint base that starts beyond it falls back to the whole ring
        thin = comp & (G.rg >= G.gr - 1.0) & (G.rg <= G.gr + p.seed_band)
        n_c, c_lab = cv2.connectedComponents((thin if thin.any() else comp & G.own_ring).astype(np.uint8),
                                             connectivity=8)
        contacts = [c_lab == c for c in range(1, n_c)]
    else:
        # nothing touches the rim: the evidence can miss a tube's base (next to a drifting grain, say); a region
        # within bridge_px of the rim is a candidate, bridged by a straight segment from the rim (the whole-movie
        # score and the ownership tests decide)
        near = m & (G.rg <= G.gr + p.bridge_px)
        if p.bridge_px <= 0 or not near.any():
            return [], Pend, m
        ids = np.unique(lab[near])
        comp = np.isin(lab, ids[ids > 0])
        contacts = []
        for i in ids[ids > 0]:
            part = lab == i
            rmin = float(G.rg[part].min())
            contacts.append(part & (G.rg <= rmin + 1.0))
    cost = np.where(comp, 1.0 / (Pend + 0.05), np.inf)
    paths = []
    for cmask in contacts:
        mcp = MCP_Geometric(cost)
        seeds = list(zip(*np.nonzero(cmask)))
        cum, _ = mcp.find_costs(seeds)
        # geodesic (unweighted) distance for tip finding
        mcp_u = MCP_Geometric(np.where(comp, 1.0, np.inf))
        dist, _ = mcp_u.find_costs(seeds)
        ok = comp & np.isfinite(dist)
        if not ok.any():
            continue
        d = np.where(ok, dist, -1.0).astype(np.float32)
        mx = cv2.dilate(d, np.ones((3, 3), np.uint8))
        sk = skeletonize(ok)
        nbh = cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
        cand = (list(zip(*np.nonzero(ok & (d >= mx) & (d >= p.min_path_px))))
                + list(zip(*np.nonzero(sk & (nbh == 2) & (d >= p.min_path_px)))))
        cand.sort(key=lambda yx: -d[yx])
        tips = []
        for y, x in cand:
            if all(math.hypot(y - ty, x - tx) >= p.tip_nms_px for ty, tx in tips):
                tips.append((y, x))
            if len(tips) >= p.n_tips:
                break
        for tip in tips:
            route = np.asarray(mcp.traceback(tip), float)[:, ::-1]  # (x, y)
            if len(route) < 3:
                continue
            paths.append(route)
    return paths, Pend, comp


def finish_path(route: np.ndarray, G: GrainStack, p: Params, extend_px: float | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Rim point, smoothed centreline, extension past the tip; resampled at p.step from the rim (s = 0)."""
    extend_px = p.extend_px if extend_px is None else extend_px
    c = np.array([G.centre, G.centre])
    path = route.copy()
    if len(path) > 7:
        k = np.ones(5) / 5
        path = np.stack([np.convolve(np.pad(path[:, i], 2, mode="edge"), k, "valid") for i in range(2)], axis=1)
    v = path[0] - c
    v = v / (np.linalg.norm(v) + 1e-9)
    path = np.vstack([c + v * G.gr, path])
    pts, ss = resample(path, p.step)
    if extend_px > 0 and len(pts) >= 5:
        d = pts[-1] - pts[-5]
        d = d / (np.linalg.norm(d) + 1e-9)
        ext = pts[-1] + np.arange(p.step, extend_px + 1e-9, p.step)[:, None] * d[None]
        pts = np.vstack([pts, ext])
        ss = np.arange(len(pts)) * p.step
    return pts, ss


# ----------------------------------------------------------------------------- anchored mode
def trace_in_grain_frame(G: GrainStack, xy_ref, b_idx: int) -> np.ndarray:
    """A trace (reference coordinates at bin index ``b_idx``, the grain's drift included, as the labelling tool's
    ``path_xy_ref``) in grain-frame crop coordinates, in the order it was clicked: the tool asks for the exit from the
    grain first, then the centreline to the apex (a tube that curls back can end nearer its grain than it left it)."""
    return np.asarray(xy_ref, np.float64) - [G.gx, G.gy] + G.centre - np.asarray(G.ls[b_idx])


def anchored_path(G: GrainStack, xy: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray, float, float]:
    """The decoding path along a trace (grain-frame crop coordinates, exit first): (points every p.step from where
    the trace leaves the grain's disc, their arclength, the traced length before that point (a first click inside the
    rim), the traced length after it). Each point moves sideways onto the end-state evidence, within
    ``anchor_snap_px`` and smoothly along the trace, so the path keeps the trace's order and shape; where the moved
    path's length strays from the trace's by more than 20%, the trace is used as drawn."""
    dense, s = resample(xy, p.step)
    r = np.hypot(dense[:, 0] - G.centre, dense[:, 1] - G.centre)
    out = np.nonzero(r >= G.gr - 1.0)[0]
    i0 = int(out[0]) if len(out) else 0
    pts, inside, after = dense[i0:], float(s[i0]), float(s[-1] - s[i0])
    if len(pts) < 2:
        return pts, np.zeros(len(pts)), inside, after
    drawn = resample(pts, p.step)
    if p.anchor_snap_px <= 0 or len(pts) < 5:
        return drawn[0], drawn[1], inside, after
    Pa = G.P[max(0, G.T - p.end_bins):G.T].mean(axis=0)
    nrm = normals(pts)
    offs = np.arange(-p.anchor_snap_px, p.anchor_snap_px + 1e-9, 0.5)
    vals = sample(Pa, pts[:, 0, None] + offs[None] * nrm[:, 0, None], pts[:, 1, None] + offs[None] * nrm[:, 1, None])
    shift = np.where(vals.max(axis=1) > p.thr, offs[np.argmax(vals, axis=1)], 0.0)
    k = max(1, int(round(2.0 / p.step)))  # a median over ~4 px, then a moving average: sideways moves are smooth
    shift = np.array([np.median(shift[max(0, i - k):i + k + 1]) for i in range(len(shift))])
    shift = np.convolve(np.pad(shift, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
    moved = resample(pts + shift[:, None] * nrm, p.step)
    if after > 0 and abs(moved[1][-1] / after - 1.0) > 0.2:
        return drawn[0], drawn[1], inside, after
    return moved[0], moved[1], inside, after


def anchors_from_labels(labels: dict, drop: set | None = None) -> dict[str, dict]:
    """Per grain the labelling tool still asks traces of (a tube seen, not excluded), its latest FULL trace with a
    drawn path among the bins the tool asks for now (``export_review.asked_bins``): {grain id: {"bin",
    "path_xy_ref", "length_px"}}. Traces the tool no longer asks for (the onset moved, the grain was called
    tubeless or excluded later) are not anchors."""
    from .export_review import asked_bins

    nb, out = int(labels["n_bins"]), {}
    for gid, lab in labels.get("labels", {}).items():
        if (drop and gid in drop) or (labels.get("grains") or {}).get(gid, {}).get("excluded"):
            continue
        saved = lab.get("traces") or {}
        full = [b for b in asked_bins(lab.get("onset") or {}, saved, nb)
                if (saved.get(str(b)) or {}).get("state") == "full"
                and len((saved.get(str(b)) or {}).get("path_xy_ref") or []) >= 2]
        if full:
            t = saved[str(full[-1])]
            out[gid] = {"bin": full[-1], "path_xy_ref": t["path_xy_ref"], "length_px": float(t["length_px"])}
    return out


# ----------------------------------------------------------------------------- evidence
def logodds(P: np.ndarray, clip: float) -> np.ndarray:
    P = np.clip(P, 1e-4, 1 - 1e-4)
    return np.clip(np.log(P / (1 - P)) / clip, -1.0, 1.0)


def read_along(G: GrainStack, X: np.ndarray, Y: np.ndarray, nrm_x: np.ndarray, nrm_y: np.ndarray, p: Params) -> np.ndarray:
    """X, Y: (T, K, n) crop positions per bin, state and path point; lateral max of P -> log-odds evidence."""
    T = X.shape[0]
    out = np.empty(X.shape, np.float32)
    for t in range(T):
        best = None
        for o in (-p.lat, 0.0, p.lat):
            v = sample(G.P[t], X[t] + o * nrm_x[t], Y[t] + o * nrm_y[t], 0.0)
            best = v if best is None else np.maximum(best, v)
        out[t] = logodds(best, p.clip)
    return out


def model_geometry(model: str, pts: np.ndarray, ss: np.ndarray, G: GrainStack, p: Params):
    """Deformed path positions (T, K, n) for each state of a deformation model, the identity state index,
    per-bin state step limit and penalty."""
    T, n = G.T, len(pts)
    c = G.centre
    nrm = normals(pts)
    if model == "rot":
        th = np.deg2rad(np.arange(-p.rot_max, p.rot_max + 1e-9, p.rot_step))
        K = len(th)
        rx, ry = pts[:, 0] - c, pts[:, 1] - c
        X = c + np.cos(th)[:, None] * rx[None] - np.sin(th)[:, None] * ry[None]
        Y = c + np.sin(th)[:, None] * rx[None] + np.cos(th)[:, None] * ry[None]
        NX = np.cos(th)[:, None] * nrm[None, :, 0] - np.sin(th)[:, None] * nrm[None, :, 1]
        NY = np.sin(th)[:, None] * nrm[None, :, 0] + np.cos(th)[:, None] * nrm[None, :, 1]
        X, Y, NX, NY = (np.broadcast_to(a[None], (T, K, n)) for a in (X, Y, NX, NY))
        return X, Y, NX, NY, int(np.argmin(np.abs(th))), p.rot_dmax, p.rot_pen
    if model == "sway":
        a = np.arange(-p.sway_max, p.sway_max + 1e-9, p.sway_step)
        K = len(a)
        w = np.clip(ss / p.sway_s0, 0, 1)
        X = pts[None, :, 0] + a[:, None] * w[None] * nrm[None, :, 0]
        Y = pts[None, :, 1] + a[:, None] * w[None] * nrm[None, :, 1]
        NX, NY = np.broadcast_to(nrm[None, :, 0], (K, n)), np.broadcast_to(nrm[None, :, 1], (K, n))
        X, Y, NX, NY = (np.broadcast_to(v[None], (T, K, n)) for v in (X, Y, NX, NY))
        return X, Y, NX, NY, int(np.argmin(np.abs(a))), p.sway_dmax, p.sway_pen
    if model == "anchor":
        w = np.clip(1.0 - ss / p.bend_px, 0, 1)
        dxy = G.ls[G.iref][None] - G.ls  # (T, 2): substrate displacement in the grain frame relative to the end
        X = pts[None, :, 0] + (1 - w)[None] * dxy[:, 0:1]
        Y = pts[None, :, 1] + (1 - w)[None] * dxy[:, 1:2]
        X, Y = X[:, None], Y[:, None]
        NX = np.broadcast_to(nrm[None, None, :, 0], (T, 1, n))
        NY = np.broadcast_to(nrm[None, None, :, 1], (T, 1, n))
        return X, Y, NX, NY, 0, 0, 0.0
    raise ValueError(model)


def birth_profile(Et: np.ndarray, frac: float = 0.7) -> np.ndarray:
    """Per path point, the first bin from which the evidence is positive in >= frac of the remaining bins."""
    pos = Et > 0
    T = len(Et)
    after = np.cumsum(pos[::-1], axis=0)[::-1]
    ok = pos & (after >= frac * np.arange(T, 0, -1)[:, None])
    return np.where(ok.any(axis=0), ok.argmax(axis=0), T)


def ownership(best: dict, G: GrainStack, p: Params) -> tuple[bool, str]:
    """Is the decoded tube the grain's own?  It must leave the rim roughly radially (a tube passing by runs along
    it) and grow from its base (the far part of a foreign tube that reaches the rim was there first)."""
    pts, lev = best["pts"], best["lev"]
    c = np.array([G.centre, G.centre])
    j = min(len(pts) - 1, int(round(6.0 / p.step)))
    d = pts[j] - pts[0]
    rad = pts[0] - c
    cosang = float(np.dot(d, rad) / (np.linalg.norm(d) * np.linalg.norm(rad) + 1e-9))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    if ang > p.exit_max_deg:
        return False, f"tangential_exit:{ang:.0f}"
    n = int(lev[-1])
    if n * p.step >= 6.0:
        b = best["birth"][:n]
        third = max(1, n // 3)
        near, far = float(np.median(b[:third])), float(np.median(b[-third:]))
        if far < near - p.birth_margin:
            return False, f"born_far_first:{near:.0f}>{far:.0f}"
    return True, ""


def decode_path(G: GrainStack, pts: np.ndarray, ss: np.ndarray, p: Params, models: tuple,
                pin_level: int | None = None) -> dict | None:
    best = None
    vmax = max(1, int(round(p.vmax_px / p.step)))
    moved = float(np.hypot(*(G.ls - G.ls[G.iref]).T).max())
    for model in models:
        if model == "anchor" and moved < p.move_min_px:
            continue
        X, Y, NX, NY, k0, dmax, pen = model_geometry(model, pts, ss, G, p)
        E = read_along(G, X, Y, NX, NY, p)
        E[:, :, ss < p.skip_px] = 0.0
        lev, st, score = dp_joint(E, vmax, dmax, pen, pin=k0 if dmax > 0 else None, pin_level=pin_level)
        # positive evidence left beyond the front along the chosen deformation (a foreign tube already there)
        Et = E[np.arange(G.T), st]  # (T, n)
        beyond = np.where(np.arange(len(ss))[None, :] >= lev[:, None], np.maximum(Et, 0), 0).sum()
        cand_score = score - p.beyond * beyond
        if best is None or cand_score > best["cand_score"]:
            best = {"model": model, "lev": lev, "states": st, "score": score, "beyond": float(beyond),
                    "cand_score": cand_score, "pts": pts, "ss": ss, "birth": birth_profile(Et)}
    if best is not None:
        best["own"], best["why"] = ownership(best, G, p)
    return best


# ----------------------------------------------------------------------------- image evidence near the exit
def image_evidence(G: GrainStack, best: dict, X, Y, NX, NY, p: Params) -> tuple[np.ndarray | None, dict]:
    """Analysis by synthesis for the first ``img_s_max`` px: the grain-frame image along the (deformed) path,
    minus the grain's own pre-emergence look, projected on the tube's own end-state cross-section.  z = fraction of
    the mature contrast present at (bin, state, point); the noise of z is measured on pre-emergence bins; evidence
    = clip((z - img_z0) / (img_k * sigma), -1, 1).  Returns (E_img (T, K, m) or None, info)."""
    T = G.T
    n_img = int(round(p.img_s_max / p.step)) + 1
    lev = best["lev"]
    on = np.nonzero(lev * p.step >= 1.0)[0]
    t_on = int(on[0]) if len(on) else T - 1
    # a short early window (after grains have landed), well before the evidence shows any tube: a young tube the
    # network does not see can be there for tens of bins before the front moves
    r0 = max(0, min(p.img_ref_start, t_on - p.img_pre_gap - p.img_ref_bins))
    r1 = min(r0 + p.img_ref_bins, t_on - p.img_pre_gap)
    if r1 - r0 < 2:
        return None, {"img": "no_pre_emergence_bins"}
    B = G.img[r0:r1].mean(axis=0)
    u = np.arange(-p.img_half, p.img_half + 1e-9, 0.5)
    m = min(n_img, X.shape[2])
    k_end = best["states"][-1]

    def profiles(t, k_sel):  # (K', m, U)
        xs = X[t, k_sel, :m, None] + u[None, None, :] * NX[t, k_sel, :m, None]
        ys = Y[t, k_sel, :m, None] + u[None, None, :] * NY[t, k_sel, :m, None]
        return sample(G.img[t] - B, xs, ys, 0.0)

    # template: the tube's own mature cross-section just past the exit, in the last full bins
    ns = len(best["ss"])
    j0, j1 = int(round(2.0 / p.step)), min(ns, int(round(14.0 / p.step)))
    if j1 - j0 < 4 or lev[-1] < j1:
        return None, {"img": "tube_too_short_for_template"}
    late = G.img[max(0, T - 6):T - 1].mean(axis=0) - B
    xs = X[T - 2, k_end, j0:j1, None] + u[None, :] * NX[T - 2, k_end, j0:j1, None]
    ys = Y[T - 2, k_end, j0:j1, None] + u[None, :] * NY[T - 2, k_end, j0:j1, None]
    prof = sample(late, xs, ys, 0.0).mean(axis=0)
    A_ = prof - prof.mean()
    nA = float(np.linalg.norm(A_))
    if nA < 1e-3:
        return None, {"img": "flat_template"}
    A_ /= nA
    amp = float(prof @ A_)  # mature contrast (projection of the mature profile)
    if abs(amp) < p.img_min_amp:
        return None, {"img": f"weak_template:{amp:.1f}"}
    K = X.shape[1]
    # templates: the tube's own mature cross-section, and the measured young dark line (a bright-cored tube can start
    # as a dark line); each projection is scored in units of its own pre-emergence noise, and the best one counts
    dark = -23.0 * np.exp(-u ** 2 / 3.4) + 1.5 * np.exp(-(np.abs(u) - 4.2) ** 2 / 0.5)
    D = dark - dark.mean()
    D /= float(np.linalg.norm(D))
    temps = [A_] + ([D] if float(A_ @ D) < 0.9 else [])
    Zs = np.empty((len(temps), T, K, m), np.float32)
    for t in range(T):
        pr = profiles(t, slice(None))
        for i, tm in enumerate(temps):
            Zs[i, t] = pr @ tm
    # null model: the same statistic on the path's first px turned round the grain (control angles clear of any
    # end-state tube), in every bin: their median is the common-mode change (focus, gain, registration at the rim)
    # and their spread over the movie the noise
    Pend = G.P[max(0, T - 1 - p.end_bins):T - 1].mean(axis=0)
    c = G.centre
    ctrl = []
    for phi in np.deg2rad(p.img_ctrl_deg):
        cx = c + np.cos(phi) * (X[T - 2, k_end, :m] - c) - np.sin(phi) * (Y[T - 2, k_end, :m] - c)
        cy = c + np.sin(phi) * (X[T - 2, k_end, :m] - c) + np.cos(phi) * (Y[T - 2, k_end, :m] - c)
        if float(sample(Pend, cx, cy, 0.0).max()) > 0.3:
            continue
        nx = np.cos(phi) * NX[T - 2, k_end, :m] - np.sin(phi) * NY[T - 2, k_end, :m]
        ny = np.sin(phi) * NX[T - 2, k_end, :m] + np.cos(phi) * NY[T - 2, k_end, :m]
        ctrl.append((cx, cy, nx, ny))
    if len(ctrl) < 2:
        return None, {"img": "no_controls"}
    Zc = np.empty((len(temps), T, len(ctrl), m), np.float32)
    for t in range(T):
        D0 = G.img[t] - B
        for j, (cx, cy, nx, ny) in enumerate(ctrl):
            pr = sample(D0, cx[:, None] + u[None] * nx[:, None], cy[:, None] + u[None] * ny[:, None], 0.0)
            for i, tm in enumerate(temps):
                Zc[i, t, j] = pr @ tm
    E_img = None
    sigs = []
    for i in range(len(temps)):
        base = np.median(Zc[i].reshape(T, -1), axis=1)  # (T,)
        res = Zc[i] - base[:, None, None]
        sig = max(float(1.4826 * np.median(np.abs(res - np.median(res)))), p.img_sig_floor_gl)
        sigs.append(round(sig, 2))
        e = np.clip(((Zs[i] - base[:, None, None]) / sig - p.img_t0) / p.img_tk, -1.0, 1.0)
        E_img = e if E_img is None else np.maximum(E_img, e)
    return E_img, {"img_amp": round(amp, 2), "img_sigma": sigs, "img_ref": [r0, r1], "img_templates": len(temps),
                   "img_controls": len(ctrl)}


def refine_with_image(G: GrainStack, best: dict, p: Params, pin_level: int | None = None) -> dict:
    """Re-run the joint DP for the chosen path and deformation model with the image term fused in."""
    model = best["model"]
    pts, ss = best["pts"], best["ss"]
    X, Y, NX, NY, k0, dmax, pen = model_geometry(model, pts, ss, G, p)
    E_img, info = image_evidence(G, best, X, Y, NX, NY, p)
    if E_img is None:
        return {**best, **info}
    E = read_along(G, X, Y, NX, NY, p)
    E[:, :, ss < p.skip_px] = 0.0
    m = E_img.shape[2]
    if p.img_fuse == "max":
        E[:, :, :m] = np.maximum(E[:, :, :m], E_img)
    elif p.img_fuse == "support":  # the image may find a young tube the network misses, never veto one it sees
        E[:, :, :m] += p.img_weight * np.maximum(E_img, 0.0)
    else:
        E[:, :, :m] += p.img_weight * E_img
    vmax = max(1, int(round(p.vmax_px / p.step)))
    allowed = None
    K = E.shape[1]
    if K > 1:
        # before the learned evidence shows the tube its deformation is unconstrained, and the image term would find
        # rim structure somewhere round the grain: keep the track near the one the learned evidence chose, held at
        # its first visible value before that
        vis = np.nonzero(best["lev"] * p.step >= 2.0)[0]
        t_vis = int(vis[0]) if len(vis) else G.T - 1
        ref = best["states"].copy()
        ref[:t_vis] = ref[t_vis]
        kk = np.arange(K)[None, :]
        allowed = np.abs(kk - ref[:, None]) <= p.img_state_band
    lev, st, score = dp_joint(E, vmax, dmax, pen, pin=k0 if dmax > 0 else None, allowed=allowed,
                              pin_level=pin_level)
    return {**best, "lev": lev, "states": st, "score_img": score, **info}


# ----------------------------------------------------------------------------- lengths from the front
def finalise(front: np.ndarray, half_w: float | None, p: Params) -> np.ndarray:
    """Reported length from the growth front.  Young tubes: the evidence crosses 0.5 about 0.7 px short of the apex
    while the tube is under ~4 px (measured along true geometry on development movies), so a correction that fades
    out by ``young_px`` is added.  Stopped tubes: a tube that has stopped shows its round end, and the evidence runs
    about a half-width past the apex; if the front has not moved over the last ``stop_bins`` bins, lengths are capped
    at the final front minus ``cap_frac`` x the tube's half-width."""
    L = np.where(front > 0, front + p.tip_px + p.young * np.clip(1.0 - front / p.young_px, 0.0, 1.0), 0.0)
    if half_w is not None and p.cap_frac > 0 and len(front) > p.stop_bins and front[-1] > 0:
        if front[-1] - front[-1 - p.stop_bins] <= p.stop_tol:
            cap = max(front[-1] - min(p.cap_frac * half_w, p.cap_max_px), 0.0)
            L = np.minimum(L, np.where(L > 0, max(cap, p.onset_px), 0.0))
    return L


# ----------------------------------------------------------------------------- per grain
def decode_grain(RP: Renderer, R_img: Renderer, meta: dict, grain: dict, others: list[dict], p: Params,
                 half: int | None = None, anchor: dict | None = None) -> dict:
    half = half or p.half
    base = {"id": grain["id"], "x": float(grain["x"]), "y": float(grain["y"]), "r": float(grain["r"]), "flags": []}
    if anchor is not None and int(anchor["bin"]) - int(meta.get("ref_start", 0)) < 1:
        base["flags"].append(f"anchor_ignored:bin {anchor['bin']} is before the first analysed bin")
        anchor = None
    cut = False
    if anchor is not None:  # a trace reaching past the crop is read with a bigger one, up to max_half
        far = float(np.abs(np.asarray(anchor["path_xy_ref"], float) - [grain["x"], grain["y"]]).max()) + 25
        if far > half:
            half = int(min(max(p.big, np.ceil(far)), max(p.max_half, p.big)))
            cut = far > half
    G = GrainStack(RP, R_img, meta, grain, others, p, half,
                   anchor_bin=None if anchor is None else int(anchor["bin"]))
    fpb, rs, nb = G.fpb, G.rs, G.nb
    frames = [b * fpb + fpb // 2 for b in range(rs, nb)]
    best, pin_level, routes, rejected = None, None, [], []
    k = 3 + int(np.ceil(np.abs(G.ls).max()))
    n_ext = int(round(p.extend_px / p.step))  # path points added past the tip
    inside, scale = 0.0, 1.0  # anchored: the traced length before the path starts, and trace px per path px
    if anchor is not None:
        # the trace is the end state: no path search, no ownership tests
        xy = trace_in_grain_frame(G, anchor["path_xy_ref"], G.T - 1)
        pts, ss, inside, after = anchored_path(G, xy, p)
        n_ext = 0
        routes = [pts]
        pin = p.anchor_pin and not cut
        if len(pts) >= 4:
            pin_level = len(pts) if pin else None
            best = decode_path(G, pts, ss, p, p.models, pin_level=pin_level)
            traced = float(anchor.get("length_px") or (inside + after))
            if pin and ss[-1] > 0:  # lengths in the trace's own terms: its bin reads the traced length
                scale = max(traced - inside, 0.0) / float(ss[-1])
            else:
                inside = 0.0
        if cut:
            base["flags"].append(f"trace_beyond_crop:{half}")
        base["anchor"] = {"bin": int(anchor["bin"]), "length_px": anchor.get("length_px"),
                          "path_px": round(float(ss[-1]), 2) if len(pts) else None,
                          "inside_rim_px": round(inside, 2), "scale": round(scale, 3)}
        Pend = None
    else:
        routes, Pend, comp = end_state_paths(G, p)
        for route in routes:
            pts, ss = finish_path(route, G, p)
            if len(pts) < 4:
                continue
            res = decode_path(G, pts, ss, p, p.models)
            if res is None:
                continue
            if not res["own"]:
                rejected.append(res["why"])
                continue
            if best is None or res["cand_score"] > best["cand_score"]:
                best = res
        base["flags"] += [f"rejected:{w}" for w in rejected]
    if best is not None and p.image_term:
        best = refine_with_image(G, best, p, pin_level=pin_level)
    if best is not None and half < p.big and anchor is None:
        tipxy = best["pts"][max(len(best["pts"]) - n_ext, 1) - 1]
        if min(tipxy[0], tipxy[1], 2 * half - 1 - tipxy[0], 2 * half - 1 - tipxy[1]) <= k:
            res = decode_grain(RP, R_img, meta, grain, others, p, half=p.big, anchor=anchor)
            res["flags"].append(f"crop_grown:{p.big}")
            return res
    if best is None:
        return {**base, "status": "no_emergence_by_end", "onset_frame": None, "onset_interval": None,
                "final_length_px": 0.0, "length": {"frames": frames, "px": [0.0] * len(frames)}}
    front = best["ss"][np.maximum(best["lev"] - 1, 0)] * (best["lev"] > 0)
    # the tube's half-width at the end state: area of the end-state region near the path / (2 x its length); an
    # anchor's end is where its trace ends, so no stopped-tube correction there
    n_end = int(best["lev"][-1])
    half_w = None
    if n_end >= 8 and Pend is not None:
        on_path = np.zeros(Pend.shape, np.uint8)
        cv2.polylines(on_path, [np.round(best["pts"][:n_end]).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
        near = cv2.distanceTransform(1 - on_path, cv2.DIST_L2, 3) <= 8.0
        area = float(((Pend > p.thr) & near & ~G.blocked).sum())
        half_w = area / (2.0 * max(front[-1], 1.0))
    L = finalise(front, half_w, p)
    if G.T < G.T_all:  # lengths after the end-state bin are held (not decoded)
        base["flags"].append(f"held_after:{frames[G.T - 1]}")
        L = np.concatenate([L, np.full(G.T_all - G.T, L[-1])])
        front = np.concatenate([front, np.full(G.T_all - G.T, front[-1])])
    on = np.nonzero(L >= p.onset_px)[0]  # germination on the decoder's own scale, as calibrated
    if anchor is not None:  # reported in the trace's terms: the part clicked inside the rim, and its px
        L = np.where(L > 0, L * scale + inside, 0.0)
    if (L[-1] < p.min_tube_px and anchor is None) or not len(on):
        status, onset, interval, L = "no_emergence_by_end", None, None, np.zeros_like(L)
    elif on[0] == 0:
        status, onset, interval = "emerged_at_start", frames[0], None
    else:
        status, onset, interval = "emerged_within", frames[int(on[0])], [frames[int(on[0]) - 1], frames[int(on[0])]]
    th = best["states"]
    return {**base, "status": status, "onset_frame": onset, "onset_interval": interval,
            "final_length_px": round(float(L[-1]), 2),
            "length": {"frames": frames, "px": [round(float(v), 2) for v in L]},
            "model": best["model"], "score": round(best["score"], 2), "beyond": round(best["beyond"], 2),
            "front_px": [round(float(v), 2) for v in front], "half_width_px": None if half_w is None else round(half_w, 2),
            **{k: best[k] for k in ("img_amp", "img_sigma", "img_ref", "img", "img_templates", "img_controls") if k in best},
            "n_candidates": len(routes), "state_track": [int(v) for v in th],
            # the decoded tube at the bin its path describes, as the labelling tool's path_xy_ref: reference
            # coordinates with the grain's drift, from the exit to the decoded apex
            "path_bin": rs + G.iref,
            "path_xy_ref": [[round(float(x - G.centre + G.gx + G.ls[G.iref][0]), 2),
                             round(float(y - G.centre + G.gy + G.ls[G.iref][1]), 2)]
                            for x, y in best["pts"][:max(int(best["lev"][-1]), 2)][::2]]}


def analyze(pcache: str | Path, image_cache: str | Path, grains_path: str | Path | None = None, log=print,
            params: Params | None = None, only: list[str] | None = None, anchors: dict | None = None) -> dict:
    """Decode every included grain of the census (``grains_path``, else the probability cache's). ``anchors``:
    {grain id: {"bin", "path_xy_ref"[, "length_px"]}} (``anchors_from_labels``); anchored grains are decoded along
    their trace, the others as usual."""
    p = params or Params()
    bins_p, meta = stack.load(pcache)
    RP = Renderer(bins_p, meta)
    R_img = Renderer(*stack.load(image_cache))
    src = Path(grains_path) if grains_path else Path(pcache) / "grains.json"
    doc = json.loads(src.read_text())
    census = list(doc["grains"].values()) if isinstance(doc["grains"], dict) else doc["grains"]
    grains = [g for g in census if not g.get("excluded")]
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    anchors = anchors or {}
    started, out = time.time(), []
    for g in grains:
        if only and g["id"] not in only:
            continue
        others = [o for o in physical if o["id"] != g["id"]]
        out.append(decode_grain(RP, R_img, meta, g, others, p, anchor=anchors.get(g["id"])))
    log(f"prefix decoder: {len(out)} grains ({sum(g['id'] in anchors for g in out)} anchored) in "
        f"{time.time() - started:.0f} s")
    return {"schema": PRED_SCHEMA, "method": "prefix decoder" + (" (anchored)" if anchors else ""),
            "frames_per_bin": meta["frames_per_bin"], "grains": out,
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.__dict__.items()}}
