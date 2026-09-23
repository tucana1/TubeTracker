"""SparseTrack v1: per-grain germination onset and tube length from a bin cache.

For each grain, whole-movie and offline (bins before the cache's settled reference
start are treated as unobserved):

1. Crop the grain from every bin (global registration), then refine a per-bin
   residual shift on the grain itself.
2. Tube map: blurred ``|late - early|`` change, grain bodies masked (the rim band is
   kept so a tube wrapping round its own grain survives). A change region shared
   with other grains is split by geodesic ownership.
3. The change component attached to the grain's rim is the tube; its tip is the
   geodesically farthest point, and the centreline is the evidence-weighted
   shortest path from the rim to it, starting at the exit on the grain circle.
4. Kymograph of the (background-subtracted) change along that path for every bin,
   with the grain + tube allowed to rotate rigidly about the grain centre (smooth
   rotation track, judged on distal off-rim points, pinned at the end). The tube
   front is the globally best non-decreasing path through it (dynamic programming,
   capped growth rate); lengths stop where the path reaches another grain's rim.
5. Onset = first sustained rise of the excess change just outside the rim at the
   exit angle over its pre-emergence noise; no length is reported before onset.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import __version__, stack
from . import grains as census
from .evaluate import PRED_SCHEMA
from .render import Renderer


@dataclass
class Params:
    half: int = 150              # crop half-size around each grain (px)
    reg_pad: float = 12.0        # local registration window = grain radius + pad
    ref_bins: int = 3            # leading bins averaged as the "before" image
    late_bins: int = 3           # trailing full bins averaged as the "after" image
    map_sigma: float = 1.0
    path_sigma: float = 1.0      # blur of the change map used as the centreline cost
    map_k: float = 5.0           # tube-map threshold = max(map_floor, map_k * background sigma)
    map_floor: float = 5.0
    min_component_px: int = 12
    evid_k: float = 4.0          # kymograph threshold = max(evid_floor, base + evid_k * noise)
    evid_floor: float = 5.0
    lateral: float = 1.0         # sample +/- this many px across the path
    vmax_px: float = 4.0         # maximum front advance per bin
    neg_weight: float = 1.0      # weight of negative evidence inside the claimed tube (gap bridging)
    skip_px: float = 1.5         # evidence this close to the exit is ignored (rim band)
    onset_px: float = 2.0        # onset when the front passes this far beyond the exit
    step: float = 0.5            # path sampling (px)
    rotate: bool = True          # let the grain + tube rotate rigidly about the grain centre
    max_angle: float = 45.0      # rotation search range (degrees, either way)
    angle_step: float = 1.5
    angle_penalty: float = 0.4   # Viterbi cost per degree of change between consecutive bins
    max_turn: float = 6.0        # largest rotation change between consecutive bins (degrees)
    rot_min_s: float = 6.0       # rotation judged only on path points at least this far along...
    rot_rim_clear: float = 4.0   # ...and at least this far outside the grain rim
    wedge_r: tuple = (1.0, 6.0)  # onset wedge: radii beyond the rim (px)
    wedge_halfwidth: float = 12.0  # degrees either side of the exit angle
    wedge_base_bins: int = 5     # bins defining the pre-emergence noise
    wedge_k: float = 5.0         # onset threshold = base + wedge_k * robust sigma (floor below)
    wedge_floor: float = 1.5
    wedge_hold: int = 8          # the rise must hold in >= 80% of the next wedge_hold bins
    persist_bins: int = 30       # ...and in >= 70% of the next persist_bins bins (0 = to the end)
    # onset detector: "matched" (stub matched filter, below; synthetic 42/71 in tolerance vs 17/71,
    # legacy 5/7), "wedge_fixed" (excess change at the end-state exit angle; legacy 5/7),
    # "wedge" (exit angle tracked back from the end; 2/7, drifts onto rim noise), "front" (1/7)
    onset_source: str = "matched"
    mf_len: float = 4.0          # matched-filter stub: first mf_len px of the path...
    mf_half: float = 3.0         # ...+/- mf_half px across it
    mf_search: float = 15.0      # exit angle search (degrees either way)
    mf_z: float = 3.0            # onset when the calibrated score exceeds this (sustained)
    mf_z_low: float | None = 2.0   # ...then reaching back while it stays above this (hysteresis)
    mf_back_bins: int = 6        # ...by at most this many bins (a slow drift is not a stub)
    mf_extra: tuple = ()         # extra stub templates: "dark" (generic young dark line), "negative"
    mf_follow_rotation: bool | str = "both"  # True: stub placed along the rotation track; "both": max of the two
    bg_subtract: bool = True     # subtract each bin's background change before reading the path
    # front evidence: "matched" = signed change projected on the tube's own end-state cross-section
    # (rejects blobs, focus and uniform brightness changes; synthetic lengths 68% -> 84% in tolerance);
    # "abs" = |change| near the path (v1); "union" = per-point max of the two: as good on synthetic,
    # and it still reads real tubes whose look changes as they mature (g014, g022, g026 read 0 matched)
    evidence: str = "union"
    mk_half: float = 3.5         # matched evidence: template half-width across the path (px)
    mk_k: float = 4.0            # threshold = pre-onset base + mk_k * per-bin control sigma...
    mk_floor: float = 3.0        # ...but at least this
    mk_control_px: float = 9.0   # the noise control slides the template this far off the tube
    mk_extra: tuple = ()         # extra cross-section templates for the evidence: "dark" (young tube)
    candidates: bool = True      # choose the centreline among branch/contact hypotheses by growth
    cand_tips: int = 4
    cand_nms_px: float = 10.0
    cand_branch_tips: int = 6    # + this many skeleton branch ends
    nest_px: float = 5.0         # a candidate within this of a longer one all along is the same tube
    beyond_weight: float = 0.0   # score = explained - beyond_weight * evidence left beyond the front
    ridge_px: float = 5.0        # tube-likeness: end-state change on the path vs this far beside it
    ridge_weight: float = 1.0    # score *= (1 - w) + w * fraction of path points on a ridge
    through_px: float = 12.0     # foreign-tube test: material this close to the exit...
    through_min_px: int = 10     # ...at least this many pixels of it, changed before the front left
    contact_px: float = 4.0      # lengths are censored where the path comes this close to another rim
    min_tube_px: float = 8.0     # a front that never gets this long is not a tube (unless contact-censored)
    settle: bool = True          # grains still arriving in the census bins are read from when they settle
    settle_bins: int = 24
    grain_min_rim: float = 1.5   # no rim at all in the early bins: not a grain (passing debris)


def _highpass(img: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    img = img.astype(np.float64)
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def local_shifts(crops: np.ndarray, centre: float, radius: float, pad: float, ref_bins: int,
                 max_dev: float = 1.5, window: int = 3, follow: bool = True) -> np.ndarray:
    """Residual (dx, dy) per bin of the grain relative to its own early mean.

    With ``follow`` the correlation window moves with the grain (centred on the previous
    bin's estimate), so a grain can drift further than the window half-width.
    """
    from skimage.registration import phase_cross_correlation

    w = int(math.ceil(radius + pad))
    c = int(round(centre))
    size = crops.shape[1]
    ref = _highpass(crops[:ref_bins].mean(axis=0)[c - w:c + w, c - w:c + w])
    raw = np.zeros((len(crops), 2))
    prev = np.zeros(2)
    for b in range(len(crops)):
        ox, oy = (int(round(prev[0])), int(round(prev[1]))) if follow else (0, 0)
        ox, oy = int(np.clip(ox, w - c, size - w - c)), int(np.clip(oy, w - c, size - w - c))
        win = crops[b][c + oy - w:c + oy + w, c + ox - w:c + ox + w]
        shift, _, _ = phase_cross_correlation(ref, _highpass(win), upsample_factor=20, normalization=None)
        raw[b] = (ox - shift[1], oy - shift[0])
        if b >= ref_bins:  # the reference bins define the origin
            lo = max(ref_bins, b - window)
            recent = raw[lo:b + 1]
            prev = np.median(recent, axis=0) if len(recent) >= 3 else raw[b]
    med = np.array([np.median(raw[max(0, b - window):b + window + 1], axis=0) for b in range(len(raw))])
    bad = np.hypot(*(raw - med).T) > max_dev
    return np.where(bad[:, None], med, raw)


def _geodesic_far(mask: np.ndarray, seeds: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    """Breadth-first geodesic distance (8-connected) inside ``mask`` from ``seeds``; farthest pixel."""
    h, w = mask.shape
    dist = np.full((h, w), -1, np.int32)
    q = deque()
    for y, x in zip(*np.nonzero(seeds & mask)):
        dist[y, x] = 0
        q.append((y, x))
    far, fd = (int(q[0][0]), int(q[0][1])) if q else (0, 0), 0
    while q:
        y, x = q.popleft()
        d = dist[y, x]
        if d > fd:
            far, fd = (y, x), d
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and dist[yy, xx] < 0:
                    dist[yy, xx] = d + 1
                    q.append((yy, xx))
    return far, dist


def geodesic_owner(mask: np.ndarray, seeds: list[np.ndarray]) -> np.ndarray:
    """Label each ``mask`` pixel with the index of the seed set nearest *through the mask*.

    Multi-source breadth-first search (8-connected); -1 where no seed reaches.
    """
    h, w = mask.shape
    owner = np.full((h, w), -1, np.int32)
    q = deque()
    for k, s in enumerate(seeds):
        for y, x in zip(*np.nonzero(s & mask)):
            if owner[y, x] < 0:
                owner[y, x] = k
                q.append((y, x))
    while q:
        y, x = q.popleft()
        k = owner[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and owner[yy, xx] < 0:
                    owner[yy, xx] = k
                    q.append((yy, xx))
    return owner


def _cheapest_path(cost: np.ndarray, mask: np.ndarray, seeds: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Dijkstra from any seed pixel to ``target`` through ``mask``; returns (n, 2) array of (y, x)."""
    h, w = mask.shape
    dist = np.full((h, w), np.inf)
    prev = np.full((h, w, 2), -1, np.int32)
    pq = []
    for y, x in zip(*np.nonzero(seeds & mask)):
        dist[y, x] = 0.0
        heapq.heappush(pq, (0.0, int(y), int(x)))
    while pq:
        d, y, x = heapq.heappop(pq)
        if (y, x) == target:
            break
        if d > dist[y, x]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                    nd = d + cost[yy, xx] * math.hypot(dy, dx)
                    if nd < dist[yy, xx]:
                        dist[yy, xx] = nd
                        prev[yy, xx] = (y, x)
                        heapq.heappush(pq, (nd, yy, xx))
    path = [target]
    while prev[path[-1]][0] >= 0:
        path.append(tuple(prev[path[-1]]))
    return np.array(path[::-1], np.float64)


def _resample(path_xy: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.hypot(*np.diff(path_xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(ss, s, path_xy[:, 0]), np.interp(ss, s, path_xy[:, 1])], axis=1), ss


def dp_front(evidence: np.ndarray, vmax: int) -> np.ndarray:
    """Non-decreasing front index per bin maximising summed evidence behind the front.

    ``evidence`` is (n_bins, n_points); the front at index f claims points [0, f).
    """
    n_bins, n = evidence.shape
    gain = np.concatenate([np.zeros((n_bins, 1)), np.cumsum(evidence, axis=1)], axis=1)
    score = gain[0].copy()
    back = np.zeros((n_bins, n + 1), np.int32)
    idx = np.arange(n + 1)
    for t in range(1, n_bins):
        pad = np.concatenate([np.full(vmax, -np.inf), score])
        windows = np.lib.stride_tricks.sliding_window_view(pad, vmax + 1)
        j = np.argmax(windows, axis=1)
        score = gain[t] + windows[idx, j]
        back[t] = idx - (vmax - j)
    front = np.zeros(n_bins, np.int32)
    front[-1] = int(np.argmax(score))
    for t in range(n_bins - 1, 0, -1):
        front[t - 1] = back[t, front[t]]
    return front


def rotation_track(score: np.ndarray, angles: np.ndarray, penalty: float, max_turn: float,
                   anchor_bins: int) -> np.ndarray:
    """Smooth per-bin rotation maximising summed ``score`` (n_bins, n_angles).

    Consecutive bins may differ by at most ``max_turn`` degrees at ``penalty`` per degree;
    the last ``anchor_bins`` bins are pinned to 0 degrees (the path was traced there).
    """
    n_bins, n_ang = score.shape
    zero = int(np.argmin(np.abs(angles)))
    diff = np.abs(angles[:, None] - angles[None, :])
    trans = np.where(diff <= max_turn + 1e-9, -penalty * diff, -np.inf)  # [prev, cur]
    pinned = np.full(n_ang, -np.inf)
    pinned[zero] = 0.0
    total = score[0] + (pinned if n_bins - anchor_bins <= 0 else 0.0)
    back = np.zeros((n_bins, n_ang), np.int32)
    for t in range(1, n_bins):
        cand = total[:, None] + trans
        back[t] = np.argmax(cand, axis=0)
        total = cand[back[t], np.arange(n_ang)] + score[t]
        if t >= n_bins - anchor_bins:
            total = total + pinned
    track = np.zeros(n_bins, np.int32)
    track[-1] = int(np.argmax(total))
    for t in range(n_bins - 1, 0, -1):
        track[t - 1] = back[t, track[t]]
    return angles[track]


def _kymograph(diffs: np.ndarray, pts: np.ndarray, normal: np.ndarray, centre: float, lateral: float,
               angles_deg: np.ndarray) -> np.ndarray:
    """|change| sampled along the path rotated by each angle about the grain centre.

    Returns (n_bins, n_angles, n_points), the max over -lateral, 0, +lateral across the path.
    """
    rad = np.deg2rad(angles_deg)
    cos, sin = np.cos(rad)[:, None], np.sin(rad)[:, None]
    rel = pts - centre
    maps = []
    for off in (-lateral, 0.0, lateral):
        q = rel + off * normal
        x = centre + cos * q[None, :, 0] - sin * q[None, :, 1]
        y = centre + sin * q[None, :, 0] + cos * q[None, :, 1]
        maps.append((x, y))
    mx = np.concatenate([m[0] for m in maps]).astype(np.float32)
    my = np.concatenate([m[1] for m in maps]).astype(np.float32)
    n_ang = len(angles_deg)
    out = np.stack([cv2.remap(d, mx, my, cv2.INTER_LINEAR, borderValue=0) for d in diffs])
    return out.reshape(len(diffs), 3, n_ang, -1).max(axis=1)


def cross_section_template(late_minus_early: np.ndarray, pts: np.ndarray, normal: np.ndarray, across: np.ndarray,
                           smooth: int = 2) -> np.ndarray:
    """The tube's own end-state cross-section at every path point: zero-mean, unit-norm rows.

    Rows are averaged over +/- ``smooth`` neighbouring points to cut noise.
    """
    q = pts[:, None, :] + across[None, :, None] * normal[:, None, :]
    prof = cv2.remap(late_minus_early.astype(np.float32), q[..., 0].astype(np.float32), q[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=0)
    if smooth > 0 and len(prof) > 2 * smooth + 1:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        prof = np.stack([np.convolve(np.pad(prof[:, j], smooth, mode="edge"), k, "valid")
                         for j in range(prof.shape[1])], axis=1)
    prof = prof - prof.mean(axis=1, keepdims=True)
    return prof / np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-3)


def matched_kymograph(signed: np.ndarray, template: np.ndarray, pts: np.ndarray, normal: np.ndarray, centre: float,
                      across: np.ndarray, angles_deg: np.ndarray, lateral_offset: float = 0.0) -> np.ndarray:
    """Signed change projected on the cross-section template, per (bin, angle, point).

    ``lateral_offset`` slides the whole placement sideways (off the tube: a noise control).
    """
    rad = np.deg2rad(angles_deg)
    cos, sin = np.cos(rad)[:, None, None], np.sin(rad)[:, None, None]
    q = ((pts - centre)[None, :, None, :] + (across[None, None, :, None] + lateral_offset) * normal[None, :, None, :])
    x = (centre + cos * q[..., 0] - sin * q[..., 1]).reshape(-1, len(across)).astype(np.float32)
    y = (centre + sin * q[..., 0] + cos * q[..., 1]).reshape(-1, len(across)).astype(np.float32)
    n_ang, n_pts = len(angles_deg), len(pts)
    out = np.empty((len(signed), n_ang, n_pts), np.float32)
    for t, d in enumerate(signed):
        smp = cv2.remap(d, x, y, cv2.INTER_LINEAR, borderValue=0).reshape(n_ang, n_pts, len(across))
        out[t] = np.einsum("apu,pu->ap", smp, template)
    return out


def wedge_signal(diffs: np.ndarray, centre: float, gr: float, exit_angles: np.ndarray, p: "Params") -> np.ndarray:
    """Per-bin excess change at the exit angle over the rim-wide median, just outside the rim."""
    phis = np.deg2rad(np.arange(0.0, 360.0, 3.0))
    radii = gr + np.arange(p.wedge_r[0], p.wedge_r[1] + 1e-9, 1.0)
    mx = (centre + radii[:, None] * np.cos(phis)[None]).astype(np.float32)
    my = (centre + radii[:, None] * np.sin(phis)[None]).astype(np.float32)
    prof = np.stack([cv2.remap(d, mx, my, cv2.INTER_LINEAR, borderValue=np.nan).mean(axis=0) for d in diffs])
    out = np.zeros(len(diffs))
    for t, a in enumerate(exit_angles):
        dphi = np.abs((np.rad2deg(phis) - a + 180.0) % 360.0 - 180.0)
        out[t] = np.nanmean(prof[t, dphi <= p.wedge_halfwidth]) - np.nanmedian(prof[t])
    return out


def exit_track_signal(diffs: np.ndarray, centre: float, gr: float, end_angle: float, p: "Params",
                      step_deg: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Track the exit angle backwards from the traced end-state exit and read its signal.

    The annulus just outside the rim is unwrapped to W[t, phi] = change at phi minus the
    rim-wide median (cancels whole-grain focus changes), smoothed over the tube's angular
    width. A smooth angle path pinned to ``end_angle`` in the final bins maximises
    summed W; the signal is W along that path. Returns (signal, angles_deg).
    """
    phis_deg = np.arange(0.0, 360.0, step_deg)
    phis = np.deg2rad(phis_deg)
    radii = gr + np.arange(p.wedge_r[0], p.wedge_r[1] + 1e-9, 1.0)
    mx = (centre + radii[:, None] * np.cos(phis)[None]).astype(np.float32)
    my = (centre + radii[:, None] * np.sin(phis)[None]).astype(np.float32)
    prof = np.stack([cv2.remap(d, mx, my, cv2.INTER_LINEAR, borderValue=0).mean(axis=0) for d in diffs])
    w = prof - np.median(prof, axis=1, keepdims=True)
    k = max(1, int(round(p.wedge_halfwidth / step_deg)))
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    w = np.stack([np.convolve(np.concatenate([row[-k:], row, row[:k]]), kernel, "valid") for row in w])
    n_bins, n_ang = w.shape
    dist = np.abs((phis_deg[:, None] - phis_deg[None, :] + 180.0) % 360.0 - 180.0)
    trans = np.where(dist <= p.max_turn + 1e-9, -p.angle_penalty * dist, -np.inf)
    end_i = int(np.argmin(np.abs((phis_deg - end_angle + 180.0) % 360.0 - 180.0)))
    pinned = np.full(n_ang, -np.inf)
    pinned[end_i] = 0.0
    anchor = p.late_bins + 1
    total = w[0].copy()
    back = np.zeros((n_bins, n_ang), np.int32)
    for t in range(1, n_bins):
        cand = total[:, None] + trans
        back[t] = np.argmax(cand, axis=0)
        total = cand[back[t], np.arange(n_ang)] + w[t]
        if t >= n_bins - anchor:
            total = total + pinned
    track = np.zeros(n_bins, np.int32)
    track[-1] = int(np.argmax(total))
    for t in range(n_bins - 1, 0, -1):
        track[t - 1] = back[t, track[t]]
    return w[np.arange(n_bins), track], phis_deg[track]


def matched_stub_signal(signed: np.ndarray, late_minus_early: np.ndarray, pts: np.ndarray, centre: float,
                        p: "Params", theta: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Calibrated matched-filter score of a short stub at the exit, per bin.

    The template is the grain's own end-state change over the first ``mf_len`` px of its
    path (signed, +/- ``mf_half`` px across), so dark and bright-cored tubes are both
    matched. Each bin's signed change is correlated with the template at the exit (best
    of +/- ``mf_search`` degrees) and at control angles round the rim (same search);
    the score is (exit - median control) / robust sigma of the controls over the movie.
    ``theta`` (degrees per bin) turns the whole placement with the grain's rotation track.
    """
    step = 0.5
    s_idx = np.nonzero(np.arange(len(pts)) * p.step <= p.mf_len)[0]
    stub = pts[s_idx]
    tang = np.gradient(stub, axis=0) if len(stub) > 1 else np.array([[1.0, 0.0]])
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    across = np.arange(-p.mf_half, p.mf_half + 1e-9, step)
    grid = (stub[:, None, :] + across[None, :, None] * normal[:, None, :]).reshape(-1, 2)  # template points
    tmpl = cv2.remap(late_minus_early.astype(np.float32), grid[None, :, 0].astype(np.float32),
                     grid[None, :, 1].astype(np.float32), cv2.INTER_LINEAR)[0]
    tmpl = tmpl - tmpl.mean()
    norm = float(np.linalg.norm(tmpl))
    info = {"template_norm": round(norm, 2)}
    if norm < 1e-3:
        return np.zeros(len(signed)), info
    tmpl /= norm
    tmpls = [tmpl]
    if "negative" in p.mf_extra:  # a tube whose young look is the opposite of its mature one
        tmpls.append(-tmpl)
    centre_row = np.abs(across) < 1.0
    bright_core = float(tmpl.reshape(len(stub), len(across))[:, centre_row].mean()) > 0
    if "dark" in p.mf_extra and bright_core:  # a bright-cored tube may have been a plain dark line when young
        u = np.tile(across, len(stub))
        dark = -np.exp(-u ** 2 / 3.4)
        dark = dark - dark.mean()
        tmpls.append(dark / (np.linalg.norm(dark) + 1e-9))
    tmpl_mat = np.stack(tmpls, axis=1)  # (points, templates)
    rel = grid - centre
    exit_offsets = np.arange(-p.mf_search, p.mf_search + 1e-9, 3.0)
    ctrl_offsets = [a for a in range(45, 360, 45)]

    def placed(angle_deg):
        a = math.radians(angle_deg)
        x = centre + math.cos(a) * rel[:, 0] - math.sin(a) * rel[:, 1]
        y = centre + math.sin(a) * rel[:, 0] + math.cos(a) * rel[:, 1]
        return x.astype(np.float32), y.astype(np.float32)

    def maps(base):
        exit_maps = [placed(base + o) for o in exit_offsets]
        ctrl_maps = [[placed(base + c + o) for o in exit_offsets] for c in ctrl_offsets]
        return (np.concatenate([m[0] for m in exit_maps] + [m[0] for cm in ctrl_maps for m in cm])[None, :],
                np.concatenate([m[1] for m in exit_maps] + [m[1] for cm in ctrl_maps for m in cm])[None, :])

    n_pts, n_off = len(tmpl), len(exit_offsets)
    if theta is None:
        mx, my = maps(0.0)
        scores = np.stack([cv2.remap(d, mx, my, cv2.INTER_LINEAR, borderValue=0)[0] for d in signed])
    else:
        cache = {}
        rows = []
        for d, th in zip(signed, theta):
            key = round(float(th), 1)
            if key not in cache:
                cache[key] = maps(key)
            rows.append(cv2.remap(d, *cache[key], cv2.INTER_LINEAR, borderValue=0)[0])
        scores = np.stack(rows)
    all_scores = scores.reshape(len(signed), -1, n_pts) @ tmpl_mat  # (bins, placements, templates)
    z = None
    for j in range(tmpl_mat.shape[1]):
        sc = all_scores[:, :, j]
        exit_score = sc[:, :n_off].max(axis=1)
        ctrl = sc[:, n_off:].reshape(len(signed), len(ctrl_offsets), n_off).max(axis=2)
        base = np.median(ctrl, axis=1)
        resid = ctrl - base[:, None]
        sigma = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))), 1e-3)
        zj = (exit_score - base) / sigma
        z = zj if z is None else np.maximum(z, zj)
        if j == 0:
            info["control_sigma"] = round(sigma, 3)
    return z, info


def sustained_onset(signal: np.ndarray, p: "Params", threshold: float | None = None) -> tuple[int | None, float]:
    """First bin where ``signal`` rises above its pre-emergence noise and stays there.

    ``threshold`` (absolute) is used for already-calibrated scores; otherwise the
    threshold comes from the first ``wedge_base_bins`` bins.
    """
    if threshold is None:
        base = signal[:p.wedge_base_bins]
        mu = float(np.median(base))
        sigma = max(1.4826 * float(np.median(np.abs(base - mu))), 0.25)
        thr = mu + max(p.wedge_floor, p.wedge_k * sigma)
    else:
        thr = float(threshold)
    above = signal > thr
    n = len(signal)
    if above[-min(10, n):].mean() < 0.5:  # a tube never retracts: the exit stays changed to the end
        return None, thr
    for t in range(n):
        tail = above[t:t + p.persist_bins] if p.persist_bins > 0 else above[t:]
        if above[t] and above[t:t + p.wedge_hold].mean() >= 0.8 and tail.mean() >= 0.7:
            return t, thr
    return None, thr


def _local_maxima_tips(comp: np.ndarray, dist: np.ndarray, min_dist: int, nms_px: float, k: int) -> list:
    """Branch ends of a component: local maxima of the geodesic distance from the rim."""
    d = np.where(comp, dist, -1).astype(np.float32)
    mx = cv2.dilate(d, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(comp & (d >= mx) & (d >= min_dist))
    order = np.argsort(-d[ys, xs])
    tips = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if all(math.hypot(y - ty, x - tx) >= nms_px for ty, tx in tips):
            tips.append((y, x))
        if len(tips) >= k:
            break
    return tips


def candidate_paths(comp: np.ndarray, ring: np.ndarray, cost: np.ndarray, gr: float, centre: float,
                    p: "Params") -> list[np.ndarray]:
    """Centreline hypotheses: each branch end, reached from the nearest rim contact and, for a
    tube that wraps round and touches its grain again, from the other contacts too."""
    far, dist = _geodesic_far(comp, ring)
    tips = _local_maxima_tips(comp, dist, 4, p.cand_nms_px, p.cand_tips) or [far]
    # every branch of the region gets its end as a hypothesis, however far the others reach
    from skimage.morphology import skeletonize
    skel = skeletonize(comp)
    nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    ends = [(int(y), int(x)) for y, x in zip(*np.nonzero(skel & (nb == 2))) if dist[y, x] >= 6]
    for y, x in sorted(ends, key=lambda e: -dist[e]):
        if len(tips) >= p.cand_tips + p.cand_branch_tips:
            break
        if all(math.hypot(y - ty, x - tx) >= p.cand_nms_px for ty, tx in tips):
            tips.append((y, x))
    n_c, c_lab = cv2.connectedComponents((ring & comp).astype(np.uint8), connectivity=8)
    contacts = [c_lab == c for c in range(1, n_c) if (c_lab == c).sum() >= 2]
    out, seen = [], []
    for tip in tips:
        base = _cheapest_path(cost, comp, ring, tip)
        options = [base]
        if len(contacts) > 1:
            for c in contacts:
                if c[int(base[0][0]), int(base[0][1])]:
                    continue  # the default path already starts here
                alt = _cheapest_path(cost, comp, c, tip)
                if len(alt) >= 1.3 * len(base) and c[int(alt[0][0]), int(alt[0][1])]:
                    options.append(alt)
        for path in options:
            key = (int(path[0][0]) // 4, int(path[0][1]) // 4, tip)
            if key in seen or len(path) < 3:
                continue
            seen.append(key)
            out.append(path)
    return out


def read_path(ctx: dict, path_yx: np.ndarray, p: "Params") -> dict | None:
    """Kymograph, rotation track and growth front along one centreline hypothesis.

    Also scores the hypothesis: evidence explained by monotone growth from the exit, minus
    positive evidence the front leaves unexplained beyond it, and a penalty when the exit
    sits on material that was already there before the front left it (a foreign tube
    passing the rim, not one emerging from it).
    """
    centre, gr, reg, early, late = ctx["centre"], ctx["gr"], ctx["reg"], ctx["early"], ctx["late"]
    n_bins = ctx["n_bins"]
    path = path_yx[:, ::-1]  # (x, y) in crop coordinates
    if len(path) > 7:
        k = np.ones(5) / 5
        path = np.stack([np.convolve(np.pad(path[:, i], 2, mode="edge"), k, "valid") for i in range(2)], axis=1)
    v = path[0] - centre
    v = v / (np.linalg.norm(v) + 1e-9)
    path = np.vstack([centre + v * gr, path])
    pts, ss = _resample(path, p.step)
    if len(pts) < 4:
        return None
    flags, extra = [], {}
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    diffs, rg = ctx["diffs"], ctx["rg"]
    angles = (np.arange(-p.max_angle, p.max_angle + 1e-9, p.angle_step) if p.rotate else np.zeros(1))
    if p.evidence in ("matched", "union"):
        signed_all = ctx["signed"]
        across = np.arange(-p.mk_half, p.mk_half + 1e-9, 0.5)
        template = cross_section_template(late - early, pts, normal, across)
        kymo_all = matched_kymograph(signed_all, template, pts, normal, centre, across, angles)
        zero = int(np.argmin(np.abs(angles)))
        ctrl = np.concatenate([matched_kymograph(signed_all, template, pts, normal, centre, across,
                                                 angles[zero:zero + 1], lateral_offset=off)[:, 0]
                               for off in (-p.mk_control_px, p.mk_control_px)], axis=1)  # (bins, 2 * points)
        ctrl_sigma = np.maximum(1.4826 * np.median(np.abs(ctrl - np.median(ctrl, axis=1, keepdims=True)), axis=1), 0.3)
        base = kymo_all[:p.ref_bins].mean(axis=0)
        tau_all = np.maximum(p.mk_floor, base[None] + p.mk_k * ctrl_sigma[:, None, None])  # (bins, angles, points)
        evid_all = np.clip((kymo_all - tau_all) / tau_all, -1.0, 1.0)
        extra["matched_control_sigma_median"] = round(float(np.median(ctrl_sigma)), 3)
        if "dark" in p.mk_extra:
            # the young part of a tube (always the tip) often looks like a plain dark line
            dark = np.tile(-np.exp(-across ** 2 / 3.4), (len(pts), 1))
            dark -= dark.mean(axis=1, keepdims=True)
            dark /= np.linalg.norm(dark, axis=1, keepdims=True)
            k_d = matched_kymograph(signed_all, dark, pts, normal, centre, across, angles)
            c_d = np.concatenate([matched_kymograph(signed_all, dark, pts, normal, centre, across,
                                                    angles[zero:zero + 1], lateral_offset=off)[:, 0]
                                  for off in (-p.mk_control_px, p.mk_control_px)], axis=1)
            s_d = np.maximum(1.4826 * np.median(np.abs(c_d - np.median(c_d, axis=1, keepdims=True)), axis=1), 0.3)
            tau_d = np.maximum(p.mk_floor, k_d[:p.ref_bins].mean(axis=0)[None] + p.mk_k * s_d[:, None, None])
            evid_all = np.maximum(evid_all, np.clip((k_d - tau_d) / tau_d, -1.0, 1.0))
        matched_evid = evid_all
    if p.evidence != "matched":
        kymo_all = _kymograph(diffs, pts, normal, centre, p.lateral, angles)  # (bins, angles, points)
        if p.bg_subtract:
            # per-bin background change (focus / illumination drift) away from tube and grains
            far = cv2.dilate(ctx["tube_mask"].astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
            bg_mask = (rg > gr + 8) & ~ctx["blocked"] & ~far
            bg_level = np.array([float(np.median(d[bg_mask])) if bg_mask.any() else 0.0 for d in diffs])
            kymo_all = np.clip(kymo_all - bg_level[:, None, None], 0.0, None)
            extra["background_change_max"] = round(float(bg_level.max()), 2)
        base = kymo_all[:p.ref_bins].mean(axis=0)
        noise = np.maximum(kymo_all[:p.ref_bins].std(axis=0), 0.5)
        tau_all = np.maximum(p.evid_floor, base + p.evid_k * noise)[None]
        evid_all = np.clip((kymo_all - tau_all) / tau_all, -1.0, 1.0)
    if p.evidence == "union":  # either reading may carry the tube: shape-agnostic |change| or the template
        evid_all = np.maximum(evid_all, matched_evid)
    evid_all[:, :, ss < p.skip_px] = 0.0
    # Rotation is judged only on points that can reveal it: >= rot_min_s along the path and
    # clear of the rim (a rim-hugging segment slides along the rim under any rotation).
    radius_pts = np.hypot(*(pts - centre).T)
    informative = (ss >= p.rot_min_s) & (radius_pts >= gr + p.rot_rim_clear)
    if p.rotate and informative.sum() >= 4:
        rot_score = np.clip(evid_all[:, :, informative], 0.0, None).sum(axis=2)
        theta = rotation_track(rot_score, angles, p.angle_penalty, p.max_turn, p.late_bins + 1)
    else:
        theta = np.zeros(n_bins)
    ai = np.array([int(np.argmin(np.abs(angles - a))) for a in theta])
    kymo = kymo_all[np.arange(n_bins), ai]
    tau = tau_all[-1, ai[-1]] if tau_all.ndim == 3 else tau_all[ai[-1]]
    evid = evid_all[np.arange(n_bins), ai]
    # missing evidence costs less than evidence gains: the front may bridge a short faint stretch
    dp_evid = np.where(evid < 0, p.neg_weight * evid, evid) if p.neg_weight != 1.0 else evid
    front = dp_front(dp_evid, max(1, int(round(p.vmax_px / p.step))))
    behind = np.arange(len(pts))[None, :] < front[:, None]
    explained = float(np.sum(np.where(behind, evid, 0.0)))
    beyond = float(np.sum(np.where(~behind, np.clip(evid, 0.0, None), 0.0)))
    # an exit on a structure that predates the front's departure is a foreign tube passing by
    through = False
    moved = np.nonzero(front >= int(round(3.0 / p.step)))[0]
    if len(moved):
        # a drifting grain's own edge leaves change round the rim: keep clear of it
        margin = 3.0 + float(np.hypot(*ctx["ls"].T).max())
        yy, xx = np.nonzero(ctx["comp"] & (np.hypot(*(np.mgrid[0:rg.shape[0], 0:rg.shape[1]][::-1] -
                                                     pts[0][:, None, None])) <= p.through_px + margin)
                            & (rg > gr + margin))
        if len(yy):
            dpath = np.min(np.hypot(xx[:, None] - pts[None, :, 0], yy[:, None] - pts[None, :, 1]), axis=1)
            yy, xx = yy[dpath > 4.0], xx[dpath > 4.0]
        if len(yy) >= p.through_min_px:
            frac = (diffs[:, yy, xx] > ctx["thr"]).mean(axis=1)
            t0 = int(moved[0])
            early_bins = frac[max(0, t0 - 8):max(0, t0 - 2)]
            through = bool(len(early_bins) and np.median(early_bins) >= 0.5)
    score = explained - p.beyond_weight * beyond
    # a tube is a ridge in the end-state change: higher on the path than just beside it
    chg = ctx["change"]
    lat = p.ridge_px
    samp = lambda q: cv2.remap(chg.astype(np.float32), q[:, 0].astype(np.float32)[None],
                               q[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
    on_path = np.max([samp(pts + o * normal) for o in (-1.0, 0.0, 1.0)], axis=0)
    beside = np.maximum(samp(pts + lat * normal), samp(pts - lat * normal))
    far_pts = ss >= p.skip_px + 2.0
    ridge = float(np.mean((on_path - beside > 0.25 * on_path)[far_pts])) if far_pts.any() else 1.0
    if p.ridge_weight > 0 and score > 0:
        score *= (1.0 - p.ridge_weight) + p.ridge_weight * ridge
    if through:
        score = score * 0.25 if score > 0 else score - 10.0
    extra["rotation_deg"] = [round(float(t), 1) for t in theta]
    if p.rotate and np.max(np.abs(theta)) >= 10:
        flags.append(f"rotates:{np.max(np.abs(theta)):.0f}deg")
    return {"pts": pts, "ss": ss, "theta": theta, "front": front, "kymo": kymo, "tau": tau, "evid": evid,
            "explained": explained, "beyond": beyond, "through": through, "ridge": ridge, "score": score,
            "flags": flags, "result": extra}


def grain_settling(crops: np.ndarray, centre: float, gr: float, p: "Params") -> dict:
    """Is there a settled grain here, and from which bin?

    The census reads the field's first bins; a grain still arriving then (a blurred,
    moving blob) is found late or off-centre, and a passing piece of debris is found
    where no grain ever settles. The rim fit per early bin tells them apart.
    """
    n = min(len(crops), p.settle_bins)
    q = np.array([census.rim_fit(c, centre, centre, gr)[2] for c in crops[:n]])
    qmed = float(np.median(q))
    if qmed < p.grain_min_rim:
        return {"no_grain": True, "b0": 0, "rim_median": round(qmed, 2)}
    out = {"no_grain": False, "b0": 0, "rim_median": round(qmed, 2)}
    if float(np.min(q[:p.ref_bins])) >= 0.4 * qmed:
        return out
    for b in range(1, n - 2):
        if np.all(q[b:b + 3] >= 0.85 * qmed):  # the rim has come into focus and stays
            out["b0"] = b
            break
    return out


def _pad_front(res: dict, frames: list, b0: int) -> dict:
    """Re-express a result computed from bin b0 on the grain's full frame list."""
    res["length"] = {"frames": frames, "px": [0.0] * b0 + list(res["length"]["px"])}
    if res.get("tip"):
        res["tip"] = {"frames": frames, "xy": [res["tip"]["xy"][0]] * b0 + list(res["tip"]["xy"])}
    if res.get("rotation_deg"):
        res["rotation_deg"] = [res["rotation_deg"][0]] * b0 + list(res["rotation_deg"])
    if res.get("wedge"):
        res["wedge"]["signal"] = [0.0] * b0 + list(res["wedge"]["signal"])
    return res


def analyze_grain(renderer: Renderer, meta: dict, grain: dict, others: list[dict], p: Params,
                  _settled: bool = False) -> dict:
    fpb, rs = int(meta["frames_per_bin"]), int(meta.get("ref_start", 0))
    n_bins = int(meta["n_bins"]) - rs  # bins before the reference (settling) are not observed
    gx, gy, gr = grain["x"], grain["y"], grain["r"]
    half = p.half
    crops = np.stack([renderer.crop(b, gx, gy, half) for b in range(rs, rs + n_bins)])
    centre = half - 0.5  # crop pixel coordinate of the grain centre
    if p.settle and not _settled:
        st = grain_settling(crops, centre, gr, p)
        frames = [b * fpb + fpb // 2 for b in range(rs, rs + n_bins)]
        if st["no_grain"]:
            return {"id": grain["id"], "x": gx, "y": gy, "r": gr, "flags": ["no_grain"], "map_threshold": 1.0,
                    "status": "unobservable", "onset_frame": None, "onset_interval": None,
                    "length": {"frames": frames, "px": [0.0] * n_bins}, "path": [], "rim_median": st["rim_median"],
                    "_diag": (crops[-2], np.zeros_like(crops[0]), np.zeros(crops[0].shape, bool), None, None, None,
                              centre)}
        b0 = st["b0"]
        if b0 > 0:
            # re-find the settled grain (the census saw it arriving) and read it from bin b0 on
            ref = crops[b0:b0 + 3].mean(axis=0)
            w = int(gr + 30)
            c = int(round(centre))
            found = census.detect(ref[c - w:c + w, c - w:c + w], r_min=max(5, int(gr - 4)), r_max=int(gr + 4),
                                  ring_min=5.0, body_min=15.0)
            if found:
                best = min(found, key=lambda f: math.hypot(f["x"] - w, f["y"] - w))
                if math.hypot(best["x"] - w, best["y"] - w) <= 14:
                    gx, gy = gx + best["x"] - w + 0.5, gy + best["y"] - w + 0.5
            res = analyze_grain(renderer, {**meta, "ref_start": rs + b0}, {**grain, "x": gx, "y": gy}, others, p,
                                _settled=True)
            res["flags"].append(f"settled_from_bin:{b0}")
            return _pad_front(res, frames, b0)
    ls = local_shifts(crops, centre, gr, p.reg_pad, p.ref_bins)
    reg = np.stack([cv2.warpAffine(c, np.float32([[1, 0, -dx], [0, 1, -dy]]), (2 * half, 2 * half),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    for c, (dx, dy) in zip(crops, ls)])
    late_idx = list(range(n_bins - 1 - p.late_bins, n_bins - 1))
    early = reg[:p.ref_bins].mean(axis=0)
    late = reg[late_idx].mean(axis=0)
    yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float64)
    rg = np.hypot(xx - centre, yy - centre)
    # pixels whose source position leaves the frame in any bin are not evidence
    shifts = np.asarray(meta["shifts"])[rs:] + ls
    ref_x, ref_y = gx - half + xx + 0.5, gy - half + yy + 0.5
    valid = ((ref_x + shifts[:, 0].min() >= 0) & (ref_x + shifts[:, 0].max() < renderer.width) &
             (ref_y + shifts[:, 1].min() >= 0) & (ref_y + shifts[:, 1].max() < renderer.height))
    change = cv2.GaussianBlur(np.abs(late - early), (0, 0), p.map_sigma)
    blocked = (rg < gr - 1.0) | ~valid
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        if -o["r"] - 5 < ox < 2 * half + o["r"] + 5 and -o["r"] - 5 < oy < 2 * half + o["r"] + 5:
            blocked |= np.hypot(xx - ox, yy - oy) < o["r"] + 2.0
    bg = change[(rg > gr + 30) & ~blocked]
    bg = bg[bg < np.percentile(bg, 95)] if bg.size else bg
    sigma_bg = 1.4826 * float(np.median(np.abs(bg - np.median(bg)))) if bg.size else 1.0
    thr = max(p.map_floor, p.map_k * sigma_bg)
    tube_mask = (change > thr) & ~blocked
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(tube_mask.astype(np.uint8), connectivity=8)
    ring = (rg >= gr - 1.0) & (rg <= gr + 4.0)
    attached = [(stats[l, cv2.CC_STAT_AREA], l) for l in range(1, n_lab)
                if stats[l, cv2.CC_STAT_AREA] >= p.min_component_px and np.any(ring & (lab == l))]
    result = {"id": grain["id"], "x": gx, "y": gy, "r": gr, "flags": [], "map_threshold": round(thr, 2),
              "local_shift_max_px": round(float(np.hypot(*ls.T).max()), 2)}
    frames = [b * fpb + fpb // 2 for b in range(rs, rs + n_bins)]
    if not attached:
        result.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None,
                      length={"frames": frames, "px": [0.0] * n_bins}, path=[])
        result["_diag"] = (late, change, tube_mask, None, None, None, centre)
        return result
    attached.sort(reverse=True)
    # with growth-scored candidates every region touching the rim is a hypothesis; otherwise the largest
    comp = np.isin(lab, [l for _, l in attached]) if p.candidates else lab == attached[0][1]
    if len(attached) > 1:
        result["flags"].append("second_attached_component")
    # a change region shared with other grains is split by geodesic ownership: each pixel
    # belongs to the grain nearest to it through the change map itself
    rivals = []
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        o_ring = np.abs(np.hypot(xx - ox, yy - oy) - o["r"]) <= 4.0
        if np.any(comp & o_ring):
            result["flags"].append(f"touches:{o['id']}")
            rivals.append(o_ring)
    if rivals:
        owner = geodesic_owner(comp, [ring] + rivals)
        own = owner == 0
        n_own, own_lab = cv2.connectedComponents(own.astype(np.uint8), connectivity=8)
        keep = [l for l in range(1, n_own) if np.any(ring & (own_lab == l))]
        comp = np.isin(own_lab, keep) if keep else own
        result["flags"].append("shared_change_split")
    cost = 1.0 / ((change if p.path_sigma == p.map_sigma else
                   cv2.GaussianBlur(np.abs(late - early), (0, 0), p.path_sigma)) + 1.0)
    diffs = np.abs(reg - early[None])
    ctx = {"reg": reg, "early": early, "late": late, "ls": ls, "late_idx": late_idx,
           "centre": centre, "gr": gr, "rg": rg, "blocked": blocked, "tube_mask": tube_mask, "diffs": diffs,
           "signed": (reg - early[None]).astype(np.float32), "n_bins": n_bins, "thr": thr, "comp": comp,
           "change": change}
    if p.candidates:
        cands = []
        for path_yx in candidate_paths(comp, ring, cost, gr, centre, p):
            read = read_path(ctx, path_yx, p)
            if read is not None:
                cands.append(read)
        # a candidate that is only a shorter stretch of another along the same tube is dropped:
        # the growth front, not the path, decides how far the tube got (and tips are faint)
        live = []
        for i, a in enumerate(cands):
            nested = False
            for j, b in enumerate(cands):
                if j != i and b["ss"][-1] > a["ss"][-1] + 1.0:
                    d = np.min(np.hypot(a["pts"][:, None, 0] - b["pts"][None, :, 0],
                                        a["pts"][:, None, 1] - b["pts"][None, :, 1]), axis=1)
                    if np.all(d <= p.nest_px):
                        nested = True
                        break
            if not nested:
                live.append(a)
        read = max(live or cands, key=lambda c: c["score"]) if cands else None
        result["path_candidates"] = [{"length_px": round(float(c["ss"][-1]), 1), "score": round(c["score"], 1),
                                      "explained": round(c["explained"], 1), "beyond": round(c["beyond"], 1),
                                      "ridge": round(c["ridge"], 2), "through": c["through"]} for c in cands]
        if len(cands) > 1 and read is not cands[0]:
            result["flags"].append("path_by_growth")
    else:
        far, _ = _geodesic_far(comp, ring)
        read = read_path(ctx, _cheapest_path(cost, comp, ring, far), p)
    if read is None:  # a one- or two-pixel path is not a tube
        result["flags"].append("degenerate_path")
        result.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None,
                      length={"frames": frames, "px": [0.0] * n_bins}, path=[])
        result["_diag"] = (late, change, tube_mask, None, None, None, centre)
        return result
    pts, ss, theta, front, kymo, tau = (read[k] for k in ("pts", "ss", "theta", "front", "kymo", "tau"))
    result["flags"] += read["flags"]
    result.update({k: v for k, v in read["result"].items()})
    # contact censoring: stop measuring where the path first reaches another grain's rim
    contact_idx = None
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        near = np.nonzero(np.hypot(pts[:, 0] - ox, pts[:, 1] - oy) <= o["r"] + p.contact_px)[0]
        if len(near) and (contact_idx is None or near[0] < contact_idx):
            contact_idx = int(near[0])
    if contact_idx is not None:
        censor = np.nonzero(front > contact_idx)[0]
        if len(censor):
            result["length_censored_from_frame"] = frames[int(censor[0])]
            result["flags"].append("contact_censored")
        front = np.minimum(front, contact_idx)
    length = np.array([ss[f - 1] if f > 0 else 0.0 for f in front])

    def rotated(pt, deg):
        a = math.radians(deg)
        r = pt - centre
        return np.array([centre + math.cos(a) * r[0] - math.sin(a) * r[1],
                         centre + math.sin(a) * r[0] + math.cos(a) * r[1]])

    tips = np.array([rotated(pts[max(f - 1, 0)], th) for f, th in zip(front, theta)])
    above = np.nonzero(length >= p.onset_px)[0]
    front_onset = int(above[0]) if len(above) else None
    end_angle = math.degrees(math.atan2(pts[0][1] - centre, pts[0][0] - centre))
    if p.onset_source == "matched":
        signed = (reg - early[None]).astype(np.float32)
        z, mf_info = matched_stub_signal(signed, (late - early), pts, centre, p,
                                         theta=theta if p.mf_follow_rotation is True and p.rotate else None)
        if p.mf_follow_rotation == "both" and p.rotate and np.max(np.abs(theta)) >= 5:
            # a rotating grain's tube emerged elsewhere on the rim: also look along the rotation track
            z_rot, _ = matched_stub_signal(signed, (late - early), pts, centre, p, theta=theta)
            z = np.maximum(z, z_rot)
        result["matched_filter"] = mf_info
        wedge, exit_track = z, np.full(n_bins, end_angle)
    elif p.onset_source == "wedge_fixed":
        wedge = wedge_signal(diffs, centre, gr, np.full(n_bins, end_angle), p)
        exit_track = np.full(n_bins, end_angle)
    else:
        wedge, exit_track = exit_track_signal(diffs, centre, gr, end_angle, p)
    result["exit_angle_deg"] = [round(float(a), 1) for a in exit_track]
    wedge_onset, wedge_thr = sustained_onset(wedge, p, threshold=p.mf_z if p.onset_source == "matched" else None)
    if wedge_onset is not None and p.onset_source == "matched" and p.mf_z_low is not None:
        # hysteresis: a confirmed stub reaches back while its score stays above the lower threshold
        confirmed = wedge_onset
        while wedge_onset > 0 and wedge[wedge_onset - 1] > p.mf_z_low and confirmed - wedge_onset < p.mf_back_bins:
            wedge_onset -= 1
    result["onset_bins"] = {"front": front_onset, "wedge": wedge_onset}
    result["wedge"] = {"signal": [round(float(v), 2) for v in wedge], "threshold": round(wedge_thr, 2)}
    b = front_onset if p.onset_source == "front" else wedge_onset
    if b is None and front_onset is not None and p.onset_source != "front":
        b = front_onset
        result["flags"].append("onset_from_front")
    if b is None:
        status, onset, interval = "no_emergence_by_end", None, None
        result["flags"].append("tube_map_without_onset")
    elif b == 0:
        status, onset, interval = "emerged_at_start", frames[0], None
    else:
        status, onset, interval = "emerged_within", frames[b], [frames[b - 1], frames[b]]
    if b is None:
        length[:] = 0.0
    elif b > 0:
        length[:b] = 0.0  # no tube before its own onset
        tips[:b] = rotated(pts[0], 0.0)
    if b is not None and length[-1] < p.min_tube_px and "contact_censored" not in result["flags"]:
        result["flags"].append("front_too_short")
        status, onset, interval = "no_emergence_by_end", None, None
        length[:] = 0.0
    to_ref = lambda xy: [round(float(xy[0] - centre + gx), 2), round(float(xy[1] - centre + gy), 2)]
    result.update(status=status, onset_frame=onset, onset_interval=interval,
                  length={"frames": frames, "px": [round(float(v), 2) for v in length]},
                  tip={"frames": frames, "xy": [to_ref(t) for t in tips]},
                  path=[to_ref(q) for q in pts[:: max(1, int(2 / p.step))]] + [to_ref(pts[-1])],
                  exit_xy=to_ref(pts[0]), final_length_px=round(float(length[-1]), 2),
                  path_length_px=round(float(ss[-1]), 2))
    result["_diag"] = (late, change, tube_mask, pts, kymo, (front, tau), centre)
    return result


def _diagnostic(res: dict, fpb: int) -> np.ndarray:
    late, change, mask, pts, kymo, front_tau, centre = res["_diag"]
    lo, hi = np.percentile(late, [0.5, 99.5])
    a = cv2.cvtColor(np.clip((late - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    c = cv2.cvtColor(np.clip(change / max(res["map_threshold"] * 3, 1e-6) * 255, 0, 255).astype(np.uint8),
                     cv2.COLOR_GRAY2BGR)
    c[mask] = (0.5 * c[mask] + (0, 90, 0)).astype(np.uint8)
    if pts is not None:
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(a, [poly], False, (0, 200, 255), 1)
        cv2.circle(a, tuple(np.round(pts[0]).astype(int)), 3, (0, 0, 255), -1)
    panels = [a, c]
    if kymo is not None:
        front, tau = front_tau
        k = np.clip(kymo / max(float(np.percentile(kymo, 99)), 1e-6) * 255, 0, 255).astype(np.uint8)
        k = cv2.cvtColor(cv2.resize(k, (late.shape[1], late.shape[0]), interpolation=cv2.INTER_NEAREST),
                         cv2.COLOR_GRAY2BGR)
        sy, sx = late.shape[0] / kymo.shape[0], late.shape[1] / kymo.shape[1]
        line = np.array([[f * sx, (t + 0.5) * sy] for t, f in enumerate(front)], np.int32).reshape(-1, 1, 2)
        cv2.polylines(k, [line], False, (0, 0, 255), 1)
        panels.append(k)
    sheet = np.concatenate(panels, axis=1)
    label = (f"{res['id']} {res['status']} onset {res.get('onset_frame')} final {res.get('final_length_px', 0)} px "
             f"{' '.join(res['flags'])}")
    band = np.full((18, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(band, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([band, sheet], axis=0)


def analyze(cache_dir: str | Path, out_dir: str | Path, grains_path: str | Path | None = None,
            params: Params | None = None, only: list[str] | None = None, video: bool = False, log=print) -> dict:
    p = params or Params()
    bins, meta = stack.load(cache_dir)
    renderer = Renderer(bins, meta)
    out_dir = Path(out_dir)
    (out_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    src = Path(grains_path) if grains_path else Path(cache_dir) / "grains.json"
    doc = json.loads(src.read_text())
    grains = list(doc["grains"].values()) if isinstance(doc["grains"], dict) else doc["grains"]
    grains = [g for g in grains if not g.get("excluded")]
    started = time.time()
    results = []
    for g in grains:
        if only and g["id"] not in only:
            continue
        others = [o for o in grains if o["id"] != g["id"]]
        res = analyze_grain(renderer, meta, g, others, p)
        cv2.imwrite(str(out_dir / "diagnostics" / f"{g['id']}.png"), _diagnostic(res, meta["frames_per_bin"]))
        res.pop("_diag", None)
        results.append(res)
        log(f"{g['id']}: {res['status']:<20} onset {str(res.get('onset_frame')):>6}  "
            f"final {res.get('final_length_px', 0):6.1f} px  {' '.join(res['flags'])}")
    pred = {"schema": PRED_SCHEMA, "method": f"sparsetrack-v1 {__version__}", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cache": str(cache_dir), "grains_source": str(src), "params": asdict(p),
            "frames_per_bin": meta["frames_per_bin"], "movie": meta["movie"], "grains": results}
    (out_dir / "predictions.json").write_text(json.dumps(pred))
    with open(out_dir / "grains.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grain", "x", "y", "status", "onset_frame", "onset_after", "onset_by", "final_length_px", "flags"])
        for r in results:
            iv = r.get("onset_interval") or [None, None]
            w.writerow([r["id"], r["x"], r["y"], r["status"], r.get("onset_frame"), iv[0], iv[1],
                        r.get("final_length_px", 0), ";".join(r["flags"])])
    with open(out_dir / "growth.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grain", "frame", "length_px", "tip_x", "tip_y"])
        for r in results:
            tips = (r.get("tip") or {}).get("xy") or [[None, None]] * len(r["length"]["frames"])
            for f, L, (tx, ty) in zip(r["length"]["frames"], r["length"]["px"], tips):
                w.writerow([r["id"], f, L, tx, ty])
    from . import report
    isolated = [r["id"] for r in results if next((g for g in grains if g["id"] == r["id"]), {}).get("isolated", True)]
    pop = report.write_population(pred, out_dir, set(isolated))
    report.write_growth_curves(pred, out_dir, isolated)
    if pop and pop.get("t50_interval"):
        log(f"population ({pop['n']} isolated grains): half germinated by frame {pop['t50_interval'][1]:.0f}")
    if video:
        report.write_video(renderer, meta, pred, out_dir / "field_overlay.mp4", ids=set(isolated))
        log(f"video -> {out_dir / 'field_overlay.mp4'}")
    log(f"{len(results)} grains in {time.time() - started:.0f} s -> {out_dir}")
    return pred
