"""Version 3 training samples: the version 2 extractions with the version 3 inputs, plus shifted-bin apex samples.

    python -m prototypes.tip_detector.build3 ld m2 m1

Per movie: ``samples3_<movie>.npy`` (int8 inputs, N x 5 x 192 x 192, 1/24 unit a step; memory-mapped in training) and
``samples3_<movie>.npz`` (packed masks, apexes, info).

Extractions as build.py (version 2): per FULL trace one on the grain, one on the apex, one on a random point of the
trace; PARTIAL two; no_tube/burst one; five bins before each onset. Same targets and scored region (build.annotate).

New (weak supervision, "shift_apex"): the apex at bins b - 2 and b - 4 of a FULL trace at bin b is not labelled, but the
traced body is: the tube then ended on the same path, shorter by about v x k, with v its mean growth since the previous
trace of the grain (or since its onset, length 0). Each such extraction (centred on that point, +/- 40 px) takes every
labelled grain's trace at bin b shortened that way (target sigma 2 + v k / 4 px; the path beyond is not tube yet and is
scored as such); a FULL trace without a growth estimate is blind (as an unsure one), PARTIAL ends are blind as before.
Only where v k <= 12 px and the shortened tube is >= 4 px long.

Also stored per extraction: ``near``, pixels within 8 px of a labelled trace or r + 8 px of a labelled grain (for the
training option that leaves unlabelled structure unscored in the crowded movies), and per apex its trace length.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .build import BAND, EXT, NEG_BACK, annotate, seg_dist
from .common import OUT
from .common3 import CHANNELS3, Q, Movie3

KINDS3 = ("full_grain", "full_apex", "full_path", "partial", "no_tube", "pre_onset", "shift_apex")
SHIFTS = (2, 4)
MAX_SHIFT_PX = 12.0
N_AP = 16


def velocities(mv: Movie3) -> dict:
    """(gid, b) -> mean growth (px a bin) of the FULL trace at b since the grain's previous trace or its onset."""
    out = {}
    for gid, lab in mv.labels.items():
        on = lab.get("onset") or {}
        prev = (int(on["first_visible_bin"]), 0.0) if on.get("verdict") == "emerged_within" else None
        for b, t in mv.traces(gid, ("full", "partial")):
            L = float(t.get("length_px") or 0.0)
            if t["state"] == "full" and prev is not None and b > prev[0]:
                out[(gid, b)] = max(0.0, (L - prev[1]) / (b - prev[0]))
            prev = (b, L)
    return out


def cut_path(path: np.ndarray, s_end: float) -> np.ndarray:
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    keep = path[s < s_end]
    end = [np.interp(s_end, s, path[:, 0]), np.interp(s_end, s, path[:, 1])]
    return np.vstack([keep, end]) if len(keep) else np.array([path[0], end])


def near_labels(mv: Movie3, b: int, cx: float, cy: float, half: int, traced_bin: int | None = None) -> np.ndarray:
    S = 2 * half
    x0, y0 = cx - half + 0.5, cy - half + 0.5
    jj, ii = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    near = np.zeros((S, S), bool)
    tb = b if traced_bin is None else traced_bin
    for gid, lab in mv.labels.items():
        t = (lab.get("traces") or {}).get(str(tb))
        gx, gy, r = mv.grain_at(gid, b, t if traced_bin is None else None)
        L = float((t or {}).get("length_px") or 0.0)
        if np.hypot(gx - cx, gy - cy) > half * 1.5 + r + 40 + L:
            continue
        near |= np.hypot(jj - (gx - x0), ii - (gy - y0)) <= r + 8
        path = np.asarray((t or {}).get("path_xy_ref") or [], float).reshape(-1, 2)
        if len(path) >= 2:
            near |= seg_dist(jj, ii, path - [x0, y0]) <= 8
    return near


def annotate_shift(mv: Movie3, b: int, k: int, cx: float, cy: float, half: int, vel: dict):
    """Scored region, body, apexes (x, y, sigma, L) for an extraction of bin b - k built from the traces at bin b."""
    S = 2 * half
    x0, y0 = cx - half + 0.5, cy - half + 0.5
    jj, ii = np.meshgrid(np.arange(S, dtype=np.float32), np.arange(S, dtype=np.float32))
    scored, blind, body = (np.zeros((S, S), bool) for _ in range(3))
    apexes = []
    for gid, lab in mv.labels.items():
        t = (lab.get("traces") or {}).get(str(b))
        if t is None:
            continue
        gx, gy, r = mv.grain_at(gid, b, t)
        L = float(t.get("length_px") or 0.0)
        if np.hypot(gx - cx, gy - cy) > half * 1.5 + r + 40 + L:
            continue
        # the grain's (and its tube's) move from b - k to b, by the labelling tool's following
        sh = np.zeros(2)
        if gid in mv.offsets:
            sh = np.asarray(mv.offsets[gid][b - k], float) - np.asarray(mv.offsets[gid][b], float)
        gd = np.hypot(jj - (gx + sh[0] - x0), ii - (gy + sh[1] - y0))
        path = np.asarray(t.get("path_xy_ref") or [], float).reshape(-1, 2)
        q = path + sh - [x0, y0] if len(path) else path
        st = t["state"]
        v = vel.get((gid, b))
        if st == "full" and len(q) >= 2 and v is not None and L - v * k >= 4 and v * k <= MAX_SHIFT_PX:
            s_tot = float(np.hypot(*np.diff(q, axis=0).T).sum())
            qc = cut_path(q, max(1.0, s_tot - v * k))
            scored |= (gd <= r + BAND) | (seg_dist(jj, ii, q) <= BAND)
            body |= seg_dist(jj, ii, qc) <= 2.5
            apexes.append((qc[-1][0], qc[-1][1], 2.0 + v * k / 4.0, L - v * k))
        elif st == "partial" and len(q) >= 2:
            d = seg_dist(jj, ii, q)
            scored |= (gd <= r + BAND) | (d <= BAND)
            body |= d <= 2.5
            blind |= np.hypot(jj - q[-1][0], ii - q[-1][1]) <= 40
        elif st in ("no_tube", "burst"):
            scored |= gd <= r + BAND
        else:  # unsure, FULL without a growth estimate, or too short / too fast to shift
            blind |= gd <= r + 40
            if len(q) >= 2:
                blind |= seg_dist(jj, ii, q) <= BAND
                blind |= np.hypot(jj - q[-1][0], ii - q[-1][1]) <= 40
    scored &= ~blind
    return scored, body, apexes, blind


def build(name: str, seed: int = 0, log=print):
    mv = Movie3(name)
    vel = velocities(mv)
    rng = np.random.default_rng(seed)
    W, H = mv.R.width, mv.R.height
    gids = sorted(mv.labels)
    plan = []  # (b, cx, cy, kind, gid, extra)

    def add(b, cx, cy, kind, gid, extra=None):
        cx = float(np.clip(round(cx), EXT + 4, W - EXT - 4))
        cy = float(np.clip(round(cy), EXT + 4, H - EXT - 4))
        plan.append((b, cx, cy, kind, gid, extra))

    for gid in gids:
        lab = mv.labels[gid]
        for b, t in mv.traces(gid, ("full", "partial", "no_tube", "burst")):
            if not mv.lo <= b <= mv.hi:
                continue
            gx, gy, r = mv.grain_at(gid, b, t)
            j = lambda s: rng.uniform(-s, s)
            if t["state"] in ("no_tube", "burst"):
                add(b, gx + j(24), gy + j(24), "no_tube", gid)
                continue
            path = np.asarray(t["path_xy_ref"], float)
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
            u = rng.uniform(0, s[-1])
            pp = (np.interp(u, s, path[:, 0]), np.interp(u, s, path[:, 1]))
            if t["state"] == "full":
                add(b, gx + j(24), gy + j(24), "full_grain", gid)
                add(b, path[-1][0] + j(40), path[-1][1] + j(40), "full_apex", gid)
                add(b, pp[0] + j(30), pp[1] + j(30), "full_path", gid)
                v = vel.get((gid, b))
                for k in SHIFTS:
                    L = float(t.get("length_px") or s[-1])
                    if v is None or b - k < mv.lo or L - v * k < 4 or v * k > MAX_SHIFT_PX:
                        continue
                    e = cut_path(path, max(1.0, s[-1] - v * k))[-1]
                    add(b - k, e[0] + j(40), e[1] + j(40), "shift_apex", gid, (b, k))
            else:
                add(b, gx + j(24), gy + j(24), "partial", gid)
                add(b, pp[0] + j(30), pp[1] + j(30), "partial", gid)
        on = lab.get("onset") or {}
        if on.get("verdict") == "emerged_within":
            for k in NEG_BACK:
                b = int(on["last_absent_bin"]) - k
                if b < mv.lo:
                    continue
                gx, gy, r = mv.grain_at(gid, b)
                add(b, gx + rng.uniform(-24, 24), gy + rng.uniform(-24, 24), "pre_onset", gid, (gx, gy, r + 40))

    N, S = len(plan), 2 * EXT
    xpath = OUT / f"samples3_{name}.npy"
    X = np.lib.format.open_memmap(xpath, mode="w+", dtype=np.int8, shape=(N, len(CHANNELS3), S, S))
    SC, BO, BL, NE, AP, INFO = [], [], [], [], [], []
    t0 = time.time()
    for n, (b, cx, cy, kind, gid, extra) in enumerate(plan):
        x = mv.inputs3(b, cx, cy, EXT)
        X[n] = np.round(x * Q).astype(np.int8)
        a = np.full((N_AP, 4), np.nan, np.float32)
        if kind == "shift_apex":
            bt, k = extra
            sc, bo, ap, bl = annotate_shift(mv, bt, k, cx, cy, EXT, vel)
            ne = near_labels(mv, b, cx, cy, EXT, traced_bin=bt)
            for i, p in enumerate(ap[:N_AP]):
                a[i] = p
            shift = k
        else:
            sc, bo, ap, bl = annotate(mv, b, cx, cy, EXT, extra if kind == "pre_onset" else None)
            ne = near_labels(mv, b, cx, cy, EXT)
            Ls = apex_lengths(mv, b, cx, cy, EXT, ap)
            for i, p in enumerate(ap[:N_AP]):
                a[i] = (p[0], p[1], 2.0, Ls[i])
            shift = 0
        SC.append(np.packbits(sc, axis=-1)); BO.append(np.packbits(bo, axis=-1)); BL.append(np.packbits(bl, axis=-1))
        NE.append(np.packbits(ne, axis=-1)); AP.append(a)
        INFO.append((gids.index(gid), b, KINDS3.index(kind), cx, cy, shift))
    X.flush()
    del X
    info = np.array(INFO, np.float32)
    np.savez_compressed(OUT / f"samples3_{name}.npz", scored=np.stack(SC), body=np.stack(BO), blind=np.stack(BL),
                        near=np.stack(NE), apex=np.stack(AP), info=info, gids=np.array(gids),
                        channels=np.array(CHANNELS3), labelled_fraction=len(mv.labels) / len(mv.grains))
    kinds = {k: int((info[:, 2] == i).sum()) for i, k in enumerate(KINDS3)}
    apx = np.stack(AP)
    log(f"{name}: {N} extractions {kinds}, apexes {int(np.isfinite(apx[:, :, 0]).sum())} "
        f"(long {int((apx[:, :, 3] >= 60).sum())}), labelled grains {len(mv.labels)}/{len(mv.grains)}, "
        f"{time.time() - t0:.0f} s -> {xpath.name} ({xpath.stat().st_size / 1e6:.0f} MB)", flush=True)


def apex_lengths(mv: Movie3, b: int, cx: float, cy: float, half: int, apexes) -> list[float]:
    """The traced length of each apex annotate() returned (same order: labelled grains with a FULL trace at b)."""
    x0, y0 = cx - half + 0.5, cy - half + 0.5
    out = []
    for p in apexes:
        best, Lb = 1e9, np.nan
        for gid, lab in mv.labels.items():
            t = (lab.get("traces") or {}).get(str(b))
            if t is None or t["state"] != "full" or not t.get("path_xy_ref"):
                continue
            e = np.asarray(t["path_xy_ref"][-1], float) - [x0, y0]
            d = float(np.hypot(*(e - p)))
            if d < best:
                best, Lb = d, float(t.get("length_px") or 0.0)
        out.append(Lb)
    return out


if __name__ == "__main__":
    for m in sys.argv[1:] or ["ld", "m2", "m1"]:
        build(m)
