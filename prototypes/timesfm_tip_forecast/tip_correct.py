"""Forward-curve tip correction: snap the tracker tip to the true apex (H180).

Problem: the v29 tracker tip can ride ~10-25 px off the true apex
(P58 s220: accepted (581, 454), true apex under the CNN 0.61 peak at
(554, 425)) while the refined centerline ends exactly on the biased
tip (|acc - cl_end| = 0.0) - refinement fits *to* the tip instead of
correcting it.  The H156 gate withholds confirmation there
(UNSUPPORTED) but corrects nothing.

Method: fit a quadratic to the distal 30 px of the accepted path,
extrapolate the curve 45 px past the tip, and snap to the nearest CNN
peak within CONFIRM_PX of the extrapolated curve - but only where the
curve point itself carries tube wall energy above max(0.4x distal
median, ABS_FLOOR).  A ghost tip (P22 s248) projects neighbors behind
the tip or onto background: refusal is structural.  Supersedes the
straight-line wall-walk (a biased endpoint kinks the heading 53 deg
off) and the forward-cone + segment-support design (straight chords
cut across tube bends, so the support margin was fragile).
"""

from __future__ import annotations

import numpy as np

try:  # script-adjacent use (render_guided_overlay, probes)
    from switch_cut import transverse_wall_energy
except ImportError:  # package use (pytest, run.py future wiring)
    from prototypes.timesfm_tip_forecast.switch_cut import (
        transverse_wall_energy,
    )

CURVE_ARCLEN = 30.0
MIN_FIT_ARCLEN = 25.0
EXTRAPOLATE = 45.0
EXTRAP_STEP = 2.0
SUPPORT_FLOOR = 0.4
ABS_FLOOR = 2.5
CONFIRM_PX = 20.0


def _wall_at(gray: np.ndarray, tail: np.ndarray, pos: np.ndarray) -> float:
    path = np.vstack([tail[-2:], pos[None, :]])
    return float(transverse_wall_energy(gray, path)[-1])


def refine_subpix(
    gray: np.ndarray, peak: tuple[float, float], window: int = 5
) -> tuple[float, float]:
    """Sub-pixel apex lock via image corner refinement (H197).

    The CNN peak grid is coarse; the apex is a corner-like structure,
    so cv2.cornerSubPix converges onto it.  Falls back to the input
    peak when it would move >3 px (weak structure, not a corner).
    """
    import cv2

    g = np.asarray(gray, dtype=np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.05)
    p = np.array([[[float(peak[0]), float(peak[1])]]], dtype=np.float32)
    try:
        out = cv2.cornerSubPix(g, p, (window, window), (-1, -1), criteria)
    except cv2.error:
        return (float(peak[0]), float(peak[1]))
    q = (float(out[0, 0, 0]), float(out[0, 0, 1]))
    if abs(q[0] - peak[0]) + abs(q[1] - peak[1]) > 3.0:
        return (float(peak[0]), float(peak[1]))
    return q


def consensus_smooth(
    corrected_xy: np.ndarray,
    fixed_mask: np.ndarray,
    window: int = 5,
    max_deviation: float = 12.0,
) -> np.ndarray:
    """Veto flickering corrections by temporal consensus (H190).

    A snap that jumps > ``max_deviation`` px from the rolling median of
    neighboring FIXED corrections (e.g. apex, corner, apex across three
    samples — s210 caught the elbow while s208/s211 held the apex) is
    reset to un-fixed.  Pure function over series arrays.
    """
    fixed = np.asarray(corrected_xy, dtype=float)
    mask = np.asarray(fixed_mask, dtype=bool)
    keep = mask.copy()
    idx = np.nonzero(mask)[0]
    pts = fixed[mask]
    for k, i in enumerate(idx):
        lo, hi = max(0, k - window), min(len(idx), k + window + 1)
        nbrs = np.concatenate([pts[lo:k], pts[k + 1:hi]])
        if len(nbrs) == 0:
            continue
        med = np.median(nbrs, axis=0)
        if float(np.hypot(*(pts[k] - med))) > max_deviation:
            keep[i] = False
    return keep


def correct_tip(
    gray: np.ndarray,
    path_tail: np.ndarray,
    peaks: list[tuple[float, float, float]],
    *,
    confirm_px: float = CONFIRM_PX,
    at_fault: bool = False,
    return_candidate: bool = False,
) -> tuple[float, float] | tuple[float, float, int] | None:
    """Return the corrected tip, or None on refusal.

    ``path_tail``: last accepted centerline points in source px, tip
    last (>=3 points).  ``peaks``: CNN (x, y, conf) apex proposals.
    ``at_fault``: TimesFM closed-loop verdict is fault-suspect for this
    sample — refuse entirely (H198: peak snaps would validate phantoms,
    e.g. P131 s206 snapping onto the foreign apex it fires on; the
    H156 fault branch already holds these).
    ``return_candidate`` (H212): also consider the FULL-curve anchor
    (beyond the 45 px H211 cap, out to 110 px) and return
    ``(x, y, tier)`` with tier 0 = standard snap, 1 = far candidate.
    Rationale: quad blowup sometimes carries the confirm anchor to a
    true far apex (s220, 39.6 px verified) and sometimes to foreign
    peaks (P92/P101/P64, 80-103 px) — per-sample geometry cannot tell
    them apart, but sticky agreement across samples can (era adopts,
    isolated refuses). Default False keeps the plain ``(x, y)`` return.
    """
    if at_fault:
        return None
    g = np.asarray(gray)
    tail = np.asarray(path_tail, dtype=float)
    if len(tail) < 4:
        return None
    ref = float(np.median(transverse_wall_energy(g, tail)))
    if ref <= 0:
        return None
    _tip = tail[-1].copy()

    def _pack(x: float, y: float, tier: int):
        qx, qy = refine_subpix(g, (x, y))
        if return_candidate:
            return (qx, qy, tier)
        return (qx, qy)
    floor = max(SUPPORT_FLOOR * ref, ABS_FLOOR)
    # Distal arc fit: last CURVE_ARCLEN px, quadratic in arclength.
    dxy = np.diff(tail, axis=0)
    seg = np.hypot(dxy[:, 0], dxy[:, 1])
    back = np.concatenate(([0.0], np.cumsum(seg)))
    t0 = back[-1] - CURVE_ARCLEN
    use = back >= t0
    if use.sum() < 4:
        use = np.ones(len(tail), dtype=bool)
    # Extrapolation needs a real baseline: a near-stationary tail (tip
    # jitter over a few px) fits wiggles that extrapolate anywhere.
    # Below MIN_FIT_ARCLEN, drop to confirm-against-tip mode.
    fit_arclen = float(back[use][-1] - back[use][0])
    if fit_arclen < MIN_FIT_ARCLEN:
        anchor = tail[-1][None, :]
        best = None
        for (x, y, _c) in peaks:
            d = float(np.hypot(x - anchor[0, 0], y - anchor[0, 1]))
            if d <= confirm_px and (best is None or d < best[2]):
                best = (float(x), float(y), d)
        if best is None:
            return None
        return _pack(best[0], best[1], 0)
    tb, xb, yb = back[use], tail[use, 0], tail[use, 1]
    with np.errstate(all="ignore"):
        px = np.polyfit(tb, xb, 2)
        py = np.polyfit(tb, yb, 2)
    if not (np.all(np.isfinite(px)) and np.all(np.isfinite(py))):
        return None
    te = np.arange(back[-1] + EXTRAP_STEP, back[-1] + EXTRAPOLATE + 1e-9,
                   EXTRAP_STEP)
    curve = np.column_stack([np.polyval(px, te), np.polyval(py, te)])
    # Curvature guard: noisy tails whip quadratics (lowdens tip-history
    # snapped a clean tip 100 px off). If the end tangent bends >45 deg
    # from the tip tangent, drop to confirm-against-tip mode.
    dpx = np.polyder(px)
    dpy = np.polyder(py)
    t_start = np.array([np.polyval(dpx, back[-1]), np.polyval(dpy, back[-1])])
    t_end = np.array([np.polyval(dpx, te[-1]), np.polyval(dpy, te[-1])])
    ns, ne = float(np.hypot(*t_start)), float(np.hypot(*t_end))
    truncated = ns < 1e-9 or ne < 1e-9 or float(
        t_start @ t_end / (ns * ne)
    ) < np.cos(np.radians(45.0))
    if truncated:
        curve = curve[:0]
    inframe = (
        (curve[:, 0] >= 0) & (curve[:, 1] >= 0)
        & (curve[:, 0] < g.shape[1]) & (curve[:, 1] < g.shape[0])
    )
    curve = curve[inframe]
    if len(curve) == 0 and not truncated:
        return None
    # Wall support along the extrapolated curve; cut at first dead step
    # (past the apex or into background).
    path = np.vstack([tail[-2:], curve])
    energy = transverse_wall_energy(g, path)[2:]
    live = energy >= floor
    if not live.any():
        # Clean tips end at the apex: confirm against the tip itself.
        curve = curve[:0]
    else:
        first_dead = int(np.argmax(~live)) if not live.all() else len(live)
        curve = curve[: max(first_dead, 1)]
    anchor = np.vstack([tail[-1][None, :], curve])
    # H211: cap the anchor at 45 px from the tip. The confirm radius used
    # to apply along the whole extrapolation, so a foreign peak near a
    # far curve point validated 80-103 px jumps (P92/P101/P64 eye-audit:
    # two suspect-foreign, one uncertain). 45 px = max eye-validated fix
    # (s220, 39.6 px) + margin: flicker-class jumps (40-70 px) are owned
    # by the TimesFM fault gate (refuse before geometry, H198) and
    # isolated jumps by consensus/sticky — the anchor cap only needs to
    # stop foreign-hijack distance on keep-verdict samples.
    reach = np.zeros(0)
    if len(curve):
        reach = np.hypot(curve[:, 0] - tail[-1, 0], curve[:, 1] - tail[-1, 1])
        anchor = np.vstack([tail[-1][None, :], curve[reach <= 45.0]])
    best = None
    for (x, y, _c) in peaks:
        d = float(np.min(np.hypot(anchor[:, 0] - x, anchor[:, 1] - y)))
        if d <= confirm_px and (best is None or d < best[2]):
            best = (float(x), float(y), d)
    if best is not None:
        return _pack(best[0], best[1], 0)
    # H212 candidate tier: full-curve anchor (out to 110 px) for
    # sustained far-apex eras (s220-class). Per-sample geometry cannot
    # separate these from foreign hijacks — the tier flag routes them to
    # sticky agreement (era adopts, isolated refuses) instead of
    # immediate trust. Never fires on fault samples (refused above).
    if return_candidate and len(curve):
        full = np.vstack([tail[-1][None, :], curve[reach <= 110.0]])
        cand = None
        for (x, y, _c) in peaks:
            d = float(np.min(np.hypot(full[:, 0] - x, full[:, 1] - y)))
            mv = float(np.hypot(x - _tip[0], y - _tip[1]))
            # Floor 25 px (H212 fix): the fallback cap already owns ≤25;
            # anything farther that the standard path refused (e.g. s220
            # at 39.6 px with a short live curve) routes to agreement
            # instead of trust.
            if d <= confirm_px and mv > 25.0 and (cand is None or d < cand[2]):
                cand = (float(x), float(y), d)
        if cand is not None:
            return _pack(cand[0], cand[1], 1)
    # CNN-silent fallback (H195): the live curve endpoint itself locates
    # the apex (s209/s210: no peak, endpoint on apex).  Capped at 25 px
    # from the tip so flicker jumps (P131-class, 40-70 px) can never
    # validate a phantom; needs >=3 live steps (real tube, not noise).
    if len(curve) >= 3:
        end = curve[-1]
        mv = float(np.hypot(*(end - tail[-1])))
        if 5.0 <= mv <= 25.0:
            return _pack(float(end[0]), float(end[1]), 0)
    return None
