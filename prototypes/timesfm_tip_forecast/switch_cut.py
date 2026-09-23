"""Trace-time switch repair (offline prototype): cut chimeric centerlines.

Diagnosis (H145/H146): lowdens P22's tail centerlines ride the upright tube,
then cut across background onto a neighboring tube (mid-course branch
switch; root sometimes detached). No tip filter can fix this — the whole
path is wrong and its length is fiction.

Two independent instruments, one repair:
  WHEN (TimesFM tip history only): 1-step closed-loop residual explosion
      marks the suspect onset sample (existing `guided_replay` rows).
  WHERE (pixels only): transverse wall-support profile along the path;
      a branch switch shows as a deep RELATIVE dip mid-path (absolute
      energy can't split faint-real from background — calibrated H146).
Repair: truncate the centerline at the dip arc; tip = last supported
point. Accepted only when both instruments agree (residual onset within
a few samples of a support dip); otherwise the path is left untouched.

H150 LIMITATION (read before using): magnitude-profile dips do NOT
isolate switches. Real-tube support valleys dip just as deep (P22
s202-222 cut truncations verified WORSE than accepted — red was on-apex,
green floated background). WEAK_SPAN_PX + POSTDIP_FRAC gates remove
narrow valleys and dip-with-strong-recovery but ALSO remove verified
switch cuts (s229/240/243) while keeping some valley cuts (s202/210/220):
the feature oscillates, it does not converge. Valley-vs-switch needs
geometry (turn at the dip on a smoothed polyline is washed out) or an
apex model — not more magnitude gates. Triage grade only.

Status: offline prototype. Threshold (0.30) calibrated on 6 clean + 8
fault lowdens paths — provisional, needs wider calibration before any
pipeline wiring. NOT wired into run.py (no N=1 tuning into product).
"""

from __future__ import annotations

import numpy as np

DIP_RATIO = 0.30      # min rolling-mean / baseline below this = switch dip
ROLL_WIN = 4          # rolling window (points) for sustained support
BASELINE_FRAC = 0.25  # baseline = median support over first quarter of path
APEX_EXCLUDE = 0.15   # ignore last 15% of arc (apex is naturally weak)
WEAK_SPAN_PX = 10.0   # weak span must cover this much arc (H150: narrow
                      # valleys on real tubes must not cut)
POSTDIP_FRAC = 0.60   # median support past the dip must stay below this
                      # fraction of baseline (H150: dip-then-strong-recovery
                      # = valley on a real tube, not a switch)
ROOT_ARC_PX = 10.0    # root zone for seed-support scoring (H148)
NROOT_RATIO = 0.10    # normalized root support below this = detached root
                      # (provisional: 10/11 eye precision, 0.099-vs-0.105
                      # margin — triage only, NOT a pipeline gate)


def cut_at_support_dip(
    support: np.ndarray,
    arc: np.ndarray,
    dip_ratio: float = DIP_RATIO,
) -> int | None:
    """Return the cut point_index, or None when the path is supported.

    Pure function of a per-point support array (no I/O) so the rule is
    unit-testable. Cut = argmin of the rolling-mean support over the
    mid-path window, accepted only when min/baseline < dip_ratio.
    """
    w = np.asarray(support, dtype=float)
    n = len(w)
    if n < 8 or not np.all(np.isfinite(w)):
        return None
    q = max(3, int(n * BASELINE_FRAC))
    base = float(np.median(w[:q]))
    if base <= 1e-9:
        return None  # whole path unsupported (e.g. detached root) — no cut to make
    hi = max(q + 1, int(n * (1.0 - APEX_EXCLUDE)))
    seg = w[:hi]
    if len(seg) < ROLL_WIN:
        return None
    # Sustained-support rolling mean (numpy; avoids pandas-stub friction).
    kern = np.ones(ROLL_WIN) / ROLL_WIN
    zs = np.full(len(seg), np.nan)
    zs[ROLL_WIN - 1 :] = np.correlate(seg, kern, mode="valid")
    i = int(np.nanargmin(zs))
    if zs[i] / base >= dip_ratio:
        return None
    # H150 gates: a real tube's support valley also dips deep. A switch
    # needs a WIDE weak span plus NO strong recovery past the weak run
    # (dip-then-background, or dip-then-weaker-foreign-walls).
    weak = seg < 0.5 * base
    if not bool(weak[i]):
        return None
    lo_j = i
    while lo_j > 0 and bool(weak[lo_j - 1]):
        lo_j -= 1
    hi_j = i
    while hi_j + 1 < len(seg) and bool(weak[hi_j + 1]):
        hi_j += 1
    if float(arc[hi_j] - arc[lo_j]) < WEAK_SPAN_PX:
        return None
    post = w[hi_j + 1 : hi] if hi_j + 1 < hi else np.array([])
    if len(post) and float(np.median(post)) >= POSTDIP_FRAC * base:
        return None
    return i


def normalized_root_support(
    support: np.ndarray,
    arc: np.ndarray,
    frame_grad_p95: float,
    root_arc_px: float = ROOT_ARC_PX,
) -> float:
    """Median root-zone wall energy / frame gradient p95 (H148).

    Frame normalization is what makes the score cross-movie comparable:
    raw root support on clean lowdens tubes (3-5) sits inside dense faults'
    band (1.5-4.5), while normalized separates faults (<=0.036) from clean
    (>=0.199) on the 11-path calibration set. Pure function (testable).
    """
    w = np.asarray(support, dtype=float)
    zone = w[np.asarray(arc, dtype=float) < root_arc_px]
    if len(zone) == 0 or frame_grad_p95 <= 0:
        return float("nan")
    return float(np.median(zone) / frame_grad_p95)


def transverse_wall_energy(
    gray: np.ndarray,
    pts: np.ndarray,
    half: float = 14.0,
    n_off: int = 29,
) -> np.ndarray:
    """Max transverse gradient along perpendicular cuts (cv2-free core)."""
    import cv2

    g = np.asarray(gray, dtype=float)
    if len(pts) < 3:
        return np.zeros(len(pts))  # degenerate path: no tangent, no support
    H, W = g.shape
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    norm = np.column_stack([-tang[:, 1], tang[:, 0]])
    offs = np.linspace(-half, half, n_off)
    out = np.empty(len(pts))
    for j, (p, nv) in enumerate(zip(pts, norm)):
        xs = np.clip(p[0] + nv[0] * offs, 0, W - 1).astype(np.float32)[None, :]
        ys = np.clip(p[1] + nv[1] * offs, 0, H - 1).astype(np.float32)[None, :]
        pr = cv2.remap(g, xs, ys, cv2.INTER_LINEAR)[0]
        out[j] = float(np.abs(np.diff(pr)).max())
    return out


def grain_masked_wall_energy(
    gray: np.ndarray,
    pts: np.ndarray,
    grain_center_xy: np.ndarray,
    grain_radius_px: float,
    half: float = 14.0,
    n_off: int = 29,
) -> np.ndarray:
    """Transverse wall energy ignoring samples inside the grain body.

    H160/H162: transverse cuts at the grain boundary catch the grain's
    own dark edge as tube "walls", inflating root support on detached
    roots (P97 s240 reads clean-band 0.18-0.24 with no protrusion in the
    traced direction). Grain center tracks come from the bootstrap
    owner-motion npz; radius defaults to center->root distance at call
    time. Offsets falling inside the circle are dropped before the max.
    """
    import cv2

    g = np.asarray(gray, dtype=float)
    if len(pts) < 3:
        return np.zeros(len(pts))
    H, W = g.shape
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    norm = np.column_stack([-tang[:, 1], tang[:, 0]])
    offs = np.linspace(-half, half, n_off)
    gc = np.asarray(grain_center_xy, dtype=float)
    out = np.empty(len(pts))
    for j, (p, nv) in enumerate(zip(pts, norm)):
        xs = np.clip(p[0] + nv[0] * offs, 0, W - 1)
        ys = np.clip(p[1] + nv[1] * offs, 0, H - 1)
        keep = np.hypot(xs - gc[0], ys - gc[1]) > grain_radius_px
        if keep.sum() < 3:
            out[j] = 0.0
            continue
        pr = cv2.remap(g, xs.astype(np.float32)[None, :],
                       ys.astype(np.float32)[None, :], cv2.INTER_LINEAR)[0]
        d = np.abs(np.diff(pr))
        keep_d = keep[:-1] | keep[1:]
        out[j] = float(d[keep_d].max()) if keep_d.any() else 0.0
    return out


def retreat_to_support(
    gray: np.ndarray,
    start_xy: np.ndarray,
    back_xy: np.ndarray,
    ref_support: float,
    recover_frac: float = 0.7,
    step_px: float = 2.0,
    max_steps: int = 15,
) -> tuple[np.ndarray, bool]:
    """Step backward (toward root side) until wall support recovers.

    The cut index sits at the dip floor (broad weak zone); launching an
    apex extension from the floor fails. Retreat along back_xy with the
    same lateral-snap walker until transverse support >= recover_frac *
    ref_support. Returns (position, recovered?).
    """
    import cv2

    g = np.asarray(gray, dtype=float)
    H, W = g.shape
    pos = np.asarray(start_xy, dtype=float).copy()
    d = np.asarray(back_xy, dtype=float)
    d /= np.linalg.norm(d) + 1e-9
    for _ in range(max_steps):
        n = np.array([-d[1], d[0]])
        offs = np.linspace(-10.0, 10.0, 21)
        xs = np.clip(pos[0] + n[0] * offs, 0, W - 1).astype(np.float32)[None, :]
        ys = np.clip(pos[1] + n[1] * offs, 0, H - 1).astype(np.float32)[None, :]
        pr = cv2.remap(g, xs, ys, cv2.INTER_LINEAR)[0]
        if float(np.abs(np.diff(pr)).max()) >= recover_frac * ref_support:
            return pos, True
        pos = pos + d * step_px
    return pos, False


def extend_tip_along_ridge(
    gray: np.ndarray,
    start_xy: np.ndarray,
    direction_xy: np.ndarray,
    ref_support: float,
    step_px: float = 2.0,
    search_half: float = 10.0,
    stop_frac: float = 0.5,
    stop_patience: int = 3,
    max_steps: int = 30,
) -> tuple[np.ndarray, list[float]]:
    """Walk the wall-energy ridge from a truncated cut point to the apex.

    H146's truncation under-extends (mid-shaft, not apex). From the cut
    point, step along the path direction; at each step snap laterally to
    the transverse max-energy position; stop when support stays below
    stop_frac * ref_support for stop_patience steps (tube ended) or the
    step budget exhausts. Returns (endpoint, support_trace). Pure
    image-following — no model input; the caller gates the endpoint
    against the TimesFM forecast (forecast-gated re-detection).
    """
    import cv2

    g = np.asarray(gray, dtype=float)
    H, W = g.shape
    pos = np.asarray(start_xy, dtype=float).copy()
    d = np.asarray(direction_xy, dtype=float)
    d /= np.linalg.norm(d) + 1e-9
    trace: list[float] = []
    weak = 0
    for _ in range(max_steps):
        n = np.array([-d[1], d[0]])
        offs = np.linspace(-search_half, search_half, 21)
        xs = np.clip(pos[0] + n[0] * offs, 0, W - 1).astype(np.float32)[None, :]
        ys = np.clip(pos[1] + n[1] * offs, 0, H - 1).astype(np.float32)[None, :]
        pr = cv2.remap(g, xs, ys, cv2.INTER_LINEAR)[0]
        e = np.abs(np.diff(pr))
        sup = float(e.max())
        trace.append(sup)
        # Center between the two strongest walls (polarity-free); argmax
        # alone rides edges off the tube (H149 synthetic failure).
        peak = int(e.argmax())
        # Second wall may sit anywhere across the cut (tube width unknown):
        # search the full array, not a narrow window (H149: ±6 missed a
        # 10px-distant wall and rode the near edge off-tube).
        order = np.argsort(e)[::-1]
        mid_off = float(offs[peak])  # lateral offset of strongest wall
        for q in order[1:]:
            qi = int(q)
            if e[qi] > 0.3 * e[peak] and abs(qi - peak) >= 3:
                mid_off = (float(offs[peak]) + float(offs[qi])) / 2.0
                break
        snap_vec = n * (0.5 * mid_off)  # damped lateral correction
        if sup < stop_frac * ref_support:
            snap_vec = np.zeros(2)  # void: coast straight, never chase noise
        step_dir = d + snap_vec / step_px
        step_dir /= np.linalg.norm(step_dir) + 1e-9
        pos = pos + step_dir * step_px
        d = 0.8 * d + 0.2 * step_dir
        d /= np.linalg.norm(d) + 1e-9
        if sup < stop_frac * ref_support:
            weak += 1
            if weak >= stop_patience:
                break
        else:
            weak = 0
    return pos, trace
