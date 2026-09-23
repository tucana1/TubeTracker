"""Sticky bias tracker (H205): temporally-coherent tip correction.

Per-sample snaps stair-step between tracker-frame and apex-frame (H204).
Instead of smoothing offsets (era steps survive any median), track the
bias explicitly with hysteresis:

  - derive adoption events: at sample i, fixes in [i, i+LOOKAHEAD] with
    >= AGREE_N agreeing within AGREE_PX whose median differs from the
    current bias by > ADOPT_PX adopt a new bias target
  - render: hold the current bias; ramp to a new target over RAMP
    samples once adopted (post-hoc, non-causal — fine for series output)

Flicker never adopts (no agreement); unfixed stretches hold bias.
Fault/veto samples (H206): ``reset_mask`` marks samples where correction
is forbidden (TimesFM fault, consensus veto). Resets break agreement
windows, force zero offset, and reset the bias state — pre-fault bias
must not leak across untrustworthy samples (P22 ghost: +12.75 era bias
held across s242-250 fault/vetoes).
Far candidates (H212): ``cand_xy``/``cand_mask`` carry tier-1 far-peak
snaps (beyond the 45 px anchor cap). Candidates join agreement windows
as ordinary points but never render directly — only an adopted event
(a ≥4-agreeing era, e.g. s220-class) moves the bias. Isolated far
snaps (P92/P101/P64, scattered 50+ px) can never agree and die.
Pure function over series arrays.
"""

from __future__ import annotations

import numpy as np

AGREE_N = 4
AGREE_PX = 6.0
ADOPT_PX = 8.0
LOOKAHEAD = 5
RAMP = 6


def _agree_median(pts: np.ndarray, agree_n: int, agree_px: float):
    if len(pts) < agree_n:
        return None
    med = np.median(pts, axis=0)
    dev = float(np.median(np.hypot(pts[:, 0] - med[0], pts[:, 1] - med[1])))
    return med if dev <= agree_px else None


def sticky_bias(
    accepted_xy: np.ndarray,
    corrected_xy: np.ndarray,
    fixed_mask: np.ndarray,
    *,
    agree_n: int = AGREE_N,
    agree_px: float = AGREE_PX,
    adopt_px: float = ADOPT_PX,
    lookahead: int = LOOKAHEAD,
    ramp: int = RAMP,
    reset_mask: np.ndarray | None = None,
    cand_xy: np.ndarray | None = None,
    cand_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return coherent corrected positions (N, 2)."""
    acc = np.asarray(accepted_xy, dtype=float)
    cor = np.asarray(corrected_xy, dtype=float)
    mask = np.asarray(fixed_mask, dtype=bool)
    n = len(acc)
    off = np.zeros_like(acc)
    off[mask] = cor[mask] - acc[mask]
    reset = (np.zeros(n, dtype=bool) if reset_mask is None
             else np.asarray(reset_mask, dtype=bool))
    off[reset] = 0.0
    use = mask & ~reset
    coff = np.zeros_like(acc)
    cuse = np.zeros(n, dtype=bool)
    if cand_xy is not None and cand_mask is not None:
        coff[np.asarray(cand_mask, dtype=bool)] = (
            np.asarray(cand_xy, dtype=float)[np.asarray(cand_mask, dtype=bool)]
            - acc[np.asarray(cand_mask, dtype=bool)])
        coff[reset] = 0.0
        cuse = np.asarray(cand_mask, dtype=bool) & ~reset

    # derive adoption events (reset samples split agreement windows)
    events: list[tuple[int, np.ndarray]] = []
    current = np.zeros(2)
    for i in range(n):
        if reset[i]:
            current = np.zeros(2)
            continue
        j = min(n, i + lookahead + 1)
        win = (slice(i, j) if not reset[i:j].any()
               else slice(i, i + int(np.argmax(reset[i:j]))))
        pool = off[win][use[win]]
        cpool = coff[win][cuse[win]]
        if len(cpool):
            pool = np.vstack([pool, cpool]) if len(pool) else cpool
        med = _agree_median(pool, agree_n, agree_px)
        if med is not None and float(np.hypot(*(med - current))) > adopt_px:
            events.append((i, np.array(med, dtype=float)))
            current = np.array(med, dtype=float)

    # render hold + ramps (resets force zero)
    out_off = np.zeros_like(acc)
    cur = np.zeros(2)
    ev = 0
    for i in range(n):
        if reset[i]:
            cur = np.zeros(2)
            out_off[i] = 0.0
            while ev < len(events) and events[ev][0] <= i:
                ev += 1  # drop stale pre-reset adoptions
            continue
        while ev < len(events) and events[ev][0] + ramp <= i:
            cur = events[ev][1]
            ev += 1
        if ev < len(events) and events[ev][0] <= i:
            s, tgt = events[ev]
            alpha = (i - s + 1) / ramp
            out_off[i] = (1 - alpha) * cur + alpha * tgt
        else:
            out_off[i] = cur
    return acc + out_off
