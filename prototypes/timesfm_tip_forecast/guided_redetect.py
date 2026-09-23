"""Forecast-gated tip re-detection (H156): CNN proposes, TimesFM disposes.

Per-sample verdicts combining both instruments:
  - CNN tip peaks in the frame are apex PROPOSALS (high recall, no owner
    assignment).
  - The TimesFM 1-step forecast + veto verdict assigns meaning per owner:
      * accepted tip CONFIRMED (CNN peak within CONFIRM_PX at >= MIN_CONF):
        pass (growth or benign jitter);
      * unconfirmed + veto fault-suspect: HOLD last guided (departure);
      * unconfirmed + veto keep/burst: UNSUPPORTED tip — the veto-blind
        class (smooth phantoms like P55: no jump, no apex under the tip).
        Flagged for review, tip held (fail-closed).
      * fault-suspect + recovery-proximity: RECOVER pass (H149 rule kept).
Rationale: a detector cannot assign (P131-foreign fires correctly as an
apex); a forecaster cannot see (smooth background projections). The gate
between them is the product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONFIRM_PX = 20.0  # H156: CNN peak within this of accepted tip = confirmed
# (10px rejected P22-clean at 16.5px; 17-23px plateau keeps P22-ghost
# (23.1) and P55 (33.7) unconfirmed; N=8 — revisit at final model).
MIN_CONF = 0.30  # minimum peak confidence (epoch-10 maxes ~0.7)
RECOVERY_PX = 8.0  # H149 flicker-back proximity (unchanged)
LOOKBACK = 10


def redetect_sample(
    accepted: tuple[float, float],
    verdict: str,
    peaks: list[tuple[float, float, float]],
    keep_hist: list[tuple[float, float]],
    last_guided: tuple[float, float] | None,
    confirm_px: float = CONFIRM_PX,
    min_conf: float = MIN_CONF,
    recovery_px: float = RECOVERY_PX,
) -> tuple[tuple[float, float], bool, str]:
    """Return (guided_xy, held, tag) for one sample. Pure function."""
    ax, ay = accepted
    strong = [(x, y) for (x, y, c) in peaks if c >= min_conf]
    confirmed = any(np.hypot(x - ax, y - ay) <= confirm_px for (x, y) in strong)
    if verdict != "fault-suspect":
        if confirmed or verdict in ("keep",):
            tag = "OK" if confirmed else "UNSUPPORTED"
            if verdict == "keep" and not confirmed:
                if last_guided is None:
                    return (ax, ay), False, "UNSUPPORTED"
                return last_guided, True, "UNSUPPORTED"
            return (ax, ay), False, ("OK" if confirmed else verdict.upper())
        return (ax, ay), False, verdict.upper()
    # fault-suspect: recovery proximity first (H149), else hold.
    if keep_hist and min(
        np.hypot(ax - px, ay - py) for (px, py) in keep_hist
    ) <= recovery_px:
        return (ax, ay), False, "RECOVER"
    if last_guided is None:
        return (ax, ay), False, "VETO"
    return last_guided, True, "HOLD"
