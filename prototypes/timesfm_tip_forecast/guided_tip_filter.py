"""Forecast-guided tip filter: per-sample acceptance simulation (offline).

Prototype of the tip stage of forecast-guided tracing: an arriving accepted
tip is HELD (previous guided tip, dead reckoning) when the closed-loop veto
says ``fault-suspect``, otherwise accepted.  ``burst`` and ``keep`` verdicts
pass through untouched, so real growth (even fast bursts) is never frozen.

This replays verdicts already computed by ``guided_replay`` (no new model
inference): it answers "would a forecast-centered acceptance rule have
frozen the phantom arrivals while passing true growth?"  One honest
limitation is stated, not hidden: the offline replay forecasts from the
original (contaminated) history, while a live loop would forecast from the
held-smooth history and do strictly better.  Results here are therefore a
lower bound on the live-loop behavior.

Outputs per owner: guided tip series + hold flags (a downstream consumer
could trace/grow from ``guided_x/y`` instead of the raw accepted tip).

Two-stage contract (H149): the veto *detects* anomalies (verdicts
unchanged), the filter decides *acceptance*. A fault-suspect tip within
RECOVERY_PX of a recent non-fault tip is a flicker-BACK (return to
known-good, e.g. lowdens P22 s229 eye-verified on the real apex) and is
PASSED, not held — the veto cannot split flicker-out from flicker-back
(both huge-resid + tiny-gain), but history can. Only flicker-OUTs
(departures) freeze. R is set interior to the measured 6-10px plateau
(identical pass sets), not fit to an owner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RECOVERY_PX = 8.0  # fault tip within this of a recent keep tip = return
RECOVERY_LOOKBACK = 10  # samples of keep history retained


def apply_tip_filter(
    measurements: pd.DataFrame,
    replay: pd.DataFrame,
    owner: int,
) -> pd.DataFrame:
    """Join accepted tips with replay verdicts; hold on fault-suspect."""

    m = measurements.query("pollen_id == @owner and accepted == 1").sort_values(
        "sample_index"
    )
    rep = replay.set_index("sample_index")
    gx, gy = [], []
    held = []
    last = None
    keep_hist: list[tuple[int, tuple[float, float]]] = []
    for _, row in m.iterrows():
        s = int(row.sample_index)
        ax, ay = float(row.tip_x_px), float(row.tip_y_px)
        v = rep.loc[s].verdict if s in rep.index else "keep"
        if v != "fault-suspect":
            keep_hist.append((s, (ax, ay)))
            keep_hist = [(ss, p) for (ss, p) in keep_hist if s - ss <= 10]
        if v == "fault-suspect" and last is not None:
            # H149 recovery-pass: return to known-good is not a departure.
            recent = [np.array(p) for (ss, p) in keep_hist if s - ss <= 10]
            if len(recent) and min(float(np.linalg.norm(np.array([ax, ay]) - p)) for p in recent) <= RECOVERY_PX:
                gx.append(ax)
                gy.append(ay)
                last = (ax, ay)
                held.append(False)
                keep_hist.append((s, (ax, ay)))
            else:
                gx.append(last[0])
                gy.append(last[1])
                held.append(True)
        else:
            gx.append(ax)
            gy.append(ay)
            last = (ax, ay)
            held.append(False)
    out = pd.DataFrame(
        {
            "sample_index": m.sample_index.to_numpy(),
            "source_frame": m.source_frame.to_numpy(),
            "accepted_x": m.tip_x_px.to_numpy(dtype=float),
            "accepted_y": m.tip_y_px.to_numpy(dtype=float),
            "guided_x": np.array(gx),
            "guided_y": np.array(gy),
            "held": np.array(held),
        }
    )
    out["guided_step"] = np.linalg.norm(
        np.diff(
            np.column_stack([out.guided_x, out.guided_y]),
            axis=0,
            prepend=[[out.guided_x.iloc[0], out.guided_y.iloc[0]]],
        ),
        axis=1,
    )
    return out
