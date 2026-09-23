"""TimesFM-in-the-loop guided replay: the forecaster as tracer prior + veto.

Two prospective simulations, both sourced at the grain (first accepted
sample = emergence start), both causal (no future leakage):

1. closed_loop_replay: walk samples in order; at each step forecast the
   arriving tip from history only. Verdict per sample from the forecast
   residual crossed with the traditional length signal (H136 lesson:
   residual alone cannot separate bursts from faults):
     fault-suspect : residual huge  AND length gain tiny  (P131 flicker)
     burst         : residual huge  AND length gain large  (real event)
     keep          : otherwise
   The residual gate is adaptive: ``max(3 * interval_width, 15px)`` uses
   the model's own uncertainty, not a tuned pixel number.

2. open_loop_rollout: seed on the first ``seed_len`` grain-sourced samples,
   then roll the median forecast forward with no further observations
   (true prospective use: predict the tip trajectory from emergence).
   Divergence step = first step where rollout leaves a 15px tube around
   the accepted track. Clean tubes should ride the rollout; flickering
   or latched tracks should break it immediately.

Prototype constants live in VERDICT: documented, not tuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import TipForecaster

VERDICT = {
    "seed_len": 8,       # grain-sourced seed before forecasting starts
    "width_k": 3.0,      # residual gate = width_k * interval width ...
    "abs_floor_px": 15.0,  # ... or this floor, whichever is larger
    "gain_floor_px": 5.0,  # length gain separating bursts from faults
    "diverge_px": 15.0,  # rollout divergence radius
}


def _context(track: pd.DataFrame, end: int):
    """Context arrays over track[:end]. end = # samples used."""
    seg = track.iloc[:end]
    targets = np.stack(
        [seg.tip_x.to_numpy(), seg.tip_y.to_numpy(), seg.length.to_numpy()]
    ).astype(np.float32)
    # H144 ablation: tip-speed / length-gain activity covariates are
    # uninformative (dense-H12 Δtip -0.02px, Δburst-F1 +0.00) — trust only.
    past = seg.accepted.to_numpy(dtype=np.float32)[None, :].astype(np.float32)
    return targets, past


def closed_loop_replay(
    fc: TipForecaster,
    track: pd.DataFrame,
    seed_len: int = VERDICT["seed_len"],
) -> pd.DataFrame:
    """One-step-ahead forecast of every sample from history; verdicts."""
    n = len(track)
    idx = list(range(seed_len, n))
    tg, pc, ft = [], [], []
    for t in idx:
        a, b = _context(track, t)
        tg.append(a)
        pc.append(b)
        ft.append(
            np.concatenate(
                [track.time_minutes.to_numpy(dtype=np.float32)[:t],
                 track.time_minutes.to_numpy(dtype=np.float32)[t : t + 1]]
            )
        )
    outs = fc.predict_contexts(tg, pc, ft, 1)
    rows = []
    xy = track[["tip_x", "tip_y"]].to_numpy(float)
    L = track.length.to_numpy(float)
    keep_widths: list[float] = []  # widths only from keep verdicts: faults
    for k, t in enumerate(idx):    # cannot inflate their own gate
        f, q = outs[k]
        px, py = float(f[0, 0]), float(f[1, 0])
        width = float((q[0, 0, 8] - q[0, 0, 0] + q[1, 0, 8] - q[1, 0, 0]) / 2)
        resid = float(np.hypot(xy[t, 0] - px, xy[t, 1] - py))
        gain = float(L[t] - L[t - 1])
        if len(keep_widths) >= 4:
            gate_width = float(np.median(keep_widths[-16:]))
        else:
            gate_width = width
        gate = max(VERDICT["width_k"] * gate_width, VERDICT["abs_floor_px"])
        big = resid > gate
        if big and gain < VERDICT["gain_floor_px"]:
            verdict = "fault-suspect"
        elif big:
            verdict = "burst"
        else:
            verdict = "keep"
            keep_widths.append(width)
        rows.append(
            dict(
                sample_index=int(track.sample_index.iloc[t]),
                pred_x=px, pred_y=py,
                resid=resid, width=width, gate=gate,
                length_gain=gain, verdict=verdict,
            )
        )
    return pd.DataFrame(rows)


def open_loop_rollout(
    fc: TipForecaster,
    track: pd.DataFrame,
    seed_len: int = VERDICT["seed_len"],
    horizon: int = 24,
) -> pd.DataFrame:
    """Roll the median forecast forward from the grain-sourced seed."""
    times = track.time_minutes.to_numpy(dtype=np.float32)
    seed = track.iloc[:seed_len]
    hist_x = list(seed.tip_x.to_numpy(dtype=float))
    hist_y = list(seed.tip_y.to_numpy(dtype=float))
    hist_l = list(seed.length.to_numpy(dtype=float))
    rows = []
    for h in range(1, horizon + 1):
        t = seed_len + h - 1
        if t >= len(track):
            break
        T = len(hist_x)
        targets = np.stack(
            [np.array(hist_x, np.float32), np.array(hist_y, np.float32),
             np.array(hist_l, np.float32)]
        )
        # H144 spec: trust-only past (speed ablated, no signal). open_loop
        # kept the old 2-channel past until H151; unified here.
        past = np.ones((1, T), np.float32)
        fut = np.concatenate([times[:T], times[t : t + 1]])
        f, q = fc.predict_one(targets, past, fut, 1)
        px, py, pl = float(f[0, 0]), float(f[1, 0]), float(f[2, 0])
        ax, ay = float(track.tip_x.iloc[t]), float(track.tip_y.iloc[t])
        err = float(np.hypot(ax - px, ay - py))
        rows.append(
            dict(
                sample_index=int(track.sample_index.iloc[t]),
                rollout_x=px, rollout_y=py, rollout_l=pl,
                actual_x=ax, actual_y=ay,
                rollout_err=err,
                diverged=err > VERDICT["diverge_px"],
            )
        )
        hist_x.append(px)
        hist_y.append(py)
        hist_l.append(pl)
    return pd.DataFrame(rows)
