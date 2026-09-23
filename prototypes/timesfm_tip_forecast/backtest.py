"""Rolling-origin backtest: TimesFM-3 vs persistence / velocity baselines.

Protocol (hindcast only — no future leakage):
  origin t: context = track[0 .. t]  (everything the pipeline knew at t)
  horizon H: actuals = track[t+1 .. t+H]
  TimesFM-3 sees context targets + past covariates + time schedule into H.
  Baselines see only the last context step.
Metrics per (owner, H): tip RMSE/MAE (px), length MAE, q10-q90 empirical
coverage + mean width, burst-event precision/recall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import Q10, Q50, Q90, TipForecaster

BURST_GAIN_PX = 5.0


def make_origins(n: int, horizon: int, min_context: int = 48, max_origins: int = 12) -> list[int]:
    lo = min(min_context, max(min_context // 2, n - horizon - 4))
    hi = n - horizon - 1
    if hi <= lo:
        return []
    stride = max(1, (hi - lo) // max_origins)
    return list(range(lo, hi + 1, stride))


def baseline_persist(ctx: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed (x, y, length). ctx: (3, T)."""
    return np.repeat(ctx[:, -1:], horizon, axis=1)


def baseline_velocity(ctx: np.ndarray, horizon: int) -> np.ndarray:
    """Extrapolate the last step. ctx: (3, T)."""
    last = ctx[:, -1]
    step = ctx[:, -1] - ctx[:, -2] if ctx.shape[1] >= 2 else np.zeros(3)
    return last[:, None] + step[:, None] * np.arange(1, horizon + 1)[None, :]


def backtest_owner(
    forecaster: TipForecaster,
    track: pd.DataFrame,
    horizon: int,
    min_context: int = 48,
    variant: str = "full",
    origins: list[int] | None = None,
    vis: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per origin with TimesFM-3 + baseline errors.

    Variants (single-factor input ablations; actuals, baselines, origins
    handling unchanged unless noted):
      full:             joint targets + trust past + schedule-dynamic
      uniform_schedule: future times = uniform cadence from last context time
      window24:         context sliced to last 24 samples (stale-history test)
      marginal:         three univariate forecasts, no cross-series coupling
    ``vis`` (H202): optional per-row apex-visibility past covariate
    (tip-CNN window-max heat near the accepted tip = tip-freshness);
    appended under trust as a second past-only channel when given.
    Retired variants (evidence kept in LEDGER H144 + ablation CSVs on disk):
      no_speed, no_covs, gain_cov — activity covariates carry no signal.
    `origins` overrides the internally computed list (lets a
    diagnostic-context run evaluate at the same sample_indices as accepted-only).
    """
    arr = track[["tip_x", "tip_y", "length"]].to_numpy(dtype=float).T  # (3, N)
    # H144: trust-only past covariate (tip-speed / length-gain ablated: no
    # signal on dense-H12 points or burst-F1; evidence in LEDGER + ablation CSVs).
    acc = track.accepted.to_numpy(dtype=float)
    times = track.time_minutes.to_numpy(dtype=float)
    n = arr.shape[1]
    vv = None
    if vis is not None:
        vv = np.asarray(vis, dtype=float)
        assert vv.shape[0] == n, "visibility must align to track rows"
    if origins is None:
        origins = make_origins(n, horizon, min_context)
    if not origins:
        return pd.DataFrame()

    targets, past_covs, futures = [], [], []
    for t in origins:
        T = t + 1
        if variant == "window24":
            lo = max(0, T - 24)
            sl = slice(lo, T)
            ctx_arr, ctx_acc = arr[:, sl], acc[sl]
            ctx_times = times[sl]
            ctx_vv = None if vv is None else vv[sl]
        else:
            ctx_arr, ctx_acc = arr[:, :T], acc[:T]
            ctx_times = times[:T]
            ctx_vv = None if vv is None else vv[:T]
        Tc = ctx_arr.shape[1]
        targets.append(ctx_arr.astype(np.float32))
        if ctx_vv is None:
            # H144: trust-only past (speed/gain activity covs uninformative).
            past_covs.append(ctx_acc[None, :].astype(np.float32))
        else:
            # H202: trust + apex-visibility past (tip-freshness channel).
            past_covs.append(
                np.stack([ctx_acc, ctx_vv], axis=0).astype(np.float32))
        # Future schedule: actual past times + uniform-cadence extension.
        dt = float(np.median(np.diff(ctx_times))) if Tc > 2 else 0.25
        if variant == "uniform_schedule":
            # Uniform cadence anchored at last context time (no true schedule).
            base = np.arange(Tc, dtype=float) * dt
            futures.append(
                np.concatenate([base, base[-1] + dt * np.arange(1, horizon + 1)])
            )
        else:
            futures.append(
                np.concatenate(
                    [ctx_times, ctx_times[-1] + dt * np.arange(1, horizon + 1)]
                )
            )
    if variant == "marginal":
        # Three univariate batches in one evaluator call; reassemble joint.
        mt, mp, mf = [], [], []
        for ta, pc, ft in zip(targets, past_covs, futures):
            for s in range(3):
                mt.append(ta[s : s + 1, :])
                mp.append(pc)
                mf.append(ft)
        preds = forecaster.predict_contexts(mt, mp, mf, horizon)
        joint = []
        for i in range(0, len(preds), 3):
            fc = np.stack([preds[i + s][0][0] for s in range(3)], axis=0)
            qu = np.stack([preds[i + s][1][0] for s in range(3)], axis=0)
            joint.append((fc, qu))
        preds = joint
    else:
        preds = forecaster.predict_contexts(targets, past_covs, futures, horizon)

    rows = []
    for t, (fc, qu) in zip(origins, preds):
        actual = arr[:, t + 1 : t + 1 + horizon]  # (3, H)
        tip_err = np.linalg.norm(fc[:2, :] - actual[:2, :], axis=0)
        for k in range(horizon):
            # Probabilistic burst score: fraction of length quantiles that
            # exceed the event threshold (quantiles don't cumsum, so score
            # the horizon-gain distribution directly).
            burst_p = float(
                np.mean(qu[2, k, :] - arr[2, t] > BURST_GAIN_PX)
            )
            rows.append(
                {
                    "origin": t,
                    "lead": k + 1,
                    "tip_err": tip_err[k],
                    "len_err": abs(float(fc[2, k] - actual[2, k])),
                    "x_cov": float(qu[0, k, Q10] <= actual[0, k] <= qu[0, k, Q90]),
                    "y_cov": float(qu[1, k, Q10] <= actual[1, k] <= qu[1, k, Q90]),
                    "l_cov": float(qu[2, k, Q10] <= actual[2, k] <= qu[2, k, Q90]),
                    "x_w": float(qu[0, k, Q90] - qu[0, k, Q10]),
                    "y_w": float(qu[1, k, Q90] - qu[1, k, Q10]),
                    "l_w": float(qu[2, k, Q90] - qu[2, k, Q10]),
                    "fc_gain": float(fc[2, k] - arr[2, t]),
                    "burst_p": burst_p,
                    "actual_gain": float(actual[2, k] - arr[2, t]),
                    "persist_tip_err": float(
                        np.linalg.norm(arr[:2, t] - actual[:2, k])
                    ),
                    "persist_len_err": abs(float(arr[2, t] - actual[2, k])),
                    "vel_tip_err": float(
                        np.linalg.norm(baseline_velocity(arr[:, : t + 1], horizon)[:2, k] - actual[:2, k])
                    ),
                    "vel_len_err": abs(
                        float(baseline_velocity(arr[:, : t + 1], horizon)[2, k] - actual[2, k])
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate a backtest frame into headline metrics."""
    if df.empty:
        return {}
    burst_actual = df.actual_gain > BURST_GAIN_PX
    burst_pred = df.fc_gain > BURST_GAIN_PX
    tp = int((burst_actual & burst_pred).sum())
    brier = float(((df.burst_p - burst_actual.astype(float)) ** 2).mean())
    q_pred = df.burst_p >= 0.5
    qtp = int((burst_actual & q_pred).sum())
    return {
        "n": len(df),
        "tip_rmse": float(np.sqrt((df.tip_err**2).mean())),
        "tip_mae": float(df.tip_err.mean()),
        "persist_tip_rmse": float(np.sqrt((df.persist_tip_err**2).mean())),
        "vel_tip_rmse": float(np.sqrt((df.vel_tip_err**2).mean())),
        "len_mae": float(df.len_err.mean()),
        "persist_len_mae": float(df.persist_len_err.mean()),
        "vel_len_mae": float(df.vel_len_err.mean()),
        "x_cov": float(df.x_cov.mean()),
        "y_cov": float(df.y_cov.mean()),
        "l_cov": float(df.l_cov.mean()),
        "x_w": float(df.x_w.mean()),
        "y_w": float(df.y_w.mean()),
        "l_w": float(df.l_w.mean()),
        "burst_rate": float(burst_actual.mean()),
        "burst_prec": float(tp / max(burst_pred.sum(), 1)),
        "burst_rec": float(tp / max(burst_actual.sum(), 1)),
        "burst_brier": round(brier, 4),
        "burst_qprec": float(qtp / max(q_pred.sum(), 1)),
        "burst_qrec": float(qtp / max(burst_actual.sum(), 1)),
    }
