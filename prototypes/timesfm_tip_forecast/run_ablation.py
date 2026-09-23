"""CLI: single-factor input ablations at one horizon (dense field).

Compares full-spec TimesFM-3 rows against input variants with IDENTICAL
evaluation origins (by sample_index for the diagnostic-context run):
  full, no_speed, no_covs, uniform_schedule, window24, marginal, with_rejected

with_rejected loads diagnostic rows too (accepted_only=False); the trust
covariate then varies 0/1 instead of sitting constant at 1.

Writes per-variant owner CSVs + comparison.json (pooled tip RMSE,
win-rate vs persistence, locked-rule burst F1, coverage).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BURST_GAIN_PX, backtest_owner, make_origins, summarize
from .forecast import TipForecaster
from .tip_tracks import load_tip_tracks

VARIANTS = [
    "full",
    "uniform_schedule",
    "window24",
    "marginal",
    "with_rejected",
]


def locked_burst_f1(df: pd.DataFrame) -> dict:
    """H143 locked rule: burst_p > 0 AND fc_gain >= 1.0."""
    if df.empty:
        return {}
    y = (df.actual_gain > BURST_GAIN_PX).to_numpy()
    p = (df.burst_p.to_numpy() > 0) & (df.fc_gain.to_numpy() >= 1.0)
    tp = int((p & y).sum())
    prec = tp / max(int(p.sum()), 1)
    rec = tp / max(int(y.sum()), 1)
    return {
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
        "prec": round(prec, 3),
        "rec": round(rec, 3),
        "base": round(float(y.mean()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--owners", required=True)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    owners = [int(s) for s in args.owners.split(",") if s.strip()]
    variants = [s for s in args.variants.split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    full_tracks = load_tip_tracks(args.measurements, owners, accepted_only=True)
    rej_tracks = (
        load_tip_tracks(args.measurements, owners, accepted_only=False)
        if "with_rejected" in variants
        else {}
    )
    forecaster = TipForecaster(device=args.device)
    comparison = {}
    for var in variants:
        t0 = time.time()
        frames = []
        # Reference origins from the accepted-only tracks (same eval points).
        for pid, ftrack in full_tracks.items():
            ref_origins = make_origins(len(ftrack), args.horizon)
            if var == "with_rejected":
                rt = rej_tracks.get(pid)
                if rt is None:
                    continue
                ref_s = ftrack.sample_index.to_numpy()
                rs = rt.sample_index.to_numpy()
                mapped = np.searchsorted(rs, ref_s[ref_origins])
                mapped = [int(m) for m in mapped if m < len(rt)]
                df = backtest_owner(
                    forecaster, rt, args.horizon, origins=mapped
                )
            else:
                df = backtest_owner(
                    forecaster, ftrack, args.horizon, variant=var,
                    origins=ref_origins,
                )
            if not df.empty:
                df.insert(0, "owner", pid)
                frames.append(df)
        pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not pool.empty:
            pool.to_csv(out / f"ablation_{var}_H{args.horizon}.csv", index=False)
        s = summarize(pool)
        tm = float((pool.tip_err < pool.persist_tip_err).mean()) if not pool.empty else 0.0
        comparison[var] = {
            "n_rows": len(pool),
            "tip_rmse": round(float(pool.tip_err.mean()), 3) if not pool.empty else None,
            "persist_tip_rmse": round(float(pool.persist_tip_err.mean()), 3) if not pool.empty else None,
            "win_rate_vs_persist": round(tm, 3),
            "x_cov": round(float(pool.x_cov.mean()), 3) if not pool.empty else None,
            "l_cov": round(float(pool.l_cov.mean()), 3) if not pool.empty else None,
            "burst": locked_burst_f1(pool),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{var}: {json.dumps(comparison[var])}", flush=True)

    (out / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(f"wrote {out / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
