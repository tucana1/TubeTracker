"""CLI: rolling-origin TimesFM-3 tip backtest over v29 tip tracks.

Example:
  .venv/bin/python -m prototypes.timesfm_tip_forecast.run_backtest \\
    --measurements runs/prototypes/v29/causal_growth_front/dense_full_v29_33_1/measurements.csv \\
    --owners 3,58,64,76,84,97 --horizons 3,6,12 \\
    --out runs/prototypes/timesfm/tip_backtest_v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np

from .backtest import backtest_owner, summarize
from .forecast import TipForecaster
from .tip_tracks import load_tip_tracks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--owners", default="3,58,64,76,84,97")
    ap.add_argument("--horizons", default="3,6,12")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--visibility-csv", default="",
                    help="H202: per-sample visibility CSV "
                    "(pollen_id,sample_index,visibility); joins onto each "
                    "track on sample_index (missing -> 0.0 = no evidence) "
                    "and appends it as a second past-only covariate.")
    args = ap.parse_args()

    owners = [int(s) for s in args.owners.split(",") if s.strip()]
    horizons = [int(s) for s in args.horizons.split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tracks = load_tip_tracks(args.measurements, owners)
    print(f"loaded {len(tracks)} tracks", flush=True)
    vis_map: dict[int, np.ndarray] = {}
    if args.visibility_csv:
        vv = pd.read_csv(args.visibility_csv)
        for pid, track in tracks.items():
            sub = vv[vv.pollen_id == pid].set_index("sample_index")
            aligned = (
                sub.reindex(track.sample_index)["visibility"]
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            vis_map[pid] = aligned
        print(f"visibility joined for {len(vis_map)} tracks", flush=True)
    forecaster = TipForecaster(device=args.device)

    summary = {}
    for pid, track in tracks.items():
        summary[pid] = {"n_points": len(track)}
        for h in horizons:
            t0 = time.time()
            df = backtest_owner(forecaster, track, h,
                                vis=vis_map.get(pid))
            dt = time.time() - t0
            if not df.empty:
                df.insert(0, "owner", pid)
                df.insert(1, "horizon", h)
                df.to_csv(out / f"backtest_P{pid}_H{h}.csv", index=False)
            s = summarize(df)
            s["seconds"] = round(dt, 1)
            summary[pid][f"H{h}"] = s
            print(f"P{pid} H{h}: {json.dumps(s)} ({dt:.0f}s)", flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
