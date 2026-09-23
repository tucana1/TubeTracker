"""Backtest TimesFM on corrected-tip tracks (H204: virtuous cycle).

Replaces accepted tip positions with corrected (consensus-smoothed) ones
where fixes exist, then runs the standard rolling-origin backtest.
If forecasts improve, the corrector earns a second role: cleaning the
forecaster's training data.
"""

import argparse
import json
import time

import pandas as pd

from .backtest import backtest_owner, summarize
from .forecast import TipForecaster
from .tip_tracks import load_tip_tracks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--corrected", required=True,
                    help="corrected series CSV (sample_index, corrected_x, "
                    "corrected_y, moved_px, keep)")
    ap.add_argument("--owner", type=int, required=True)
    ap.add_argument("--horizons", default="3,12")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    tr = load_tip_tracks(a.measurements, [a.owner])[a.owner]
    c = pd.read_csv(a.corrected)
    keep = c[(c.moved_px > 0) & (c.keep == 1)].set_index("sample_index")
    tr2 = tr.copy()
    idx = tr2.sample_index.isin(keep.index)
    tr2.loc[idx, "tip_x"] = tr2[idx].sample_index.map(keep.corrected_x)
    tr2.loc[idx, "tip_y"] = tr2[idx].sample_index.map(keep.corrected_y)
    print(f"swapped {int(idx.sum())}/{len(tr2)} positions", flush=True)

    fc = TipForecaster(device=a.device)
    out: dict = {}
    for name, tt in (("accepted", tr), ("corrected", tr2)):
        out[name] = {}
        for h in [int(s) for s in a.horizons.split(",")]:
            t0 = time.time()
            s = summarize(backtest_owner(fc, tt, h))
            s["seconds"] = round(time.time() - t0, 1)
            out[name][f"H{h}"] = s
            print(f"{name} H{h}: {json.dumps(s)}", flush=True)
    pd.DataFrame(out).to_json(a.out, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
