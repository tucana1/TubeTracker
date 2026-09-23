"""Run guided replay (closed-loop veto + open-loop rollout) over owners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .forecast import TipForecaster
from .guided_replay import closed_loop_replay, open_loop_rollout
from .tip_tracks import load_tip_tracks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--owners", required=True)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    owners = [int(x) for x in a.owners.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tracks = load_tip_tracks(a.measurements, owners)
    fc = TipForecaster()
    summary = {}
    for pid, tr in tracks.items():
        rep = closed_loop_replay(fc, tr)
        rep.to_csv(out / f"replay_P{pid}.csv", index=False)
        roll = open_loop_rollout(fc, tr, horizon=a.horizon)
        roll.to_csv(out / f"rollout_P{pid}.csv", index=False)
        vc = rep.verdict.value_counts().to_dict()
        div = roll[roll.diverged]
        summary[pid] = {
            "n": len(rep),
            "fault": int(vc.get("fault-suspect", 0)),
            "burst": int(vc.get("burst", 0)),
            "keep_rate": round(float((rep.verdict == "keep").mean()), 3),
            "max_resid": round(float(rep.resid.max()), 1),
            "diverge_step": (int(div.index.tolist()[0]) + 1 if len(div) else None),
            "rollout_err_end": (
                round(float(roll.rollout_err.iloc[-1]), 1) if len(roll) else None
            ),
        }
        print(f"P{pid}: {json.dumps(summary[pid])}", flush=True)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
