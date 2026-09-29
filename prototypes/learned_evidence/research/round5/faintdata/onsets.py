"""Signed onset errors per grain (score()'s rows: distance of the predicted onset from the truth bracket, in bins;
hit = within 2 bins) and length errors summary, for prediction tags in my folder, original truth.

    python onsets.py MOVIE TAG [TAG ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata")
sys.path.insert(0, str(ME.parent.parent / "round5"))
import score5  # noqa: E402
from sparsetrack.evaluate import score  # noqa: E402

mv, tags = sys.argv[1], sys.argv[2:]
doc = score5.docs(mv)["orig"]
fpb = doc["frames_per_bin"]
rows = {t: {r["grain"]: r for r in score(doc, json.loads((ME / "preds" / mv / f"{t}.json").read_text()),
                                          onset_tol=50)["rows"]} for t in tags}
stats = {t: [] for t in tags}
for gid in sorted(rows[tags[0]]):
    r0 = rows[tags[0]][gid]
    if r0.get("human") != "emerged_within":
        continue
    cells = []
    for t in tags:
        r = rows[t][gid]
        full = r.get("full", [])
        nok = sum(abs(f["error"]) <= max(2.0, 0.1 * f["human"]) for f in full)
        if "onset_error" in r:
            e = r["onset_error"] / fpb
            stats[t].append(e)
            cells.append(f"{t}: {e:+6.1f} len {nok:2d}/{len(full):<2d}")
        else:
            stats[t].append(np.nan)
            cells.append(f"{t}: {r.get('pred'):>8s} len {nok:2d}/{len(full):<2d}")
    br = r0["human_bracket"]
    print(f"{gid} truth ({br[0] / fpb:5.1f}, {br[1] / fpb:5.1f}] | " + " | ".join(cells))
for t in tags:
    a = np.array(stats[t])
    ok = a[~np.isnan(a)]
    print(f"{t}: emerged {len(a)}, not timed {int(np.isnan(a).sum())}, hits (<= 2 bins) {int((np.abs(ok) <= 2).sum())}, "
          f"late > 2 {int((ok > 2).sum())}, early < -2 {int((ok < -2).sum())}, median {np.median(ok) if len(ok) else np.nan:+.1f}")
