"""Per-movie scores (orig and human_t2) of prediction tags in my folder (default = round5's baseline, linked).

    python permovie.py TAG [TAG ...] [--movies m1,m2]
"""
import json
import sys
from pathlib import Path

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata")
sys.path.insert(0, str(ME.parent.parent / "round5"))
import common5  # noqa: E402
import score5  # noqa: E402
from sparsetrack.evaluate import score  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
movies = common5.DEV
for a in sys.argv[1:]:
    if a.startswith("--movies"):
        movies = a.split("=", 1)[1].split(",")
for doc in ("orig", "human_t2"):
    print(f"=== {doc}: lengths within tol / n, onset hits / n")
    for mv in movies:
        cells = []
        for tag in args:
            p = ME / "preds" / mv / f"{tag}.json"
            if not p.exists():
                cells.append(f"{tag}: -")
                continue
            x = score(score5.docs(mv)[doc], json.loads(p.read_text()), onset_tol=50)
            lf, on = x["length_full"], x["onset"]
            cells.append(f"{tag}: len {lf['within_tolerance']:4d}/{lf['n']:<4d} ({100 * lf['within_tolerance'] / max(lf['n'], 1):5.1f}%)"
                         f" on {on['hits']:3d}/{on['n_human_emerged_within']:<3d}")
        print(f"{mv:12s} " + " | ".join(cells), flush=True)
