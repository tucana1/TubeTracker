"""Per-movie breakdown of default vs my tags (lengths within tolerance, onset hits), orig and human_t2.

    python permovie.py TAG [TAG...] [--movies m1,m2]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/tta")
sys.path.insert(0, str(ME.parent.parent / "round5"))
sys.path.insert(0, str(ME))
import score5  # noqa: E402
from common5 import DEV  # noqa: E402
from score_tta import PREDS, link_defaults  # noqa: E402
from sparsetrack.evaluate import score  # noqa: E402


def row(movie: str, tag: str, doc: str):
    p = PREDS / movie / f"{tag}.json"
    if not p.exists():
        return None
    x = score(score5.docs(movie)[doc], json.loads(p.read_text()), onset_tol=50)
    return x["length_full"]["within_tolerance"], x["length_full"]["n"], x["onset"]["hits"], \
        x["onset"]["n_human_emerged_within"]


def main(tags, movies):
    link_defaults()
    for doc in ("orig", "human_t2"):
        print(f"--- {doc}: lengths in tol / onsets (default -> tag)")
        for mv in movies:
            base = row(mv, "default", doc)
            if base is None:
                continue
            cells = []
            for t in tags:
                r = row(mv, t, doc)
                if r is not None:
                    cells.append(f"{t} {r[0] - base[0]:+4d} len {r[2] - base[2]:+3d} on")
            if cells:
                print(f"{mv:11s} default {base[0]:4d}/{base[1]:<4d} on {base[2]:3d}/{base[3]:<3d} | " + " | ".join(cells),
                      flush=True)


if __name__ == "__main__":
    mv = sys.argv[sys.argv.index("--movies") + 1].split(",") if "--movies" in sys.argv else list(DEV)
    tags = [a for a in sys.argv[1:] if not a.startswith("--") and a != ",".join(mv)]
    main(tags, mv)
