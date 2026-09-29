"""Per-grain differences between default and a tag on one movie (orig truth): length hits and errors per trace.

    python grain_diff.py MOVIE TAG
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
from score_tta import PREDS, link_defaults  # noqa: E402
from sparsetrack.evaluate import score  # noqa: E402


def main(movie: str, tag: str, doc: str = "orig"):
    link_defaults()
    d = score5.docs(movie)[doc]
    ra = {r["grain"]: r for r in score(d, json.loads((PREDS / movie / "default.json").read_text()), onset_tol=50)["rows"]}
    rb = {r["grain"]: r for r in score(d, json.loads((PREDS / movie / f"{tag}.json").read_text()), onset_tol=50)["rows"]}
    tot = 0
    for g in sorted(set(ra) & set(rb)):
        fa, fb = ra[g].get("full", []), rb[g].get("full", [])
        ha = sum(abs(f["error"]) <= max(2.0, 0.1 * f["human"]) for f in fa)
        hb = sum(abs(f["error"]) <= max(2.0, 0.1 * f["human"]) for f in fb)
        oa, ob = ra[g].get("onset_error"), rb[g].get("onset_error")
        if ha != hb or oa != ob:
            tot += hb - ha
            ea = " ".join(f"{f['error']:+.0f}" for f in fa)
            eb = " ".join(f"{f['error']:+.0f}" for f in fb)
            tl = " ".join(f"{f['human']:.0f}" for f in fa)
            print(f"{g}: hits {ha} -> {hb} ({hb - ha:+d}); onset err {oa} -> {ob}\n   truth {tl}\n   default {ea}\n   {tag} {eb}")
    print(f"net {tot:+d}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "orig")
