"""score5.compare of my tags against R5 "default", with score5's PREDS pointed at my folder (defaults symlinked in).

    python score_tta.py TAG [TAG...] [--groups faint,thin,...] [--movies m1,m2]    -> scores/<TAG>.json and a table
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True
ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/tta")
R5 = ME.parent.parent / "round5"
sys.path.insert(0, str(R5))
import score5  # noqa: E402
from common5 import DEV, GROUPS  # noqa: E402

PREDS = ME / "preds"


def link_defaults() -> None:
    for mv in [*DEV, "real"]:
        src = R5 / "preds" / mv / "default.json"
        dst = PREDS / mv / "default.json"
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)


def run(tags, movies=None, quiet=False) -> dict:
    link_defaults()
    score5.PREDS = PREDS
    groups = GROUPS
    if movies:
        groups = {g: [m for m in ms if m in movies] for g, ms in GROUPS.items()}
        groups = {g: ms for g, ms in groups.items() if ms}
    res = {}
    for tag in tags:
        r = score5.compare("default", tag, groups=groups, quiet=quiet)
        (ME / "scores").mkdir(exist_ok=True)
        (ME / "scores" / f"{tag}.json").write_text(json.dumps(r, indent=1, default=str))
        res[tag] = r
        if not quiet:
            print("rule preview (orig):", " | ".join(score5.rule_check(r)), flush=True)
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mv = sys.argv[sys.argv.index("--movies") + 1].split(",") if "--movies" in sys.argv else None
    args = [a for a in args if a not in (mv and [",".join(mv)] or [])]
    run(args, mv)
