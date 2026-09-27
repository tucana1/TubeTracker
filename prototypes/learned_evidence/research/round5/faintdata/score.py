"""Paired comparison of my predictions against the round-5 default with the shared score5.compare (PREDS pointed at my
folder, where default.json is a link to round5's baseline, made only once that baseline exists).

    python score.py NEW_TAG [BASE_TAG=default]      -> scores/<BASE>__<NEW>.json, and the rule's development preview
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ME = Path("/tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad/agents/faintdata")
R5 = ME.parent.parent / "round5"
sys.path.insert(0, str(R5))
import common5  # noqa: E402
import score5  # noqa: E402

PREDS = ME / "preds"


def link_defaults() -> None:
    for mv in [*common5.DEV, "real"]:
        src, dst = common5.PREDS / mv / "default.json", PREDS / mv / "default.json"
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)


if __name__ == "__main__":
    new = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "default"
    link_defaults()
    score5.PREDS = PREDS
    res = score5.compare(base, new)
    print("\nround-5 rule, development preview:", " | ".join(score5.rule_check(res)))
    (ME / "scores").mkdir(exist_ok=True)
    (ME / "scores" / f"{base}__{new}.json").write_text(json.dumps(res, indent=1, default=str))
