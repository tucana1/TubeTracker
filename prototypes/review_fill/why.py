"""Why M2 misses: python -m prototypes.review_fill.why TAG

Per movie and protocol, M2's missed targets by cause: route (its route at that bin > 3 px from the human trace over
the human length), blind (route right, the map marks < half of the traced tube), long, short (route right, map
marks it: the front itself)."""
from __future__ import annotations

import json
import sys

from prototypes.review_fill.build import OUT

s = json.loads((OUT / "eval" / sys.argv[1] / "summary.json").read_text())
for movie, m in s["movies"].items():
    for proto, rows in m["rows"].items():
        R = rows.get("M2") or []
        if not R:
            continue
        why = {"hit": 0, "route": 0, "blind": 0, "long": 0, "short": 0}
        for r in R:
            if r["hit"]:
                why["hit"] += 1
            elif r.get("cover") is not None and r["cover"] > 3.0:
                why["route"] += 1
            elif (r.get("marked") or 0.0) < 0.5:
                why["blind"] += 1
            elif r["err"] > 0:
                why["long"] += 1
            else:
                why["short"] += 1
        stubs = [r for r in R if r["h"] <= 8.0]
        print(f"{movie} {proto}: {len(R)} targets ({len(stubs)} tubes <= 8 px): {why}")
