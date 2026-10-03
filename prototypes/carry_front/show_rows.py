"""Per-trace rows of an O1 run: python -m prototypes.carry_front.show_rows tune_ld m2 b [--miss]"""
from __future__ import annotations

import json
import sys

from prototypes.carry_front.carry import OUT

tag, movie, name = sys.argv[1:4]
miss = "--miss" in sys.argv
s = json.loads((OUT / "o1" / tag / "summary.json").read_text())[f"{movie}/{name}"]
print(f"{movie}/{name} params {s['params']} lengths {s['lengths']}/{s['n']} reasons {s['reasons']}")
for gid, rows in sorted(s["rows"].items()):
    for r in rows:
        if miss and r["hit"]:
            continue
        print(f"  {gid} b{r['bin']}: human {r['h']:.1f} pred {r['pred']:.1f} err {r['err']:+.1f} "
              f"{'HIT' if r['hit'] else '   '} tip {r['tip_err'] if r['tip_err'] is None else round(r['tip_err'], 1)} "
              f"cover {r['cover']}")
