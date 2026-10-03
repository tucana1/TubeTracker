"""Per-target rows of an evaluation: python -m prototypes.review_fill.show TAG MOVIE PROTO [--miss M2]

Prints, per target trace, the human length and each method's reading (and M2's route distance), to look at failures.
"""
from __future__ import annotations

import json
import sys

from prototypes.review_fill.build import OUT

tag, movie, proto = sys.argv[1:4]
miss = sys.argv[sys.argv.index("--miss") + 1] if "--miss" in sys.argv else None
s = json.loads((OUT / "eval" / tag / "summary.json").read_text())
rows = s["movies"][movie]["rows"][proto]
methods = [m for m in ("M0", "M1", "M2", "M3") if rows.get(m)]
print(f"{tag} {movie} {proto}: M2 {s['params']} M3 {s['m3']}")
for k in range(len(rows["M0"])):
    r0 = rows["M0"][k]
    if miss and rows[miss][k]["hit"]:
        continue
    cells = []
    for m in methods:
        r = rows[m][k]
        te = "-" if r["tip_err"] is None else f"{r['tip_err']:.0f}"
        cells.append(f"{m} {r['pred']:6.1f}{'*' if r['hit'] else ' '}{'t' if r['both'] else ' '} tip {te:>3}")
    r2 = rows["M2"][k]
    print(f"  {r0['gid']} b{r0['bin']:3d} human {r0['h']:6.1f} | " + " | ".join(cells) +
          f" | route {r2.get('cover')} marked {r2.get('marked')} ahead {r2.get('ahead')}"
          f" onset_given {r2.get('onset_given')}")
