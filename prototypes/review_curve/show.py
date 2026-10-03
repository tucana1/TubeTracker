"""Per-target rows of candidate curves: python -m prototypes.review_curve.show MOVIE PROTO [CANDIDATE ...]
[--onset human] [--diff]   (--diff: only targets where the methods disagree on the hit)"""
from __future__ import annotations

import sys

from prototypes.review_curve.evaluate import ROUNDS, evaluate
from prototypes.review_fill.evaluate import load_movie

args = [a for a in sys.argv[1:] if not a.startswith("--")]
movie, proto, names = args[0], args[1], args[2:] or ["clip"]
mode = "human" if "--onset" in sys.argv and "human" in sys.argv else "review"
CANDIDATES = {**ROUNDS["r1"], **ROUNDS["r2"]}
gs = {g.gid: g for g in load_movie(movie)}
res = evaluate(list(gs.values()), mode, [n for n in names if n in CANDIDATES], CANDIDATES)
R = res[proto]
methods = ["M0", "M1", *[n for n in names if n in CANDIDATES]]
for k in range(len(R["M0"])):
    rows = [R[m][k] for m in methods]
    if "--diff" in sys.argv and len({r["hit"] for r in rows}) == 1:
        continue
    r0 = rows[0]
    g = gs[r0["gid"]]
    an = [(b, round(L, 1)) for b, L, _ in g.anchors(proto)]
    on = g.review_onset(an[0][0], mode)
    cells = " | ".join(f"{m} {r['pred']:6.1f}{'*' if r['hit'] else ' '}{'t' if r['both'] else ' '}"
                       for m, r in zip(methods, rows))
    print(f"{r0['gid']} b{r0['bin']:3d} h {r0['h']:6.1f} | {cells} | anchors {an} onset {on[1]} (model "
          f"{g.model_onset()}, human {g.human_onset()[1]})")
