"""The protocol x method table of an evaluation, as markdown: python -m prototypes.review_fill.table TAG"""
from __future__ import annotations

import json
import sys

from prototypes.review_fill.build import OUT

s = json.loads((OUT / "eval" / sys.argv[1] / "summary.json").read_text())
print(f"M2 {s['params']}; M3 {s['m3']}; onset: {s['onset']}; tuned on {s.get('tuned_on')}\n")
print("| movie | protocol | targets (grains) | M0 0.8.8 | M1 app (rescaled) | M2 image fill | M3 | ML lines | "
      "M2 - M1 lengths [95% CI] | M2 - M1 len & tip | M2 - M0 lengths | M3 - M1 lengths |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
fmt = lambda c: f"{c['hits']} / {c['both']} ({c['med_abs']})"
ci = lambda p, k: f"{p[k][0]:+d} [{p[k][1]:+.0f}, {p[k][2]:+.0f}]" if p else "-"
for movie, m in s["movies"].items():
    for proto, t in m["table"].items():
        print(f"| {movie} | {proto} | {t['M0']['n']} ({t['M2-M1'].get('grains', 0)}) | {fmt(t['M0'])} | {fmt(t['M1'])} | "
              f"{fmt(t['M2'])} | {fmt(t['M3'])} | {fmt(t['ML'])} | {ci(t['M2-M1'], 'lengths')} | "
              f"{ci(t['M2-M1'], 'both')} | {ci(t['M2-M0'], 'lengths')} | {ci(t['M3-M1'], 'lengths')} |")
print("\ncells: lengths in tolerance / length-and-tip (median |error| px)")
