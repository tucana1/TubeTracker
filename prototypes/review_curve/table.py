"""The protocol x method table of a review_curve evaluation, as markdown: python -m prototypes.review_curve.table TAG"""
from __future__ import annotations

import json
import sys

from prototypes.review_curve.evaluate import OUT

s = json.loads((OUT / sys.argv[1] / "summary.json").read_text())
new = s["chosen"]
print(f"{new} {s['params']}; onset: {s['onset']}; chosen on {s['tuned_on']}\n")
print("| movie | protocol | targets | M0 0.8.8 | M1 rescaled (old app) | new | new - M1 lengths | new - M1 len & tip | "
      "new - M0 lengths |")
print("|---|---|---|---|---|---|---|---|---|")
fmt = lambda c: f"{c['hits']} / {c['both']} ({c['med_abs']})"
ci = lambda p, k: f"{p[k][0]:+d} [{p[k][1]:+.0f}, {p[k][2]:+.0f}]"
for movie, m in s["movies"].items():
    for proto, t in m["table"].items():
        print(f"| {movie} | {proto} | {t['M0']['n']} | {fmt(t['M0'])} | {fmt(t['M1'])} | {fmt(t[new])} | "
              f"{ci(t['new-M1'], 'lengths')} | {ci(t['new-M1'], 'both')} | {ci(t['new-M0'], 'lengths')} |")
print("\ncells: lengths in tolerance / length-and-tip (median |error| px); differences paired over grains, 95% CI")
