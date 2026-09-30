"""Markdown rows of pixel checks (``pixels.py --json`` files) and bench dumps, for the README.

    python -m prototypes.tube_net.table pix LABEL=FILE:MOVIE [...]
    python -m prototypes.tube_net.table e2e BASELINE.json LABEL=FILE:MOVIE [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


def pix_row(label: str, s: dict) -> str:
    a, lg, y, rim = s["all"], s["long>=50"], s["young<=8"], s["rim_before_onset"]
    te = s.get("tip_extent")
    tip = (f"{te['median']:+.1f} ({te['p25']:+.1f}..{te['p75']:+.1f}), {te['within2']}/{te['traces']}" if te else "-")
    return (f"| {label} | {100 * a['point_recall']:.0f}% | {100 * lg['point_recall']:.0f}% | "
            f"{a['stub_seen']}/{a['traces']}, {a['tip_seen']} | {y['stub_seen']}/{y['traces']}, {y['tip_seen']} | "
            f"{100 * a['beside']:.1f}% | {100 * rim['share']:.1f}% | {tip} |")


def e2e_row(label: str, base: dict, new: dict, movie: str) -> str:
    """Totals and paired differences with ``synth_bench.paired``'s bootstrap (same seed and resamples)."""
    import numpy as np
    g = new[movie]
    rng = np.random.default_rng(0)
    keys = sorted(set(base[movie]["grains"]) & set(g["grains"]))
    idx = rng.integers(0, len(keys), (4000, len(keys)))
    cells = []
    for field in ("len_hit", "onset_hit", "both_hit"):
        d = np.array([int(bool(g["grains"][k][field])) - int(bool(base[movie]["grains"][k][field]))
                      if field == "onset_hit" else g["grains"][k][field] - base[movie]["grains"][k][field]
                      for k in keys])
        lo, hi = np.percentile(d[idx].sum(axis=1), [2.5, 97.5])
        cells.append(f"{d.sum():+d} ({lo:+.0f} to {hi:+.0f})")
    return (f"| {label} | {g['len_hit']}/{g['len_n']} | {g['on_hit']}/{g['on_n']} | {g['both']} | "
            f"{cells[0]} | {cells[1]} | {cells[2]} |")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pix":
        print("| network | traced | long | stubs, tips seen | young stubs, tips | beside | rim before onset | "
              "marks end vs apex: median (IQR), within 2 px |")
        print("|---|---|---|---|---|---|---|---|")
        for kv in sys.argv[2:]:
            label, rest = kv.split("=", 1)
            path, movie = rest.rsplit(":", 1)
            print(pix_row(label, json.loads(Path(path).read_text())[movie]["summary"]))
    else:
        base = json.loads(Path(sys.argv[2]).read_text())
        print("| network | lengths | onsets | length and tip | paired: lengths | onsets | length and tip |")
        print("|---|---|---|---|---|---|---|")
        for kv in sys.argv[3:]:
            label, rest = kv.split("=", 1)
            path, movie = rest.rsplit(":", 1)
            print(e2e_row(label, base, json.loads(Path(path).read_text()), movie))
