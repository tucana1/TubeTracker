"""Per-grain probability curves of one evaluation (human bracket, 0.8.8's onset, the network's onset) as one PNG.

    python -m prototypes.onset_net.curves m2 v1 [--src drift] [--out file.png]
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import OUT, baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("tag")
    ap.add_argument("--src", default="drift")
    ap.add_argument("--out")
    a = ap.parse_args()
    ev = json.loads((OUT / f"eval_{a.movie}_{a.tag}.json").read_text())
    base = {g["id"]: g for g in baseline(a.movie)["grains"]}
    fpb = ev["fpb"]
    G = ev["grains"]
    ids = sorted(G, key=lambda g: (G[g]["kind"] != "grain", G[g]["fv"] if G[g]["fv"] is not None else 1e9))
    n = len(ids)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 1.7 * rows), squeeze=False)
    for ax, g in zip(axs.ravel(), ids):
        e = G[g]
        p = np.asarray(e[f"p_{a.src}"])
        ax.plot(p, lw=0.8, color="#3060c0")
        if a.src == "drift":
            ax.plot(np.asarray(e["p_follow"]), lw=0.5, color="#999999", alpha=0.7)
        if e["fv"] is not None:
            ax.axvspan((e["la"] if e["la"] is not None else e["fv"] - 1) - 3, e["fv"] + 2, color="#30a030", alpha=0.2)
            ax.axvline(e["fv"], color="#208020", lw=0.8)
        b = base.get(g)
        if b and b.get("status") == "emerged_within":
            ax.axvline((b["onset_frame"] - fpb // 2) / fpb, color="#d08000", lw=0.8, ls="--")
        t = e[f"onset_{a.src}"]
        if t is not None:
            ax.axvline(t, color="#c02020", lw=0.8, ls=":")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{g} {e['verdict'][:10]} fv={e['fv']} net={t} "
                     f"088={None if not b or b.get('status') != 'emerged_within' else (b['onset_frame'] - fpb // 2) // fpb}",
                     fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out or OUT / f"curves_{a.movie}_{a.tag}_{a.src}.png", dpi=80)


if __name__ == "__main__":
    main()
