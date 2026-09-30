"""Two pixel checks of the same movie (``pixels.py --json``) compared paired over grains, with 95% bootstrap intervals.

    python -m prototypes.tube_net.pixpair A.json B.json MOVIE [--names A B]

Per grain, over its FULL traces: traced points marked (from the grain's visible edge), stubs seen, tips seen, and
young (<= 8 px) stubs seen. B minus A: the pooled share of traced points marked (percentage points) and the counts,
each with a bootstrap over grains (the unit that was sampled).
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def per_grain(rows: list[dict]) -> dict:
    out = {}
    for r in rows:
        g = out.setdefault(r["grain"], np.zeros(5))
        g += [r["hit_edge"], r["n_edge"], r["stub_seen"], r["tip_seen"], r["stub_seen"] and r["length"] <= 8]
    return out


def compare(a: list[dict], b: list[dict], n_boot: int = 4000, seed: int = 0) -> dict:
    A, B = per_grain(a), per_grain(b)
    g = sorted(set(A) & set(B))
    xa, xb = np.array([A[k] for k in g]), np.array([B[k] for k in g])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))

    def rec(x, i=None):
        s = x if i is None else x[i]
        return s[..., 0].sum(axis=-1) / np.maximum(s[..., 1].sum(axis=-1), 1)

    d = rec(xb) - rec(xa)
    dd = rec(xb, idx) - rec(xa, idx)
    out = {"grains": len(g), "traced": [100 * float(d), *(100 * np.percentile(dd, [2.5, 97.5]))]}
    for k, name in ((2, "stubs"), (3, "tips"), (4, "young_stubs")):
        diff = xb[:, k] - xa[:, k]
        out[name] = [float(diff.sum()), *np.percentile(diff[idx].sum(axis=1), [2.5, 97.5])]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("movie")
    ap.add_argument("--names", nargs=2, default=None)
    x = ap.parse_args(argv)
    ra, rb = (json.load(open(f))[x.movie]["rows"] for f in (x.a, x.b))
    c = compare(ra, rb)
    na, nb = x.names or (x.a, x.b)
    print(f"{x.movie}, {nb} minus {na} over {c['grains']} grains: traced points {c['traced'][0]:+.1f} pts "
          f"(95% CI {c['traced'][1]:+.1f} to {c['traced'][2]:+.1f}); " + "; ".join(
              f"{k.replace('_', ' ')} {c[k][0]:+.0f} ({c[k][1]:+.0f} to {c[k][2]:+.0f})"
              for k in ("stubs", "tips", "young_stubs")))


if __name__ == "__main__":
    main()
