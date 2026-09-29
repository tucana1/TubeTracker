"""Per-grain comparison of two scored prediction files (e.g. st053 and a KymoReader row).

    python -m prototypes.kymo_reader.compare ld runs/lab_checks_2026-09-29/ld_st053.json \
        runs/kymo_reader/results/C_st_ld.json
"""

from __future__ import annotations

import sys

import numpy as np

from .evaluate import LABELS, _within


def rows(labels, pred):
    from sparsetrack.evaluate import score
    rep = score(labels, pred)
    out = {}
    for r in rep["rows"]:
        full = r.get("full", [])
        out[r["grain"]] = {"onset_err": r.get("onset_error"), "pred": r.get("pred"), "human": r.get("human"),
                           "full": [(f["frame"], f["human"], f["pred"], f.get("tip_error")) for f in full]}
    return out, rep


def main(argv=None):
    from sparsetrack.evaluate import load
    a = argv or sys.argv[1:]
    movie, pa, pb = a[0], a[1], a[2]
    labels = load(LABELS[movie])
    ra, repa = rows(labels, load(pa))
    rb, repb = rows(labels, load(pb))
    print(f"{'grain':6s} {'onset err A':>11s} {'onset err B':>11s}  lengths A/B  traces (human: A B)")
    for g in sorted(ra):
        fa, fb = ra[g]["full"], rb[g]["full"]
        ha = sum(_within(p - h, h) for _, h, p, _ in fa)
        hb = sum(_within(p - h, h) for _, h, p, _ in fb)
        tr = "  ".join(f"{h:.0f}: {pa_:.0f} {pb_:.0f}" for (_, h, pa_, _), (_, _, pb_, _) in zip(fa, fb))
        oa, ob = ra[g]["onset_err"], rb[g]["onset_err"]
        mark = " <" if hb < ha else (" >" if hb > ha else "")
        print(f"{g:6s} {str(oa):>11s} {str(ob):>11s}  {ha}/{hb} of {len(fa)}{mark}   {tr}")
    for name, rep in (("A", repa), ("B", repb)):
        e = np.array([f["error"] for r in rep["rows"] for f in r.get("full", [])])
        print(f"{name}: lengths {rep['length_full']['within_tolerance']}/{rep['length_full']['n']}, median |err| "
              f"{np.median(np.abs(e)):.2f}, bias {np.mean(e):+.2f}; onset {rep['onset']['hits']}/{rep['onset']['n_timed']}")


if __name__ == "__main__":
    main()
