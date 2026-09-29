"""Paired comparison of two prediction tags over movie groups: lengths within max(2 px, 10%) and onsets within 2 bins
(``sparsetrack.evaluate.score``, onset_tol 50 synthetic frames), for the original truth and human-style documents
(``humanstyle``), with a 95% paired bootstrap over grains pooled across a group's movies.

    python -m prototypes.learned_evidence.research.score BASE_TAG NEW_TAG [orig] [human_t2]
Points are percentage points of the group's length (or onset) count. The development preview of round 5's rule is
printed; the rule is decided on the sealed held-out set (look.py), once.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .common import FIELD, GROUPS, PREDS, ROOT, truth_path
from .humanstyle import load_docs

_docs: dict = {}


def docs(movie: str, sealed: bool = False) -> dict:
    if movie not in _docs:
        _docs[movie] = load_docs(truth_path(movie, sealed), FIELD)
    return _docs[movie]


def _merged(reps):
    rows = []
    for mv, rep in reps:
        rows += [dict(r, grain=f"{mv}:{r['grain']}") for r in rep["rows"]]
    return {"rows": rows}


def compare(a: str, b: str, which=("orig", "human_t2"), groups=GROUPS, sealed: bool = False, quiet: bool = False,
            preds=PREDS) -> dict:
    from sparsetrack.evaluate import score

    from ..evaluate import paired_bootstrap
    out = {}
    for doc in which:
        if not quiet:
            print(f"\n=== {doc}: {a} -> {b} (lengths within max(2 px, 10%); onsets within 2 bins)")
        for gname, movies in groups.items():
            movies = [m for m in movies if (preds / m / f"{a}.json").exists() and (preds / m / f"{b}.json").exists()]
            if not movies:
                continue
            ra, rb, sa, sb = [], [], np.zeros(4, int), np.zeros(4, int)
            for mv in movies:
                d = docs(mv, sealed)[doc]
                xa = score(d, json.loads((preds / mv / f"{a}.json").read_text()), onset_tol=50)
                xb = score(d, json.loads((preds / mv / f"{b}.json").read_text()), onset_tol=50)
                ra.append((mv, xa))
                rb.append((mv, xb))
                for s, x in ((sa, xa), (sb, xb)):
                    s += [x["length_full"]["within_tolerance"], x["length_full"]["n"], x["onset"]["hits"],
                          x["onset"]["n_human_emerged_within"]]
            pb = paired_bootstrap(_merged(rb), _merged(ra), 50)
            n, no = max(sa[1], 1), max(sa[3], 1)
            pts = {"len_pt": 100 * pb["length_diff"] / n, "len_lo_pt": 100 * pb["length_ci"][0] / n,
                   "len_hi_pt": 100 * pb["length_ci"][1] / n, "on_pt": 100 * pb["onset_diff"] / no}
            if not quiet:
                print(f"{gname:24s} lengths {sa[0]:4d} -> {sb[0]:4d} /{sa[1]:<5d} diff {pb['length_diff']:+4.0f} "
                      f"[{pb['length_ci'][0]:+.0f}, {pb['length_ci'][1]:+.0f}] = {pts['len_pt']:+.1f} pt "
                      f"[{pts['len_lo_pt']:+.1f}, {pts['len_hi_pt']:+.1f}] | onsets {sa[2]:3d} -> {sb[2]:3d} "
                      f"/{sa[3]:<3d} diff {pb['onset_diff']:+3.0f} [{pb['onset_ci'][0]:+.0f}, {pb['onset_ci'][1]:+.0f}] "
                      f"({len(movies)} movies)", flush=True)
            out[f"{doc}|{gname}"] = {"base": sa.tolist(), "new": sb.tolist(), "paired": pb, "points": pts,
                                    "movies": movies}
    return out


def rule5(res: dict, thin: str = "thin (v5 s3-8, s13-15)", thick: str = "thick s30+s31",
          wide: str = "wide s26+s27") -> list[str]:
    """Round 5's rule, parts 1-2 (original truth): thin+faint pooled lengths with a 95% lower bound above zero; thin
    onsets >= -1.0 pt; thick and wide lengths >= -1.0 pt with lower bound >= -3.0 pt. Parts 3-4 (the real-movie audit,
    laptop time) are judged apart."""
    def pts(g):
        return (res.get(f"orig|{g}") or {}).get("points")

    lines = []
    p = pts("thin+faint pooled")
    lines.append("thin+faint pooled lengths: " + ("not scored" if p is None else
                 f"{p['len_pt']:+.2f} pt (lower {p['len_lo_pt']:+.2f}): {'PASS' if p['len_lo_pt'] > 0 else 'FAIL'}"))
    p = pts(thin)
    lines.append("thin onsets: " + ("not scored" if p is None else
                 f"{p['on_pt']:+.2f} pt: {'PASS' if p['on_pt'] >= -1.0 else 'FAIL'}"))
    for g in (thick, wide):
        p = pts(g)
        lines.append(f"{g} lengths: " + ("not scored" if p is None else
                     f"{p['len_pt']:+.2f} pt (lower {p['len_lo_pt']:+.2f}): "
                     f"{'PASS' if p['len_pt'] >= -1.0 and p['len_lo_pt'] >= -3.0 else 'FAIL'}"))
    return lines


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    which = [w for w in sys.argv[3:] if w.startswith(("orig", "human_t"))] or ["orig", "human_t2"]
    res = compare(a, b, which)
    print("\nround-5 rule, development preview:", " | ".join(rule5(res)))
    (ROOT / "scores").mkdir(parents=True, exist_ok=True)
    (ROOT / "scores" / f"{a}__{b}.json").write_text(json.dumps(res, indent=1, default=str))
