"""Judge a network on a set of labelled grains, maps in memory (nothing is written to the caches).

    python -m prototypes.tube_adapt.evaluate pix MODEL.pt --movie m1 --grains g003 ... --out OUT.json
    python -m prototypes.tube_adapt.evaluate e2e MODEL.pt --grains g003 ... --out OUT.json [--tag NAME]

``pix``: the pixel check of ``prototypes/tube_net/pixels.py`` (traced points marked from the grain's visible edge, long
traces, young stubs, stubs and tips seen, marks beside tubes, where the marks end relative to the apex, rim marks
before onset - movie 1's grains followed bin by bin) on just these grains' bins and rows.
``e2e``: SparseTrack's defaults with only the network changed (``prototypes/tube_net/bench.py``'s in-memory maps over
every bin), these grains analysed (``analyze(..., only=)``; each grain is read on its own, so this equals analysing all
and keeping these), scored on these grains only: per-grain onset hit, FULL lengths in tolerance, length and tip
(``scripts/synth_bench.score_real``'s per-grain fields).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from prototypes.tube_adapt.common import MOVIES, OUT, REPO, SCRATCH, labels, offsets, subset, write_subset


def pix(model: str, movie: str, grains: list[str] | None = None, net=None, log=print) -> dict:
    from prototypes.synth_v6.recall import measure, rim_false_marks, summarise
    from prototypes.tube_net.pixels import extra, prob_movie, needed_bins, rim_followed, tip_extent
    from sparsetrack import stack
    L = labels(movie)
    if grains:
        L = subset(L, grains)
    lab_path = write_subset(movie, list(L["labels"]), OUT / "labels" / f"pix_{movie}_{abs(hash(tuple(sorted(L['labels'])))) % 10**8}.json")
    _, meta = stack.load(REPO / MOVIES[movie][0])
    pm = prob_movie(model, movie, needed_bins(L, meta), net=net, log=log)
    rows = measure(pm, lab_path)
    s = summarise(rows) | extra(rows)
    s["rim_before_onset"] = rim_followed(pm, L, offsets(movie)) if movie == "m1" else rim_false_marks(pm, lab_path)
    te = tip_extent(pm, L)
    ext = np.array([r["extent"] for r in te]) if te else np.zeros(0)
    if len(ext):
        s["tip_extent"] = {"traces": len(te), "median": float(np.median(ext)), "p25": float(np.percentile(ext, 25)),
                           "p75": float(np.percentile(ext, 75)), "within2": int(np.sum(np.abs(ext) <= 2.0)),
                           "past4": int(np.sum(ext > 4.0)), "short4": int(np.sum(ext < -4.0))}
    return {"summary": s, "rows": rows, "tip_extent": te, "grains": sorted(L["labels"])}


def within(err: float, h: float) -> bool:
    return abs(err) <= max(2.0, 0.1 * h)


def e2e(model: str, grains: list[str], tag: str, movie: str = "m1", params: dict | None = None) -> dict:
    sys.path.insert(0, str(REPO / "scripts"))
    from prototypes.tube_net.bench import in_memory
    from sparsetrack.analyze import Params, analyze
    from sparsetrack.evaluate import score
    in_memory(movie, model)
    work = SCRATCH / "tube_adapt" / tag
    pred = analyze(REPO / MOVIES[movie][0], work, grains_path=REPO / MOVIES[movie][1],
                   params=Params(model=model, **(params or {})), only=grains, log=lambda *a: None)
    rep = score(subset(labels(movie), grains), pred)
    out = {}
    for r in rep["rows"]:
        if r["grain"] not in grains:
            continue
        full = r.get("full", [])
        out[r["grain"]] = {"onset_hit": (abs(r["onset_error"]) <= 600) if "onset_error" in r else None,
                           "len_hit": sum(within(f["error"], f["human"]) for f in full), "len_n": len(full),
                           "both_hit": sum(within(f["error"], f["human"]) and f.get("tip_error", 1e9)
                                           <= max(5.0, 0.1 * f["human"]) for f in full),
                           "full": full, "pred_status": r.get("pred"), "onset_error": r.get("onset_error")}
    return {"on_hit": rep["onset"]["hits"], "on_n": rep["onset"]["n_timed"],
            "len_hit": rep["length_full"]["within_tolerance"], "len_n": rep["length_full"]["n"],
            "both": rep["tips"]["length_and_tip"], "len_med": rep["length_full"]["median_abs_error"],
            "len_bias": rep["length_full"]["bias"], "grains": out, "pred": str(work / "predictions.json")}


def paired(base: dict, new: dict, grains: list[str], n_boot: int = 4000, seed: int = 0) -> dict:
    """New minus base over ``grains`` (per-grain dicts of e2e): onsets, lengths, length and tip, with 95% bootstrap
    intervals over grains."""
    rng = np.random.default_rng(seed)
    g = [k for k in grains if k in base and k in new]
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    res = {"grains": len(g)}
    for key in ("onset_hit", "len_hit", "both_hit"):
        d = np.array([int(new[k][key] or 0) - int(base[k][key] or 0) for k in g])
        res[key] = [int(d.sum()), *np.percentile(d[idx].sum(axis=1), [2.5, 97.5]).tolist(),
                    int(sum(int(base[k][key] or 0) for k in g)), int(sum(int(new[k][key] or 0) for k in g))]
    res["len_n"] = int(sum(base[k]["len_n"] for k in g))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pix", "e2e"))
    ap.add_argument("model")
    ap.add_argument("--movie", default="m1")
    ap.add_argument("--grains", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag")
    a = ap.parse_args(argv)
    if a.what == "pix":
        res = pix(a.model, a.movie, a.grains)
        from prototypes.tube_net.pixels import line
        print(f"{Path(a.model).stem} on {a.movie} ({len(res['grains'])} grains): {line(res['summary'])}", flush=True)
    else:
        res = e2e(a.model, a.grains, a.tag or Path(a.out).stem, a.movie)
        print(f"{Path(a.model).stem} on {a.movie} ({len(a.grains)} grains): onsets {res['on_hit']}/{res['on_n']}, "
              f"lengths {res['len_hit']}/{res['len_n']}, length and tip {res['both']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
