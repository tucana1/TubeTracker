"""What false census grains do to a movie's germination statistics, and what leaving flagged grains out does.

    python -m prototypes.census_check.germ predict MOVIE      # SparseTrack (defaults) on every judged census grain
    python -m prototypes.census_check.germ table              # the populations below, for every movie predicted

``predict`` reads the movie as the product does - the cache's own census, every census grain a grain (nobody has
excluded anything) - but only the grains the annotator judged (every ld grain; m2's and m1's random samples of
isolated grains away from the edge, with their replacements), with the tube maps held in memory
(``prototypes.tube_net.bench.in_memory``: nothing is written to the cache). Predictions go to
runs/census_check/pred_<movie>/.

``table`` compares, over each movie's judged census-isolated grains away from the edge (the grains the product's
population statistics use and the samples were drawn from):
- the annotator: the grains they kept, their onset brackets;
- the model over all of them (what the product reports now);
- the model without the grains the census check flags (per classifier: ``flags``, grain id -> flagged);
- the model over the grains the annotator kept (a perfect census check).
Per population: grains, germinated share by the end, T50 (Turnbull; frames).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
MOVIES = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
          "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json"),
          "m1": ("runs/sparsetrack/m1", "benchmark/labels/m1_v1.json")}
OUT = REPO / "runs/census_check"


def judged_population(movie: str) -> tuple[list[str], set[str], dict]:
    """(judged census grains isolated and away from the edge, the ones the annotator kept, the labels)."""
    cache, labels = (REPO / p for p in MOVIES[movie])
    L = json.loads(labels.read_text())
    census = {g["id"]: g for g in json.loads((cache / "grains.json").read_text())["grains"]}
    ids = [gid for gid, g in L["grains"].items() if g.get("exclude_reason") != "not_sampled"
           and census[gid].get("isolated") and not census[gid].get("border")]
    kept = {gid for gid in ids if not L["grains"][gid].get("excluded")}
    return sorted(ids), kept, L


def has_prob_movie(cache: Path, model: str) -> bool:
    """Whether ``learned.prob_cache`` would only read (a finished probability movie built by this very network)."""
    import hashlib
    meta = cache / f"prob_{Path(model).stem}" / "meta.json"
    return meta.exists() and json.loads(meta.read_text()).get("model_sha1") == \
        hashlib.sha1(Path(model).read_bytes()).hexdigest()


def predict(movie: str, model: str | None = None, params: dict | None = None, tag: str = "") -> Path:
    from prototypes.tube_net.bench import in_memory
    from sparsetrack import learned
    from sparsetrack.analyze import Params, analyze
    model = str(Path(model or learned.MODEL).resolve())
    cache = REPO / MOVIES[movie][0]
    if not has_prob_movie(cache, model):  # built already (read only), or held in memory
        in_memory(movie, model)
    ids, _, _ = judged_population(movie)
    out = OUT / f"pred_{movie}{tag}"
    analyze(cache, out, grains_path=cache / "grains.json", params=Params(model=model, **(params or {})), only=ids,
            log=lambda *a: None)
    return out / "predictions.json"


def human_interval(on: dict, fpb: int, nb: int):
    v, fv, la = on.get("verdict"), on.get("first_visible_frame"), on.get("last_absent_frame")
    first, last = fpb // 2, (nb - 1) * fpb + fpb // 2
    if v == "emerged_within" and fv is not None:
        return (float(la if la is not None else fv - fpb), float(fv))
    if v == "emerged_at_start":
        return (-math.inf, float(first))
    if v == "no_emergence_by_end":
        return (float(last), math.inf)
    return None  # unobservable


def curve(intervals: list, last: float, n_grid: int = 600) -> dict:
    """Turnbull curve: germinated share by the end and T50 (frames; None if under half germinate)."""
    from sparsetrack.report import turnbull
    grid = np.linspace(0.0, last, n_grid)
    cdf = np.zeros(n_grid)
    for q, p, mass in turnbull(intervals):
        if not np.isfinite(p):
            continue
        lo = q if np.isfinite(q) else p
        cdf += mass * np.clip((grid - lo) / max(p - lo, 1e-9), 0.0, 1.0)
    t50 = float(grid[int(np.argmax(cdf >= 0.5))]) if cdf[-1] >= 0.5 else None
    return {"n": len(intervals), "share": float(cdf[-1]), "t50": t50}


def populations(movie: str, flags: dict[str, set[str]] | None = None, pred_path: Path | None = None) -> dict:
    from sparsetrack.report import onset_intervals
    ids, kept, L = judged_population(movie)
    fpb, nb = int(L["frames_per_bin"]), int(L["n_bins"])
    last = (nb - 1) * fpb + fpb // 2
    pred = json.loads((pred_path or OUT / f"pred_{movie}" / "predictions.json").read_text())
    by_id = {g["id"]: g for g in pred["grains"]}
    hum = [iv for gid in sorted(kept) if (iv := human_interval((L["labels"].get(gid) or {}).get("onset") or {},
                                                                 fpb, nb)) is not None]

    def model(sel):
        return curve(onset_intervals({"grains": [by_id[g] for g in sel if g in by_id]}), last)

    out = {"annotator": curve(hum, last), "model_all": model(ids), "model_kept": model(sorted(kept)),
           "excluded": sorted(set(ids) - kept), "frames_per_bin": fpb}
    for name, fl in (flags or {}).items():
        out[f"model_{name}"] = model([g for g in ids if g not in fl])
        out[f"flagged_{name}"] = sorted(fl & set(ids))
    return out


def fmt(p: dict, fpb: int) -> str:
    t = "-" if p["t50"] is None else f"{p['t50'] / fpb:.1f}"
    return f"{p['n']} grains, {100 * p['share']:.0f}% germinated, T50 bin {t}"


if __name__ == "__main__":
    if sys.argv[1] == "predict":
        print(predict(sys.argv[2]))
    else:
        for mv in MOVIES:
            if (OUT / f"pred_{mv}" / "predictions.json").exists():
                P = populations(mv)
                print(mv, "| annotator:", fmt(P["annotator"], P["frames_per_bin"]), "| model, all:",
                      fmt(P["model_all"], P["frames_per_bin"]), "| model, annotator's grains:",
                      fmt(P["model_kept"], P["frames_per_bin"]), "| excluded:", " ".join(P["excluded"]))
