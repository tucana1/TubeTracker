"""Calibrate the per-bin decoder's end offset on a movie's human traces.

The per-bin decoder adds a constant ``end_px`` to every length it reads (``reach.py``: the medial
axis stops short of a tube's end). How far short depends on how the evidence looks at a tube's end,
and on the synthetic movies it ranged from reading 7.7 px short (faint tubes) to 5.6 px long (thick
bright-cored tubes). One constant cannot memorise frames or grains, so a movie's traces can fit it.

The traces pick the offset (``ENDS``, whole pixels) that puts the most lengths and onsets in
tolerance. The check splits the labelled grains into folds: each fold is read with the offset the
other folds picked, against the default. Picking the best of ten on the same traces flatters small
gains, so the calibration is adopted only if the paired 95% interval for lengths lies above zero and
onsets are no worse. Then ``decoder.json`` holds the offset picked on every grain, for the model it
was fitted on; ``pipeline.py --decoder`` reads it, and ``finetune.py`` and ``Analyze_Movie_Learned.command``
pick it up from ``ADOPTED``. Not adopted, it is written as ``decoder_not_adopted.json``.

Fit it on a model that was not tuned on the same traces (its readings of them are in-sample).

    python -m prototypes.learned_evidence.calibrate --field runs/sparsetrack/ld \\
        --labels benchmark/labels/ld_v1.json --work runs/learned_evidence/ld_cal
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

ENDS = tuple(float(e) for e in range(-6, 4))
DEFAULT_END = 1.0  # reach_grain's
ADOPTED = Path("runs/learned_evidence/ld_cal/decoder.json")  # where the launcher and finetune.py look


def model_default() -> Path:
    """The model the launcher uses when nothing is fine-tuned: the dev field's, else the shipped one."""
    from .finetune import SHIPPED
    return next(p for p in (Path("runs/learned_evidence/ld/unet.pt"), SHIPPED) if p.exists())


def decoder_settings(path: str | Path | None, model: str | Path | None = None, log=print) -> dict:
    """Per-bin decoder keywords from a ``decoder.json`` (none without one). Says so when the file was
    fitted on another model: the offset belongs to the evidence it was fitted on."""
    if not path:
        return {}
    doc = json.loads(Path(path).read_text())
    if model is not None and Path(doc.get("model", "")).resolve() != Path(model).resolve():
        log(f"note: {path} was fitted on {doc.get('model')}, not {model}")
    return {"end_px": float(doc["end_px"])}


def _read(job):
    import cv2
    cv2.setNumThreads(1)
    from . import reach
    pcache, field, labels_path, vmax, end = job
    return end, reach.analyze(pcache, field, grains_path=labels_path, log=lambda *a: None, big=300, burst=True,
                              vmax=vmax, end_px=end)


def readings(pcache: Path, field: Path, labels_path: Path, vmax: float, ends=ENDS, workers: int = 4) -> dict:
    """The per-bin decoder's predictions with each end offset (in parallel processes)."""
    import multiprocessing as mp
    jobs = [(str(pcache), str(field), str(labels_path), vmax, e) for e in ends]
    if workers <= 1:
        return dict(map(_read, jobs))
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:  # forked workers hang once torch has threads
        return dict(pool.imap_unordered(_read, jobs))


def hits(labels: dict, preds: dict) -> dict[float, dict[str, tuple]]:
    """end -> grain -> (onset hit 0/1/None, length hits, length traces), as ``sparsetrack.evaluate.score`` counts."""
    from sparsetrack.evaluate import score

    from .evaluate import per_grain_hits
    out = {}
    for end, pred in preds.items():
        rep = score(labels, pred)
        out[end] = per_grain_hits(rep, rep["onset"]["tolerance_frames"])
    return out


def pick(table: dict[float, dict[str, tuple]], grains) -> float:
    """The offset with the most lengths plus onsets in tolerance over ``grains`` (ties: nearest the default)."""
    return max(table, key=lambda e: (sum(table[e][g][1] + (table[e][g][0] or 0) for g in grains if g in table[e]),
                                     -abs(e - DEFAULT_END)))


def cross_check(table: dict[float, dict[str, tuple]], folds: int = 3, seed: int = 0) -> dict:
    """Each fold's grains read with the offset the other folds picked, against the default: paired
    differences per grain, with a bootstrap over grains."""
    grains = sorted(table[DEFAULT_END])
    fold = {g: i % folds for i, g in enumerate(grains)}
    dl, do, picks = {}, {}, []
    for k in range(folds):
        best = pick(table, [g for g in grains if fold[g] != k])
        picks.append(best)
        for g in (g for g in grains if fold[g] == k):
            (oa, la, _), (ob, lb, _) = table[best][g], table[DEFAULT_END][g]
            dl[g], do[g] = la - lb, (oa or 0) - (ob or 0)
    rng = np.random.default_rng(seed)
    d = np.array([dl[g] for g in grains], float)
    boot = [d[rng.integers(len(d), size=len(d))].sum() for _ in range(4000)] if len(d) else [0.0]
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    length_diff, onset_diff = int(sum(dl.values())), int(sum(do.values()))
    return {"grains": len(grains), "picks": picks, "length_diff": length_diff, "length_ci": [lo, hi],
            "onset_diff": onset_diff, "adopted": lo > 0 and onset_diff >= 0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--field", required=True, help="prepared cache of the labelled movie")
    ap.add_argument("--labels", required=True, help="that movie's human labels (never movie 2)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default=None, help="the model whose evidence is decoded (default: as the launcher, "
                                                  "without a fine-tuned one)")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4, help="processes for the ten decodings")
    ap.add_argument("--keep-caches", action="store_true")
    args = ap.parse_args(argv)
    from . import evaluate
    from .finetune import speed_cap
    from .model import load as load_model

    field, work, labels_path = Path(args.field), Path(args.work), Path(args.labels)
    if "m2" in labels_path.name:
        raise SystemExit("movie 2 is the held-out benchmark: never calibrate on it")
    model = Path(args.model) if args.model else model_default()
    import torch
    tuned_on = (torch.load(model, map_location="cpu", weights_only=False).get("args") or {}).get("labels")
    if tuned_on and Path(tuned_on).resolve() == labels_path.resolve():
        raise SystemExit(f"{model} was fine-tuned on {labels_path}: its readings of those traces are in-sample. "
                         "Calibrate the model it started from instead.")
    labels = json.loads(labels_path.read_text())
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    pcache = evaluate.prob_cache(field, load_model(str(model)), work / "prob", log=print)
    try:
        vmax = speed_cap(pcache, field, labels_path)
        table = hits(labels, readings(pcache, field, labels_path, vmax, workers=args.workers))
    finally:
        if not args.keep_caches:
            shutil.rmtree(pcache, ignore_errors=True)
    check = cross_check(table, args.folds)
    grains = sorted(table[DEFAULT_END])
    end = pick(table, grains)
    lines = [f"{labels_path.name}: per-bin decoder end offset on {model} ({time.time() - started:.0f} s)",
             "end offset: lengths, onsets in tolerance (every labelled grain)"]
    for e in ENDS:
        n_len = sum(v[1] for v in table[e].values())
        n_on = sum(v[0] or 0 for v in table[e].values())
        lines.append(f"  {e:+.0f} px: {n_len:4d} {n_on:4d}" + ("  (default)" if e == DEFAULT_END else "")
                     + ("  <- picked" if e == end else ""))
    lines += [f"check ({args.folds} folds of grains; picks {', '.join(f'{p:+.0f}' for p in check['picks'])}): "
              f"lengths {check['length_diff']:+d} [{check['length_ci'][0]:+.0f}, {check['length_ci'][1]:+.0f}], "
              f"onsets {check['onset_diff']:+d} over {check['grains']} grains",
              ("adopted: the interval for lengths lies above zero and onsets are no worse" if check["adopted"] else
               "not adopted: the gain is not clear of noise (or onsets are worse); keep the default offset")]
    print("\n".join(lines))
    (work / "report.txt").write_text("\n".join(lines) + "\n")
    doc = {"end_px": end, "model": str(model), "labels": str(labels_path), "check": check,
           "default_end_px": DEFAULT_END}
    out = work / ("decoder.json" if check["adopted"] else "decoder_not_adopted.json")
    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
