"""Calibrate the per-bin decoder's end offset (and onset length) on a movie's human traces.

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

Then the length at which a germination is called. The decoder's default, 2 px of its fitted length, calls
tubes visible at about 1.3 px of true length on synthetic movies; an annotator marks "first visible" later
(about 4 px on ``ld_v1``, from its traces), and every human bracket is one bin wide, so default onsets come
early. The annotator's length is measured directly: for every labelled onset, the decoder's own fitted
length midway between the last-absent and first-visible bins; the median is where germination is called
(``onset_px``). Against simulated annotators calling at 2-6 px it recovered their length and roughly
doubled onsets within 2 bins; from 28 grains it was never worse than the default in 200 draws, where a
search over thresholds gated by an interval adopted only a third of the time. It is adopted with at least
10 labelled onsets, unless a cross-validated check (each third of the grains called at the length the
other two give) finds it clearly worse. Lengths do not change with it.

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
ONSETS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)  # germination lengths (px) shown in the report
DEFAULT_ONSET = 2.0  # reach_grain's
MIN_ONSETS = 10  # labelled onsets needed to measure the annotator's germination length
ADOPTED = Path("runs/learned_evidence/ld_cal/decoder.json")  # where the launcher and finetune.py look


def model_default() -> Path:
    """The model the launcher uses when nothing is fine-tuned: the dev field's, else the shipped one."""
    from .finetune import SHIPPED
    return next(p for p in (Path("runs/learned_evidence/ld/unet.pt"), SHIPPED) if p.exists())


def _same(a, b) -> bool:
    return bool(a) and bool(b) and Path(a).resolve() == Path(b).resolve()


def judged_with(model: str | Path) -> str | None:
    """The decoder settings file a fine-tuned model's check was read with (``finetune.py`` records it)."""
    try:
        import torch
        return (torch.load(model, map_location="cpu", weights_only=False).get("args") or {}).get("decoder")
    except Exception:
        return None


def decoder_settings(path: str | Path | None, model: str | Path | None = None, log=print) -> dict:
    """Per-bin decoder keywords from a ``decoder.json`` (none without one). Says so when the file was
    fitted on another model: the offset belongs to the evidence it was fitted on. Not for a model
    fine-tuned after it: that model's check read its evidence with this file, so the pair was judged."""
    if not path:
        return {}
    doc = json.loads(Path(path).read_text())
    if model is not None and not _same(doc.get("model"), model) and not _same(judged_with(model), path):
        log(f"note: {path} was fitted on {doc.get('model')}, not {model}")
    return {"end_px": float(doc["end_px"]), **({"onset_px": float(doc["onset_px"])} if "onset_px" in doc else {})}


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


def pick(table: dict[float, dict[str, tuple]], grains, default: float = DEFAULT_END) -> float:
    """The setting with the most lengths plus onsets in tolerance over ``grains`` (ties: nearest the default)."""
    return max(table, key=lambda e: (sum(table[e][g][1] + (table[e][g][0] or 0) for g in grains if g in table[e]),
                                     -abs(e - default)))


def cross_check(table: dict[float, dict[str, tuple]], folds: int = 3, seed: int = 0, default: float = DEFAULT_END,
                gain: str = "length") -> dict:
    """Each fold's grains read with the setting the other folds picked, against the default: paired
    differences per grain, with a bootstrap over grains. Adopted when the interval for ``gain``
    ("length" or "onset") lies above zero and the other is no worse."""
    grains = sorted(table[default])
    fold = {g: i % folds for i, g in enumerate(grains)}
    dl, do, picks = {}, {}, []
    for k in range(folds):
        best = pick(table, [g for g in grains if fold[g] != k], default)
        picks.append(best)
        for g in (g for g in grains if fold[g] == k):
            (oa, la, _), (ob, lb, _) = table[best][g], table[default][g]
            dl[g], do[g] = la - lb, (oa or 0) - (ob or 0)
    rng = np.random.default_rng(seed)
    draws = [rng.integers(len(grains), size=len(grains)) for _ in range(4000)] if grains else []

    def ci(d):
        d = np.array([d[g] for g in grains], float)
        boot = [d[i].sum() for i in draws] or [0.0]
        return [float(v) for v in np.percentile(boot, [2.5, 97.5])]

    length_diff, onset_diff = int(sum(dl.values())), int(sum(do.values()))
    length_ci, onset_ci = ci(dl), ci(do)
    adopted = (length_ci[0] > 0 and onset_diff >= 0) if gain == "length" else (onset_ci[0] > 0 and length_diff >= 0)
    return {"grains": len(grains), "picks": picks, "length_diff": length_diff, "length_ci": length_ci,
            "onset_diff": onset_diff, "onset_ci": onset_ci, "gain": gain, "adopted": adopted}


def onset_readings(labels: dict, pred: dict) -> dict[str, float]:
    """Per labelled grain germinated within the movie: the decoder's fitted length midway between the
    annotator's last-absent and first-visible bins, i.e. how long the tube is, as the decoder reads it,
    when the annotator calls it visible."""
    from sparsetrack.evaluate import match_grains

    grains = {g: v for g, v in labels["grains"].items() if not v.get("excluded") and v.get("isolated", True)}
    matched = match_grains({"grains": grains}, pred.get("grains", []), radius=float(pred.get("match_radius_px", 12.0)))
    out = {}
    for gid, lab in labels.get("labels", {}).items():
        on, p = lab.get("onset") or {}, matched.get(gid)
        if on.get("verdict") != "emerged_within" or p is None or on.get("first_visible_frame") is None:
            continue
        fv = float(on["first_visible_frame"])
        la = float(on["last_absent_frame"]) if on.get("last_absent_frame") is not None else fv
        frames, fit = np.asarray(p["length"]["frames"], float), np.asarray(p["length"]["px"], float)
        out[gid] = 0.5 * float(np.interp(la, frames, fit) + np.interp(fv, frames, fit))
    return out


def onset_estimate(readings, seed: int = 0) -> tuple[float, list[float]]:
    """The germination length: the median reading (1-12 px), with a bootstrap 95% interval."""
    v = np.asarray(list(readings.values() if isinstance(readings, dict) else readings), float)
    rng = np.random.default_rng(seed)
    boot = [np.median(v[rng.integers(len(v), size=len(v))]) for _ in range(2000)]
    lo, hi = (float(np.clip(x, 1.0, 12.0)) for x in np.percentile(boot, [2.5, 97.5]))
    return round(float(np.clip(np.median(v), 1.0, 12.0)), 1), [round(lo, 1), round(hi, 1)]


def onset_check(labels: dict, pred: dict, readings: dict[str, float], folds: int = 3, seed: int = 0) -> dict:
    """Each fold's grains called at the germination length the other folds' readings give, against the
    default: paired onset hits per grain, with a bootstrap over grains."""
    grains = sorted(readings)
    fold = {g: i % folds for i, g in enumerate(grains)}
    default = hits(labels, {0: with_onset(pred, DEFAULT_ONSET)})[0]
    diff, picks = {}, []
    for k in range(folds):
        est, _ = onset_estimate([readings[g] for g in grains if fold[g] != k])
        picks.append(est)
        mine = hits(labels, {0: with_onset(pred, est)})[0]
        for g in (g for g in grains if fold[g] == k):
            diff[g] = (mine.get(g, (0,))[0] or 0) - (default.get(g, (0,))[0] or 0)
    d = np.array([diff[g] for g in grains], float)
    rng = np.random.default_rng(seed)
    boot = [d[rng.integers(len(d), size=len(d))].sum() for _ in range(4000)] if len(d) else [0.0]
    return {"grains": len(grains), "picks": picks, "onset_diff": int(d.sum()),
            "onset_ci": [float(v) for v in np.percentile(boot, [2.5, 97.5])]}


def with_onset(pred: dict, thr: float) -> dict:
    """The predictions with germinations called where the fitted length first reaches ``thr`` px, as
    ``reach_grain`` calls them (lengths unchanged; ``thr`` at most its ``min_tube_px``)."""
    out = []
    for r in pred["grains"]:
        r = dict(r)
        if r.get("status") in ("emerged_within", "emerged_at_start"):
            fit, frames = np.asarray(r["length"]["px"], float), r["length"]["frames"]
            on = np.nonzero(fit >= thr)[0]
            if len(on) and on[0] == 0:
                r.update(status="emerged_at_start", onset_frame=frames[0], onset_interval=None)
            elif len(on):
                r.update(status="emerged_within", onset_frame=frames[on[0]],
                         onset_interval=[frames[on[0] - 1], frames[on[0]]])
        out.append(r)
    return {**pred, "grains": out}


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
        preds = readings(pcache, field, labels_path, vmax, workers=args.workers)
    finally:
        if not args.keep_caches:
            shutil.rmtree(pcache, ignore_errors=True)
    table = hits(labels, preds)
    check = cross_check(table, args.folds)
    grains = sorted(table[DEFAULT_END])
    end = pick(table, grains)
    # then the onset threshold, on the lengths read with the offset kept (lengths do not change with it)
    base = preds[end if check["adopted"] else DEFAULT_END]
    otable = hits(labels, {t: with_onset(base, t) for t in ONSETS})
    on_reads = onset_readings(labels, base)
    onset, onset_ci, ocheck = DEFAULT_ONSET, None, None
    if len(on_reads) >= MIN_ONSETS:
        onset, onset_ci = onset_estimate(on_reads)
        ocheck = onset_check(labels, base, on_reads, args.folds)
    adopt_onset = ocheck is not None and ocheck["onset_ci"][1] >= 0 and onset != DEFAULT_ONSET  # unless clearly worse
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
               "not adopted: the gain is not clear of noise (or onsets are worse); keep the default offset"),
              "germination called at: onsets in tolerance (every labelled grain)"]
    for t in ONSETS:
        lines.append(f"  {t:.0f} px: {sum(v[0] or 0 for v in otable[t].values()):4d}"
                     + ("  (default)" if t == DEFAULT_ONSET else ""))
    if ocheck is None:
        lines.append(f"not adopted: {len(on_reads)} labelled onsets (at least {MIN_ONSETS} are needed); "
                     "keep calling germination at 2 px")
    else:
        lines += [f"the annotator's first-visible length, as the decoder reads it: {onset:.1f} px "
                  f"[{onset_ci[0]:.1f}, {onset_ci[1]:.1f}] (median over {len(on_reads)} labelled onsets)",
                  f"check ({args.folds} folds, each called at the length the others give: "
                  f"{', '.join(f'{p:.1f}' for p in ocheck['picks'])} px): onsets {ocheck['onset_diff']:+d} "
                  f"[{ocheck['onset_ci'][0]:+.0f}, {ocheck['onset_ci'][1]:+.0f}] against 2 px",
                  (f"adopted: germination is called at {onset:.1f} px, where the annotator calls tubes visible"
                   if adopt_onset else "not adopted: " + ("it is the default already" if onset == DEFAULT_ONSET else
                                                          "clearly worse in the check; keep calling germination at 2 px"))]
    print("\n".join(lines))
    doc = {"end_px": end if check["adopted"] else DEFAULT_END, "model": str(model), "labels": str(labels_path),
           "check": check, "default_end_px": DEFAULT_END, "onset_check": ocheck, "default_onset_px": DEFAULT_ONSET,
           "onset_estimate": onset, "onset_interval": onset_ci}
    if adopt_onset:
        doc["onset_px"] = onset
    if not (check["adopted"] or adopt_onset):  # a record of what was found; its end_px stays the default
        doc["picked_end_px"], doc["picked_onset_px"] = end, onset
    out = work / ("decoder.json" if check["adopted"] or adopt_onset else "decoder_not_adopted.json")
    out.write_text(json.dumps(doc, indent=1))
    (work / "report.txt").write_text("\n".join(lines) + "\n")  # last: adapt.py takes it to mean the step is done
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
