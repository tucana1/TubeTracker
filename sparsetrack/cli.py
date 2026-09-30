"""Command line: ``python -m sparsetrack prepare|bench ...``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import grains, stack


def registered_mean(bins: np.ndarray, shifts: list, which: list[int]) -> np.ndarray:
    height, width = bins.shape[1:]
    acc = np.zeros((height, width), np.float64)
    for b in which:
        dx, dy = shifts[b]
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        acc += cv2.warpAffine(np.asarray(bins[b], np.float32), m, (width, height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return acc / len(which)


def cmd_prepare(args) -> None:
    out = Path(args.out)
    if (out / "meta.json").exists() and (out / "grains.json").exists() and not args.force:
        print(f"cache already exists at {out} (use --force to rebuild)")
        return
    ref_start = args.ref_start if args.ref_start == "auto" else int(args.ref_start)
    stack.prepare(args.movie, out, frames_per_bin=args.frames_per_bin, ref_bins=args.ref_bins, ref_start=ref_start)
    write_census(out, args.ref_bins, args.flatfield)


def write_census(out: Path, n_ref: int, flatfield: bool) -> None:
    bins, meta = stack.load(out)
    ref_bins = list(range(meta.get("ref_start", 0), meta.get("ref_start", 0) + n_ref))
    reference = registered_mean(bins, meta["shifts"], ref_bins)
    found = grains.detect(reference, flatfield=flatfield)
    (out / "grains.json").write_text(json.dumps({
        "schema": "sparsetrack.grains.v1", "reference_bins": ref_bins, "flatfield": flatfield,
        "method": "Hough circles + dark-rim/body contrast on the registered reference", "grains": found},
        indent=1))
    isolated = sum(g["isolated"] for g in found)
    print(f"grains: {len(found)} detected ({isolated} isolated, "
          f"{sum(g['clump_size'] > 1 for g in found)} in clumps, {sum(g['border'] for g in found)} near the edge)")
    for w in census_warnings(found):
        print(f"WARNING: {w}")


def census_warnings(found: list[dict], r_min: float = 9.0, r_max: float = 18.0) -> list[str]:
    """A census that suggests the movie is not like the ones SparseTrack was built on (grains 9-18 px in radius:
    the lab's 1280 x 1024 movies, radius median 12-13 px): none found, or many at the edge of the radius range
    (grains larger or smaller than expected, e.g. another magnification)."""
    if not found:
        return ["no grains found. SparseTrack looks for grains 9-18 px in radius (as in the lab's movies so far); "
                "a movie at another magnification needs other settings."]
    r = [g["r"] for g in found]
    at_edge = sum(x <= r_min + 0.5 or x >= r_max - 0.5 for x in r) / len(r)
    if len(r) >= 5 and at_edge > 0.5:
        return [f"{100 * at_edge:.0f}% of the grains found are at the edge of the 9-18 px radius range: grains in this "
                f"movie may be larger or smaller than SparseTrack expects (another magnification?); check the census "
                f"(field_early.png) before trusting the analysis."]
    return []


def cmd_census(args) -> None:
    write_census(Path(args.cache), args.ref_bins, args.flatfield)


def cmd_synth(args) -> None:
    from .synth import make_movie, preset
    cfg = preset(args.preset, seed=args.seed, encode=not args.lossless)
    tag = "" if args.preset == "v1" else args.preset
    make_movie(args.cache, args.out, cfg,
               name=args.name or f"synth{tag}_s{args.seed}{'_lossless' if args.lossless else ''}")


def auto_frames_per_bin(movie: str | Path, target_bins: int = 175) -> int:
    """Whole keyframe groups per bin, about ``target_bins`` bins per movie (the sparse movie: 300)."""
    from .video import probe
    info = probe(movie)
    step = info.keyframe_interval or max(1, info.n_frames // max(len(info.keyframes), 1))
    return int(max(step, round(info.n_frames / target_bins / step) * step))


CALIBRATION = Path("calibration.json")  # the lab's microscope: {"um_per_px": ..., "s_per_frame": ...}


def calibration(um_per_px: float | None = None, s_per_frame: float | None = None) -> tuple[float, float] | None:
    """Pixel size and frame interval: as given, else from calibration.json in the working folder; None if
    either is unknown (tables stay in px and frames)."""
    if not (um_per_px and s_per_frame) and CALIBRATION.exists():
        cal = json.loads(CALIBRATION.read_text())
        um_per_px, s_per_frame = um_per_px or cal.get("um_per_px"), s_per_frame or cal.get("s_per_frame")
    return (float(um_per_px), float(s_per_frame)) if um_per_px and s_per_frame else None


def run_folder(movie: str | Path, out: str | Path | None = None) -> Path:
    """Where ``run`` keeps a movie's cache (``cache/``), analysis (``analysis/``) and review (``review/``)."""
    import re
    movie = Path(movie).expanduser()
    if out:
        return Path(out)
    if movie.is_dir() and (movie / "cache").is_dir():  # the folder itself
        return movie
    return Path("runs/sparsetrack") / re.sub(r"[^A-Za-z0-9_.-]+", "_", movie.stem).strip("_.")


def prepared(movie: Path, out: Path, frames_per_bin: int | None = None, flatfield: bool = False) -> Path:
    """The movie's cache in its run folder, built the first time."""
    cache = out / "cache"
    if not ((cache / "meta.json").exists() and (cache / "grains.json").exists()):
        fpb = frames_per_bin or auto_frames_per_bin(movie)
        print(f"preparing {movie.name}: {fpb} frames per bin")
        stack.prepare(movie, cache, frames_per_bin=fpb, ref_bins=3, ref_start="auto")
        write_census(cache, 3, flatfield)
    return cache


def cmd_run(args) -> None:
    """One step for a new movie: cache (first time only), analysis, review gallery."""
    import webbrowser
    from .analyze import analyze
    movie = Path(args.movie).expanduser()
    out = run_folder(movie, args.out)
    cache = prepared(movie, out, args.frames_per_bin, args.flatfield)
    units = calibration(args.um_per_px, args.s_per_frame)
    analyze(cache, out / "analysis", video=args.video, units=units)
    page = (out / "analysis" / "index.html").resolve()
    print(f"review gallery: {page}")
    if not args.no_browser:
        webbrowser.open(page.as_uri())


def cmd_review(args) -> None:
    """Check and correct an analysis in the labelling tool, then export the reviewed results."""
    import shutil
    import time
    from .bench.server import serve
    from .review import export, prefill
    out = run_folder(args.movie, args.out)
    cache = Path(args.cache) if args.cache else out / "cache"
    pred = Path(args.pred) if args.pred else out / "analysis" / "predictions.json"
    if not pred.exists() or not (cache / "meta.json").exists():
        raise SystemExit(f"analyse the movie first (sparsetrack run {args.movie}): needs {pred} and {cache}")
    folder = out / "review"
    labels = folder / "review_labels.json"
    model = labels.with_suffix(".model.json")
    if labels.exists() and model.exists() and pred.stat().st_mtime > model.stat().st_mtime:
        if args.new:
            keep = out / f"review_{time.strftime('%Y%m%d-%H%M%S')}"
            shutil.move(str(folder), str(keep))
            print(f"the earlier review is kept, whole, in {keep}")
        else:
            print("this movie was analysed again after its review was pre-filled: carrying on with the earlier "
                  "review (--new starts one on the new analysis and keeps the earlier one beside it)")
    if not labels.exists():
        prefill(cache, pred, labels)
    if not args.export_only:
        print("Confirm or fix each answer (Enter confirms the model's). Press Ctrl-C here when you stop; "
              "answers are saved as you go.")
        serve(cache, labels, port=args.port, open_browser=not args.no_browser, annotator=args.annotator)
    export(labels, *(calibration(args.um_per_px, args.s_per_frame) or (None, None)))


def cmd_bench(args) -> None:
    from .bench.server import serve
    serve(args.cache, args.labels, port=args.port, open_browser=not args.no_browser, annotator=args.annotator,
          sample=args.sample, seed=args.seed)


def cmd_analyze(args) -> None:
    from .analyze import analyze
    only = [g.strip() for g in args.only.split(",")] if args.only else None
    analyze(args.cache, args.out, grains_path=args.grains, only=only, video=args.video)


def cmd_eval(args) -> None:
    from .evaluate import load, markdown, score
    labels, preds = load(args.labels), [load(p) for p in args.pred]
    text = "".join(markdown(score(labels, pred, onset_tol=args.onset_tol, subset=args.subset)) + "\n"
                   for pred in preds)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


def cmd_summary(args) -> None:
    """Several analysed movies side by side: one table and one figure."""
    from .summary import write_summary
    folders = [run_folder(m) for m in args.movies]
    missing = [str(f) for f in folders if not (f / "analysis" / "predictions.json").exists()]
    if missing:
        raise SystemExit(f"not analysed yet (run `sparsetrack run` first): {', '.join(missing)}")
    write_summary(folders, Path(args.out), calibration(args.um_per_px, args.s_per_frame))


def cmd_retest(args) -> None:
    from .evaluate import load, retest_report
    r = retest_report(load(args.labels))
    o, t = r["onset"], r["traces"]
    print(f"onset retest: {o['within']}/{o['n']} first-visible bins within +/-600 frames of the first answer")
    print(f"length retest: {t['n']} traces repeated, {t['both_full']} full both times; length within max(2 px, 10%): "
          f"{t['length_within']}/{t['both_full']}; length and apex within max(5 px, 10%): {t['length_and_tip']}/"
          f"{t['both_full']}; median |difference| {t['median_abs_diff'] if t['median_abs_diff'] is None else round(t['median_abs_diff'], 2)} px")
    for row in t["rows"]:
        print("  ", row)


def cmd_compare(args) -> None:
    from .evaluate import load
    from .render import Renderer
    from .report import write_comparison
    bins, meta = stack.load(args.cache)
    page = write_comparison(load(args.labels), load(args.pred), Renderer(bins, meta), args.out)
    print(f"comparison page: {page.resolve()}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sparsetrack")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="build the keyframe-bin cache and grain list for a movie")
    p.add_argument("movie")
    p.add_argument("--out", required=True)
    p.add_argument("--frames-per-bin", type=int, default=300)
    p.add_argument("--ref-bins", type=int, default=3, help="bins averaged as the 'before' reference")
    p.add_argument("--ref-start", default="auto",
                   help="first reference bin, or 'auto' = first bin after the field has settled")
    p.add_argument("--force", action="store_true")
    p.add_argument("--flatfield", action="store_true", help="correct vignetting before grain detection")
    p.set_defaults(func=cmd_prepare)
    c = sub.add_parser("census", help="re-run grain detection on an existing cache (renumbers grains: "
                                      "do this before labelling starts)")
    c.add_argument("cache")
    c.add_argument("--ref-bins", type=int, default=3)
    c.add_argument("--flatfield", action="store_true")
    c.set_defaults(func=cmd_census)
    b = sub.add_parser("bench", help="open the benchmark labelling tool")
    b.add_argument("cache")
    b.add_argument("--labels", required=True)
    b.add_argument("--port", type=int, default=8765)
    b.add_argument("--no-browser", action="store_true")
    b.add_argument("--annotator", default="investigator")
    b.add_argument("--sample", type=int, default=0,
                   help="new labels only: label a random sample of this many isolated grains (others start excluded)")
    b.add_argument("--seed", type=int, default=20260923)
    b.set_defaults(func=cmd_bench)
    a = sub.add_parser("analyze", help="per-grain onset and tube length for a prepared movie")
    a.add_argument("cache")
    a.add_argument("--out", required=True)
    a.add_argument("--grains", help="grain list: a benchmark labels file (human census) or grains.json")
    a.add_argument("--only", help="comma-separated grain ids")
    a.add_argument("--video", action="store_true", help="also render field_overlay.mp4 (model tubes on the field)")
    a.set_defaults(func=cmd_analyze)
    y = sub.add_parser("synth", help="synthetic movie with exact truth from a prepared cache's real field")
    y.add_argument("cache")
    y.add_argument("--out", required=True)
    y.add_argument("--seed", type=int, default=0)
    y.add_argument("--name")
    y.add_argument("--lossless", action="store_true", help="write FFV1 (no codec artifacts) as a control")
    y.add_argument("--preset", default="v1", choices=("v1", "v2", "v3", "v4", "v5"),
                   help="v1: clean isolated tubes; v2: adds foreign tubes, crossings, curls, pauses/stops, "
                        "drifting grains and docking particles")
    y.set_defaults(func=cmd_synth)
    r = sub.add_parser("run", help="prepare (first time), analyse and open the review gallery for a movie")
    r.add_argument("movie")
    r.add_argument("--out", help="output folder (default runs/sparsetrack/<movie name>)")
    r.add_argument("--frames-per-bin", type=int, help="default: whole keyframe groups, about 175 bins per movie")
    r.add_argument("--flatfield", action="store_true", help="correct vignetting before grain detection")
    r.add_argument("--video", action="store_true", help="also render field_overlay.mp4")
    r.add_argument("--um-per-px", type=float, help="pixel size: lengths and growth in um in the tables")
    r.add_argument("--s-per-frame", type=float, help="frame interval: onsets in minutes, growth in um/min")
    r.add_argument("--no-browser", action="store_true")
    r.set_defaults(func=cmd_run)
    y = sub.add_parser("summary", help="several analysed movies side by side: germinated share, T50, growth rate "
                                       "(summary.csv, summary.png)")
    y.add_argument("movies", nargs="+", help="movies (as given to run) or their run folders")
    y.add_argument("--out", default="runs/sparsetrack/summary", help="default runs/sparsetrack/summary")
    y.add_argument("--um-per-px", type=float)
    y.add_argument("--s-per-frame", type=float)
    y.set_defaults(func=cmd_summary)
    v = sub.add_parser("review", help="check and correct an analysis in the labelling tool, pre-filled with the "
                                      "model's answers; exports the reviewed results when you stop")
    v.add_argument("movie", help="the movie, as given to run (or its output folder)")
    v.add_argument("--out", help="output folder, as given to run (default runs/sparsetrack/<movie name>)")
    v.add_argument("--cache", help="prepared cache (default OUT/cache)")
    v.add_argument("--pred", help="predictions to review (default OUT/analysis/predictions.json)")
    v.add_argument("--new", action="store_true",
                   help="start a new review of a newer analysis; the earlier review is kept beside it")
    v.add_argument("--export-only", action="store_true", help="write the results of the review so far, no tool")
    v.add_argument("--um-per-px", type=float, help="pixel size, for lengths in um")
    v.add_argument("--s-per-frame", type=float, help="frame interval, for times in minutes")
    v.add_argument("--port", type=int, default=0, help="default: any free port")
    v.add_argument("--annotator", default="reviewer")
    v.add_argument("--no-browser", action="store_true")
    v.set_defaults(func=cmd_review)
    m = sub.add_parser("compare", help="page of every human trace next to the model's path and tip")
    m.add_argument("cache")
    m.add_argument("--labels", required=True)
    m.add_argument("--pred", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_compare)
    r = sub.add_parser("retest", help="the annotator's blind repeats against their first answers")
    r.add_argument("labels")
    r.set_defaults(func=cmd_retest)
    e = sub.add_parser("eval", help="score predictions against benchmark labels")
    e.add_argument("--labels", required=True)
    e.add_argument("--pred", required=True, nargs="+")
    e.add_argument("--onset-tol", type=float, default=600.0)
    e.add_argument("--subset", choices=("isolated", "all"), default="isolated")
    e.add_argument("--out")
    e.set_defaults(func=cmd_eval)
    args = ap.parse_args(argv)
    args.func(args)
