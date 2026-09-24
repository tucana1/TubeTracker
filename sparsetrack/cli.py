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


def cmd_run(args) -> None:
    """One step for a new movie: cache (first time only), analysis, review gallery."""
    import re
    import webbrowser
    from .analyze import analyze
    movie = Path(args.movie).expanduser()
    out = Path(args.out) if args.out else Path("runs/sparsetrack") / re.sub(r"[^A-Za-z0-9_.-]+", "_", movie.stem).strip("_.")
    cache = out / "cache"
    if not ((cache / "meta.json").exists() and (cache / "grains.json").exists()):
        fpb = args.frames_per_bin or auto_frames_per_bin(movie)
        print(f"preparing {movie.name}: {fpb} frames per bin")
        stack.prepare(movie, cache, frames_per_bin=fpb, ref_bins=3, ref_start="auto")
        write_census(cache, 3, args.flatfield)
    analyze(cache, out / "analysis", video=args.video)
    page = (out / "analysis" / "index.html").resolve()
    print(f"review gallery: {page}")
    if not args.no_browser:
        webbrowser.open(page.as_uri())


def cmd_bench(args) -> None:
    from .bench.server import serve
    serve(args.cache, args.labels, port=args.port, open_browser=not args.no_browser, annotator=args.annotator)


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
    r.add_argument("--no-browser", action="store_true")
    r.set_defaults(func=cmd_run)
    e = sub.add_parser("eval", help="score predictions against benchmark labels")
    e.add_argument("--labels", required=True)
    e.add_argument("--pred", required=True, nargs="+")
    e.add_argument("--onset-tol", type=float, default=600.0)
    e.add_argument("--subset", choices=("isolated", "all"), default="isolated")
    e.add_argument("--out")
    e.set_defaults(func=cmd_eval)
    args = ap.parse_args(argv)
    args.func(args)
