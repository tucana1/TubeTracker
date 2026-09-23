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
    meta = stack.prepare(args.movie, out, frames_per_bin=args.frames_per_bin, ref_bins=args.ref_bins)
    bins, meta = stack.load(out)
    ref_bins = list(range(args.ref_bins))
    reference = registered_mean(bins, meta["shifts"], ref_bins)
    found = grains.detect(reference)
    (out / "grains.json").write_text(json.dumps({
        "schema": "sparsetrack.grains.v1", "reference_bins": ref_bins,
        "method": "Hough circles + dark-rim/body contrast on the registered reference", "grains": found},
        indent=1))
    isolated = sum(g["isolated"] for g in found)
    print(f"grains: {len(found)} detected ({isolated} isolated, "
          f"{sum(g['clump_size'] > 1 for g in found)} in clumps, {sum(g['border'] for g in found)} near the edge)")


def cmd_bench(args) -> None:
    from .bench.server import serve
    serve(args.cache, args.labels, port=args.port, open_browser=not args.no_browser, annotator=args.annotator)


def cmd_analyze(args) -> None:
    from .analyze import analyze
    only = [g.strip() for g in args.only.split(",")] if args.only else None
    analyze(args.cache, args.out, grains_path=args.grains, only=only)


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
    p.add_argument("--ref-bins", type=int, default=3, help="leading bins averaged as the reference")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prepare)
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
    a.set_defaults(func=cmd_analyze)
    e = sub.add_parser("eval", help="score predictions against benchmark labels")
    e.add_argument("--labels", required=True)
    e.add_argument("--pred", required=True, nargs="+")
    e.add_argument("--onset-tol", type=float, default=600.0)
    e.add_argument("--subset", choices=("isolated", "all"), default="isolated")
    e.add_argument("--out")
    e.set_defaults(func=cmd_eval)
    args = ap.parse_args(argv)
    args.func(args)
