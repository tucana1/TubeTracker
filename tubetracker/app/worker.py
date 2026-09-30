"""The analysis of one movie, run by the app as its own process (so the window never waits on it, and it can be
stopped at any point)::

    python -m tubetracker.app.worker --run RUN_FOLDER [--movie MOVIE] [--flatfield]

Steps, each skipped when already done: read the movie's keyframes into the cache (first time only), find the
grains, build the tube-probability maps (first time only; needs torch), then SparseTrack's analysis grain by grain,
then the review labels file pre-filled with the model's answers (the app's corrections go there). The analysis is
written to ``analysis.running/`` and moved to ``analysis/`` only when complete; an earlier analysis (and the review
and results made from it) is kept whole in ``earlier/<time>/``.

Progress goes to stdout as lines ``@@ {"phase": ..., "label": ..., "k": ..., "n": ...}``; everything else printed
is the log. Phases: probe, prepare, register, census, maps, speed, grains, finish, done (or error).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

GRAIN_LINE = re.compile(r"^(\S+): (emerged_within|emerged_at_start|no_emergence_by_end|unobservable)\b")


def emit(**kw) -> None:
    print("@@ " + json.dumps(kw), flush=True)


def log(msg) -> None:
    print(msg, flush=True)


class Throttle:
    """``progress(done, total)`` callbacks turned into progress lines at most every ``every`` seconds."""

    def __init__(self, phase: str, label: str, every: float = 0.5):
        self.phase, self.label, self.every, self.last = phase, label, every, 0.0

    def __call__(self, done: int, total: int) -> None:
        now = time.time()
        if done >= total or now - self.last >= self.every:
            self.last = now
            emit(phase=self.phase, label=self.label, k=int(done), n=int(total))


def keep_earlier(root: Path) -> Path | None:
    """Move an earlier analysis, with the review and results made from it, into ``earlier/<time>/``."""
    parts = [p for p in (root / "analysis", root / "review", root / "results") if p.exists()]
    if not parts:
        return None
    dest = root / "earlier" / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for p in parts:
        p.rename(dest / p.name)
    return dest


def run(root: Path, movie: Path | None, flatfield: bool = False) -> None:
    from sparsetrack import cli, stack

    from .runfolder import RunFolder
    from .units import Units

    folder = RunFolder(root)
    cache = folder.root / "cache"
    if not ((cache / "meta.json").exists() and (cache / "grains.json").exists()):
        if movie is None or not movie.exists():
            raise FileNotFoundError(f"the movie {movie} is not there any more: open it again from where it is now")
        emit(phase="probe", label="Reading the movie's frame list")
        fpb = cli.auto_frames_per_bin(movie)
        log(f"preparing {movie.name}: {fpb} frames per bin")
        label = "Reading the movie's keyframes (first time only)"
        emit(phase="prepare", label=label, k=0, n=0)

        def prep_log(msg):
            log(msg)
            if str(msg).startswith("binned into"):
                emit(phase="register", label="Registering the frames to each other")
        stack.prepare(movie, cache, frames_per_bin=fpb, ref_bins=3, ref_start="auto", log=prep_log,
                      progress=Throttle("prepare", label))
        emit(phase="census", label="Finding the grains")
        cli.write_census(cache, 3, flatfield)
    else:
        emit(phase="prepare", label="Movie already prepared", skip=True)
    census = json.loads((cache / "grains.json").read_text())["grains"]
    n = sum(1 for g in census if not g.get("excluded"))
    if n == 0:
        raise RuntimeError("no grains were found in this movie (SparseTrack looks for grains 9-18 px in radius)")
    try:  # the tube maps first, so their progress shows (analyze would build them itself)
        from sparsetrack import learned
        out = cache / f"prob_{Path(learned.MODEL).stem}"
        label = "Building the tube maps (first time only, about 2 minutes)"
        if not (out / "meta.json").exists():
            emit(phase="maps", label=label, k=0, n=0)
        learned.prob_cache(cache, learned.MODEL, log, progress=Throttle("maps", label))
    except ImportError:
        log("torch is not installed: every grain is read from change evidence (reader=change)")
        emit(phase="maps", label="Tube maps skipped (torch is not installed)", skip=True)
    from sparsetrack.analyze import analyze

    setup = folder.load_setup()
    units = Units.from_setup(setup, folder.n_frames()).pair()
    running = folder.root / "analysis.running"
    if running.exists():  # a run that was stopped
        shutil.rmtree(running)
    emit(phase="speed", label="Measuring how fast tubes grow in this movie")
    done = {"k": 0}

    def grain_log(msg):
        log(msg)
        text = str(msg)
        if text.startswith("growth scale"):
            emit(phase="grains", label="Reading the grains", k=0, n=n)
        m = GRAIN_LINE.match(text)
        if m:
            done["k"] += 1
            emit(phase="grains", label="Reading the grains", k=done["k"], n=n, gid=m.group(1))
    analyze(cache, running, log=grain_log, units=units)
    emit(phase="finish", label="Saving the results and the check list")
    kept = keep_earlier(folder.root)
    if kept:
        log(f"the earlier analysis (and its review and results) is kept in {kept}")
    running.rename(folder.root / "analysis")
    from .corrections import Reviewer
    from .model import RunData
    reviewer = Reviewer(RunData(RunFolder(folder.root)))
    reviewer.start(background=False)
    if reviewer.error:
        log(f"the review file could not be prepared ({reviewer.error}); the app will try again when it opens")
    emit(phase="done", label="Analysis complete", kept=str(kept) if kept else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tubetracker.app.worker")
    ap.add_argument("--run", required=True, help="the movie's run folder")
    ap.add_argument("--movie", help="the movie (needed the first time, to prepare it)")
    ap.add_argument("--flatfield", action="store_true", help="correct vignetting before finding grains")
    args = ap.parse_args(argv)
    try:
        run(Path(args.run), Path(args.movie).expanduser() if args.movie else None, args.flatfield)
    except Exception as exc:  # noqa: BLE001 - reported to the window
        traceback.print_exc()
        emit(phase="error", label=f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
