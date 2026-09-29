"""Regenerate the research data in a new container (hours of CPU; about 0.5 GB of disk per prepared movie).

    python -m prototypes.learned_evidence.research.regen field               prepare the real sample movie (in git),
                                                                            which is also the synthetic movies' field
    python -m prototypes.learned_evidence.research.regen movies [MOVIE ...]   render and prepare (default: every
                                                                            development movie, common.DEV)
    python -m prototypes.learned_evidence.research.regen maps [MODEL ...]     sparse maps (default v2 and B3) on every
                                                                            development movie and "real"
    python -m prototypes.learned_evidence.research.regen baselines           the current default's predictions ("default")

The sealed held-out set (common.FRESH) is rendered only by the look at a frozen candidate. A movie rendered on a field
prepared differently can differ from the original at the pixel level; its truth follows the seed, and the baselines
are decoded again here, so comparisons stay paired.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time

from .common import DEV, FIELD, REPO, SYNTH, config, image_cache, parse, sparse_path


def field() -> None:
    """The real sample movie's cache as Analyze_Movie_Learned.command prepared it (1 frame per bin, 3 reference bins
    from the start)."""
    from sparsetrack import stack
    from sparsetrack.cli import auto_frames_per_bin, write_census
    if not (FIELD / "grains.json").exists():
        movie = REPO / "sample_movie.avi"
        stack.prepare(movie, FIELD, frames_per_bin=auto_frames_per_bin(movie), ref_bins=3, ref_start=0)
        write_census(FIELD, 3, False)


def render(movie: str, sealed: bool = False) -> None:
    from sparsetrack import stack
    from sparsetrack.cli import write_census
    from sparsetrack.synth import make_movie
    if not (FIELD / "meta.json").exists():
        raise SystemExit(f"no prepared field at {FIELD}: run `regen field` first")
    kind, seed = parse(movie)
    name = f"synth_{kind}_s{seed}"
    SYNTH.mkdir(parents=True, exist_ok=True)
    if not (SYNTH / f"{name}.mp4").exists():
        make_movie(FIELD, SYNTH, config(movie), name=name, log=lambda *a: None)
    cache = image_cache(movie, sealed)
    if not (cache / "grains.json").exists():
        with contextlib.redirect_stdout(io.StringIO()):
            stack.prepare(SYNTH / f"{name}.mp4", cache, frames_per_bin=25, ref_bins=3, ref_start=0, log=lambda *a: None)
            write_census(cache, 3, False)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    what, rest = argv[0], argv[1:]
    t0 = time.time()
    if what == "field":
        field()
    elif what == "movies":
        for mv in rest or DEV:
            render(mv)
            print(f"MOVIE_DONE {mv} ({time.time() - t0:.0f} s)", flush=True)
    elif what == "maps":
        from .sparse import build
        for model in rest or ("v2", "B3"):
            for mv in [*DEV, "real"]:
                if not sparse_path(model, mv).exists():
                    build(model, mv)
                print(f"MAP_DONE {model} {mv} ({time.time() - t0:.0f} s)", flush=True)
    elif what == "baselines":
        from .decode import main as decode
        decode(["default", "v2", "B3", ",".join([*DEV, "real"])])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
