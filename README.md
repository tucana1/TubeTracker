# TubeTracker

TubeTracker is a research prototype for pollen germination time-lapse movies:
per grain, detect when a tube emerges and measure the tube's length over time.

**Status (23 Sep 2026): sparse-first reset.** Work now concentrates on isolated
grains before crossings and clumps. The research record up to 22 Sep 2026 is in
`prototypes/LEDGER.md`; the complete pre-reset tree (all prototype generations,
round scripts and their tests) is preserved at the git tag `snapshot-2026-09-23`.

## SparseTrack (active development)

`sparsetrack/` is the new sparse-field pipeline. The movies are x264 exports with a
keyframe every 12 frames at the encoder's quality floor, so it reads keyframes only
and works on registered averages of 25 keyframes (300 source frames per "bin").

```bash
# build the cache: keyframe bins, registration, grain census (~30-40 s for the sparse movie)
.venv/bin/python -m sparsetrack prepare MOVIE --out runs/sparsetrack/ld
# benchmark labelling tool (local web page; answers saved to the labels file after every click)
.venv/bin/python -m sparsetrack bench runs/sparsetrack/ld --labels benchmark/labels/ld_v1.json
# SparseTrack v1: per-grain onset and exit-to-apex length (~13 s for the sparse movie)
.venv/bin/python -m sparsetrack analyze runs/sparsetrack/ld --out runs/sparsetrack/ld_v1 [--grains benchmark/labels/ld_v1.json]
# score any predictions against the benchmark
.venv/bin/python -m sparsetrack eval --labels benchmark/labels/ld_v1.json --pred runs/sparsetrack/ld_v1/predictions.json
```

`analyze` traces each tube once on the end-of-movie change map, lets the grain and tube
rotate rigidly about the grain centre, reads growth backwards along that path with a
non-decreasing dynamic-programming front, and calls onset from the excess change just
outside the rim at the exit. It writes `predictions.json`, `grains.csv`, `growth.csv`, a diagnostic image per grain,
`growth_curves.png` (small multiples), `population.csv`/`population.png` (interval-censored
cumulative germination, Turnbull estimate, with T50) and, with `--video`, `field_overlay.mp4`. Scores so far are in `benchmark/reports/`.

Double-clicking `Label_Sparse_Benchmark.command` prepares the sparse movie (first time
only) and opens the labelling tool; `Label_Movie2_Heldout.command` does the same for the
held-out movie 2 (vignetting-corrected census; its first ~9 settling bins are skipped as
the "before" reference). `sparsetrack census CACHE [--flatfield]` re-runs grain detection
(it renumbers grains, so only before labelling starts). The tool asks for a grain census (confirm, exclude or add grains), each grain's onset
bracket on a whole-movie filmstrip then single bins, and exit-to-apex traces at a
few fixed times. Answers use the `GerminationEvent` vocabulary of
`tubetracker/annotation_schema.py`. `benchmark/labels/` is the benchmark; keep it
under version control.

## Frozen v30 movie-analysis app

The v30 native review app is kept unchanged while its replacement is built. It is
not being developed further.

```bash
./Start_TubeTracker_Analysis.command     # napari UI (.venv-annotator) + inference (.venv)
.venv/bin/python scripts/analyze_movie.py --config CONFIG --project-dir PROJECT --out OUT
```

The launcher opens `~/Documents/TubeTracker-annotator-projects/rev14analysis`.
Movie, model and snapshot locations are pinned in
`prototypes/v30_video_apex/analysis_release.json`. Back up annotation projects with
`scripts/backup_project_dbs.py`; never keep the only copy in a temporary folder.
Known limits: the app is tied to one configured movie and field, grain discovery is
off, and it withholds automatic lengths.

## Legacy desktop engine (upstream TubeTracker)

`./Start_TubeTracker_local` opens the original wxPython application (Hough grains,
LapTrack linking, tip templates, CSV export). A command-line pilot run:

```bash
.venv/bin/python scripts/run_pilot.py MOVIE --sample-id ID --genotype WT \
  --biological-replicate plant-1 --time-per-frame 30 --pixel-size 0.8 --distance-unit um
```

## Repository layout

| Path | Contents |
|---|---|
| `sparsetrack/` | Keyframe-bin cache, registration, grain census, benchmark labelling tool |
| `benchmark/labels/` | Human benchmark labels (tracked) |
| `tubetracker/` | Upstream engine (`gui.py`, `analysis.py`, `models.py`, `views.py`) and the frozen v30 app modules |
| `prototypes/v30_video_apex/` | Model and solver modules used by the frozen app |
| `prototypes/timesfm_tip_forecast/grain_detect.py` | Radial grain detector used by the frozen app |
| `prototypes/LEDGER.md` | Research record, H1–H491 |
| `scripts/` | App and pilot entry points, annotation backup |
| `tests/` | Checks for the upstream engine, the annotation store/schema and the frozen analysis service |
| `runs/` (git-ignored) | Checkpoints and outputs; the frozen app's checkpoints live under `runs/prototypes/v30/` |

## Tests

```bash
.venv/bin/python -m pytest -q
```

Software tests do not establish biological accuracy; accuracy is measured against
human-labelled benchmarks.
