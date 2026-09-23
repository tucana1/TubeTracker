# TubeTracker

TubeTracker is a research prototype for pollen germination time-lapse movies:
per grain, detect when a tube emerges and measure the tube's length over time.

**Status (23 Sep 2026): sparse-first reset.** Work now concentrates on isolated
grains before crossings and clumps. The research record up to 22 Sep 2026 is in
`prototypes/LEDGER.md`; the complete pre-reset tree (all prototype generations,
round scripts and their tests) is preserved at the git tag `snapshot-2026-09-23`.

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
