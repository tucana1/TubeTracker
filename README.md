# TubeTracker

TubeTracker is a research prototype for pollen germination time-lapse movies:
per grain, detect when a tube emerges and measure the tube's length over time.

## For the lab: analyse a movie, then check it

**Setup, once per Mac** (Python 3.10-3.14 and Homebrew):

```bash
brew install ffmpeg                    # reads the movies' keyframes
python3 -m venv .venv
.venv/bin/pip install -e ".[cnn]"      # SparseTrack and its tube network (torch)
```

1. **Analyse.** Double-click `Analyze_Movie_SparseTrack.command` and choose a movie, or
   several (Cmd-click; they are analysed in turn), or run `.venv/bin/python -m sparsetrack run MOVIE`.
   The first run prepares a movie (a few minutes); the analysis takes 5-20 minutes on a
   laptop, longer for long, crowded movies. A review gallery opens with the movie's result
   in one sentence (germinated share, T50, median growth rate), then one card per grain,
   the model's least sure grains first. Results are in `runs/sparsetrack/<movie name>/analysis/`:
   `grains.csv` (per grain: germinated or not, onset interval, final length, growth rate,
   the model's confidence), `growth.csv` (every grain's length at every time),
   `population.png` (germination curve with T50) and `growth_curves.png`. The first time,
   the launcher asks for the pixel size and frame interval (Enter skips) and keeps them in
   `calibration.json`; with them the tables also give onsets in minutes, lengths in um and
   growth in um/min (`sparsetrack run MOVIE --um-per-px 0.65 --s-per-frame 30` does the same
   for one run).
2. **Check and correct.** Double-click `Review_Movie_SparseTrack.command` and choose the
   same movie (or `.venv/bin/python -m sparsetrack review MOVIE`). The labelling tool
   opens with the model's answers already filled in: each grain's onset bracket, and its
   tube traced at a few times. **Enter** confirms an answer as it stands; **-** and **=**
   shorten or lengthen a traced tube along the model's route (Shift: 5 px); otherwise fix it
   as you would label it (click the first bin where the tube is visible; click along the
   tube from the grain to its tip). The bar at the top counts what you have checked, and
   "next unfinished" goes to the next answer still the model's. Grains come least sure
   first: the model's confidence in a reading rises with the tube's length and falls the
   longer the reading has stood still (fitted on one labelled movie, checked on the other).
   Checking in that order brought both movies to 79% of traces within tolerance (the
   annotator's own repeatability) after checking 28% and 41% of the traces, against 48%
   and 65% in random order. Confident answers can still be wrong: check them too when
   the numbers matter. Answers are saved as you
   go; press Ctrl-C in the window when you stop. `runs/sparsetrack/<movie name>/review/`
   then holds `reviewed_grains.csv`, `reviewed_traces.csv` and `population.png`, saying
   which answers you checked and which you changed, and `reviewed_growth.csv` /
   `growth_curves.png`: every grain's length at every time, the model's curve pinned to the
   lengths you checked, in um and minutes too with `calibration.json`. Run it again to carry on.

**If the gallery opens with a red warning** that many grains could not be followed, the grains in
that movie drift or are still landing after the first minutes: SparseTrack reads each grain at its first
place, so a grain's results after it moved are wrong. Check those grains (flag `drift_rejected`) in the
review, or use a movie whose grains have settled. Following moving grains is being worked on.

**How far to trust the model unchecked** (29 Sep 2026, SparseTrack 0.6.0, against one
annotator's traces; length within max(2 px, 10%), onset within 2 bins; both movies were
used in development, so a third, blind-labelled movie is the honest test):
- sparse movie: lengths 69/104 (66%), onsets 15/26 (58%). The same annotator
  repeating 15 traces blind agreed with themself on 11/14 lengths (79%) and on 4/7 onsets.
- crowded movie 2: lengths 25/54 (46%), onsets 5/19 (26%). Tubes that touch or cross
  other tubes, and very young tubes, are the hard cases; check those first (the gallery
  and the review open with the model's least sure grains).
- the germination curve holds up better than single onsets, whose errors partly cancel:
  T50 within about one bin of the annotator's on the sparse movie (7735 vs 7383 frames;
  96% vs 100% germinated by the end) and 2.3 bins on movie 2 (24752 vs 24049 frames; 95%
  both); the curves differ by at most 0.14 and 0.19. `sparsetrack eval` reports this.

**Status (29 Sep 2026; details in `docs/status-2026-09-29.md`).** SparseTrack (below) is the
tracker. The cloud session's learned-evidence pipeline (`prototypes/learned_evidence/`, merged
29 Sep) was tested on the lab's movies and read fewer lengths than SparseTrack (ld 40/100 vs
69/104); it is kept as a research record. Work concentrated on isolated grains before crossings and
clumps (sparse-first reset, 23 Sep). The research record up to 22 Sep 2026 is in
`prototypes/LEDGER.md`; the complete pre-reset tree is preserved at the git tag
`snapshot-2026-09-23`.

## SparseTrack (active development)

`sparsetrack/` is the new sparse-field pipeline. The movies are x264 exports with a
keyframe every 12 frames at the encoder's quality floor, so it reads keyframes only
and works on registered averages of 25 keyframes (300 source frames per "bin").

```bash
# one step for any movie: cache (first time), analysis, review gallery in the browser
.venv/bin/python -m sparsetrack run MOVIE            # or double-click Analyze_Movie_SparseTrack.command
# build the cache: keyframe bins, registration, grain census (~30-40 s for the sparse movie)
.venv/bin/python -m sparsetrack prepare MOVIE --out runs/sparsetrack/ld
# benchmark labelling tool (local web page; answers saved to the labels file after every click)
.venv/bin/python -m sparsetrack bench runs/sparsetrack/ld --labels benchmark/labels/ld_v1.json
# per-grain onset and exit-to-apex length (~40 s for the sparse movie)
.venv/bin/python -m sparsetrack analyze runs/sparsetrack/ld --out runs/sparsetrack/ld_A [--grains benchmark/labels/ld_v1.json]
# score any predictions against the benchmark
.venv/bin/python -m sparsetrack eval --labels benchmark/labels/ld_v1.json --pred runs/sparsetrack/ld_A/predictions.json
# synthetic movie with exact truth on the real field (x264-encoded like the real movies), then its cache
.venv/bin/python -m sparsetrack synth runs/sparsetrack/ld --out runs/sparsetrack/synth --seed 0 --preset v2
.venv/bin/python -m sparsetrack prepare runs/sparsetrack/synth/synthv2_s0.mp4 --out runs/sparsetrack/synth/v2s0_cache --frames-per-bin 25 --ref-start 0
# score parameter variants on the synthetic seeds (by failure class) and the legacy real grains
.venv/bin/python scripts/synth_bench.py --suite v2 --breakdown --set evidence=matched
# ... or on the human benchmarks only, both movies in parallel (~2 min per variant), paired
# against a saved run with a bootstrap over grains
.venv/bin/python scripts/synth_bench.py --real --no-synth --no-legacy --dump-real base.json
.venv/bin/python scripts/synth_bench.py --real --no-synth --no-legacy --set tip_offset_px=2.5 --baseline base.json
# is it the route or the reading? the human's traced route as each grain's only path
.venv/bin/python scripts/oracle_route.py m2 contact_px=-1000
```

`analyze` works per grain, whole-movie and offline:
- **Registration and settling.** Local registration follows the grain as it drifts;
  a track that jumps or wanders (it has locked onto a neighbour) falls back to the
  field registration. Grains still landing in the census bins are read from when they
  settle, and detections with no grain rim are reported unobservable.
  `--set grain_track=auto` (or `follow`) follows each grain by its own look instead
  (`sparsetrack/track.py`: a bank of templates of the grain, matched within 8 px of its last
  place each bin, through passing blobs, crossings, pushes and changes of look, never onto a
  neighbour that is still at its place). A grain it can no longer find (burst, swept off,
  out of the frame) is lost from that bin: its readings are held from there, flagged
  `grain_lost_after:<frame>` (marked in the gallery), and an ungerminated lost grain is
  censored at that frame in the population. `follow` reads every grain in its own frame;
  `auto` only once it has moved further than its diameter, and by the phase track nearer
  (a tube stuck to the substrate stays sharp in the field while its grain is pushed a few px).
- **Path.** Candidate centrelines are traced on the end-of-movie change map, one per
  branch end and rim contact. The one kept is the candidate whose monotone growth from
  the exit explains the most evidence, weighted by how ridge-like its end-state
  cross-section is.
- **Length.** Growth is read backwards along that path with a non-decreasing
  dynamic-programming front. The grain and tube may rotate rigidly. The evidence is
  |change| combined with the change projected on the tube's own end-state cross-section.
- **Onset.** Onset is called by a matched stub filter at the exit (end-state exit and
  rotation track), with hysteresis.
- **Crowded or noisy grains** (reader `hybrid`, the default). Where a grain's change
  region touches a neighbour, change evidence picks up foreign tubes; where the background
  change is noisy enough to lift the tube-map threshold above its floor, it picks up bands
  and blobs. Such grains are read instead from learned tube probabilities
  (`sparsetrack/learned.py`; a grain that is only noisy keeps the change reader's
  germination call and onset, taken at its clean rim): a small U-Net
  trained only on codec-exact synthetic movies of the two benchmark fields, read by an
  arrival flood in which the tube claims only material that arrives at its own tip.
  The first run on a movie writes its probability movie next to the cache (uint8, about
  half the cache's size; ~1-2 min on an Apple GPU). This needs torch (`pip install
  .[cnn]`); without it every grain is read from change evidence. `--set reader=change`
  or `reader=flood` choose one reader for all grains.

It writes:
- `predictions.json`, `grains.csv`, `growth.csv` and a diagnostic image per grain;
- `growth_curves.png` (small multiples);
- `population.csv`/`population.png` (interval-censored cumulative germination, Turnbull
  estimate, with T50);
- `index.html`, a review gallery. Grains whose flags ask for a second look are marked
  and come first, lowest path coverage first. Path coverage is the share of the grain's
  own change region lying within 6 px of its traced path. A tube that curls, turns back,
  wraps round its grain or shares a region leaves much of it unexplained;
- with `--video`, `field_overlay.mp4`.

Scores so far are in `benchmark/reports/`.

The synthetic presets:
- **v1:** clean isolated tubes.
- **v2:** adds foreign tubes from clumps, crossings, curls, pauses and stops, drifting
  grains and docking particles.
- **v3:** adds tubes that start dark and turn bright-cored, tubes stuck to the substrate,
  landing grains and fat stubs.
- **v4:** adds sideways sway.

Seeds 3-4 of v2-v4 are held out.

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
