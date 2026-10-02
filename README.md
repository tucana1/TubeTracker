# TubeTracker

TubeTracker is a research prototype for pollen germination time-lapse movies:
per grain, detect when a tube emerges and measure the tube's length over time.

## For the lab: the TubeTracker app

**Setup, once per Mac** (Python 3.10-3.14 and Homebrew):

```bash
brew install ffmpeg                    # reads the movies' keyframes
python3 -m venv .venv
.venv/bin/pip install -e ".[cnn]"      # SparseTrack and its tube network (torch)
```

**Open it** by double-clicking `TubeTracker.command` (or `./Start_TubeTracker_local`, or
`.venv/bin/python -m tubetracker`). Everything happens in its window:

1. **Open** a movie (File > Open Movie, or drop it on the window; .mp4 or .avi), or one analysed before from the
   start screen.
2. **Settings** (asked the first time, Movie > Settings later): how long the movie ran (the time per frame
   follows from its frame count), the pixel size in um (optional), sample ID, genotype, replicate. Times are then
   in minutes and lengths in um throughout.
3. **Analyse.** SparseTrack runs in the background, step by step with a progress bar and Cancel: a few minutes
   the first time a movie is read, then roughly 10-30 s per grain. The movie opens when it is done.
4. **Look.** The side panel shows the movie at a glance (grains counted, germinated, T50, growth, checks left).
   Every grain is outlined by its state at the time shown (germinated, not yet, never, lost partway, excluded),
   with its tube drawn to its length then along the tube's middle, the tip and the exit; grains and their tubes
   move with the grain as it drifts. Play, scrub, scroll to zoom, drag to pan; the Growth view shows what changed
   over the last six time steps, where growing tips stand out. The timeline under the movie holds the germination
   curve with T50, germinations, grains lost partway, tubes that stop growing, the grains to check and focus
   changes; click one to go there.
5. **Check.** N goes to the next grain to check, the model's least sure first, framed with its tube at the time to
   look at. The side panel says why (click a reason to go to its time). Enter confirms; then by what is wrong:
   Onset (O here, Shift-O never), Tube (T and a click set the tip, its length read along the tube; D draws it;
   None here) and Grain (B burst, X not a grain, K clump); right-click a grain for the same. Cmd-Z undoes, Esc
   goes back to the movie's numbers. Answers are saved at once. Help > TubeTracker Help (F1) explains everything
   on screen, the numbers and the files.
6. **Export** (File > Export Results): `results/grains.csv` (per grain: onset, final length, growth rate, the
   model's confidence, whether you checked it, the sample's metadata), `results/growth.csv` (length at every
   time), `germination.png` (with T50) and `growth_curves.png`. File > Results shows the same in the app;
   File > Compare Movies puts several movies side by side (`runs/sparsetrack/summary/`).

Each movie has a folder `runs/sparsetrack/<movie name>/`: `setup.json`, `cache/` (keyframe bins, grain census),
`analysis/` (SparseTrack's `predictions.json` and tables), `review/review_labels.json` (your checks, in the
labelling tool's format; `reviewed_*.csv` after an export) and `results/`. A folder made on the command line
(`sparsetrack run MOVIE`) opens the same way. Reviewing least sure first brought both labelled movies to 79% of
traces within tolerance (the annotator's own repeatability) after checking 28% and 41% of the traces, against 48%
and 65% in random order; confident readings can still be wrong.

The command-line tools remain for batch work and research: `sparsetrack run MOVIE` (or
`Analyze_Movie_SparseTrack.command`, several movies in turn) writes the same `analysis/` with an HTML gallery,
`sparsetrack review MOVIE` opens the web labelling tool on it, `sparsetrack summary` compares movies.

**Grains that move** are followed (SparseTrack 0.7.0): each grain is tracked by its own look from bin
to bin, and one that moves further than its own diameter is read where it is. A grain that is knocked (it jumps and
turns over in a bin, as in movie 1) can be searched for again by its last look, turned (`--set track_refind=true`;
off until measured on all three movies, docs/status-2026-09-29.md). When labelling, the views follow the tracker
wherever it has the grain; where it lost one, the tool says so, and **G** then a click on the grain says where it is
now (the views follow it from there; Shift+G takes it back). A grain that can no longer be
found (it burst, drifted out of view or was swept off) is read until then and its numbers are held from
there, flagged `grain_lost_after`. **If the gallery opens with a red warning** that many grains were lost
partway, their final lengths and growth are only known up to that time; check them in the review. Tubes that
swing or turn on their own while their grain stays put are still read in a fixed place. The gallery also warns
when **the movie's focus changed** (movie 2 was out of focus from about bin 13 to bin 63): tubes that emerged
meanwhile are seen only once it is sharp again, so onsets at that time mean "visible by" (flag
`onset_at_focus_change`).

**How far to trust the model unchecked** (30 Sep - 1 Oct 2026, SparseTrack 0.8.1-0.8.3, against one annotator's traces; length
within max(2 px, 10%), onset within 2 bins). The tube network was fine-tuned on the first two movies' traces, so the
honest numbers for those come from the same recipe trained on the *other* movie's traces:
- sparse movie: lengths 73/104 (70%), length and tip 58, onsets 14/26; growth rate per grain within
  max(0.1 px/bin, 20%) of the annotator's for 25 of 27 grains (correlation 0.91). The same annotator repeating 15
  traces blind agreed with themself on 11/14 lengths (79%) and on 4/7 onsets.
- crowded movie 2: lengths 24/54 (44%), length and tip 22, onsets 4/18; growth rates 10 of 18 grains within
  tolerance, correlation 0.72. Tubes that cross or touch others, and very young tubes, remain the hard cases. Movie 2
  was out of focus from about bin 13 to bin 63, so its onsets there mean "visible by".
- movie 1, labelled blind (30 grains, 50 traces; its grains drift and get knocked, and some tubes grow over their
  grain from a pore facing the camera): the version fixed beforehand (0.8.0) read 8/50 lengths and 3/12 onsets. The
  cause was one rule: on a noisy background the change reader's "no tube yet" overrode the tube network's reading.
  0.8.1 keeps the network's own call there: 15/50 lengths (18/50 measured from the grain's edge), onsets 8/26, T50
  9304 vs 8602 frames, growth rates 7/18 (correlation 0.31); on the other movies lengths +0 and +1. This fix came
  from movie 1's scores, so these numbers are no longer blind (`benchmark/reports/m1_v1_frozen.md` has the blind
  ones). Movie 1 is a hard movie; check its grains in the app.
- the germination curve holds up better than single onsets, whose errors partly cancel: T50 within about one bin
  of the annotator's on the sparse movie and 2.3 bins on movies 2 and 1. `sparsetrack eval` reports it and the
  growth-rate agreement.

**Status (1 Oct 2026; details in `docs/status-2026-09-29.md`).** SparseTrack 0.8.3 (below) is the
tracker, and the TubeTracker app (above) is its desktop front end. The cloud session's learned-evidence pipeline
(`prototypes/learned_evidence/`, merged 29 Sep) was tested on the lab's movies and read fewer lengths than
SparseTrack (ld 40/100 vs 69/104); its code is kept as a research record (its large outputs were removed). Work concentrated on isolated grains before
crossings and clumps (sparse-first reset, 23 Sep). The research record up to 22 Sep 2026 is in
`prototypes/LEDGER.md`; the complete pre-reset tree is preserved at the git tag `snapshot-2026-09-23`.

## SparseTrack (active development)

`sparsetrack/` is the new sparse-field pipeline. The movies are x264 exports with a
keyframe every 12 frames at the encoder's quality floor, so it reads keyframes only
and works on registered averages of 25 keyframes (300 source frames per "bin").

```bash
# one step for any movie: cache (first time), analysis, review gallery in the browser
.venv/bin/python -m sparsetrack run MOVIE            # or double-click Analyze_Movie_SparseTrack.command
# build the cache: keyframe bins, registration, grain census (~30-40 s for the sparse movie); --sample all or N
# averages every (N-th) frame instead of the keyframes (no gain on x264 movies: docs/status-2026-09-29.md)
.venv/bin/python -m sparsetrack prepare MOVIE --out runs/sparsetrack/ld
# benchmark labelling tool (local web page; answers saved to the labels file after every click)
.venv/bin/python -m sparsetrack bench runs/sparsetrack/ld --labels benchmark/labels/ld_v1.json
# per-grain onset and exit-to-apex length (~40 s for the sparse movie)
.venv/bin/python -m sparsetrack analyze runs/sparsetrack/ld --out runs/research/ld_A [--grains benchmark/labels/ld_v1.json]
# score any predictions against the benchmark
.venv/bin/python -m sparsetrack eval --labels benchmark/labels/ld_v1.json --pred runs/research/ld_A/predictions.json
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
  With `grain_track=auto` (the default from 0.7.0; `phase` is the old behaviour, `follow`
  reads every grain in its own frame) each grain is followed by its own look instead
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
  Where the front reached the path's end well before the movie ends and new material keeps
  arriving beyond it in order outwards (a tube turning back along its grain, which no path
  from the rim reaches the long way), a second front reads that continuation (`tip_continue`,
  0.7.0: ld 69 -> 73/104, synthetic movies +10/647).
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

The original wxPython application (Hough grains, LapTrack linking, tip templates, CSV export) opens from
TubeTracker's Tools > Legacy Manual Pipeline (or `.venv/bin/python -m tubetracker --legacy`). A command-line
pilot run:

```bash
.venv/bin/python scripts/run_pilot.py MOVIE --sample-id ID --genotype WT \
  --biological-replicate plant-1 --time-per-frame 30 --pixel-size 0.8 --distance-unit um
```

## Repository layout

| Path | Contents |
|---|---|
| `sparsetrack/` | Keyframe-bin cache, registration, grain census, benchmark labelling tool |
| `benchmark/labels/` | Human benchmark labels (tracked) |
| `tubetracker/app/` | The TubeTracker app (window, data layer, corrections, exports) |
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
