# WT / LLG pilot guide

This guide covers the data needed for the first WT versus LLG-overexpression
evaluation and the reproducible analysis workflow. It prioritizes germination
time and tube growth; burst candidates remain reviewer suggestions.

## Data request

Request a small representative set before asking for the entire experiment:

- Two or three WT videos and two or three LLG-overexpression videos.
- One clean/easy video and one difficult video from each genotype.
- Original or losslessly exported files when possible.
- Unique filenames without spaces or special characters.

Suggested filename:

```text
YYYYMMDD_genotype_plant-replicate_field-replicate.avi
```

### Required metadata

Collect the following for every video:

- Genotype and transgenic line.
- Plant or biological replicate identifier.
- Field/video technical replicate identifier.
- Imaging date/session and microscope/camera setup.
- Biological time represented by one frame.
- Pixel-to-micrometer calibration at the recorded magnification.
- Definition of time zero: media addition, recording start, or another event.
- Any frame averaging, downsampling, cropping, rotation, or contrast processing.
- Temperature and treatment conditions.

Playback FPS in an AVI/MP4 is not necessarily the biological frame interval and
must not be substituted for it.

### Manual reference measurements

For one WT and one LLG video, request:

- Germination frame for approximately 10 clearly visible grains.
- One-hour tube length for the same grains, traced in ImageJ if possible.
- Notes identifying grains that leave the field, overlap, burst, or become
  impossible to follow.

These annotations provide an initial accuracy check; they are not intended to
be a full training dataset.

Ask how many independent plants, imaging sessions, and videos are available.
Pollen grains within one video are nested observations and should not be treated
as independent biological replicates without an appropriate model.

## Analysis protocol

### 1. Intake

Place incoming videos under `data/raw/`. This directory is ignored by git. Do
not rename or alter the originals after analysis begins.

Inspect each video:

```bash
.venv/bin/python scripts/inspect_video.py data/raw/video.avi \
  --output-dir runs/intake/video
```

Review `contact_sheet.jpg` for focus drift, lateral movement, overlap, changing
illumination, and particles leaving the field. Complete the missing biological
time and pixel calibration fields in the associated laboratory metadata record.

### 2. Parameter tuning

Start with a representative WT and LLG video. Use the GUI to tune background
cutoff, blur radius, grain radius/threshold, and tip detection. Use one shared
parameter set across genotypes unless an imaging-session difference is
documented and scientifically justified.

Grain radius and tip side-length parameters refer to the resized analysis frame
(1000 x 725 by default), matching the desktop GUI. Do not use genotype-specific
settings merely because one genotype produces a different phenotype.

### 3. Reproducible pilot run

Run the headless pipeline with explicit metadata:

```bash
.venv/bin/python scripts/run_pilot.py data/raw/video.avi \
  --sample-id 20260710_WT_plant1_field1 \
  --genotype WT \
  --biological-replicate plant1 \
  --imaging-session 20260710 \
  --time-per-frame 30 \
  --time-unit sec \
  --pixel-size 0.8 \
  --distance-unit um
```

Every run records its parameters, source metadata, git revision, and whether the
working tree contained uncommitted changes.

Rupture suggestions are off by default. Add `--burst-candidates` only when that
experimental reviewer-only output is intentionally being evaluated.

### 4. QC review

Review at minimum:

- Every grain marked `review_not_germinated` in `pilot.grains.csv`.
- Broken, implausibly short, or jumping tracks in the annotated track video.
- Grain-to-track assignments in `pilot.tracks.csv`.
- Any orange burst candidates; these are not accepted burst events.
- Differences between automatic calls and the manually annotated reference
  grains.

Record corrections rather than silently changing parameters after seeing the
genotype result.

#### Burst review controls

Rupture review is intentionally downstream of the primary analysis. In the
desktop application, first run `Find Grains`, `Find Tips`, `Track Tips`, and
`Track Germination`. Then use the `Burst Review` panel:

1. Click `Find Candidates`.
2. Select a candidate to jump to its proposed rupture frame. The application
   enables the `Grain status` overlay and marks the candidate in orange.
3. Inspect the frames immediately before and after the suggestion.
4. Click `Confirm` only when a rupture is visually supported, or `Dismiss` when
   it is not.

Confirmed events are saved as reviewed burst events. A visually apparent event
that was not suggested can still be recorded with `Add burst frame` under
manual tracking. Candidate scores remain experimental and must not be reported
as confirmed rupture measurements without this review.

#### Session recovery and export

The `Session` menu provides three recovery levels:

- `Restart Tracking` clears tip tracks, germination calls, and burst review but
  keeps detected grains and tips.
- `Restart Analysis` clears all detections and results while keeping the loaded
  video and current parameters.
- `Start Over` closes the current video and clears its unsaved analysis state.

Use `File > Choose Output Folder` once per session. The selected location is
shown in the status bar. `Save All Results` writes the complete TubeTracker
output set, while `Export Coordinates CSV` writes one tidy table containing
grain and track IDs, frame, biological time, original-pixel coordinates,
cumulative tip movement, interval growth rate, and detection method.

### 5. Primary outputs

- `pilot.grains.csv`: one row per grain with germination time and QC state.
- `pilot.tracks.csv`: tip position, cumulative length, and interval growth rate.
- `*.survival.raw.data.csv`: native TubeTracker grain/event export.
- `*.survival.curves.csv`: fraction germinated over time.
- `*.tracks.raw.data.csv`: native TubeTracker tip trajectory export.
- `*.tracks.synchronized.csv`: track lengths aligned to first detection.
- `*.tracks.unsynchronized.csv`: track lengths on the original movie timeline.
- Annotated AVI files for visual verification.
- `run_manifest.json` and `summary.json`: provenance and run-level counts.

### 6. Scientific validation before full analysis

Compare automatic and manual measurements for the reference grains:

- Germination-frame error.
- One-hour length error.
- Missed and false germination calls.
- Track fragmentation and identity switches.
- Manual correction time per video.

TubeTracker length is based on cumulative tip movement. Validate it against
manual curved-tube traces before treating it as interchangeable with physical
tube contour length.
