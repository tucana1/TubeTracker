# TubeTracker

TubeTracker is a research prototype for reviewing pollen grains and their tubes
in time-lapse microscopy movies. The supported rev14 workflow is the native
movie-analysis app described here. Earlier pipelines remain below as experiment
history and reproducible baselines.

## Supported movie-analysis prototype

On the configured workstation, open `Start_TubeTracker_Analysis.command`, or run:

```bash
.venv/bin/python scripts/launch_tubetracker_analysis.py
```

The launcher opens the existing project at
`~/Documents/TubeTracker-annotator-projects/rev14analysis` with nine requested
human annotations. It uses `.venv-annotator` for the native UI and `.venv` for
inference. Movie, model and snapshot locations/hashes are pinned in
`prototypes/v30_video_apex/analysis_release.json`; preserve those artifacts when
moving the project to another workstation. A read-only launch preview is
available with `--print-command`.

Use **Run / Recompute**, select a grain/frame result, review its tip or whole
path, and export after recomputing. The **Open requested annotations** queue
asks for three entire grain-exit-to-cap traces, two local crossing reviews, two
whole visible owned-body masks, one exact non-cap polygon task and one exhaustive
grain census. It states the target grain, source frame and field for each task.
Later frames can establish identity; annotations describe the current frame.

Each physical grain keeps its own identity, including grains in touching clumps.
FULL paths, PARTIAL paths, precise tips, hidden states, scoped background and
Unknown pixels have separate meanings. Grain-count completeness never licenses
tip-background training. All genuine caps are positive for the appearance
detector; ownership is resolved separately. Missing cadence or spatial scale
leaves physical rates null.

The same analysis service is available without the UI:

```bash
.venv/bin/python scripts/analyze_movie.py \
  --config "$HOME/Documents/TubeTracker-annotator-projects/rev14analysis/analysis-config.json" \
  --project-dir "$HOME/Documents/TubeTracker-annotator-projects/rev14analysis" \
  --out runs/prototypes/v30/reviewed-export
```

The application persists corrections and their revisions, checks movie/model/
source identities, and invalidates stale results. An explicit `--verification-only`
project is for synthetic workflow checks; its edits are excluded from biological
training, route validation and certified population coverage.

**Current accuracy limit:** automatic cap rejection, owned-body fit and genuine
whole-route validation have not all passed. The recorded three-grain interval
has 45 owner/frame rows and zero complete automatic lengths. A selected tip alone
does not prove the intervening tube. The app withholds unsupported measurements;
its software tests do not establish biological accuracy. The nine genuine tasks
are prepared but still unanswered. See `prototypes/LEDGER.md` and the evidence
under `runs/prototypes/v30/rev14_completion_run/` for measured failures.

`snap29_rev14` preserves the seven biological record files from snap27 byte for
byte, excludes workflow evidence and keeps census proposals unconfirmed. The
pinned release remains on snap27 until a deliberate, verified release update.
Back up live annotation projects with `scripts/backup_project_dbs.py`; never keep
the sole copy in a temporary folder.

Run backend and native-widget checks in their respective environments:

```bash
.venv/bin/python -m pytest tests -q
.venv-annotator/bin/python -m pytest tests/test_analysis_queue_widgets.py -q
```

## Historical workflows

The legacy desktop engine (`tubetracker/gui.py`, `tubetracker/analysis.py`),
material/global-ribbon pipelines, owner-conditioned video graph and growth-front
experiments below retain their original commands and evidence. Claims called
"current" within those experiment descriptions refer to that historical version;
they do not supersede the rev14 gates above. CNN point tools remain available for
reproducing older experiments.

These are not three separate copies of the application. They share the same
models, image-processing utilities, calibration rules, and output conventions.

## Legacy desktop launcher

On macOS, run:

```bash
./Start_TubeTracker_local
```

The launcher creates `.venv`, installs the pinned dependencies, and opens the
desktop application. On the first screen:

1. Select a microscopy video.
2. Enter the time represented by one source frame.
3. Enter the image scale, if known, so lengths can be reported in physical units.
4. Choose an output directory.
5. Review the detected grains and tracking overlay before accepting the export.

Run the automated checks with:

```bash
.venv/bin/python -m pytest
```

## Legacy analysis

The legacy desktop interface predates the supported native review workflow. Its
workflow detects pollen grains, links them through time with LapTrack, records
germination and tip motion, estimates centerline length, and exports detailed and
summary CSV files. OpenCV handles video decoding and image processing.

For a reproducible command-line pilot run:

```bash
.venv/bin/python scripts/inspect_video.py data/raw/example.avi \
  --output-dir runs/intake/example

.venv/bin/python scripts/run_pilot.py data/raw/example.avi \
  --sample-id example \
  --genotype WT \
  --biological-replicate plant-1 \
  --time-per-frame 30 \
  --pixel-size 0.8 \
  --distance-unit um
```

The pilot run records its inputs and settings in `run_manifest.json`. It also
writes grain detections, frame-by-frame measurements, one-row-per-track
summaries, review media, and quality-control fields. Keep the manifest with the
CSV outputs so an analysis can be reproduced later.

## Material-Curve Pipeline

The maintained research pipeline follows one selected pollen and the complete
tube attached to it. CoTracker follows multiple points on the pollen body so
pollen movement is separated from tube growth. In pollen-fixed coordinates, the
pipeline then:

- learns the pre-germination background;
- requires a new tube to begin at the selected pollen;
- keeps an ordered root-to-tip centerline through time;
- uses tube-shaped image evidence and paired tube boundaries to refine the line;
- uses earlier material points and incoming direction to avoid switching tubes
  at crossings;
- allows only the distal end to add length; and
- marks uncertain frames for review instead of silently reporting a new path.

Install the optional point tracker:

```bash
.venv/bin/pip install -e '.[pollen-motion]'
```

Prepare a raw video once, then inspect `pollen_catalog.jpg` and choose the ID of
the pollen to analyze:

```bash
.venv/bin/python scripts/prepare_material_video.py data/raw/example.mp4 \
  --sample-count 480 \
  --output-dir runs/cache/example
```

Run the prepared cache and pollen catalog with calibrated values:

```bash
.venv/bin/python scripts/run_pollen_anchored_pipeline.py \
  --gray-cache runs/cache/P0034/gray_samples.npy \
  --source-frames-cache runs/cache/P0034/source_frames.npy \
  --grain-cache runs/cache/P0034/grain_tracks.npz \
  --candidate-catalog runs/cache/P0034/pollen_candidates.csv \
  --pollen-id P0034 \
  --sample-count 240 \
  --output-sample-count 40 \
  --time-per-source-frame 1.0 \
  --pixel-size 1.0 \
  --distance-unit px \
  --trace-mode material \
  --install-model \
  --output-dir runs/current/P0034
```

Replace the example calibration values with the microscope's real frame interval
and pixel size before interpreting growth rates or physical lengths. Tracking can
use more frames than are included in the scientific output; this preserves
identity while allowing a smaller set of evenly spaced measurement time points.

The material run writes:

- `measurements.csv`: every tracked frame and its length, state, and quality;
- `selected_measurements.csv`: the requested reporting time points;
- `centerline_points.csv`: every point in every accepted centerline;
- `selected_centerline_points.csv`: centerlines at reporting time points;
- `summary.csv`: one compact summary for the selected pollen;
- `review.mp4`: the centerline, pollen anchor, and quality state overlaid on video;
- `run_manifest.json`: parameters, versions, inputs, and run counts; and
- point-track caches that make an identical rerun faster.

### Growth-Front Research

The optional growth-front analysis converts the traced tube into an
arclength-by-time kymograph and fits one globally monotone growth trajectory.
This can bridge transient blur and expose crossing material beyond the inferred
tip without allowing it to rewrite the attached tube.

```bash
.venv/bin/pip install -e '.[research]'

.venv/bin/python scripts/prototype_growth_front.py \
  --cache-dir runs/cache/P0034 \
  --run-dir runs/current/P0034 \
  --output-dir runs/research/P0034
```

This remains an offline prototype validated on only a small number of targets;
its results must be reviewed alongside the causal material pipeline.

### Birth-Time Topology Research

The current experimental tracker treats a growing tube as a connected birth
event. It removes common field motion, ignores material already present during
warmup, requires new evidence to persist locally, and recovers one simple path
from the pollen to the tip. This prevents a trace from accumulating loops around
tube walls and makes the sampling cadence part of the result.

```bash
.venv/bin/python prototypes/v17_birth_topology/track.py \
  --movie data/raw/example.mp4 \
  --source-start 0 \
  --sample-seconds 3 \
  --source-frame-interval-seconds 0.5 \
  --pixel-size 0.1 \
  --distance-unit um \
  --output-dir runs/prototypes/v17/example
```

Replace the example frame interval and pixel size with microscope calibration
values. If they are omitted, the output explicitly uses encoded playback time
and native pixels instead of presenting them as experimental units.

The run writes an accepted-events review video, separate labeled images for
accepted and review-required events, a birth-time map, a compact event summary
CSV, per-timepoint length and tip measurements, full raw and monotone path birth
coordinates, and a JSON report containing every threshold and cadence verdict.
All pollen-bound alternatives remain in the CSV files even when a quality flag
keeps them out of the automatic result set. `prototypes/LEDGER.md` records the
results and failure mechanisms of every research iteration.

For a primary germination whose recovered path is credible but whose
threshold-derived tip jumps too far, v17 can first fit a second temporal front
along that already-fixed path to rescue timing-only failures. After trajectory
selection is complete, the same fit is applied to every accepted path so that
ordinary and rescued trajectories receive the same temporal test. It uses
warmup-relative image evidence, cannot move backward or leave the selected
tube, and cannot advance farther than the configured biological step limit.
Geometry, pollen ownership, germination timing, and the accepted germination
count are frozen before this refinement. Automatic validation requires direct
and eventual image support across at least 80% of the path; lower-confidence
fits remain exported as withheld candidates. `event_measurements.csv` keeps the
reported and threshold-derived time series side by side, while
`event_paths.csv` retains raw, isotonic, candidate-front, and selected tip times
for every path point.

Every detailed tip row also records its evidence state. `threshold-confirmed`
means the persistent birth map has reached that point. An earlier temporal-front
tip is `front-direct-supported` when local image evidence is present at its
fitted arrival, `front-later-supported` when the evidence appears only before
the later threshold confirmation, or `front-inferred` when neither check finds
local support. `tip_measurement_accepted` remains the filter for scientific
frame-by-frame analysis; all other states stay in the exports for review. Exact
confirmation frames and remaining confirmation lags are included in both sample
and time units.

After all ordinary and short-path promotions, every selected trajectory must
also admit a supported, outward-only temporal front along its complete frozen
path. This validation does not anchor the first measurable prefix to a synthetic
zero-length tip: exact onset validity remains a separate claim. It does enforce
the configured step limit between later tip positions and requires at least 80%
direct and eventual path support. When the fit passes, it becomes the reported
tip timeline only if every earlier tip that it would expose has direct or later
image support before threshold confirmation. A fit may therefore validate the
path while leaving the conservative threshold timeline selected; an unsupported
inferred tip is never promoted automatically. A validation failure clears only
`trajectory_accepted`; the germination decision, path, candidate timing, and
detailed measurements remain available for review.

`front_refinement_timelines.jpg` places every applied, validated, or rejected
front candidate on raw temporal crops. The red dot is the fitted tip, the blue
point is the root, and the yellow ring is the original threshold-derived tip.
This sheet is the required visual check before treating a newly promoted
trajectory as a scientific measurement.

`germination_review_timelines.jpg` shows eight time-ordered landmarks for every
automatic or warmup-withheld primary event: start, warmup, a view before the
onset evidence, the connected-root cue, the pollen-rim cue, the oriented local
contrast cue, the first measurable tube, and the final state. Hollow markers
locate the eventual path without covering faint raw tube evidence. A tube
already supported during warmup is reported as
`left-censored` when both its connected prefix and paired boundaries agree, or
`warmup-ambiguous` when only one cue agrees. These states withhold the onset
from automatic germination counts without deleting its measurements or an
independently valid tip trajectory.

The exports keep germination existence, onset timing, and tip tracking as three
separate claims. `germination_accepted` identifies events that can be counted as
germinations. `germination_onset_time_accepted` additionally requires the
connected-root cue to agree closely with either the rim cue or a
background-subtracted contrast change measured along the first tube segment.
The contrast cue compares the eventual path with pixels on both local flanks,
suppressing uniform focus and illumination changes. `germination_onset_quality`
records whether root-rim, root-contrast, or rim-contrast cues support the
timestamp, or whether agreement is low, unavailable, or reversed. Agreement
between the rim and oriented-contrast cues can timestamp an already-accepted
germination when connected centerline evidence matures later; it cannot create
a germination or validate a tip. `trajectory_accepted` identifies a fully
trusted path. `contact_bridge_trajectory_accepted` identifies a separately
validated prefix that crosses one foreign pollen and is reacquired on the far
side, while `precontact_trajectory_accepted` identifies a prefix ending before
an unresolved contact. `trajectory_measurement_scope` distinguishes `full`,
`contact-bridged-prefix`, `pre-contact`, and `none`;
`tip_measurement_accepted` is the definitive row-level filter for frame-by-frame
length and tip analysis. The
`germination_onset_*` columns report the conservative consensus and the span of
the optical cues used to accept or review it;
`measurement_start_*` reports when the connected centerline first reaches the
configured minimum length. Legacy `germination_*` fields remain aliases for
measurement start. Detailed rows label the intervening period
`emerged-below-measurement`, keep length at zero and tip coordinates blank, and
never invent a tip from an onset cue alone. When the onset cues remain split,
the exact onset and delay fields stay blank and pre-measurement rows are labeled
`onset-timing-review` rather than reporting an unsupported midpoint.
`germination_onset_timing_scope` makes this distinction machine-readable:
`point-estimate` is eligible for exact-time analysis, while
`optical-review-window` spans every observed root, rim, and local-contrast cue
for an accepted germination whose cues do not agree. That window narrows manual
review; it is not a biological confidence interval, because visible contrast
may mature after the tube was constructed.

A contact bridge is accepted only when the tube has a validated approach
trajectory, enters and exits one foreign pollen with no more than a 45-degree
total direction change, and no independently plausible pollen root explains
the same far-side path. The reacquired segment must then contain at least five
temporally resolved growth steps and independently pass the same direct,
eventual, pointwise, and tip-motion tests as an ordinary trajectory. The search
stops at the first support loss or second pollen contact. Thus a successful
bridge does not authorize the remainder of a connected image component:
`contact_bridge_last_path_index`, its censor time, two-sided angles, competing
root count, and pre/post-contact evidence are all exported for review.

Automatic germination also requires a persistent change in the exact pollen-rim
sector where the recovered path begins. That rim change must fall within the
same temporal horizon used to connect the event; an older or later change cannot
be borrowed to validate an unrelated tube. The summary CSV includes the rim
change frame, signed offset, and timing error in both samples and seconds so the
decision remains inspectable rather than hidden behind a pass/fail flag.

The tracker also checks whether that rim change is direction-specific. It
rotates the same short path probe into rim sectors at least 60 degrees away,
follows the pollen's measured translation, and requires a continuous new-
evidence chain rather than isolated dark pixels. Nearby rotations are excluded
because they can sample the opposite wall of the same tube. A spatially
independent chain appearing within the configured onset window marks the rim
change as ambiguous. This withholds only a short, otherwise unsupported
germination claim; a longer validated tube remains accepted with the competing
sector recorded for review. The summary CSV exports the selected-chain time,
competing rotation and time, competitor count, and final verdict.

After pollen ownership and the final path are fixed, the first 8 pixels outside
the pollen must also admit a physically bounded, outward-moving temporal front.
The path identity and shape stay fixed, while this short pollen-adjacent sampling
strip follows the measured translation of its pollen body between frames. A
counterfactual fit also holds the same strip fixed in the stabilized field. This
separates tube evidence that remains attached to the pollen from a pre-existing
or unrelated structure that only happens to intersect its reference position.
The summary records both support fractions and labels each comparison
`both-supported`, `pollen-following-only`, `static-only`, or
`neither-supported`. A `static-only` result cannot establish pollen attachment.
The check may advance by no more than the existing tip-step limit and requires
the same 80% direct and eventual image support used by the full-path front. A
failure moves only the germination claim to review with
`unresolved-proximal-emergence`; the path, any independently valid tip
trajectory, and all measurements remain exported. The summary records the
maximum pollen-reference motion used by the check, and
`proximal_emergence_review.jpg` shows every withheld case at full cadence around
its proposed onset, alongside every passing case under the same evidence window.
Large gray rings mark the fixed reference strip; smaller blue and yellow rings
mark its pollen-following root and end. This keeps grain edges, pre-existing
stubs, and unrelated dark objects visible to the reviewer.

The first pixels outside the pollen can still be poor ridge measurements because
they overlap the curved pollen boundary. A candidate withheld only for that
reason may therefore use a rim-bridged proximal check, but only after its complete
path and tip trajectory have independently passed every temporal and motion gate.
The pollen-rim change must coincide with a supported pollen-following front beyond
the configured grain-exit margin, an oriented on-path-minus-flank change must
confirm it within the existing temporal horizon, and no other quality flag may
be present. This can accept germination existence without inventing an exact
onset: disagreement among the root, rim, and contrast clocks remains an
`optical-review-window`. The summary exports the bridge decision, distal support
fractions, and fitted front interval.

Only this local attachment probe follows pollen motion. Reported tube paths and
tips remain in globally stabilized image coordinates. Across the two reference
videos, translating the complete path by 0%, 25%, 50%, 75%, or 100% of the
measured pollen-center displacement never improved complete-path temporal
support for any of the 22 tested trajectories. Large apparent center shifts can
reflect association between look-alike pollen rather than rigid motion of an
attached tube, so they are evidence to audit, not a transform to apply blindly.

`trajectory_review_timelines.jpg` shows every accepted full, contact-bridged, or
pre-contact tip trajectory at nine evenly spaced growth stages. It uses raw
crops with hollow
root and tip rings, without drawing over the intervening tube, so delayed tips,
off-tube positions, and crossover switches can be checked directly. Germinations
withheld by a separate onset review remain labeled `G-REVIEW`; partial rows are
labeled `BRIDGED` or `PRE-CONTACT`. Each panel labels the tip as confirmed,
directly
supported, later-supported, or inferred so temporal reconstruction is never
visually presented as an ordinary threshold detection.

`blinded_validation_timelines.jpg` provides a deterministic field-wide audit
without revealing tracker decisions on the image. Its companion
`blinded_validation_manifest.csv` separates accepted events, onset reviews,
rejected events, pollen with no recovered hypothesis, and pollen too close to
an image boundary for an ordinary measurement. Rows without an event span the
full recording rather than showing an arbitrary adjacent pair of frames. The
manifest contains blank human-review columns so precision and recall can be
scored before the hidden tracker stratum is consulted.

Event extraction retains any pollen-attached path long enough to establish the
configured germination distance; it does not silently discard a real short
emergence because the path is below the ordinary full-tube threshold. A short
path can become a trusted tip trajectory only after pollen ownership and onset
are fixed and the same temporal resolution, motion, growth-order, and image
support checks used for longer paths all pass. Component width is measured
against the complete component skeleton, so unrelated branches are not charged
to whichever root-to-tip branch happens to be selected.

A path that enters the width-scaled safety envelope around another pollen cannot
be promoted as a complete automatic tip trajectory. This prevents a trace that
terminates in or passes through another pollen from being presented as an
observed continuation. The prefix before first contact may still become a
`pre-contact` measurement when contact and timing jumps are its only blockers,
the prefix independently passes the same direct/eventual support and motion
limits as a full trajectory, and it contains enough post-measurement samples for
the cadence rule. Accepted rows stop before the first unsafe path point; an
orange cross marks the censor location in the review video. The summary reports
the foreign pollen ID, censor frame, supported prefix length, and
`maximum_accepted_length_*`. The entire candidate path remains in the detailed
exports for review. Temporal-front refinement still cannot override contact,
ownership, geometry, or occlusion warnings.

Crowded birth components may touch several pollen grains. The tracker therefore
retains one root-to-tip hypothesis for each attached pollen instead of silently
keeping only the closest root. If hypotheses from different pollen explain most
of the same path, they are marked `ambiguous-component-ownership` and excluded
from automatic counts unless exactly one has a coherent time-resolved
trajectory. The alternatives and their shared component ID remain in the CSV
for review.

Run the full selected interval in one invocation whenever memory permits. The
baseline, registration, pollen catalog, and birth map then remain fixed for the
entire experiment. This is essential for a slowly developing tube whose root
may become visible many minutes before its final tip. Automatic events must
also extend away from the pollen body, remain observable inside the frame, and
avoid broad dark occluders. If one pollen has multiple distinct plausible
branches, all competing branches remain measured but require review unless one
has uniquely coherent tip motion.

At a skeleton junction, path search does not take diagonal shortcuts through a
pixel corner. Competing endpoints are softly ranked by how well they continue
the tube's incoming direction at the junction, while bends away from a true
branch are unaffected. The exported `max_junction_turn_degrees` value makes
that crossover decision available for review.

Overlapping-window and fixed-map topology-slab variants were tested and retired:
the former changes the baseline and pollen census at each boundary, while the
latter truncates tubes whose contrast matures over longer than its local context.
Their negative results remain in the experiment ledger, but the unsupported run
modes are no longer carried in the maintained prototype.

`trajectory-growth` means the tip updates passed motion and cadence checks.
`germination-only`, `emergence-only`, and `atlas-event` preserve useful
biological evidence without claiming a reliable frame-by-frame tip trajectory.
Every rejected or ambiguous path remains in the detailed CSV output.

### Phase-Consensus Research

`v18_phase_consensus` is the current experimental successor to the single-phase
v17 workflow. It runs two interleaved analyses, keeps each proposed pollen-to-tip
branch topologically fixed, and asks whether the other phase independently
supports the same path. The root follows measured pollen motion while that motion
smoothly fades along the proximal shank; interior path points may make only small,
smooth normal corrections inside the original branch corridor.

```bash
.venv/bin/python prototypes/v18_phase_consensus/track.py \
  --movie data/raw/example.mp4 \
  --source-start 0 \
  --output-seconds 3 \
  --analysis-seconds 1.5 \
  --pixel-size 0.1 \
  --distance-unit um \
  --output-dir runs/prototypes/v18/example
```

The reported tip is a globally smooth, root-connected visible prefix of the
fixed branch. A crossing beyond an unsupported gap cannot become the tip by
itself. Phase stability is judged as tube-length disagreement at matched times,
because one-pixel localization differences on very slow tubes can correspond to
many seconds. Point-arrival disagreement remains in the CSV as uncertainty, and
exact germination time still requires independent time-domain onset agreement.
Every phase-supported candidate remains exported even when it does not pass the
automatic gate.

Length-trajectory acceptance is intentionally separate from germination-time
acceptance. A pollen-connected path can be used automatically when both phases
support its geometry, visible front, and length at matched times even if the
optical emergence cues disagree. In that case the length curve is accepted, but
the germination and exact-onset fields remain unresolved instead of inheriting a
timestamp from the trajectory.

For every automatic trajectory, v18 also reports an operational measurable-
growth onset: the first time each phase places the tube beyond the configured
minimum germination length. The summary retains both phase bounds, their
midpoint, and the full uncertainty width; detailed rows include time relative to
that midpoint. This is useful for aligning growth curves, but it is explicitly
separate from biological emergence and is never presented as an exact onset when
the two optical clocks disagree.

Each run report also summarizes how reproducibly the two phases locate the
connected-root, pollen-rim, path-contrast, and combined onset cues. On the two
current benchmark movies, none of 30 paired cue measurements from the eight
visually trusted trajectories met the 1.5-second exact-time tolerance. Median
disagreement ranged from 3.8 to 258.8 seconds depending on the cue and movie.
This makes exact biological germination time an unresolved measurement rather
than an automatic output. The raw cue frames remain in the summary CSV for
review, while the phase-bounded measurable-growth interval is the appropriate
automated timing output for a pilot study.

This is the strongest current research prototype, but it is not yet the routine
GUI workflow. It still requires validation on representative WT and
LLG-overexpression videos before biological conclusions are drawn. Review
crossings, long occlusions, abrupt motion, and all automatically selected events.

#### Validation Gate

A full-recording ablation compared the retained v18 method with multiscale tube
enhancement, a jointly optimized space-time curve, an interleaved two-phase tip
front, and shared registration. None improved both benchmark movies: the most
promising curve optimization kept 7 automatic low-density trajectories but
reduced the dense result from 4 to 3, while direct multiscale evidence reduced
the low-density result from 7 to 4. These variants are recorded in the research
ledger and are not selectable production options.

The next milestone is a blinded reference set made from representative WT and
LLG-overexpression videos. At fixed time points, a reviewer should mark whether
germination is present, the pollen-to-tip centerline, the tip location, and the
same-tube identity through crossings. Before viewing aggregate results, define
the evaluation measures: germination precision and recall, centerline-length
error, tip-location error, identity switches, and the fraction requiring human
review. Biological comparison should begin only after those measurements show
that the retained method is reliable enough for the intended videos.

### Causal-Worldsheet Research

`v19_causal_worldsheet` is an isolated proof of a new crossing representation.
Instead of treating a skeleton junction as permission to switch branches, it
represents every root-to-tip hypothesis as a directed lane with its incoming
orientation, pollen ancestry, construction order, and earlier whole-curve shape.
It then selects the complete sequence of rooted curves jointly across analyzed
times. Pollen translation is removed before comparing material coordinates, so
whole-field or grain motion does not by itself change tube identity.

```bash
.venv/bin/python -m prototypes.v19_causal_worldsheet.track \
  --dense-movie data/raw/example.mp4 \
  --v17-run runs/prototypes/v17/example \
  --output-dir runs/prototypes/v19/example
```

On the controlled crossing benchmark, the directed lane retained the intended
tube in 21/21 cases when optical birth timing was unavailable, compared with
15/21 for the current planar selector. In six short sequence tests, greedy
framewise selection switched to the foreign branch every time, while the global
worldsheet retained the correct rooted curve in 6/6. The same lane rule agrees
with prior raw-video adjudication on dense G31, which remains continuous through
contact, and G95, whose 78.7-degree handoff is rejected.

This validates crossing identity only. The current prototype begins from an
observed skeleton and does not improve missing-tube segmentation, establish
biological germination time, or replace v18's phase-validated tip and length
front. The intended integration is for v19 to resolve one material branch and
then hand that branch to v18 for measurement. Broader integration is deliberately
paused until the blinded reference set can show whether it improves real
crossing cases without reducing recall elsewhere.

### Orientation-Worldsheet Research

`v20_orientation_worldsheet` tests a more complete crossing representation. It
never collapses the image into a planar skeleton. Each possible tube point keeps
its position, direction, and first persistent appearance time, so two tubes may
cross at one image pixel without becoming the same state. Bright and dark
phase-contrast boundaries are treated consistently, each trace remains owned by
its source pollen, and other pollen bodies cannot become shortcuts. A v18 path
is used only as a weak proposal when v18 independently accepted it; rejected
paths must be rediscovered from the complete pollen rim.

```bash
.venv/bin/python prototypes/v20_orientation_worldsheet/track.py \
  --v18-run runs/prototypes/v18/example \
  --consensus-id 1 \
  --evidence-samples 120 \
  --output-samples 30 \
  --output-dir runs/prototypes/v20/example
```

The prototype exports a review video, detailed sampled measurements, a
directional birth visualization, and a JSON audit report. On one difficult
low-density candidate whose v18 path had an abrupt tip jump, it rediscovered a
30 px pollen-attached tube with full global path support and no backward growth.
On one independently supported dense candidate, it recovered 60 px of the 64.3
px reference path with 85% pointwise support and 97.5% global support while
avoiding the neighboring pollen cluster. A known foreign-contact candidate was
correctly withheld rather than extended into an unsupported branch.

These are targeted audits, not a field-wide accuracy estimate. v18 remains the
authoritative research pipeline until v20 is evaluated against blinded manual
centerlines and tips over representative WT and LLG-overexpression videos.

### Deformable-Worldsheet Research

`v21_deformable_worldsheet` changes the unit of optimization from a point or a
single frozen path to the complete pollen-attached tube over the complete sampled
recording. A causal atlas gives every material position a permanent arclength
identity. One graph-cut optimization then chooses the normal displacement of
every active material position at every sampled time, coupling neighboring times
and neighboring positions along the tube. Tube bending therefore changes the
centerline geometry without being counted as new growth; only advancement of the
causal distal front changes length. Each displacement must also retain the
material point's persistent appearance-time signature. This prevents a brighter
crossing branch with a different construction history from taking over the fit.

The independent mode does not use the v18 centerline to choose a direction. It
starts hypotheses around the complete pollen rim and retains three evidence
families: a full-timeline change-point model, a persistence model, and a
left-censored fallback for tubes already present at the start. A short hidden
neck at the pollen halo may be crossed once, but later unsupported gaps cannot be
used to jump branches. Pollen attachment outranks raw contrast during selection,
and round pre-existing endpoints are penalized. An open-curve certificate also
prevents a strongly supported pollen boundary from being reported as a long
tube. All hypotheses remain in the audit report. Geometry support and scientific
measurement support are separate decisions, so a credible tube can remain
available while uncertain timepoints are withheld.

```bash
.venv/bin/python prototypes/v21_deformable_worldsheet/track.py \
  --v18-run runs/prototypes/v18/example \
  --consensus-id 1 \
  --evidence-samples 120 \
  --output-samples 30 \
  --independent-atlas \
  --output-dir runs/prototypes/v21/example
```

The complete v21.5 field audit reconstructed geometry for 10 of 18 retained
low-density candidates and accepted measurements for 8. In the denser movie it
reconstructed 8 of 22 and accepted measurements for 4, compared with 3 geometry
and 2 measurement results in the prior v21.2 run. Dense C2 is now accepted at 6
of 7 active reporting times and remains on the faint upward tube; low-density C17
is accepted at all 22 active times. Visual audit also found one dense false path
that made an almost closed 90 px circuit around a pollen body. v21.6's open-curve
gate rejects that loop and selects a short causal alternative, which is then
withheld for weak evidence. A complete v21.6 field rerun is still pending. The
prototype exports sampled measurements, every deformed centerline point, a review
video, and a JSON report. These results justify a blinded evaluation; they do not
yet establish biological accuracy.

### High-Resolution Ribbon Research

`v22_coupled_ribbon_worldsheet` addresses the information lost when a
1280-pixel source movie is reduced to the 480-pixel discovery width. At the
smaller scale, a real pollen tube can collapse to a one-pixel line. v22 returns
to higher-resolution source frames and treats a tube as a coupled physical
ribbon: center, direction, left wall, right wall, width, pollen owner, and
appearance history move together. It searches the complete pollen rim again,
so the high-resolution pass can repair an incorrect coarse topology instead of
only shifting it sideways. A separate whole-video optimization then fits every
sampled time and permits only a connected distal front to add length.

```bash
.venv/bin/python prototypes/v22_coupled_ribbon_worldsheet/audit.py \
  --v18-run runs/prototypes/v18/example \
  --v21-case runs/prototypes/v21/example \
  --consensus-id 1 \
  --analysis-width 960 \
  --validate-temporal-splits \
  --output-dir runs/prototypes/v22/example
```

The independent ribbon search appeared to repair several targeted topology
challenges, but later raw-frame review exposed two shared-assumption failures.
Dense C17 inherited a minimum-radius Hough detection on a tube shank rather
than a pollen grain. Low-density C10 began at a real pollen but crossed onto an
unrelated longer tube. Alternating-timepoint reconstruction repeated those
same assumptions, so agreement measured reproducibility rather than biological
correctness. The earlier C17 and C10 promotion claims are withdrawn. Paired-wall
evidence remains useful as a tube-likelihood feature, but v22 is not an
authoritative measurement pipeline.

The optional temporal-split check reconstructs the tube independently from
alternating sampled timepoints, scores each path on the frames it did not use,
and compares length, centerline, and endpoint agreement. A six-case internal
audit produced three reproducible geometries, two plausible full-data traces
that remained review-only because the independent lengths or paths disagreed,
and one correctly rejected false path. One reproducible short tube was already
visible at the beginning, so its length remains usable while its germination
time is withheld. Failed geometry now produces blank measurement fields and a
clearly withheld review instead of drawing the rejected fallback as a result.
This is a useful precision gate, not an accuracy estimate: manual reference
centerlines and tips are still required to measure absolute error and recall.

### Owner-Conditioned Video Graph Research

`v23_owner_memory` separates pollen identity from tube identity. A learned
Cellpose mask establishes that a candidate is biologically pollen when the
grain is clearly visible. Hough circles remain as recall-oriented proposals,
but a Hough-only track cannot become biological truth. One global path-cover
optimization links detections over the recording and allows new grains to
enter after the warmup period. Each confirmed pollen then owns a separate tube
layer, so two tubes may cross at the same image pixels without becoming the
same object.

Tube tracing keeps several complete paths through a crossing and resolves them
over later frames. The current prototype combines paired-wall evidence,
material appearance order, previously grown path shape, and learned foreign
pollen occupancy. Near-duplicate paths from neighboring root seeds count as
one topology; genuinely different branches remain separate and are withheld
when their evidence is too close.

Two real counterexample gates now pass. On dense C17, the old Hough owner has
no learned pollen support across the audited sequence. On low-density C10, one
owner track spans all 11 sampled frames with six learned-mask confirmations and
geometric support through the late cluster merge. Its independent tube path
grows from 50 to 90 source pixels across six late samples, stays attached to
the owner, and stops at the crossover instead of following v22's unrelated long
continuation.

A complete low-density field pass now applies the same owner-conditioned search
to all 39 semantically confirmed pollen across the 52,583-frame recording. Its
dense temporal extension re-registers each moving pollen at every sampled time
point and estimates emergence from three independent cues: a constrained path
front, a self-supervised early-versus-confirmed tube appearance test, and a
direction-specific protrusion that persists and grows from the pollen rim. Two
agreeing cues set onset; a durable rim event can rescue a degenerate late
template. No mature path is allowed to create length before the selected onset.

The current low-density audit retains 23 non-conflicting automatic trajectories
at 252 time points and withholds 16 cases for review. An independent field check
collapses neighboring owners that claim the same mature branch to one winner.
Normal review videos draw accepted trajectories only; rejected hypotheses are
available only with the explicit `--show-review` diagnostic option. The temporal
run exports one measurement row per pollen and time point, every reported
centerline coordinate, onset evidence and competing-cue times, and the complete
parameterized JSON report. These outputs establish field-wide execution and
conservative conflict handling, not biological accuracy: physical calibration,
blinded manual centerlines, recall measurement, and broader-video validation
remain open gates. v23 remains isolated from the desktop application.

## Native Owner-Portal Prototype

The isolated v25 successor keeps v23's learned pollen identities but changes
how tube paths are selected and timed. Several late path hypotheses are retained
for every pollen. Each is then checked on multiple native-resolution frames for
the two visible tube walls, connected temporal completion, an open distal end,
and clearance from every other pollen. Selection uses the length of tube that is
actually supported instead of preferring a short high-contrast fragment. This
also permits a legitimate U-shaped tube while rejecting a circuit that returns
to its owner. If the coarse atlas cannot produce an accepted path, a second pass
rebuilds that owner's atlas at twice the working resolution and replays every
new path through the same full-movie checks. This recovers thin early protrusions
without displaying unverified alternatives.

Germination timing is measured separately in pollen-centered polar coordinates.
The algorithm follows one mature attachment direction backward through time and
requires a persistent protrusion from a still-recognizable pollen boundary. The
summary exports the last confidently dormant time and the first confirmed
emergence time as an interval. Detailed lengths are monotone and remain available
for left-censored or length-only cases without inventing an exact germination
time.

On the complete 62.6-minute low-density movie, the current audit retains 36
non-conflicting owner paths: 28 have a measured emergence interval, seven are
left-censored, and one supports length without a reliable emergence event.
Native crop timelines for representative isolated and clustered cases were
visually reviewed, including correction of a path that previously ran from P13
into P8. A native-resolution connected-prefix pass also recovered P8's clear
U-shaped tube after the coarse atlas found the correct geometry but understated
its temporal completion. The multiscale pass recovered four additional paths,
including two small persistent emergence events and two distinct paths in the
P9/P10/P12 neighborhood. Three traces are retained only up to the last supported
tip before an abrupt foreign-branch jump. Five ambiguous or unsupported owners
remain withheld, and one abrupt owner-change event is exported separately as a
rupture candidate rather than a tube. The prototype is not yet the desktop
default and still requires blinded manual centerlines plus additional WT and LLG
videos before scientific promotion.

## Bidirectional Dynamic-Ribbon Prototype

`v26_bidirectional_ribbon` addresses a limitation in v25's fixed mature path.
For each automatically selected tube, it begins with the clearest mature
centerline and reconstructs the complete pollen-connected curve backward through
every sampled frame. The prior curve moves with its pollen and constrains both
the root-to-tip shape and direction, while the current frame still determines
how much tube is visibly supported. Bright-interior, dark-interior, and generic
paired-wall evidence are combined so a weak polarity model cannot erase an
otherwise visible tube.

The reconstruction rejects paths confined to a pollen rim, paths that travel
out along one boundary and return along the other, and distal paths that end in
another pollen grain. One- or two-sample optical dropouts are filled by
interpolating the complete surrounding curves. New distal growth must persist
in a later sample before it becomes the reported length; the original observed
length remains in `measurements.csv` with a confirmation flag. The prototype
exports one dynamic centerline per accepted time point rather than replaying one
late path with a changing endpoint.

An initial visual audit covered 11 real low-density candidates across the field
and all three optical modes; ten produced persistent trajectories under the
final confirmation rule. In the known P39 failure, the old late
boundary-return jump was removed, the tube remained pollen-connected, and
persistent emergence moved to the visibly plausible interval near 19.5 minutes.
P4's inherited path into another pollen was automatically rejected, and P36,
P10, P12, and the wider audit set followed their visible tubes without branch
switches. A one-frame final extension on P7 is retained as raw evidence but not
reported as confirmed growth. These checks establish a substantial qualitative
improvement, not biological accuracy; blinded manual centerlines and additional
WT/LLG movies remain required before promotion to the desktop workflow.

```bash
.venv/bin/python -m prototypes.v26_bidirectional_ribbon.track VIDEO.mp4 \
  --identity-report runs/prototypes/v23/owner_memory/field_identity/report.json \
  --base-run runs/prototypes/v25/causal_portal/field \
  --output runs/prototypes/v26/bidirectional_ribbon/field
```

## Body-Aware Global Ribbon Prototype

`v27_global_ribbon_worldsheet` removes the mature-centerline requirement from
v26. For each pollen, it searches the complete rim for possible tube roots,
keeps candidates from widely separated directions, and auditions those complete
curves across the movie. The selected history is therefore based on recoverable
growth over time rather than final-frame contrast alone. An earlier run can
still be supplied with `--base-run`, but it is optional.

Pollen identities also remain active during tube selection. If a candidate
reaches another credible pollen body, the valid owner-connected prefix is kept
and exported with `contact_censored` status; material beyond that contact is not
silently assigned to the first pollen. This corrected a real P8 failure in
which a 189-pixel trace was actually following P13's downstream tube. Without
an inherited centerline, v27 found P8's germination at 18.25 minutes and retained
an 80-pixel prefix up to P13. It independently found clean P39 at 19.5 minutes
and 85.3 pixels, close to the 88-pixel inherited-path result. Both full-movie
histories had no backward length steps.

A third crowded case, P4, remained connected to another pollen from the first
frame and therefore could not establish which pollen owned the visible material.
The workflow now preserves such coordinates for review but excludes the path
from normal result overlays as `review_left_censored_contact`.

A complete low-density field audit processed all 42 retained pollen identities
and showed that targeted successes were not enough for field-wide promotion.
Several early candidates followed the pollen rim instead of an outward tube,
while P37's valid sparse trace collapsed during denser temporal optimization.
The current v27.6 implementation identifies the last irreversible departure
from the pollen halo, reconnects the distal tube to its image-supported body
exit, and ranks mature alternatives by sustained growth and whole-recording
support. Causal-atlas proposals broaden recall for curved or faint tubes, while
retained pollen bodies remain ownership barriers at contacts and crossings.

The practical default analyzes complete curves at evenly spaced one-minute
anchors and interpolates material coordinates only between accepted anchors.
Every interpolated CSV row is marked, and interpolated centerlines are rescaled
to the linearly interpolated material length so bends cannot create artificial
length loss. `--dense-timeline` retains the slower every-sample experiment when
needed. P37 now begins at its visible lower-left attachment, has an onset near
17 minutes, and ends at 82.7 pixels rather than counting a trip around the
pollen body. P8 and P39 independently reproduce 19-minute onsets and final
lengths of 80.0 and 85.3 pixels. Event evidence distinguishes a plausible
optical path from observed tube growth, and a field-edge test distinguishes a
complete tip from a tube censored by the image boundary.

The final v27.6 replay completed all 42 owners with zero processing failures:
15 ordinary measurements, 13 contact-censored prefixes, 3 left-censored tubes,
1 boundary-censored tube, 6 review cases, and 4 `no_growth_detected` owners.
Thus 32 centerline histories are geometrically usable under their explicit
censoring labels. Four representative times were visually inspected for every
owner; no accepted trace visibly switched onto another tube. Every usable
length history is nondecreasing, and each final summary length exactly matches
both its detailed measurement and exported centerline arclength. This is a
complete qualitative audit of one low-density movie, not an accuracy estimate
against blinded ground truth.

All candidate coordinates remain available for diagnosis. Detailed measurement
and centerline rows repeat `measurement_status`, so a frame-level accepted
candidate from a review or no-growth owner cannot be mistaken for an automatic
biological result. Review and no-growth paths are hidden from the normal movie
overlay. P3 remains visible as `boundary_censored`; P2, P16, P17, and P28 are
shown as no growth.

A zero-shot Cellpose-SAM 4.2.1 check was also run on the final low-density
frame. It was slower and lower-recall than the existing fused detector on this
brightfield footage, so it remains an optional hypothesis source rather than a
required dependency. Blinded manual centerlines plus additional WT and LLG
videos remain required before promotion to the routine application.

```bash
.venv/bin/python -m prototypes.v27_global_ribbon_worldsheet.track VIDEO.mp4 \
  --identity-report runs/prototypes/v23/owner_memory/field_identity/report.json \
  --output runs/prototypes/v27/global_worldsheet/field \
  --track-ids 8,39 \
  --selection-stride 4
```

For a complete resumable field pass, run owners independently and aggregate
their three CSV exports with:

```bash
.venv/bin/python scripts/run_v27_field_validation.py VIDEO.mp4 \
  --identity-report runs/prototypes/v23/owner_memory/field_identity/report.json \
  --causal-atlas-run runs/prototypes/v24/causal_birth_forest/field \
  --output runs/prototypes/v27/global_worldsheet/field \
  --track-ids 1,2,3,4 \
  --workers 4
```

## Detection-Constrained Owner-Motion Prototype

The optional v28 owner-motion path addresses a failure that local circle tests
cannot solve: after germination, an updating template or a new circle detection
can slide from the pollen body onto its rounded tube tip. Native-detail rigid
landmarks provide a motion prior, trusted semantic observations protect known
owner locations, and a high-recall circle detector proposes pollen bodies on
every sampled frame. One joint assignment gives each detection to at most one
owner, so neighboring tracks cannot collapse onto the same round object. Tube
tracing remains at the less expensive analysis scale, and the resulting compact
motion cache can be reused by later experiments.

The v28.4 full-field run added a lifecycle check to complete-curve selection.
A tube claimed to predate the recording must already extend at least
1.5 pollen radii beyond its body in the first frame; later germination and
contact-censored paths are unaffected. P2's visible pre-existing tube passes at
1.93 radii. Four old frame-zero rim paths on P4, P24, P25, and P36 fail at
0.52-1.01 radii and are replaced by visually plausible later emergence near
2, 20, 12, and 33 minutes. P16 and P28 remain no-growth controls after being
recomputed under the same rule.

All 42 low-density owners completed with no processing failures. Field-level
arbitration retains 23 ordinary measurements, 3 boundary-censored tubes, and 4
contact-censored prefixes. Four no-growth owners, four review cases, and four
unresolved duplicate claims remain outside the automatic result set. When P19,
P20, and P78 independently claimed one distal branch, P78 was retained because
it was the only causal owner path that did not pass through another pollen;
P19 and P20 remain conflicts. P1 and P6 remain unresolved because neither route
has unique unobstructed evidence. The exports contain 10,584 owner-time rows and
139,528 centerline points, with all raw coordinates retained regardless of
field status.

The current v28.5 result also fits accepted material lengths to an
evidence-weighted nondecreasing history. This removed a 0.67-pixel sparse-anchor
oscillation on P39 without changing its owner attachment; original observed
lengths remain exported for audit. All 30 measured or explicitly censored
histories now have zero backward length steps, and every reported length agrees
exactly with its exported centerline arclength and final summary. This growth
constraint is appropriate before rupture; a future validated rupture event must
explicitly open a new post-rupture segment rather than appearing as ordinary
negative growth.

The v28.6 field allocator addresses the remaining case where independently
reasonable owner traces claim the same tube. Each pollen can retain a compact
archive of its complete audited alternatives. Within each conflict group, an
exact search assigns the largest possible set of distinct, event-certified
branches, preserves independently supported ownership before using small score
differences, and rejects an ordinary route that crosses another pollen. An
explicit contact-censored prefix remains legal because it ends at that contact.
Only changed seed labels are written to a reproducible override file; the
per-owner tracer rechecks their geometry, emergence, and ownership before using
them.

On the complete low-density movie, this recovered separate branches for P1/P6
and P19/P78 while continuing to withhold P20's duplicate claim. Field conflicts
fell from four owners to one. The promoted result contains 24 ordinary
measurements, 6 contact-censored prefixes, 3 boundary-censored tubes, 4
no-growth owners, 4 review owners, and 1 ownership conflict. All 42 owners ran
successfully; all 10,584 time rows are nondecreasing and exactly agree with the
corresponding centerline and final summary. The competing assignments and both
regenerated owner movies were visually inspected, but this remains qualitative
validation rather than a blinded accuracy measurement.

To archive alternatives, allocate conflict groups, and replay only the changed
owners, add these stages to the field command:

```bash
.venv/bin/python scripts/run_v27_field_validation.py VIDEO.mp4 \
  --identity-report IDENTITY/report.json --causal-atlas-run CAUSAL_RUN \
  --owner-motion-cache MOTION/pollen_motion_tracks.npz \
  --field-cache FIELD_CACHE --output CANDIDATE_RUN \
  --track-ids 1,6,19,20,78 --export-alternatives

.venv/bin/python scripts/allocate_v28_candidate_branches.py CANDIDATE_RUN \
  --field-cache FIELD_CACHE --components '1,6;19,20,78' \
  --output ALLOCATION_RUN

.venv/bin/python scripts/run_v27_field_validation.py VIDEO.mp4 \
  --identity-report IDENTITY/report.json --causal-atlas-run CAUSAL_RUN \
  --owner-motion-cache MOTION/pollen_motion_tracks.npz \
  --field-cache FIELD_CACHE --output FIELD_RUN --track-ids 1,6,19,20,78 \
  --mature-seed-overrides ALLOCATION_RUN/mature_seed_overrides.json
```

The seven event-timed audit sheets were inspected across the complete movie.
They support the lifecycle and ownership corrections, but they are not blinded
ground truth. The supplied reference workflow and additional WT/LLG recordings
remain required before reporting biological accuracy or making this pipeline
the desktop default.

Generate the native-detail owner cache once:

```bash
.venv/bin/python -m scripts.benchmark_pollen_motion VIDEO.mp4 \
  --identity-report runs/prototypes/v23/owner_memory/field_identity/report.json \
  --output runs/prototypes/v28/pollen_motion/field \
  --track-ids 1,2,3,4 \
  --width 1280 \
  --tracking-space fixed-crop-mosaic \
  --body-crop-size 192 \
  --mosaic-batch-size 6
```

Then reuse it in a resumable field trace:

```bash
.venv/bin/python scripts/run_v27_field_validation.py VIDEO.mp4 \
  --identity-report runs/prototypes/v23/owner_memory/field_identity/report.json \
  --causal-atlas-run runs/prototypes/v24/causal_birth_forest/field \
  --owner-motion-cache runs/prototypes/v28/pollen_motion/field/pollen_motion_tracks.npz \
  --output runs/prototypes/v28/global_worldsheet/field \
  --track-ids 1,2,3,4 \
  --workers 4
```

## Bidirectional Causal Growth-Front Prototype

The v29 prototype reconstructs each tube as an ordered material path rather
than choosing a new tip independently in every frame. It evaluates both the
retained field path and any causal-atlas alternative, keeps the pollen root
attached while allowing narrow deformation, and assigns every reached material
point one first-appearance time. The exported length is the connected prefix
from the owner to that causal front, so it cannot shrink or jump across an
unsupported gap.

Two field-level checks handle crowded pollen. A path stops before entering a
foreign pollen body, and a direction-aware shared-tail test prevents two owners
from reporting the same branch after a crossing or contact. v29.2 also fits the
same appearance history from the distal end toward the proposed owner. A
stronger, well-supported inward fit identifies a neighboring tube growing
toward that pollen rather than away from it. On the low-density movie this
independently rejects the reversed P9 and P17 claims, while shared-tail
arbitration gives the P19/P23 branch to P23 and the P22/P78 tail to P78.

v29.3 adds a selective native-resolution verification pass for paths that the
field workflow withheld despite otherwise passing the root and growth-direction
checks. It follows paired tube walls along the complete owner-relative path and
compares them with nearby control lanes. A path is rescued only when this
independent pass reproduces at least 75% of the claimed length and its growth
onset is no more than 12 samples later than the causal estimate. On the current
movie, only P10 passes: native evidence reproduces its complete 40-pixel
contact-censored path with a two-sample onset difference. P33 and P41 remain
withheld, and the earlier P9/P17 direction rejections are unchanged.

v29.4 treats ownership as a field-wide causal graph rather than only a pairwise
tail comparison. Every candidate is checked against earlier-growing accepted
paths. A foreign claim is made only when the later candidate follows the same
local direction for at least 0.75 pollen radii; perpendicular crossings and
brief contacts are ignored. This independently attributes P9 to P11, P17 to
P78, P33 to P31, and P41 to P40. The first two corroborate the reverse-growth
test, while P33 and P41 now receive explicit `foreign_branch_capture` decisions
instead of generic review labels. After shared material is removed, P19 has no
owner-departing prefix; P20 reaches P78 before it has a reportable independent
length. They receive the same explicit foreign-branch decision. No retained
path is newly flagged.

```bash
.venv/bin/python prototypes/v29_causal_growth_front/run.py \
  --field-run FIELD_RUN \
  --field-cache FIELD_CACHE \
  --owner-motion-cache MOTION/pollen_motion_tracks.npz \
  --causal-atlas-run CAUSAL_RUN \
  --output runs/prototypes/v29/causal_growth_front/field
```

The complete v29.4 low-density replay processes 42 owners and retains 33
measured or explicitly censored histories. Its 10,584 detailed time rows have
zero backward length steps, all 93,140 accepted centerline rows agree with their
reported lengths, and a post-arbitration scan finds no duplicate claims. The
foreign-branch audit image draws every later candidate against the earlier path
that explains it. Rejected owners have blank reportable lengths and tips; their
values remain in clearly named diagnostic columns. The full suite passes 363
tests plus three subtests. The final field contains no generic review or
unresolved ownership status: the nine withheld owners are three no-growth
grains, two reverse-growth assignments, and four foreign-branch captures. These
checks establish internal consistency and remove visually verified ownership
errors, but blinded centerlines are still required before reporting biological
accuracy.

v29.5 keeps those ownership and growth decisions fixed, then revisits the
retained centerlines in the native video. It searches only across each existing
material path, requires a bright tube interior bounded by both phase-dark
walls, and fits one smooth correction supported across at least three complete
observations. The pollen attachment is locked and the path cannot change
branch or point order. A correction is withheld when its evidence is unstable,
reaches the search boundary, or would change total length by more than 20%.
On the complete low-density replay, 16 of the 33 retained histories receive a
native center correction; the median absolute final-length change is 3.29 px
and the maximum is 8.43 px. P7's otherwise attractive correction is rejected
because it would increase length by 43%. The ownership graph and all 33 retained
histories remain unchanged, the automatic post-refinement scan finds zero
duplicate claims, and `native_centerline_audit.jpg` shows every final curve on
the source-resolution image. All 10,584 time rows remain nondecreasing, all
93,220 exported centerline rows agree with their measurements, and the full
suite passes 366 tests plus three subtests. This improves spatial placement; it
does not replace blinded biological validation.

v29.6 adds an independent native-geometry certificate after centering. It
checks at least 85% of each candidate over up to eight late observations and
requires paired tube-wall support across a median 40% of the tested path. The
threshold sits in the observed gap between P15 (44.6%, visibly tube-aligned)
and P5 (38.5%, visibly unsupported), so it is a one-movie calibration that must
be challenged on new recordings. Seven prior retained candidates fail this
check: P3, P5, P22, P24, P25, P35, and P36. The native audit marks them in
orange, and their reportable measurements are blank rather than silently used.
No information is deleted: `diagnostic_candidate_length_px` keeps every time
series and `diagnostic_centerlines.csv` contains all candidate coordinates,
while `centerlines.csv` contains only the 26 verified histories. The complete
run writes 10,584 time rows, 78,297 reportable centerline rows, and 118,239
diagnostic centerline rows. It has zero backward reportable steps, zero
post-gate duplicate claims, and exact length-to-centerline agreement. The full
suite passes 368 tests plus three subtests. This is a precision-oriented gate,
not evidence that the remaining paths are biologically correct.

v29.7 separates first-emergence timing from mature-tube validation. For each of
the 26 geometry-verified histories, a native audit examines only the first 20 px
of the owner-attached path and looks for a persistent 3 px extension beyond the
dormant baseline. It compares that event with the existing causal 5 px onset,
allowing at most 12 sampled time points of disagreement. Thirteen owners are
independently corroborated: P1, P6, P11, P12, P21, P26, P29, P30, P31, P32,
P37, P38, and P39. Nine disagree and four have no native root event, so their
germination timing remains review-required even though their accepted length
histories are unchanged. `verified_germination_time_minutes` is populated only
for the 13 corroborated owners; the existing causal time columns remain as
diagnostic estimates for every accepted path. This prevents a mature-tube
visibility rule or a simple length threshold from being presented as a precise
germination time. Cross-video and blinded event labels are still required.

v29.8 and v29.9 tested two narrower repairs before changing the tracing model.
A native tip-front correction could refine some endpoints but frequently reached
its search boundary, while a broad lateral path search could improve local image
support without repairing a wrong branch. Both remain diagnostic and promote no
measurements. These negative results are intentional safeguards: endpoint or
lateral corrections cannot substitute for tracing a complete pollen-attached
tube.

v29.11 adds source-resolution paired-boundary tracing for paths rejected by the
legacy geometry check. It starts at the tracked pollen boundary, follows the two
phase-dark tube walls around their brighter interior, masks foreign pollen
bodies, and generates distinct complete-path alternatives. Every alternative
must then pass movie-wide causal growth, outward direction, repeated native wall
support, root-emergence agreement, and field-wide duplicate checks. This recovers
P5, P22, P24, P25, P35, and P36 while correctly leaving P3 unresolved, increasing
the verified field from 26 to 32 histories.

v29.13 separates tube topology, growth timing, and frame-specific pose. Each
native topology is evaluated in a stable pollen-relative pose and a smoothly
deformable pose. Growth timing and root emergence stay anchored to the stable
material path; deformation can win only when the same rigid topology is already
native-verified and deformation improves its wall coverage by a configurable
margin. Five rescues remain rigid, while P36 alone earns deformation and follows
its visibly bending tube at intermediate times. The final low-density replay at
`runs/prototypes/v29/causal_growth_front/lowdens_full_v29_13_2` retains 32 of 42
histories, leaves P3 unresolved, and has zero duplicate claims, zero backward
length steps, and exact agreement between all 5,676 accepted time points and
their exported centerline arclengths. The three-stage
`native_boundary_rescue_time_audit.jpg` shows owner attachment, emergence,
mid-growth, and maturity for every rescue. The complete suite passes 376 tests
plus three subtests. This is a substantial internally validated improvement on
one movie, not a blinded biological accuracy estimate.

v29.17 makes pollen ancestry authoritative when local circle detections drift
onto rounded tips or neighboring structures. It then checks only the retained
growth prefix for a prompt, irreversible exit from the pollen body; paths that
run through the grain, walk around its rim, or return to its halo are rejected.
A short clean emergence may pass this biological topology check, but any trace
that departs by less than one pollen radius must still compete with independent
source-resolution boundary reconstructions before its geometry is reported.
Native root evidence is also allowed to correct an early or late coarse onset
instead of rejecting an otherwise complete tube. On the authoritative replay
at `runs/prototypes/v29/causal_growth_front/lowdens_full_v29_17_1`, these rules
recover the previously missed P3 tube, replace truncated P10 and P19 paths, and
retain 34 of 42 histories. The built-in export gate verifies all 5,892 accepted
time points and 92,798 centerline coordinates: there are no backward length
steps, missing paths, arclength disagreements, invalid accepted topologies, or
duplicate claims. P2 remains explicitly unresolved because neither of its two
plausible paths has independent source-video support. These results are a
strong internal checkpoint on the low-density movie, and the complete suite
passes 386 tests plus three subtests. Blinded references and a second recording
are still required before claiming biological accuracy.

v29.18 adds owner-aligned temporal consensus as a new native-boundary proposal,
without allowing temporal averaging to overwrite a stronger complete path. The
last three source views are centered on the tracked pollen, normalized, and
combined before paired tube boundaries are traced. Consensus candidates still
have to pass the same movie-wide growth, owner-topology, direction, repeated
native-quality, root-timing, and duplicate-ownership checks. They are also
compared with the strongest independently verified single-frame proposal; a
substantially shorter consensus branch is rejected and recorded in
`report.json`. This guard caught a real P8 failure during development: a clean
29.3 px wrong branch was prevented from replacing its complete 72.0 px tube and
from creating a false collision with P5.

The authoritative low-density replay is
`runs/prototypes/v29/causal_growth_front/lowdens_full_v29_18_2`. It retains 34
of 42 histories and exports 5,893 accepted time points with 92,801 centerline
coordinates. There are zero backward length steps, missing or mismatched
centerlines, invalid accepted topologies, or duplicate claims. Visual review of
the only changed geometries, P19, P27, and P40, confirms continuous attachment
to the correct pollen and alignment with the visible tube at emergence,
mid-growth, and maturity.

A targeted transfer probe on the dense C17 recording also exposed and repaired
an inherited seed error: the old coordinate was on a rounded tube endpoint,
while five-frame persistence and expected grain size consistently selected the
nearby pollen body. On a mature C17 interval, individual source frames produced
only 49.5-60 px traces, whereas the owner-aligned temporal consensus produced
six agreeing 94.5-105 px paths along the faint continuation. This is promising
cross-video evidence, but it is not yet a complete dense-field replay. The full
suite passes 390 tests plus three subtests.

v29.19 adds a separate path for tubes that genuinely leave the microscope
field. The search is told where the real source-image edge lies even when a
working crop has reflected padding, and it carries a global direction memory so
that a weak tube cannot switch to a crossing branch through a sequence of small
turns. An edge candidate is accepted only when it reaches that true edge, is
substantially longer than the best ordinary alternative, and source-resolution
paired-wall evidence shows a connected, monotone growth history across the
movie. Failed edge searches cannot re-enter the ordinary candidate pool.

On the dense C17 challenge owner, the final replay at
`runs/prototypes/v29/causal_growth_front/dense_c17_owner_v29_19_10` follows the
faint pollen-attached tube for 136.5 source pixels through the ambiguous region
to the bottom image boundary. Its native growth audit reaches 100% of the
retained path with no backward length steps. Because the tube leaves the field,
the export keeps the material-length estimate while separately reporting the
visible in-frame centerline and marking the result as a lower bound.

The complete low-density regression at
`runs/prototypes/v29/causal_growth_front/lowdens_full_v29_19_3` retains 35 of 42
histories, including a short P16 emergence corroborated at the pollen root and
on repeated native views. It exports 6,096 accepted time points and 94,889
centerline coordinates with zero backward steps, duplicate claims, invalid
topologies, missing paths, or length disagreements. P3 and P8 are no longer
called boundary-censored because their retained paths stay inside the real
source frame. The full suite passes 399 tests plus three subtests. These are
strong internal and cross-video checks, not a blinded biological accuracy
estimate; a complete dense-field replay and independent manual references are
still required.

v29.20 closes two dense-field failure modes without relaxing tube acceptance.
First, a faint continuation must agree with the complete previously trusted
root-to-tip path before it can extend it; agreement near the pollen alone is no
longer enough. This prevents a late crossover from replacing the end of an
otherwise correct tube. The dense P107 trace remains attached through its full
53.3 px path after this stronger check.

Second, the dense bootstrap now retains every learned pollen detection instead
of requiring it to appear in almost every census image. The C17 owner set grows
from 139 to 170: 139 high-confidence, 20 supported, and 11 provisional owners.
Geometric-only circle proposals remain excluded. Confidence tier and detection
support are carried into the owner cache and CSV output rather than being hidden
by a single accepted/rejected label. Lower-confidence identities remain
traceable but cannot mask tube evidence belonging to a stronger owner; this
prevents rounded tube structures from shortening established paths. In a
targeted probe of four formerly omitted owners, P20, P56, P68, and provisional
P264 produced visually attached 39.0, 28.5, 36.0, and 76.5 px traces. The same
priority rule restored established P5 and P117 results exactly to 112.0 and
48.0 px. This is a recall and bookkeeping improvement on selected cases, not a
field-wide or blinded accuracy estimate. The full suite passes 408 tests plus
three subtests.

v29.21 removes repeated random seeking through the original long H.264 movie.
One run-scoped cache now decodes each sampled native-resolution grayscale frame
once, reuses it across wall tracing, growth timing, quality checks, and audit
rendering, and retries a failed decoder seek once with a fresh reader. The P56
transfer replay decoded 241 frames and served 506 later requests from memory
with zero retries. Its 28.5 px result, onset, measurement table, and all 2,048
centerline rows are exactly unchanged. The cache uses about 301 MiB of temporary
memory for this 1280x1024 sample set and writes nothing to disk; decoder and
cache counts are recorded in `report.json`. The full suite passes 409 tests plus
three subtests.

## Blinded Reference Validation

Internal consistency cannot establish biological accuracy. The reference
workflow creates raw, pollen-centered crops with opaque case names and never
shows the predicted tube. Each owner receives evenly spaced checkpoints plus a
separately labelled set of challenge times near the proposed transition. A
reviewer selects no visible tube, visible tube, tube leaving the image, target
not visible, incorrect target, or ambiguous; visible tubes are clicked from
their pollen attachment to the visible tip. Progress saves after every action
and resumes at the first unfinished crop. Incorrect target is scored as an
identity failure rather than being hidden inside the tube-detection metrics.
Markers use multi-observation identity tracks when available and registered
bidirectional tracking for single-observation owners. A local pollen-ring check
changes marker color but never moves the owner: rounded tube tips can pass
circle tests, so temporal ancestry takes precedence over local roundness.

```bash
.venv/bin/python scripts/review_reference_centerlines.py prepare VIDEO.mp4 \
  --identity-report runs/prototypes/v23/owner_memory/field_identity/report.json \
  --prediction-run runs/prototypes/v27/global_worldsheet/field \
  --output runs/reference_validation/field

.venv/bin/python scripts/review_reference_centerlines.py annotate \
  runs/reference_validation/field

.venv/bin/python scripts/review_reference_centerlines.py finalize \
  runs/reference_validation/field --annotator "Reviewer name"

.venv/bin/python scripts/review_reference_centerlines.py score \
  runs/reference_validation/field \
  --prediction-run runs/prototypes/v27/global_worldsheet/field
```

Finalization requires every sample to be reviewed and checksum-locks the
references before model results can be scored. The scorecard reports visible
tube precision and recall, length error, tip error, whole-centerline error, and
sampled germination intervals. Uniform checkpoints and model-targeted challenge
frames remain separate in the report. Field-censored centerlines contribute
geometry evidence but not a false exact tip or total-length target.

## Optional CNN Labels

The repository retains one fallback for difficult images: manually review point
labels for pollen centers and tube tips, train a small heatmap model, and run it
as a proposal generator. It does not replace the pollen-attached centerline
checks, and labeling is not required for the main material pipeline.

```bash
.venv/bin/pip install -e '.[cnn]'

.venv/bin/python scripts/annotate_cnn_points.py data/raw/example.mp4 \
  --dataset-dir runs/labels/example

.venv/bin/python scripts/train_cnn_points.py runs/labels/example \
  --output-dir runs/cnn-model/example

.venv/bin/python scripts/run_cnn_prototype.py \
  data/raw/example.mp4 runs/cnn-model/example/best.pt \
  --time-per-frame 1.0 \
  --output-dir runs/cnn-review/example
```

## Repository Layout

- `tubetracker/analysis.py`, `models.py`: routine detection, tracking, biological
  states, calibration, and exports.
- `tubetracker/gui.py`, `views.py`: desktop workflow and image controls.
- `tubetracker/curve_prototype.py`, `anchored_tracing.py`: shared image evidence,
  grain tracking, geometry, and pollen-rooted path tracing.
- `tubetracker/pollen_motion.py`, `grain_pose.py`: CoTracker point motion and
  pollen stabilization.
- `tubetracker/pollen_anchored_chain.py`, `topology_aware_tracing.py`: complete
  rooted-chain updates and crossover-aware continuation.
- `tubetracker/native_boundary_tracing.py`, `causal_growth_front.py`: paired-wall
  native tracing, causal material-front reconstruction, and conservative pose
  selection.
- `tubetracker/material_curve_tracking.py`, `material_ribbon.py`: persistent
  material identities, inextensible length, and paired-boundary refinement.
- `tubetracker/growth_front.py`: offline kymograph construction, path-locked
  temporal and directly visible front fitting, contamination diagnostics, and
  subpixel tip reconstruction.
- `tubetracker/causal_growth_front.py`: material-point appearance times,
  multi-hypothesis path selection, owner-rooted growth certificates, and
  bidirectional causal ownership checks.
- `tubetracker/causal_filament_graph.py`: directed crossing lanes and global
  root-relative curve-sequence selection.
- `tubetracker/orientation_worldsheet.py`: polarity-invariant orientation
  evidence, explicit paired-wall width, direction-specific material birth, and
  lifted-space tracing.
- `tubetracker/temporal_ribbon.py`: mature-seeded backward curve deformation,
  boundary-return and foreign-owner rejection, dropout interpolation, and
  persistent-growth confirmation.
- `tubetracker/global_ribbon_worldsheet.py`: global selection among complete
  root-to-tip curve hypotheses, including explicit prebirth and dropout states.
- `tubetracker/deformable_worldsheet.py`: joint time-by-material graph-cut
  fitting, independent-atlas scoring, and separate promotion gates.
- `tubetracker/owner_memory.py`: global pollen-instance linking and delayed,
  owner-conditioned tube-path resolution.
- `tubetracker/field_arbitration.py`: duplicate-path detection, exact allocation
  of retained owner alternatives, direction-aware sustained-tail ownership,
  and field-safe measurement status.
- `tubetracker/cnn_prototype.py`: optional click-supervised proposal model.
- `scripts/`: supported inspection, pilot, material, and CNN commands.
- `tests/`: focused regression and architecture checks.
- `tubetracker_assets/`: packaged application resources.
- `prototypes/LEDGER.md`: experiment history, withdrawn claims, and current
  validation status.
- `prototypes/v17_birth_topology/`, `v18_phase_consensus/`,
  `v19_causal_worldsheet/`, `v20_orientation_worldsheet/`,
  `v21_deformable_worldsheet/`, `v22_coupled_ribbon_worldsheet/`,
  `v23_owner_memory/`, `v24_causal_birth_forest/`, `v25_causal_portal/`, and
  `v26_bidirectional_ribbon/`, `v27_global_ribbon_worldsheet/`,
  `v28_global_worldsheet/`, and `v29_causal_growth_front/`: the retained
  birth-topology history and isolated successor research layers. Read
  `prototypes/LEDGER.md` before treating any prototype result as current.

`TubeTracker.py` remains a small compatibility entry point. New analysis code
should import from `tubetracker`. Headless analysis modules must not import
wxPython, and runtime resources must not be embedded as hardcoded arrays or
machine-specific paths.

Generated inputs and outputs belong under `data/raw/` and `runs/`; both are
ignored by Git except for placeholder files. Do not commit videos, model
checkpoints, caches, or run artifacts. Frame caches are disposable copies made
from the selected source video; `scripts/prepare_material_video.py` recreates
them when the material pipeline needs them. Deleting a frame cache does not
delete the original movie, reviewed labels, or exported measurements.


## rev14 outcomes (2026-09-20, after the completed annotation batch)

- **Batch integrated reproducibly**: `runs/prototypes/v30/snap31_rev14` was built
  from the completed-backup project with the established 21-project source list,
  so the historical corpus is retained while the new masks, crossings, census and
  reviews join. The pinned-model recompute with the live reviews
  (`runs/prototypes/v30/rev14_w4_integration/`) certifies cf70's reviewed FULL
  path (43.6764 px) and withholds length for 3756 while keeping its tip; the
  model-only and human-constrained exports are reported side by side.
- **Frame-specific reviewed roots**: a human FULL trace licenses a root for its
  own frame only (point 1 of the reviewed centreline), with provenance, and the
  route/pixel cache keys change on revision or withdrawal. A PARTIAL start is
  never installed as a verified root.
- **Perception failures diagnosed separately** (see
  `runs/prototypes/v30/rev14_w4_diagnosis/`): cf70's cap failure is a *missing
  raw response* (0.017-0.043 at the genuine tips, valid coverage, nearest
  emitted cap 88-91 px), not filtering; 3756's body failure is a supervision gap
  (the body model takes a grain-centred input only, and the visible span
  receives mean support 1.19e-05). The bounded change is supervision only: the
  two new body masks plus both reviewed PARTIAL centrelines as scoped positive
  bands (`rev14_w4_body_fit2`, tube half-width 4.0 px, outside = unknown).
- **Validation truth reconciled** in `route-validation-panel-v2.json`: the
  superseded hidden label is kept as history, the new visible-tip truth is
  recorded, and any score comparison across the change must report it.
- **Queue workflow completed**: pending tasks by default with separate completed
  history, save-and-next without skipping, batch-complete state, queue-mode
  control locks, Look-only default for result inspection, and an explicit
  **Undo saved correction** that restores the prior accepted revision as a new
  audited revision (or removes a first accidental correction with audited
  history). Native widget contracts: `tests/test_rev14_w5_queue_widgets.py`.
