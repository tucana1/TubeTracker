# Learned evidence for SparseTrack (experiment)

**Question:** can a small network trained only on SparseTrack's codec-exact synthetic movies give
the unchanged SparseTrack decoder better evidence than its hand-built change maps?

The network reads the same three registered images SparseTrack uses: the bin being read, the
"before" reference (first 3 bins) and the "after" reference (last full bins). It outputs, per pixel,
the probability that tube has already been built there, plus a tip heatmap. Nothing in
`sparsetrack/` changes.
- The per-bin probability, ×16, is written as a SparseTrack cache and analysed like a movie.
- At that scale SparseTrack's grey-level thresholds sit at P ≈ 0.3 (tube map), and its front evidence
  runs from −1 at P = 0 to +1 at P ≥ 0.63.
- Per-grain registration is measured on the image cache.

## For the lab: analyse a new movie (prototype)

1. Double-click `Analyze_Movie_Learned.command` in the repository folder and choose the movie.
   - It prepares the movie the first time, which takes a few minutes. The cache is the same one SparseTrack uses.
   - It then runs three analyses, taking 10–20 minutes on a laptop:
     - SparseTrack as it is;
     - learned evidence through SparseTrack's decoder;
     - learned evidence through the per-bin decoder.
2. The review gallery opens by itself, one card per grain, with grains that need a second look first. Each card shows
   six moments from germination to the end, with the tube the per-bin decoder measured outlined in green and its
   centreline in yellow, then its length over time. Check that the outline follows the grain's own tube.
3. Results are in `runs/learned_evidence/<movie name>/`:
   - `per_grain.csv`: per grain and per analysis, the status, onset interval and final length;
   - `perbin/population.png`: the germination curve with T50;
   - `perbin/growth_curves.png`.
4. To check and correct them, double-click `Review_Movie_Learned.command` and choose the same movie.
   - Your labelling tool opens with the model's answers already filled in: an onset bracket and traced tubes for
     every grain. Confirm or fix each grain as you would label it.
   - Press Ctrl-C in the window when you stop. `reviewed_grains.csv`, `reviewed_traces.csv` and `population.png`
     are then written next to the results, saying which answers you checked and which you changed.
   - Run it again to carry on; answers not yet checked stay the model's.
   - Each analysed movie keeps its probability cache (`prob_cache/`, about 0.5 GB) for the review. Once the review
     is pre-filled, it can be deleted; re-running the analysis rebuilds it.

For physical units, run the pipeline command below with `--um-per-px` and `--s-per-frame`.

**Once, after the dev labels are done:** double-click `Adapt_Learned_To_Dev_Movie.command` (about 2.5 hours the first
time; it can be stopped and started again). It runs the dev test on your labels, then calibrates the decoder and
fine-tunes the network on your traces, keeping each only if its check says so. `runs/learned_evidence/SUMMARY.md` then
says what each step found and what the launcher above uses from now on.

**This is a prototype.** Its accuracy has been measured only on synthetic movies. Your dev benchmark (`ld_v1`,
appendix B of the assessment) is the real test. Until it is done, trust the numbers only for grains whose card looks
right. Burst frames and growth-arrest frames are hints for review, not measurements.

| File | What it does |
|---|---|
| `truth.py` | Exact per-frame truth rasters (built tube body, instances, tips) from a `sparsetrack.synth.Scene`, reusing its geometry: rotation, drift, substrate anchoring, sway |
| `data.py` | Training crops (bin, before, after) plus targets from a synthetic movie's cache and scene |
| `model.py` | 0.49 M-parameter U-Net (BatchNorm, so tiled inference does not depend on tile size); tiled prediction |
| `train.py` | Training: dihedral, gain, offset and noise augmentation; BCE + Dice for the body; weighted BCE for tips. CPU, MPS or CUDA |
| `evaluate.py` | Probability caches; end-to-end SparseTrack runs (baseline, learned, and "perfect" = exact truth masks as evidence); the adaptive crop (below); oracle-path fronts; paired bootstrap over grains |
| `pipeline.py` | One command for a real movie: synthetic movies on its field → shards → training → probability cache → three runs scored on its human labels (SparseTrack as it is; learned evidence through SparseTrack's decoder; learned evidence through `reach.py`) |
| `reach.py` | Decoder v2, the per-bin decoder: in every bin, the medial-axis length of the region with P > 0.5 attached to the grain, then a monotone fit over bins |
| `review.py` | Review pictures for the per-bin decoder on the movie itself (six registered bins with the region read and its medial axis, then the length curve), in SparseTrack's own review gallery |
| `adapt.py` | One command for the dev movie: the dev test, then `calibrate.py`, then `finetune.py`, each skipped once done; writes `SUMMARY.md` with the verdicts, what the launcher uses and the movie-2 command |
| `prefill.py` | Writes the per-bin decoder's answers as a labels file for the labelling tool (onset brackets; traced tubes at the bins the tool asks for), through the tool's own API, marked as the model's, for review |
| `export_review.py` | Results from reviewed labels: per-grain and per-trace CSVs (checked, changed from the model's) and the germination curve |
| `calibrate.py` | Fits the per-bin decoder's end offset on a movie's human traces, with a check over folds of grains; writes `decoder.json` only if the gain is clear of noise |
| `finetune.py` | Fine-tuning on a movie's human traces, with a check that holds out grains and traced frames; writes the tuned model only if it reads more right |
| `show.py` | Side-by-side panels (registered bin, SparseTrack's evidence, learned probability) for real footage |
| `models/unet_v2_sample_field.pt` | The trained v2 model (ten synthetic movies on the sample movie's field), for a quick first look |

## Run it on the dev movie (on the machine that has the movies)

```bash
.venv/bin/pip install torch==2.13.0          # the project's `cnn` extra
.venv/bin/python -m prototypes.learned_evidence.pipeline \
    --field runs/sparsetrack/ld --labels benchmark/labels/ld_v1.json --work runs/learned_evidence/ld
```

About 1.5 hours on a laptop. That is ten synthetic movies at about 5 min each, training (about 30 min on 4 CPU
cores, less on an Apple GPU) and a probability cache for the real movie. It prints both SparseTrack runs
scored on `ld_v1` and a paired bootstrap of the difference.

`python -m prototypes.learned_evidence.adapt` runs this, then the calibration and fine-tuning below, in that order
(each skipped once its report exists), and writes `runs/learned_evidence/SUMMARY.md`.

**Any movie, no labels.** Leave out `--labels` to run all three on a new movie. The pipeline then writes each run's
`predictions.json` and a `per_grain.csv` (status, onset interval and final length per run) instead of scores. For
example, add `--field runs/sparsetrack/<movie> --model runs/learned_evidence/ld/unet.pt`.
- `--um-per-px` and `--s-per-frame` add final lengths in µm and onsets in minutes to the CSV.
- With or without labels, `perbin/` also holds what the lab would look at:
  - `index.html`, a review gallery of every grain drawn on the movie;
  - `population.png` and `population.csv`, the germination curve (Turnbull, with T50);
  - `growth_curves.png`.

**Quick first look (about 10 minutes).** Add `--model prototypes/learned_evidence/models/unet_v2_sample_field.pt`
to skip the synthetic movies and training. That model (2 MB) is v2 below: trained on ten synthetic movies built on
the field of `sample_movie.avi`, the repository's only movie. The full run builds them on your own `ld` field, so it
remains the proper test.

**Long tubes.** SparseTrack reads each grain in a ±150 px crop, and a tube that leaves it stops at the edge with no
flag. Movie 2's longest tubes pass ±128 px, so both runs use `evaluate.adaptive_crop`:
- A grain whose path ends at the crop edge is read again at ±300 px.
- The matched kymograph is read in angle groups, because OpenCV's `remap` crashes on paths longer than ~268 px.
- On `ld` no path reaches the edge, so nothing changes. `--fixed-crop` turns it off.
- On a synthetic movie of movie 2's length it takes the traces that leave the crop from 9/20 to 20/20 in tolerance.
  The other 33 grains are bit-identical.

**Calibrate the decoder on your traces (after the dev test, about 10 minutes).** The per-bin decoder adds a constant
to every length it reads, because the medial axis stops short of a tube's end. How far short depends on how the
evidence looks at tube ends. `calibrate.py` fits that one number on the `ld` traces:

```bash
.venv/bin/python -m prototypes.learned_evidence.calibrate --field runs/sparsetrack/ld \
    --labels benchmark/labels/ld_v1.json --work runs/learned_evidence/ld_cal
```

- It decodes with offsets from −6 to +3 px and picks the one with the most lengths and onsets in tolerance.
- The check reads each third of the grains with the offset the other two thirds picked.
- It writes `decoder.json` only if the 95% interval for the gain in lengths lies above zero and onsets are no worse.
  Picking the best of ten on the same traces flatters small gains, hence the stricter rule.
- `pipeline.py --decoder` uses it, and `finetune.py` and `Analyze_Movie_Learned.command` pick it up by themselves.
- Fit it on a model that was not tuned on the same traces. It refuses a model fine-tuned on them.

**Fine-tune on your traces (after the dev test, about 45 minutes).** `finetune.py` tunes the network on the `ld`
traces and checks whether that helps before anything uses the result:

```bash
.venv/bin/python -m prototypes.learned_evidence.finetune --field runs/sparsetrack/ld \
    --labels benchmark/labels/ld_v1.json --work runs/learned_evidence/ld_ft
```

- Run it after calibrating. It reads an adopted `decoder.json` as the launcher does, so the check judges the model
  with the decoder it will be used with.
- **What it learns from.** Tube along each traced line, and background in a band beside it that starts past the
  tube's walls, measured on the image, since real tubes are wider than synthetic ones. Since tubes only grow, it also
  labels bins that were not traced:
  - before a grain's onset, its whole future path is background;
  - between two traces, the part both agree on is tube, and the path past the later trace is background.
- Everything else is held to what the network predicted before, so untraced tubes are not taught as background.
- **How it is checked.** The grains are split into three folds, and so are the traced bins. Each fold's grains are
  read at that fold's bins by a model tuned without those grains and without any label within 3 bins of those bins.
  - The frames matter as much as the grains. Tuned on traces from a few frames, a network reads other grains better
    in exactly those frames, and worse elsewhere. A split by grain alone would reward that.
- **What you get.** `report.txt` gives both models' scores and the paired difference.
  - The final model (every trace) is written as `unet_ft.pt` only if the check says it reads more right than the
    starting model. Otherwise it is written as `unet_ft_not_adopted.pt`.
  - `Analyze_Movie_Learned.command` uses `unet_ft.pt` when it exists.
- It refuses any labels file with `m2` in its name.

**Movie 2 stays held out.**
- Never pass its cache as `--field` or `--train-field` for training.
- Score it once, with a model trained on the dev field:

```bash
.venv/bin/python -m prototypes.learned_evidence.pipeline --field runs/sparsetrack/m2 \
    --labels benchmark/labels/m2_v1.json --model runs/learned_evidence/ld/unet.pt \
    --work runs/learned_evidence/m2 --heldout-once
```

- If fine-tuning was adopted, pass `--model runs/learned_evidence/ld_ft/unet_ft.pt` instead.
- If calibration was adopted, add `--decoder runs/learned_evidence/ld_cal/decoder.json`.
- Whatever is frozen is scored once.

## Results so far

Full tables are in `docs/assessment-2026-09-24.md`, section 5. All numbers come from synthetic movies built on the field of
`sample_movie.avi`, the only movie in the repository. The real test is the command above.

- **Evidence only** (true per-bin geometry, SparseTrack's `dp_front`; held-out seed 3):
  - learned: 157/173 lengths in tolerance (91%), median error 1.12 px;
  - SparseTrack's `union` evidence: 124/173 (72%), median 1.78 px.
- **End to end** (unchanged SparseTrack decoder; held-out v5 seeds 3, 4, 6, 7 and 8, 135 grains):

  | Evidence | Lengths in tolerance | Median error | Onsets | Germinations missed |
  |---|---|---|---|---|
  | Learned v2 (10 synthetic movies, the default) | 734/1014 (72%) | 1.60 px | 86/112 | 7 |
  | Learned v1 (5 synthetic movies) | 699/1014 (69%) | 1.62 px | 82/112 | 12 |
  | SparseTrack's own | 623/1024 (61%) | 2.12 px | 79/112 | 6 |
  | Perfect evidence (ceiling) | 760/1014 (75%) | 1.33 px | 102/112 | 8 |

  - Paired over grains, lengths: v1 +76 traces over SparseTrack's evidence (95% CI −16 to +165), v2 +111 (+25 to
    +195), v2 over v1 +35 (+6 to +70).
  - Bright-cored tubes go from 41% to 70–71%.
  - Weak spot left: drifting grains (67% with SparseTrack's evidence, 63% with v2).
  - Development seed 5: v1 144/192, v2 141/192, SparseTrack's evidence 102/192.
- **Replication** (three fresh movies, v5 seeds 13–15, rendered after v2 was chosen; then four more, 22–25, below):
  - v2 398/601 (66%) against SparseTrack's evidence 389/612 (64%): +9 traces (95% CI −47 to +64), within noise.
  - Pooled over all eight held-out movies (216 grains): 1132/1615 (70%) against 1012/1636 (62%), +120 traces
    (+17 to +226). Perfect evidence: 1176/1615 (73%).
  - The gain per movie runs from −3 to +43 traces. It follows the mix of tubes: bright-cored tubes and sways gain
    most, and drifting grains lose.
- **Two integration choices**, made on the development seed only:
  - Onset comes from the growth front (`onset_source="front"`). SparseTrack's matched stub filter z-scores against
    control angles that are exactly zero on probability maps, and called grains "emerged at start" (3/22 onsets).
  - Tip offset is 0 px: a 2 px offset scored 136/192 against 144/192.
- **The decoder's own ceiling:**
  - With perfect evidence (exact truth masks), SparseTrack's decoder reaches only 73% of lengths in tolerance over
    the eight held-out movies (58–89% per movie). Rotating, drifting and curling tubes remain its losses.
- **Decoder v2: `reach.py`, the per-bin decoder.** It reads the region with P > 0.5 attached to the grain in every
  bin, measures its length along the medial axis (+1 px), then fits a monotone L1 curve. Settings were frozen on the
  development seed. With learned v2 evidence:
  - First eight held-out movies: a tie with SparseTrack's decoder, 1144 against 1132 of 1615 (+12, 95% CI −67 to
    +88).
  - Seven new development movies (seeds 5, 16–21): 1073 against 989 of 1399 (+84, +32 to +139), better on every
    tube type.
    - No label-free per-grain switch between the two decoders beat using `reach.py` everywhere. The switches
      tried were width, path coverage, drift, rotation and the longer reading, each scored leave-one-movie-out.
  - Four untouched test movies (seeds 22–25), frozen beforehand: 664 against 620 of 860 (+44, −6 to +98), onsets 74
    against 67. SparseTrack as it is gets 504 there.
  - Pooled over the twelve held-out and test movies: 1808/2475 (73%) against 1516/2496 (61%) for SparseTrack as it
    is (+292, +170 to +413). It is ahead on all twelve movies, and drifting grains no longer lose.
  - `pipeline.py` scores it as a third run (`perbin/`), so `ld_v1` decides on real footage.
- **Bursting tubes** (tubes inpainted away from a random bin on, in four synthetic movies; 28 tubes, 159 traces before
  their bursts):
  - SparseTrack as it is keeps 43% of those lengths in tolerance, and learned evidence through SparseTrack's
    decoder 47%. SparseTrack's decoder reads the path at the end of the movie, when the tube has gone.
  - The per-bin decoder keeps 71%.
  - Its burst-aware fit (`burst=True`, on in `pipeline.py`) adds a little: +11 and +5 lengths. It cannot time bursts
    (5 of 28 within ±2 bins).
- **Real footage** (qualitative, `show.py`): tube-specific, near zero on grain bodies, but misses wide, dark-walled
  tubes whose profile is outside the synthetic range.
  - Training with four extra movies of 1.3–2.5× wider tubes (`v5w`) did not fix it (model v3, not adopted).
  - Thin-tube lengths were unchanged, and onsets improved +7 with the per-bin decoder.
- **Fine-tuning on sparse human traces** (`finetune.py`; synthetic test). Traces were made from the synthetic truth to
  look like yours: the same bins as `ld` (70, 122 and 174, plus one near onset) and the same number of clicks for a
  tube's length. They were made on movies whose tubes the shipped model was not trained on: faint (0.35–0.7× contrast),
  thick bright-cored (2–3× wider) and wide (`v5w`). About 90 traces per movie, 27 grains.
  - The first version was worse, and each fault was fixed on the development movies:
    - Tube labelled past the traced apex made every tube read ~1.5 px long.
    - Traces from only three frames taught the network those frames: other grains read better in them (+13 points)
      and worse elsewhere (−4.5 points beyond 12 bins). A check split by grain alone showed +10 lengths against a
      true −9. Hence the folds over frames, and labels spread over time.
    - A fixed background band fell inside thick tubes' walls. Background now starts past the width measured on the
      image.
    - Tuned on faint tubes, the network saw tubes on grain rims from the first bin (3 of 27 grains). Hence the ring
      labels.
  - Final version, all bins scored against the full truth (per-bin decoder):

    | Tuned on | Same movie (cross-validated) | Verdict of the check | Fresh movie of the same kind | Thin-tube movie (v5 seed 5) |
    |---|---|---|---|---|
    | Faint tubes | 49.7% → 48.7% | not adopted | 61.1% → 60.7% | −29 lengths (95% CI −47 to −12) |
    | Thick tubes | 28.1% → 33.5% | adopted | 21.3% → 22.2% | −3 (−13 to +7) |
    | Wide tubes | 71.2% → 74.5% | adopted | 75.5% → 79.0% | −10 (−22 to +1) |

  - The check's verdict matched the truth on all three.
  - The gains are real but small, and a tuned model is worse on movies unlike the one it was tuned on. Use it only
    for movies taken under the same conditions as `ld`.
- **Calibrating the decoder's end offset on sparse traces** (`calibrate.py`; the same synthetic movies and traces).
  The offset was picked on the development movie's traces, then applied to a fresh movie of the same kind:

  | Movie | Check on the development traces (lengths, 95% CI) | Verdict | Fresh movie, lengths in tolerance |
  |---|---|---|---|
  | Thick tubes (read 5.6 px long) | +19 (+8 to +31), offset −5 | adopted | 41 → 90 |
  | Faint tubes | +3 (−3 to +10), offset +3 | not adopted | would have been 119 → 114 |
  | Wide tubes | +1 (−11 to +13), offset 0 | not adopted | would have been 156 → 149 |

  - Where a movie's tubes all read long or short by about the same amount, one fitted number more than doubles the
    lengths in tolerance. That is far more than fine-tuning gave on the same thick tubes, whose bias it barely
    moved (+5.6 to +5.0 px).
  - Faint tubes read short in proportion to their length, which no single offset fixes.
  - After calibration, fine-tuning added nothing on thick tubes (fresh movie 90 → 84 lengths, 95% CI −18 to +7).
    Its check, reading with the calibrated offset, did not adopt it (−1, −4 to +2). So calibrate first, and let
    fine-tuning's check decide on top.
  - Letting the traces pick the probability threshold as well made the full-truth scores worse (wide tubes: 191
    against 201 lengths and onsets in tolerance), so only the offset is fitted.
- **Pre-filled review labels** (`prefill.py`; synthetic movies, per-bin decoder). The decoder's path starts where
  the tube region crosses the grain's rim and follows its medial axis.
  - Wide tubes: traces start 1.0 px from the true exit and follow the centreline within 0.48 px (90th percentile
    1.38 px). 54 of 87 lengths are already within tolerance, so most grains need a confirmation, not a trace.
  - Thick bright-cored tubes: starts are as good (1.0 px), but the path runs 3.3 px from the centreline. The
    network marks one of the dark walls, and 22 of 73 lengths are within tolerance there.
  - Your labelling tool opens the file unchanged, asks for a new trace where a corrected onset moves the plan, and
    keeps the model's and your answers apart.
  - On `sample_movie.avi` (real footage, no labels) the traces lie on the visible tubes. Where the decoder's
    reading of a bin fades, the trace is taken from the last bin it saw the tube that long, never stretched. Where
    two grains touch, a trace can follow a tube between them, and the reviewer decides whose it is.

