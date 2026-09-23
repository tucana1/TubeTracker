# TubeTracker — Handoff for the Consultant

Date: 2026-09-16. Branch: `feature/burst-candidate-detection`.
Supersedes: `prototypes/HANDOFF_2026-09-15_pre-instrument-fix.md` (the
previous handoff from this same session; its *conclusions about a flat
body head are void* — see §3). Companion: `prototypes/LEDGER.md`,
newest entries H296–H315 carry everything below.

Read order: this file → LEDGER H313–H315 (the instrument corrections) →
LEDGER H296–H312 → the traps in §7. Section §3 is the one that will save
you the most time; section §8 has the commands to verify state before
touching anything.

---

## 1. Where things stand (one screen)

The frontier is the **v30 video pipeline** (`prototypes/v30_video_apex/`
+ `scripts/train_v30_front.py`), a headless CLI. The desktop app
(`tubetracker/gui.py`) is maintained but runs none of this.

This round moved the blocker twice, and the second move is the
substantive one:

- At H295 the blocker was **selection**: with curved walkers in the
  proposal set the correct route existed (oracle 5/8) but was selected
  0/8 times.
- Through this round the blocker turned out to be **the body loss and
  the way we measured it**. After fixing both: **the pipeline can fit
  tubes** (body loss 0.73 → 0.067 on three masks, front error 88 → 36 px,
  3360 updates) and **it does not yet generalise** (held-out masks sit at
  crop-IoU 0.005–0.115; best 0.115 at ep40). Those are the first honest
  numbers for either question in this ledger.

So step 2 of the work order ("prove whole-instance learning on a few
neighbouring tubes") is no longer blocked by the model, the loss or the
optimiser. It is blocked by **generalisation**, i.e. data/coverage and
the full mixture — see §6.

## 2. What this round established

1. **Route selection: the evidence gate is a filter, not a selector**
   (H296). `evidence-gated route_p` beats shipped `q_max × route_p`;
   evidence alone is useless; `--select-evidence-gate` defaults to 0.0
   and is bit-identical. One selection policy now lives in
   `prototypes/v30_video_apex/selection.py` and is shared by the runner
   and the evaluator.
2. **Annotation contract, three fixes** (H303/H304/H306), all driven by
   the user reporting what they saw on screen:
   - a foreign *labelled* tube in the crop is known **not-my-tube**
     (validity 1, target 0), not "unknown material" — without this the
     confusable-negative term multiplied zero and an A/B came out
     byte-identical (`add_confusable_validity`, tested);
   - the mask task's ring belongs **on the grain**: `focus_xy` means
     different things across observation families (ball for pair
     observations, a path point for route observations), and the builder
     wrote `target_xy` twice (the last write won). Now
     `ring_anchor_for_observation()` + `refine_ring_anchor()` (step one
     grain radius upstream along the tube, then snap onto the measured
     blob) + `measure_blob()` (threshold → **opening to sever the
     contiguous tube** → component centroid and area-equivalent radius).
     Ring radius = 1.5 × r_eq, floor 20 px, so the marker is bigger than
     the uncertainty in what it points at;
   - the camera centres the **ring** (the guide still sets the zoom).
3. **Every reviewed extent ever saved was unusable** (H306): napari
   0.9.1 reports a **3-component camera centre (z, y, x)** and the code
   read `[0], [1]` as (y, x), so z (always 0) became the y centre.
   Every historical extent falls outside its own crop and the loader
   silently degraded `complete` masks to band-only validity. Fixed
   (read the last two components). Historical extents are **not**
   reconstructible (the x centre was never recorded) and were
   deliberately left alone — a guessed extent would supervise pixels
   nobody looked at. The manifest now reports
   `n_masks_extent_usable` / `n_masks_extent_unusable` (snap20: 0 / 8)
   so the loss is visible instead of hidden.
4. **Annotation rounds completed by the user** (all on snapshots):
   `rev8masks` 3 body masks, `rev8pairs` 3 paired-clump paint tasks (the
   swap criterion's clump, frame 42000, three grains 55–128 px apart),
   `rev8masks2` 2 long-tube masks (274 and 359 stamps). Snapshot chain
   snap16 → **snap20** (12 masks, 14 duels, 124 observations).
5. **Real bugs in the plumbing, found by measurement** (see §3–§4): the
   body loss had a degenerate flat valley; the heads had no learning
   rate of their own; the swap metric flattered a dead model; two of my
   own instruments were broken.

## 3. The measurement problem — read this before believing anything

This session produced three *independent* ways that a trained checkpoint
looked dead, plus two metric traps. All five are now fixed and tested,
but they will bite any new evaluator. **When a measurement says
"everything is the same", suspect the instrument before the world.**

**Instrument failure 1 — the loader (H313).** My external evaluator did
`load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)`.
Checkpoints store weights under **`model_state`**, so it passed the whole
checkpoint dict and `strict=False` matched **nothing**: every measured
checkpoint was a randomly-initialised model. That is where "the body
head is flat at every scale", "logit std 0.003", "IoU 0.009" and the
ten-checkpoint bisection came from. The tell: *identical* numbers across
wildly different checkpoints. Any evaluator must read `model_state`
explicitly and **assert** the missing/unexpected key counts.

**Instrument failure 2 — the temporal window (H314).** The trainer feeds
a **9-frame window** (`dataset.QUERY_OFFSETS = (-8…+8)`,
`query_index=4`), not a single frame. Evaluating one frame is permanently
out of distribution.

**Instrument failure 3 — starving the run (H314).** With
`--mask-jitter 0` the trainer produces **one update per epoch** (the epoch
line reports `updates_this_epoch: 7.0` with jitter, `1.0` without):
jitter is what *generates* the sights. A "clean" jitter-free replication
froze at loss 0.5995 for 1400 epochs and proved only that.

**Metric trap 1 — valid-region IoU (H309).** `mask_iou_split` scores
*inside the valid region*, and the valid region of a complete mask is
essentially a band around the paint. A spatially **constant** field above
the threshold therefore scores ~1.0 there (H299's "plumbing IoU 1.000")
while localising nothing; the same constant scores 0.009 crop-wide. The
dev probe now reports `body_logit_std`, `body_logit_mean`, `crop_iou`,
`pred_frac_crop` and a `flat_field` flag, and prints a loud
`WARNING ep N: FLAT body field …` into the run's own log.

**Metric trap 2 — the swap summary counted ties as peaks (H307).** A row
was "peaking on its own tube" if `diag >= offdiag`; `0 >= 0` holds for
every column, so an all-zero (collapsed) matrix scored **3/3** on the
very test built to catch query-invariance. Now strict, plus a
`degenerate` flag the report prints as `inconclusive(model predicts
nothing)`.

**The trustworthy instrument** is the trainer's own dev rows (it loads
the model it is training and runs the same pipeline). Prefer them over
any external script; if you must evaluate externally, replicate
§3-failures 1–3 exactly and report crop-IoU.

## 4. Loss and optimiser changes (and why)

- **The split-normalised body BCE had a degenerate flat valley** (H312).
  Normalising each class by its own mass gives the *uniform* direction
  exactly zero net gradient: positives pull up, negatives pull down,
  cancel — the loss sits at **0.6931** (the entropy of a coin flip) while
  the prediction flips between all-foreground and all-background.
  Measured across loss shapes (split, ×20, pos-weighted pixel-BCE):
  every one stayed at crop-IoU 0.009 and `pred_frac` ~1.0.
  **Fix:** `--body-objective dice_pixel` = soft Dice + a small
  pixel-normalised BCE (0.1). Dice cannot be reduced by a constant; this
  escaped the valley (logit spread 0.001 → 4.4 in 1500 steps). `split`
  remains the default so old runs stay reproducible — for any new run,
  pass `dice_pixel`.
- **Heads need their own learning rate** (H312): same 1500 steps give
  logit spread 0.65 at lr 0.5 versus 0.046 at 0.05; the trunk's 0.003 is
  ~150× too slow for a 1×1 head. `--head-lr 0.5` puts the heads in their
  own AdamW group (16 head tensors at 0.5, 40 trunk tensors at 0.003).
  Verified that updates really happen: at head-lr 5.0 the body term moves
  (0.66 → 0.79, worse, then plateaus); at trunk lr 1.0 the loss explodes
  to 6e11 before recovering.
- **Confusable negatives** (`train.BODY_CONFUSABLE_WEIGHT`): a
  wrong-instance activation must cost something. A **fixed 10×**
  collapsed the full-dataset run (held-out IoU 0.000 at ep11/ep15/final
   versus 0.217 for weight 0 — but see §3: both arms were measured
  through broken instruments, so **the fixed-weight verdict needs
  re-running**). `--confusable-mode balance` scales the square so the
  confusable pixels carry at most as much total weight as the paint
  (`confusable_neg_weights`, tested).
- **A 2-minute testbed beats a 2-hour run**: `scripts/head_fit_probe.py`
  freezes the trunk and fits only the 1×1 body head against a real target,
  sweeping loss shapes / LRs and reporting crop-IoU and logit spread.
  Every question this session asked with a full run could have been asked
  here first — that is the single biggest process lesson.

## 5. Results, stated honestly

`learn3`: 3 train masks (`mask-rev8m-000/001/002`), 240 epochs, 3360
updates, dice_pixel, head lr 0.5, jitter 6, dev = the held-out p05 masks.

| epoch | train body loss | dev IoU (v30m2-001 / rev8m-002) | dev crop-IoU | dev logit spread |
|---|---|---|---|---|
| 0 | 0.732 | 0.398 / 0.217 | 0.016 / 0.010 | 0.008 |
| 40 | 0.565 | 0.215 / 0.153 | 0.115 / 0.079 | 1.14 |
| 160 | 0.067 | 0.143 / 0.102 | 0.010 / 0.006 | 59.0 |
| 239 | 0.110 | 0.114 / 0.083 | 0.008 / 0.005 | 36.3 |

Read the dev IoU *falling* as a good sign: at ep0 the model was a constant
field and a constant scores IoU = paint/valid for free; 0.398 was never
a prediction. No flat-field warnings fired anywhere in the run.

Front head, same round: `mean_dev_front_px` 88.3 → 33–36 px in the
jittered configurations (H314).

## 6. Open questions, in priority order

1. **Generalisation** (the live blocker). Fitting is proven; held-out
   localisation is not. Next: run the *full mixture* on the fixed
   objective (dice_pixel + head lr) and read the dev rows; more masks
   move this (annotation is the lever).
2. **Re-derive everything measured before H313.** The swap matrices, the
   distal readouts, the confusable A/B, the H299 plumbing claim — all
   were produced through the broken evaluator and/or the tie-counting
   metric. `whole_instance_report.py` now uses the strict metric, but it
   must be pointed at *freshly trained* checkpoints.
3. **My external evaluator still disagrees with the trainer's dev rows
   at the same crop**, even after matching weights, window and jitter.
   Unresolved. Treat the trainer's rows as the standard; do not trust an
   external number until it reproduces one.
4. **`--prompt-mix` looks inert**: `auto-grain:1.0`, `human:1.0` and
   `none:1.0` produced byte-identical loss trajectories. Either the
   prompt never reaches the body term or the flag is not plumbed. Unproven
   — worth 20 minutes.
5. **H306's loss is permanent for the 8 historical masks** (band-only
   validity). The next annotation round records usable extents; until
   then "complete" means less than it says.
6. **Step 3 of the work order is untouched**: offline tracking and the
   complete movie workflow. `scripts/run_v30_movie.py` uses the shared
   selection policy, but it has not been exercised since the loss fixes.

## 7. Traps (updated for this machine)

- `/tmp` is wiped; **`timeout` does not exist on macOS**; the low-density
  movie filename contains a **space before `.mp4`**
  (`test1lowdensjoshua-28c-hz.mp4 .mp4`).
- The machine sleeps between agent turns: wrap long jobs in `caffeinate -i`.
- **Hermes reaps its own tracked background processes when the agent
  closes** (`termination_source: agent_close`, exit −15, seen four times).
  Always `--save-every`, and resume from the last checkpoint with
  `--init-checkpoint … --fresh-optimizer`.
- Run **one** trainer at a time; two starve each other (~20 min/epoch).
- The trainer **refuses to overwrite an existing out-dir** (and refuses a
  snapshot with a (movie, frame) in both splits, and duplicate owner
  keys). Clear the directory rather than renaming the flag.
- napari 0.9.1: no `camera.rect`; **3-component camera centre**; offscreen
  (`QT_QPA_PLATFORM=offscreen`) **segfaults** — never evaluate camera code
  headless, use the fakes in `tests/test_review_extent.py`.
- The snapshot builder **refuses a project dir passed twice** (a real
  double-count happened) and fails closed on a missing db.
- zsh + `set -u`: guard empty arrays before expansion.
- Recipe for any new external evaluator: load `model_state` and assert
  key counts → feed the 9-frame `QUERY_OFFSETS` window with
  `query_index=4` → never disable jitter (it starves the run) → report
  crop-IoU and logit spread, never valid-region IoU.

## 8. Verify before you touch

```bash
cd /Users/joshjiang/Documents/TubeTracker
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests/ -q -p no:cacheprovider
#   expect: 644 passed

# snapshot + its honest counters
.venv/bin/python -c "import json;m=json.loads(open('runs/prototypes/v30/snapshots/snap20/snapshot_manifest.json').read());print({k:m[k] for k in sorted(m) if k.startswith('n_masks') or k.startswith('n_body') or k.startswith('n_duel')})"

# a project's labels reach the snapshot and land in the right crops
.venv/bin/python scripts/verify_label_pipeline.py \
  --project-dir "$HOME/Documents/TubeTracker-annotator-projects/rev8masks2" \
  --movie "ld=/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"

# the 2-minute head-fit testbed (sweeps loss shapes / head LR)
.venv/bin/python scripts/head_fit_probe.py

# the acceptance run: fit three tubes, read the dev rows (see §5 table)
caffeinate -i .venv/bin/python -u scripts/train_v30_front.py \
  --snapshot runs/prototypes/v30/snapshots/snap20 \
  --out-dir runs/prototypes/v30/<new-dir> \
  --only-refs "mask-rev8m-000,mask-rev8m-001,mask-rev8m-002" \
  --dev-refs r4-p05,dt-005 --dev-frames ld:49350,ld:11831 \
  --epochs 240 --seed 0 --multiscale --base 4 \
  --lr 0.003 --head-lr 0.5 --mask-jitter 6 --save-every 40 \
  --body-objective dice_pixel --body-pixel-bce 0.1
```

## 9. Where things live

- Pipeline: `prototypes/v30_video_apex/` (`targets.py` — masks, anchors,
  blob measurement, confusable validity; `model.py` — OwnerPrompt,
  `resolve_prompt_kind`; `train.py` — `masked_multihead_loss`,
  `soft_dice_loss`, `confusable_neg_weights`, objective switches;
  `selection.py`; `dataset.py` — `QUERY_OFFSETS`).
- Trainer/app: `scripts/train_v30_front.py`, `tubetracker/annotation_app.py`
  (`_fit_camera_to_task`, `_current_review_region`), `scripts/run_annotation_app.py`.
- Snapshots: `runs/prototypes/v30/snapshots/snap{14..20}` (snap20 is
  current); backups `runs/prototypes/annotation_backups/`;
  `scripts/rebuild_v30_snapshot.sh` (dedupes projects),
  `scripts/build_v30_snapshot.py`, `scripts/snapshot_diff.py`.
- Reports: `scripts/whole_instance_report.py`, `scripts/eval_whole_instance.py`
  (strict swap summary + `degenerate`), `scripts/head_fit_probe.py`.
- Annotation projects: `~/Documents/TubeTracker-annotator-projects/`
  (`rev8masks`, `rev8masks2`, `rev8pairs`, `v30w2`).
- Movies: `/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4`
  (`ld`, 52,583 frames, 3 s cadence) and
  `/Users/joshjiang/Downloads/Pollen tube movie 2 7-14-26.mp4` (`m1`/`m2`).
- Envs: `.venv` (research: torch, cv2) and `.venv-annotator` (napari).
- Ledger: `prototypes/LEDGER.md`, H296–H315 for this round
  (H313–H315 are the corrections that invalidate earlier readings).
