# TubeTracker handoff — rev9 work orders, 2026-09-16 (evening)

Supersedes `prototypes/HANDOFF.md` (2026-09-16 morning). That file is
kept as written; nothing here overwrites it. `HANDOFF_2026-09-15_pre-
instrument-fix.md` is the recovered pre-instrument-fix version.

Written for the rev9 reviewer. Every number below is from an artifact
on disk, named where it matters. Where something did not work, it is
reported as a failure, not reframed as progress.

---

## 1. One-screen state

**The pipeline fits tubes. Ownership is not established. One root
cause blocked everything until today and is now fixed.**

* Body geometry now trains and generalises to held-out material:
  best checkpoint of `rev9_wpB8` reaches dev mask IoU **0.549**
  (proximal 0.571, **distal 0.530**) on a tube it never trained on.
* On the pinned three-grain clump every query returns a **tube-shaped**
  answer, 7% of the crop, IoU 0.54-0.60 on its best tube.
* **Ownership fails**: all three queries answer with the *same* tube.
  `diag 0.254 vs off-diag 0.252`, 1/3 rows peak, row spread 0.064.
* The body head's geometry works; the **owner/query conditioning does
  not discriminate between neighbouring instances**.
* Suite: **712 passed / 0 failed**.
* Ledger `prototypes/LEDGER.md`: H1-H349 (348 rows).

## 2. Root cause found today (explains every earlier flat result)

A **dead ReLU** on the temporal Conv3d, upstream of every head.

Measured on a real crop, stage by stage: encoder feature maps carry
spatial std 0.124 -> the temporal Conv3d's output is constant ->
`relu` zeroes it -> the body head can only return its bias. Direct
proof on `rev9_wpB7`'s checkpoint: the temporal pre-activation is
uniformly **negative** (min -0.690, max -0.042), so `relu` returns
exactly 0 for every input, and `body` emits a constant (std 0.0,
mean 1.3663) **bit-identical across three different masks, prompts and
clips**.

Gradient through a dead ReLU is exactly zero, so the layer cannot
recover and no objective, learning rate, Dice region or supervision
change could ever have altered the outcome. This single defect explains:

* all six flat 60-epoch arms (dev rows bit-identical across epochs);
* the rev9 audit's image-independence finding (0.984/0.983 IoU) —
  same signature;
* why `learn3`'s honest numbers were as bad as they were.

Secondary measurement: the encoder's features carry only 3% spatial vs
97% channel-offset variation (0.004 vs 0.133) — the collapse had
reached into the trunk.

**Fix**: `leaky_relu(negative_slope=0.01)` on that layer (`model.py`).
It cannot revive existing checkpoints (their weights encode the
collapsed solution).

## 3. Current measurements (all reproducible)

| run | what | result |
|---|---|---|
| `learn3` | pre-fix baseline | fit yes, generalise no; dev IoU falls 0.398 -> 0.08 |
| `rev9_wpB4a` | literal `dice_selectors`, 60 ep | flat; final checkpoint all-background, matrix 0.000 |
| `rev9_wpB4b` | `dice_pixel` control, 60 ep, same set | flat; all-background, matrix 0.000 |
| `rev9_wpB5` | + `--body-dice-balanced` | flat (pre-fix) |
| `rev9_wpB6` | + `--body-bg-region band` | flat (pre-fix) |
| `rev9_lr0.01 / 0.03` | trunk-lr ladder, 10 ep | 0.01 still constant; **0.03 diverges** (dev std 2.5e+07) |
| `rev9_relu_check` | 5 ep **after the fix** | dev std 0.36 -> 2.63, `flat False` every epoch, body 1.025 -> 0.754, dev IoU 0.575 |
| `rev9_wpB8` | 60 ep **after the fix** | body 1.025 -> 0.285, non-flat every epoch; dev IoU oscillates (0.59 ep18, 0.55 ep48, 0.08 ep54/59) |

The equal-updates comparison (arms A/B, 2,100 updates each, manifests
identical except the objective) is in `scripts/compare_runs.py`'s
output and in H342: **at that supervision scale the objective choice
made no difference** — both flat, both all-background.

Honest note on checkpoint selection: `rev9_wpB8`'s **last** checkpoint
is one of the bad oscillation points (matrix diag 0.015). `ep49.pt`
(dev IoU 0.549) is the good one, and choosing it by dev IoU **is** a
selection — the last checkpoint's numbers are reported alongside it.

## 4. The three-query proof on one image (as required)

Pinned crop `[475, 518, 288, 288]`, frame 42000, `snap24`, all three
owners present, through `scripts/eval_whole_instance.py` (one shared
inference path, `prototypes/v30_video_apex/inference.py`):

| query | IoU on g0 | on g1 | on g2 |
|---|---|---|---|
| g0 | 0.09 | **0.60** | 0.09 |
| g1 | 0.10 | **0.58** | 0.09 |
| g2 | 0.09 | **0.54** | 0.10 |

`ep49.pt`. Every query picks g1's tube. Geometry is right (tube-shaped,
proximal 0.84 on that tube), ownership is absent. Evidence:
`runs/prototypes/v30/framing_checks/threequery_wpB8_ep49.png`.

Fit criteria on the same checkpoint (in `eval_whole_instance.py`'s
JSON): attachment error 11.2-11.4 px, spill into the other tubes
0.12-0.15, `pred_frac 0.070` — `pred_frac_plausible: True` for the
first time (every earlier arm covered ~30% of the crop or nothing).
`iou_ge_0.90_each: False`; `owned_absence_case`: **none in this
snapshot** — zero false emissions here is not evidence.

## 5. Work packages, honestly

**A.1 review regions — done.** One predicate, `tubetracker/
review_region.py`, shared by app and pipeline. Both paint paths refuse
an extent that misses its own paint, reason recorded; the app is
fail-closed (it cannot store an unusable extent). Exit evidence over
the reviewer's exact 14 learn3 views: **0 licensed by a bad extent**
(was 4: 1300/644/1080/2475 px).

**A.2 round trip — done.** `--reopen` added; the user re-saved; both
masks revision 2, `note=ok`. A duplicate-stamp defect (re-saving doubled
stamps: 274->548) was found and fixed.

**A.3 one builder — done.** `batch_builder.py`; trainer (both branches
+ dev probe) and evaluator share it.

**A.4 checkpoint contract — done.** `model_factory.py`; refuses
undeclared architecture and unfit weights. Legacy `learn3` = 0/10 knobs
-> refused; a fresh checkpoint = 10/10, strict 0 missing / 0 unexpected.

**A.5 candidate schema — done.** `candidates.py` (`candidate_schema: 2`);
the reviewer's 5/8 raw -> 8/8 adapted repro is a test, and the runner
prints `parity vs movie_v18_evgate: 8/8 winners match`.

**A.6 consumers — done.** `label_ledger.py` consumption table (it
caught a real leak on its first run: `body_mask -> vis_valid` default
visibility supervision). Owned absence no longer trains a generic
cap-negative. Route preference is pairwise and never certifies a lane.
Foreign-mask union now requires same canonical movie + same frame +
different physical owner.

**B objective and supervision — measured end to end.** Selectors
implemented with gradient checks (`tests/test_body_selectors.py`:
paint pulls up and background down at every flat field in all three
modes). Then the sequence of honest negatives in §3, and §2's root
cause. Two corrections I made to my own claims are recorded in the
ledger: an "all-one optimum" claim that was wrong (H336) and a
"real dynamics" claim that was a transient (H347 note).

**C microscopy-pretrained comparison — first stage done.**
`scripts/compare_pretrained.py`. cellpose `cpsam_v2`, zero
adaptation, no prompts: 5 instances, 2.9% of crop, all on the GRAINS,
diag IoU 0.0045. micro-sam `vit_b_lm`: with the app's grain-point
prompt diag 0.008 (`grain` prompt puts the point outside a tube-only
mask — its 3/3 "peak" is degenerate); with the review's declared
**verified in-tube prompt** diag **0.108**, per-grain 0.053/0.011/
**0.261** vs our control 0.363/0.059/0.251, off-diag 0.0. The prompt
*representation* is decisive, exactly as the review argued.
`facing`/fine-tune stage is **not done**: it needs the GPU allocation
decision rather than this machine's limits, per the review.

**D movie workflow — integrated early and running.**
`scripts/run_v30_movie.py` end to end on current code: factory load,
46 proposals -> 40 candidates, selection, all four artifacts
(candidates/measurements/report/selection.json). One event (n=1, not a
quality claim): oracle 1.0, selected 1.0, median error 3.37 px inside
a 5-px tolerance. The known route blocker is unchanged (0/8 automatic
selection, median 73 px on the 8-event set).

## 6. What the reviewer's list still needs

1. **Owner conditioning** — the now-isolated blocker. Recommended
   design (the review's own): owner-independent image encoder with
   cached frame features feeding an owner-conditioned decoder. The
   current U-Net mixes the owner into the encoder, and the three-query
   table above shows it does not discriminate.
2. **Training stability** — `rev9_wpB8` oscillates in and out of good
   solutions (dev IoU 0.08 <-> 0.59). Likely LR schedule / seed
   averaging; not yet investigated.
3. **WP-C fine-tune stage** — needs the GPU decision.
4. **Owned-absence example** — no labels of that kind exist yet; the
   reviewer asked for one and it is still an open annotation round.
5. **Fit target** IoU >= 0.90 each: not met (0.09-0.60).

## 7. Traps (all cost real time here)

* **`py_compile` is not a smoke test.** An import patch that never
  landed crashed a launch with `NameError`; a config patch referencing
  a non-existent `--trunk-lr` would have killed every future run. Run
  one epoch to a scratch dir before committing an hour.
* **Code fixes do not reach a running process.** Restart the app after
  save-path fixes; restart the trainer after loss changes.
* **A dead ReLU is invisible in the loss** (H347). Check activation
  statistics, not just losses; `flat_field` warnings exist for this.
* **The band channel** is now computed unconditionally; it used to be
  the quarantine fallback, so with all extents usable it was silently
  empty (H344).
* **`history.json`** now writes every epoch (atomic replace) — an
  interrupted arm used to leave nothing. `--save-every 5` for the same
  reason: a reap at ep17 once cost a whole run.
* Hermes reaps tracked background jobs on agent close; machine sleeps
  between turns — always `caffeinate -i`.
* Two trainers on this machine contend; cap threads if they must run
  together.

## 8. Verify-first commands

```
cd /Users/joshjiang/Documents/TubeTracker
.venv/bin/python -m pytest tests/ -q -p no:cacheprovider          # 712 passed

# the three-query proof on the pinned clump
.venv/bin/python scripts/eval_whole_instance.py \
  --checkpoint runs/prototypes/v30/rev9_wpB8/ep49.pt \
  --snapshot runs/prototypes/v30/snapshots/snap24 \
  --movie "ld=/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4" \
  --frame ld:42000 --crop-size 288 --out /tmp/tq.json

# extent + licensing evidence for snap24
.venv/bin/python scripts/render_extents_sheet.py

# equal-updates comparison
.venv/bin/python scripts/compare_runs.py \
  --run "selectors:runs/prototypes/v30/rev9_wpB4a" \
  --run "pixel:runs/prototypes/v30/rev9_wpB4b"

# the dead-relu diagnostic (stage by stage)
.venv/bin/python /tmp/diag_body_constant.py   # kept: see §9 note
```

## 9. Where things live

* `prototypes/LEDGER.md` — H1-H349 (348 rows); this stretch is
  H315-H349.
* Annotator: `rev8masks`, `rev8masks2`, `rev8pairs` (all round-tripped);
  snapshots snap14-snap24 under `runs/prototypes/v30/snapshots/`;
  **snap24 is current: 8 body-mask samples, 8 usable, 0 quarantined**,
  with licensed reviewed background 7,900-82,559 px per mask against
  paints of 298-1,076 px.
* Evidence: `runs/prototypes/v30/framing_checks/` —
  `threequery_wpB8_ep49.png` (ownership), `threequery_relucheck.png`,
  `extents_snap24.png` (+ .json), `wpC_pretrained_comparison.png`,
  `rev9_roundtrip_evidence.png`.
* JSONs: `threequery_wpB8_ep49.json`, `threequery_relucheck.json`,
  `wpC_*.json`, `threequery_baseline_learn3.json`.
* `scripts/`: `compare_pretrained.py`, `compare_runs.py`,
  `eval_whole_instance.py` (carries the fit criteria),
  `render_extents_sheet.py`, `render_threequery.py`,
  `objective_fit_probe.py`, `fit_criteria.py` (superseded by the
  evaluator's JSON — its checks live there now).
* `prototypes/v30_video_apex/`: `inference.py` (one inference path),
  `batch_builder.py`, `model_factory.py`, `candidates.py`,
  `label_ledger.py`, `route_pref.py`, `targets.py` (named channels:
  `fg`/`band`/`bg_reviewed`/`unknown`), `train.py`.
* Two venvs: `.venv` (research) and `.venv-annotator` (napari). μSAM
  is **not** installed in `.venv` on purpose (its napari dependency
  broke an import-hygiene test); it belongs in its own venv.
* `runs/prototypes/v30/rev9_wpB8_partial17/` — the reaped 17-epoch
  run, kept as evidence (17 epochs of real learning before the reap).
