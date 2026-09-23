# TubeTracker — Handoff for the Consultant

Date: 2026-09-15. Branch: `feature/burst-candidate-detection`.
Supersedes: `prototypes/HANDOFF_2026-09-07.md` (still on disk; covers the
v29 + TimesFM program). Companion: `prototypes/LEDGER.md`, newest
entries H262–H296 carry everything below.

Read order: this file → LEDGER H262–H296 → LEDGER "Hard-won
cross-cutting lessons" → the section-7 traps below. Then verify with
the commands in section 8 before touching anything.

---

## 1. Where things stand (one screen)

Two systems live in this repo and only one of them is the frontier:

- **Desktop app** (`tubetracker/gui.py` + `analysis.py`) — maintained,
  wxPython, LapTrack-based routine analysis. Unchanged this round.
  It is *not* running any of the newer research engines.
- **Research pipeline** (`prototypes/v17…v30`, headless CLI) — where
  all progress has happened. Current state: **v29 is the measurement
  engine** (owner-anchored lengths, gates, censoring, sticky tip
  evidence), **TimesFM-3 is a live fail-closed veto gate inside it**,
  and **v30 is the attempt to see a tube directly** (learned front +
  route proposals) that is now one measured step past its blocker.

The v30 blocker, measured at H295: with curved walkers in the proposal
set, the *correct* route exists at 0–2 px (oracle 5/8) but the model
selected it 0/8 times (median error 73 px). This round we found out
what the wrong choices were made of, and shipped a fix that is small,
causally isolated, and verified:

| Run | selected hits | median tip err | p90 | oracle |
|---|---|---|---|---|
| `movie_v17` (H295 baseline) | 0/8 | 73.0 px | 117.0 px | 5/8 |
| `movie_v18_control` (gate off, new snapshot) | 0/8 | 73.0 px | 117.0 px | 5/8 |
| **`movie_v18_evgate` (gate 0.6)** | **1/8** | **33.4 px** | **79.8 px** | 5/8 |

The control run is winner-identical to the baseline, which proves the
gain is the new gate and not snapshot drift. Offline, on route
coverage rather than tip distance, the gated rule picks a correct route
in **4/6** events with human routes (2/6 shipped, 3/6 ranker-only).

**The honest summary: selection is no longer blocked by junk
candidates, it is now blocked by evidence quality.** That is a much
cheaper problem than the one we started the round with.

---

## 2. What the consultant is being asked

You are reviewing (a) whether the evidence-gate direction is the right
one to push vs. reformulating selection around body evidence, (b) what
the next annotation round should be, and (c) how to sequence the
remaining validation work against the goal of a *working prototype*.
Sections 5–7 are the raw material; section 6 is our current plan.

---

## 3. What has been done (condensed arc)

Physics that constrains everything (measured, see LEDGER preamble):
apical growth only, ≤7.5 px/frame, monotone arclength, rigid-body sway;
any tracker whose tip moves >3 px median while "growing" is modelling
noise. Second law, discovered the hard way: **contrast matures after
construction** — wall material is laid down optically invisible and
scatters into visibility over tens of frames, so every level-based
tracker measures the *developed* front, not the construction front.
Cadence law: trajectory tracking is valid only when the sampling
interval ≤ growth-duration/20 (movie 1's 31 s/frame is atlas-only).

| Stage | Contribution | Standing |
|---|---|---|
| v4 / v7 / v13 | gray-contrast gated length, spine-consensus kymograph, birth-time tomography | first number matching the user's eye (~72 px) |
| v17–v18 | birth-event topology; two-phase consensus | first phase-stable lengths |
| v19–v22 | directed crossing lanes; orientation/SE(2); deformable worldsheet; paired-wall ribbons | one real crossover repaired; counterexamples forced v23 |
| v23–v28 | owner-conditioned graph: learned pollen masks, CoTracker rigid owner pose, owner memory, field arbitration | complete 42-owner lowdens field pass |
| v29 | causal growth front: pre-growth owner census, halo/phantom rejection, dual-polarity walls, proximal gates, censoring labels, sticky tip evidence (v29.39) | **the measurement engine** |
| TimesFM-3 | zero-shot forecasting as auditor → fail-closed veto gate (v29.36+), burst head, rollout triage | in the pipeline |
| tip-CNN | corrector stack (peak-snap → consensus-once → sticky hysteresis), 11-probe promotion gate | v1 **frozen**, training moratorium |
| v30 | learned ribbon **front** localization + candidate **route** proposals; selection is the open problem | current frontier |
| annotation line (P0A) | napari workbench, SQLite store (schema v2.1, append-only revisions), task builders, fold → dataset with lineage, blinded reference kit | the human-data supply line |

---

## 4. What we most recently did (this session, 2026-09-14/15)

1. **Verification pass.** Full suite: **582 passed, 2 failed** — both
   failures are `tests/test_v30_targets.py` reading `/tmp/v30snap6`,
   a snapshot destroyed by the `/tmp` wipe. Code, not data, is fine;
   fix is to regenerate or skip.
2. **Opened the annotator for a 20-task uncertainty-stratified apex
   batch** on lowdens (`~/Documents/TubeTracker-annotator`). The user
   annotated all 20 in ~100 s: **13 tips + 7 no-tube verdicts**. All
   20 foci verified to land exactly on FRST-detected grains.
3. **Banked those labels durably**: `runs/prototypes/timesfm/tip_dataset_v7demo`
   (= `tip_dataset_v6repair` + 13 human tips with full
   task/observation/revision lineage; all referenced images present).
   The 7 ball-scoped no-tube verdicts remain **inert by design**
   (fold rule: a crop-scoped "can't tell" must never become a
   full-frame negative). They want wiring as ball-scoped negative
   discs in the v30 target space.
4. **Discovered and contained a data-loss trap.** `/tmp` gets wiped on
   this machine: it had already destroyed `annotator_big` (12 traces,
   **6 raster masks**, 8 duels, 4 census tiles) and all snapshots
   `snap1–snap13`. The eight other project DBs survived in
   `~/Documents/New project/TubeTracker-rev7-audit-2026-09-11/annotation_databases/`
   and were used to rebuild the snapshot lineage **on durable storage**:
   `runs/prototypes/v30/snapshots/snap14` and `snap15`
   (snap15: 124 observations, 4 body masks, 14 duels, 16 regions,
   11 tube tasks, 24 orphans, 0 census).
5. **Mined and ran a real selection round.** `build_rival_duels.py`
   produced 8 `route_duel` tasks from `movie_v17` (each event's shipped
   winner vs its best-coverage rival) in
   `~/Documents/TubeTracker-annotator-projects/v30w2`. The user judged
   all 8: **5 "neither", 3 preferences** (p04→winner, p01→walker,
   p02→walker), with the field observation that candidates "go in
   loops, or obviously incorrect straight lines."
6. **Diagnosed both junk classes from the candidate table**, which
   confirmed the observation exactly:
   - the `a*` attachment family is **25–31 of ~35 candidates per event
     and is literally 2-point straight 120 px lines** with no image
     content; the model's top pick was a fan in 3/8 events (p02: top
     nine slots all fans, 0.66→0.28; dt-007: fans in the top seven up
     to 0.994);
   - the walker family curves but loops;
   - the trained ranker was *already right* about the fans
     (`route_p` 0.005–0.03 on fans vs 0.4–0.9 on walkers) — the shipped
     rule `q_max × route_p` lets a saturating front softmax outvote it.
7. **Built and ran an offline evaluator** (`prototypes/v30_video_apex/eval_route_evidence.py`):
   every candidate re-scored with frame-normalized wall support along
   its own route (dense resample, so 2-point lines are scored rather
   than trivially excluded). Rule comparison over 6 gold events:
   evidence alone 0/6 (rejected — it re-ranks toward any high-contrast
   edge), `tip` alone 1/6, shipped `sel` 2/6, `route_p` 3/6,
   **evidence-gated `route_p` 4/6** (including both held-out dev
   events). Artifact: `runs/prototypes/v30/route_evidence_eval.json`.
8. **Wired the gate into the runner**: `scripts/run_v30_movie.py
   --select-evidence-gate FLOAT`, default **0.0 = behavior-preserving**.
   It restricts *selection only* (a candidate below
   `0.6 × event-best` support cannot win); all candidates remain
   exported with `evidence.wall_ev` and `evidence.gate_kept`, and the
   flag is recorded per event and in `report.json`.
9. **Validated end-to-end with a control**: on the 8 events,
   gate=0.6 gives 1/8 selected, median 33.4 px, p90 79.8 px, oracle
   unchanged at 5/8, kept-pool 6–21 of 29–40; the gate=0 control on
   the same snapshot reproduces the baseline winner-for-winner.
10. **Recorded everything**: LEDGER entry **H296** (protocol), memory
    notes, durable paths for snapshots.

---

## 5. Verified numbers you can trust

| Claim | Value | Provenance |
|---|---|---|
| Lowdens reference tube (P0016) | 36.142 px @480 = 72.284 px @960 vs user's ~72 px | H36, H46, H76 |
| v29.38 sticky evidence wired | 16,870/16,870 rows status- and length-identical vs v37; live for 32/70 owners | HANDOFF_2026-09-07 §2, H217 |
| Lowdens field (v29.19 replay) | 35 of 42 owners retained; 6,096 accepted timepoints; 94,889 centerline coordinates | H119 |
| Dense field (v29.31) | 70 owners → 55 measured, 6 review, 5 no-growth, 4 invalid-identity; 6,083 timepoints; 124,143 centerline rows | H131 |
| TimesFM veto | trips only when ≥3 faults AND ≥5% rate; field-wide it flags exactly P131 (dense) / P22 (lowdens) | H139, H142 |
| Burst head (locked rule) | `burst_p > 0 AND fc_gain ≥ 1.0`; dense F1 0.34/0.38/0.44 at H3/6/12 | H143 |
| Tip-CNN v1 | val 0.0129, frozen; 11-probe gate; 3 fine-tunes (E1/E2/E3) all FAIL on the faint class | H243, H256, H259 |
| v30 selection (this round) | 0/8 → 1/8 selected; median 73.0 → 33.4 px; oracle 5/8 both | §1 table, H296 |
| Snapshot (durable) | `snap15`: 124 obs, 4 masks, 14 duels, 16 regions, 11 tube tasks | §4.4 |
| Test suite | 582 passed, 2 failed (both `/tmp/v30snap6` fixtures) | this session |

Current front checkpoint: `runs/prototypes/v30/front_v16/front.pt`
(with its route head live — `route_p` varies 0.002–0.898 across
candidates, it is *not* in fallback). Dev split of that trainer:
`r4-p01`, `r4-p04`, `dt-005`, `dt-006`; dev frames `ld:11831`,
`ld:13145`, `ld:46830`, `ld:47250`. **Duels at dev frames never reach
training** (held out by design).

---

## 6. What's next (ranked, with the reasoning)

**A. Train v18 on `snap15` (cheap, background, tests ranker
generalisation).** `scripts/train_v30_front.py --snapshot
runs/prototypes/v30/snapshots/snap15 --init-checkpoint
runs/prototypes/v30/front_v16/front.pt ...` consumes `duels.json`
automatically and `--hard-routes` for mined negatives. Five of the
eight new verdicts are "neither" — i.e. *both* offers were wrong —
which is exactly the supervision the ranker has never had. Expected
gain is modest on top of the gate; the value is in knowing whether the
route head generalises from real verdicts at all.

**B. Body masks on the held-out curved events (the annotation round).**
The gate fixed *eligibility*; 5 of 6 events still pick a route that
does not follow the human path (33 px median tip error). The ledger's
evidence says body evidence is the discriminator with a real margin
effect (2 masks raised heat margins 3–5×, H284), and the mask-paint UI
exists (`--refs obs-r4-p01,obs-r4-p02,obs-r4-p04,obs-r4-p05`). **Do the
headless round-trip first** — the brush UI has never been
display-verified, and a broken paint session costs the annotator real
time.

**C. The blinded 42-owner reference kit** (`scripts/review_reference_centerlines.py`,
H96/H97). Built, resumable, checksum-locking, **still unannotated**.
This is the only thing that converts every "internally consistent"
statement in the ledger into a precision/recall number, and it is the
declared promotion gate for v27/v29/v30 alike. ~504 views; a
multi-session job.

**D. Wire the 7 inert no-tube verdicts** as ball-scoped negative discs
in the v30 target space (small; makes the user's round-2 labels
trainable).

**E. Prototype assembly (the stated goal).** A working prototype is
not a new model — it is: v29 lengths + v29.36 veto + v30 front on a
gated route + annotator QC, under one run manifest, with
`--select-evidence-gate` on by default *after* B/C give it cover.
Do this after A and B, and freeze scope at that point.

**F. Housekeeping.** (i) make the two `/tmp`-dependent tests
regenerate-or-skip; (ii) keep everything off `/tmp` (done for
snapshots; `/tmp` probe sheets are still the habit in older scripts);
(iii) copy the surviving annotation DB backups into the repo tree
(`runs/prototypes/annotation_backups/`) so they live with the project.

---

## 7. Ground rules and traps (do not re-learn these)

**Method rules, earned the hard way:**
- Summary metrics never override eyes. Every major correction in the
  ledger came from looking at raw frames; several came *after* a green
  CSV.
- Never tune a threshold on a handful of events (H125, H147). Gates
  must self-calibrate relative to per-frame/per-event statistics
  (H148's frame-normalization is the pattern).
- Fail closed: uncertain events are withheld with an explicit reason
  label and their data kept, never quietly deleted (H108, H115).
- Claim discipline: exact germination onset is not knowable from this
  imaging; report intervals. Length is what we can measure.
- Stop-rules are real: no more ranker negative-variants (H281), no
  tip-CNN retrain without a perturbation-free design (H230/H241/H259),
  no μSAM on this machine without >10 GiB free (H277/H279).
- A control run is not optional. This round's gate claim rests on
  `movie_v18_control` reproducing the baseline winner-for-winner.

**Environment traps (all hit this session):**
- **`/tmp` is wiped.** Snapshots, project DBs, probe sheets and
  mid-run scratch all vanish. Durable homes: `runs/…` for snapshots
  and datasets, `~/Documents/TubeTracker-annotator*` for projects.
- macOS has **no `timeout`** (use `gtimeout`, or no wrapper).
- Two venvs, do not mix: `.venv` (torch/cv2, runs everything
  pipeline-side) and `.venv-annotator` (Python 3.13 + napari 0.9.1 +
  PyQt6, runs only the workbench). napari uses `border_color`, not
  `edge_color`; canvas gestures must be bound at viewer level or
  overlay layers swallow clicks (H237/H238).
- The lowdens movie's filename contains **a space before `.mp4`** —
  quote it, every time.
- `run_v30_movie.py` refuses to overwrite an existing `--out-dir`.
- The fold refuses to turn a crop-scoped "can't tell" into a
  frame-level negative — that is a rule, not a bug.
- `evaluate.py`'s headline metric is **tip distance ≤ 5 px**, while
  the offline route evaluator measures **route coverage**. They will
  disagree; say which one you mean.

---

## 8. Key paths and verify-it-yourself commands

```bash
# 0. suite (expect 582 pass / 2 known /tmp failures)
.venv/bin/python -m pytest tests/ -q

# 1. the annotated projects (read-only)
#    ~/Documents/TubeTracker-annotator            (20 apex tasks, done)
#    ~/Documents/TubeTracker-annotator-projects/v30w2  (8 duels, done)
#    surviving legacy DBs:
#    ~/Documents/New project/TubeTracker-rev7-audit-2026-09-11/annotation_databases/

# 2. durable snapshots
ls runs/prototypes/v30/snapshots/            # snap14, snap15

# 3. the selection evidence evaluation (offline, ~1 min)
.venv/bin/python prototypes/v30_video_apex/eval_route_evidence.py \
  --candidates runs/prototypes/v30/movie_v17/candidates.json \
  --snapshot   runs/prototypes/v30/snapshots/snap15 \
  --movie      "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4" \
  --out        runs/prototypes/v30/route_evidence_eval.json

# 4. the movie run, gate off vs gate on (~6 min each)
.venv/bin/python scripts/run_v30_movie.py \
  --front-checkpoint runs/prototypes/v30/front_v16/front.pt \
  --v1-weights runs/prototypes/timesfm/tip_cnn_v1/best-point-heatmap-model.pt \
  --snapshot runs/prototypes/v30/snapshots/snap15 \
  --movie "ld=/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4" \
  --event-ref obs-dt-004 --event-ref obs-dt-007 \
  --event-ref obs-r4-p00 --event-ref obs-r4-p01 --event-ref obs-r4-p02 \
  --event-ref obs-r4-p03 --event-ref obs-r4-p04 --event-ref obs-r4-p05 \
  --out-dir runs/prototypes/v30/movie_vNEXT --select-evidence-gate 0.6

# 5. the annotator (needs a display)
./.venv-annotator/bin/python scripts/run_annotation_app.py \
  --movie "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4" \
  --project-dir "$HOME/Documents/TubeTracker-annotator-projects/<project>"
```

Reference artifacts: `runs/prototypes/v30/movie_{v17,v18_control,v18_evgate}/`
(report.json + candidates.json + measurements.json each),
`runs/prototypes/v30/route_evidence_eval.json`,
`runs/prototypes/timesfm/tip_dataset_v7demo/`,
`prototypes/LEDGER.md` (H296 is this round).

---

## 9. Open problems (unchanged unless noted)

- **v30 selection**: junk-eligibility fixed; evidence quality still
  binds (median 33 px tip error at 1/8). Next lever is body evidence
  or a representation that can express "is this route on the tube".
- **Faint apexes** (P3-class): model-blind; moratorium stands until a
  perturbation-free training design exists.
- **Smooth phantoms / stable foreign latches** (P61/P97 class): veto-
  blind, support-blind, rollout-blind by design; needs trace-time
  switch detection (H146/H147 prototypes exist, not wired).
- **Tip-CNN supervision form**: three fine-tunes, zero promotions;
  binding constraint is supervision design, not label count.
- **Blinded accuracy**: no precision/recall number exists for any
  version. Section 6C is the fix.
- **/tmp durability**: contained for snapshots; scripts that still
  write probe sheets there should be migrated opportunistically.
