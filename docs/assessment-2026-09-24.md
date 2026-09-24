# TubeTracker: assessment of the work so far and the direction to a working prototype

> **DRAFT — work paused.** Sections 1–4 are drafted. The experiment in section 5 has code but no results yet
> (training had not started). Nothing here changes the labelling tool, SparseTrack, or the frozen `sparsetrack-0.4.0`
> tag.

*24 September 2026. Written from a full read of every branch (`main`, `feature/burst-candidate-detection`,
`snapshot-2026-09-23`, `sparse-reset-2026-09-23`), the research ledger (H1–H491), the benchmark labels and reports,
plus runs of the legacy engine and SparseTrack on `sample_movie.avi` and a new experiment
(`prototypes/learned_evidence/`, section 5).*

---

## 1. Bottom line

1. **The sparse-first reset was the right call, and SparseTrack is the right foundation.** After about 490 logged
   hypotheses (v1 to v30), the pre-reset program ended with zero certified automatic lengths (H464, H468). SparseTrack,
   built in one day, is the first method measured against real human labels. It is also the best so far on every
   metric: onsets within ±600 frames for 13/28 grains, lengths within max(2 px, 10%) for 58/104 traces, median error
   2.8 px.
2. **Those numbers are in-sample and noisy.** They come from 28 grains, and the same labels were used to tune 0.3 and
   0.4.
   - 95% bootstrap intervals over grains: onset 29–64%, lengths 43–69%.
   - Even the gain from 0.2 to 0.4 is borderline (paired: onset +6 grains, 95% CI 0 to +12).
   - Held-out movie 2, the labels you are making now, is the first honest number. Score the frozen 0.4.0 on it once
     before changing anything else.
3. **The labelling tool is good, with one fixable validity problem.** Every one of the 32 Session A onset brackets is
   exactly one bin wide, because the tool defaults "last absent" to the bin before. Yet your blind retest moved
   first-visible by 8, 19 and 5 bins on three of seven grains. The benchmark therefore treats human onsets as far more
   precise than they are. For movie 2, shift-click an honest last-absent bin whenever the transition is not crisp.
4. **The direction: learned evidence, physics decoding, human review.**
   - Keep SparseTrack's decoder: candidate paths, the monotone growth-front DP and the onset logic. It encodes the
     biology the ledger established.
   - Replace its hand-built change evidence with a small network trained on SparseTrack's own codec-exact synthetic
     movies. They give unlimited, exact labels.
   - Put a model-prefilled review-and-correct step, built into the labelling tool, in front of the final numbers.
   - This fixes what killed every earlier learned model: 5–135 human labels and a network asked to make global
     decisions. It keeps what works, and each piece can be measured on the benchmark you already have.
5. **The experiment behind this recommendation (section 5).** Trained only on synthetic movies, on CPU in minutes,
   the network is fed into the unchanged SparseTrack pipeline. On held-out synthetic seeds it was compared end to end
   with SparseTrack's own evidence. *[Results: see section 5.]* Whether this transfers to real movies takes one command
   on your Mac (appendix B), scored on `ld_v1`.
6. **The biggest single lever may be the microscope, not the code.** The movies are ~14 fps x264 at a QP-30 quality
   floor with a keyframe every 12 frames: about 0.9 KB per frame for faint 2–5 px tubes. For new experiments, record
   lossless time-lapse (one 16-bit frame every 10–20 s, about 1 GB/h) with hardware autofocus. Every method gets
   easier.

**This week:** finish the movie-2 labels (about an hour, section 4.2). Score 0.4.0 on them. Answer the five questions
in section 8. Then build prototype v1: SparseTrack 0.4, a review loop, physical units and one-click run/export. It is
about a week of work and useful to the lab immediately for sparse movies.

---

## 2. What exists (inventory)

| Generation | Where | Period | What it is | Status |
|---|---|---|---|---|
| Upstream TubeTracker | `main` (`TubeTracker.py`) | 2024–25 | wxPython GUI. Hough grains, skeleton-endpoint or template tips, motpy linking, manual QC and bursts. Published: Yimga Ouonkap et al., *Plant Reproduction* 38:16 (2025) | Reference only |
| Modernised engine | `tubetracker/{gui,analysis,models,views}.py`, `scripts/run_pilot.py` | Jul 2026 | Same algorithms, modular, LapTrack linking, CLI pilot runner, reviewer-first burst candidates | Kept, not developed |
| Research program v1–v30 | ledger H1–H491; code at tag `snapshot-2026-09-23` | 13 Jul – 22 Sep | Kymograph/DP trackers, tomography and min-cut formulations, TimesFM priors, a weakly supervised tip CNN, the v30 owner-conditioned temporal U-Net with a napari annotator and review queues (1,274 tests) | Frozen; pruned at the reset |
| **SparseTrack 0.4.2** | `sparsetrack/` | 23 Sep | Keyframe-bin cache, per-grain registration, end-state path candidates, monotone DP growth front, matched-stub onset, Turnbull population curve, review gallery, codec-exact synthetic benchmark | **Active** |
| **Benchmark labelling tool** | `sparsetrack/bench/` | 23 Sep | Local web app for census, onset brackets, exit-to-apex traces and blind retest; autosave plus journal | **Active; movie-2 labelling in progress** |
| Human benchmark | `benchmark/labels/ld_v1.json` | 23 Sep | Session A: 39-grain census, 32 onsets, 127 traces, 7 retests | Dev set (tuned on) |

---

## 3. Assessment

### 3.1 Upstream TubeTracker (baseline)

I ran it headless on `sample_movie.avi` with the manual's settings (blur 7, background 55).
- It took ~23 s end to end.
- It found 32 grains and called 26 germinated, but produced **114 tip tracks for roughly 40 real tubes**. That is the
  fragmentation the manual tells users to fix by hand with "link tracks".
- Every frame is resized to the screen size (1000×725 on the reference display): 0.78× horizontally and 0.71×
  vertically. Tubes lose resolution and the aspect ratio is distorted.
- Bursts are manual.
- A few latent bugs:
  - `update_burst` uses `self.is_germinated == False` (a comparison, not an assignment).
  - `segment_inputs` uses `**` instead of `*` in `gv12`.
  - `Start_TubeTracker` runs `find ~/` over the whole home folder.
- It is a reasonable published baseline for fast 1-frame-per-minute movies. It is not a basis for the lab's current
  14-fps compressed movies.

### 3.2 The research program, 13 Jul – 22 Sep (v1 to v30)

The ledger is an unusually honest record. Its durable findings should be kept and cited:

- **Tip growth only, so arclength never decreases.** The tube is the tip's trail. Frame-to-frame growth is sub-pixel
  to a few pixels, and apparent motion of the tube is rigid swing of the whole tube.
- **Appearance lags existence (H25).** New wall matures optically over tens of frames. Any tracker measures the
  *developed* front; onset is best reported as an interval.
- **One frame is too noisy.** Single-frame evidence buries faint tubes, so temporal pooling is the minimum. Absolute
  thresholds do not transfer between movies.
- **Material-point trackers and flow fail on this footage.** CoTracker landmarks drifted 47–76 px, and dense optical
  flow blew up on textureless brightfield (H3). CoTracker3 did hold up for rigid grain pose (H98).
- **Zero-shot foundation segmenters do not see these tubes.** Tube IoU: Cellpose-SAM 0.005, µSAM 0.008–0.108 (H334).
  SAM2 did not lock onto faint tubes (H2).
- **Tip coherence is a first-class metric.**

What did not work, and why. This matters for the recommendation.

- **Learned models were starved of labels, and the labels had the wrong form.**
  - The v1 tip CNN was trained on 1,539 weak labels mined from tracker output. It was frozen after every retrain
    failed its probe gates (H181–H264).
  - The v30 owner-conditioned U-Net was trained on 5–135 human samples. It collapsed to constant outputs (a dead ReLU,
    H347), learned prompt and position shortcuts (H316b), never separated owners (H349, H372, H411), and picked the
    right tip 0/6–0/8 times (H268–H407b).
  - Late in the program, a tip-cap model reached 65/75 tips within 5 px after hard-negative mining (H462). **Networks
    can see these tips; what failed was the data regime and asking the network to make global decisions.**
- **Process swamped progress.** There were certification gates, ownership contracts, review queues, 1,274 tests and
  about 490 hypotheses, scored mostly on a handful of cases (P0016, P0034, cf70). The ledger records an 80–100 px
  "phantom" corridor that was measured bit-exactly by four methods (H20). Bit-exact agreement proved method equals
  method, not method equals reality.

Verdict: the reset (freeze v30, prune, benchmark first, sparse first) was correct.

### 3.3 SparseTrack 0.4.x

**Design.** One grain at a time, on registered 300-frame keyframe averages:
- Take the end-state change map and trace candidate centrelines (one per branch end or rim contact).
- Read a kymograph along each candidate, allowing a rigid swing about the exit.
- Find a non-decreasing DP growth front, and pick the candidate whose monotone growth explains the most evidence.
- Call onset with a matched stub filter plus hysteresis.
- Censor length at contact with other grains.
- Report a Turnbull NPMLE population curve (interval-censored, with T50) and a review gallery ordered by path
  coverage.

**Strengths.**
- It is physically grounded and fast: about 40 s per movie after a 30–40 s cache build.
- It is interpretable, with per-grain diagnostics.
- It has sound statistics for censored onsets.
- It comes with a codec-exact synthetic benchmark (v1–v5) and held-out discipline. It is the first method in the
  project with a scoreboard.

**Scores on the dev benchmark** (`ld_v1`: 28 isolated grains, 104 FULL traces; from `benchmark/reports/ld_v1_scores.md`):

| Version | Onset within ±600 frames | Lengths within max(2 px, 10%) | Median length error |
|---|---|---|---|
| v29.19.3 (pre-reset field pipeline) | 4/22 | 49/96 | 3.76 px |
| SparseTrack 0.1 | 4/28 | 46/104 | 3.61 px |
| SparseTrack 0.2 | 7/28 | 48/104 | 3.62 px |
| SparseTrack 0.4.0 (frozen for movie 2) | 13/28 | 58/104 | 2.81 px (bias −3.4 px) |
| SparseTrack 0.4.1 | 14/27 | 61/104 | 2.76 px |
| Your own blind retest (onset, 7 grains) | 4/7 | — | — |

Looser tolerances for 0.4.0:
- Onset within ±2,400 frames: 22/28.
- Lengths within max(5 px, 20%): 79/104.

**How to read these numbers.** I bootstrapped over grains, because traces from one grain are correlated.
- 95% intervals for 0.4.0: onset 29–64%, lengths 43–69%. Each trace is about 1% of the length metric.
- Paired against 0.2.0 on the same grains: onset +6 grains (95% CI 0 to +12) and lengths +10 traces (−2 to +23). This
  is borderline even in-sample.
- Most of the 23 September commits moved the score by 1–3 traces while choosing parameters on this same set. That is
  within noise, and it risks overfitting 28 grains.

**Known failure classes** (from the reports and `compare.html`):
- tubes that curl over their own grain;
- ownership where a tube contacts another grain;
- sway;
- paths that stop short of the frame edge;
- early spurious rotation, since fixed by the exit pivot.

**Structural limits.**
- Isolated grains only: 28/39 on the dev movie, 82/122 on movie 2.
- Lengths stop at first contact.
- Everything hangs on the end-state change map: a tube that fades or bursts, or a tube pushed around non-rigidly,
  breaks the path.
- Time resolution is one bin (300 frames).
- There is no burst or growth-arrest output, which the lab's survival analysis needs.
- There are no physical units: the real frame interval is unrecorded (H358), and pixel size is not in the cache.

**My runs here.**
- The SparseTrack tests pass (27; one needs ffmpeg). The six failing tests belong to the frozen v30 app and fail only
  because `torch` is missing.
- On `sample_movie.avi` it runs end to end in about 2 minutes. That movie is out of distribution (fast growth, one
  frame per bin). Many curves look right, but some paths land on background in the overlay, and four grains get late
  spurious onsets (bins 97–120). This is a qualitative smoke test only.
- `sparsetrack prepare` with the default 300 frames/bin crashes (`IndexError`) on a 129-frame movie; only `run`
  auto-sizes bins. The launchers are macOS-only, and movie paths are hard-coded to `~/Downloads`.

### 3.4 The newest annotation app (the SparseTrack benchmark labelling tool)

**What it does.**
- The grain census: confirm, exclude with a reason, or add grains, with a close-up view.
- The onset: a whole-movie filmstrip of 4-bin tiles, then 18 single bins, with keyboard verdicts.
- Exit-to-apex polyline traces at planned times: onset + 6 bins, 40%, 70% and the last full bin.
- Tube states: full, partial, no tube or unsure, plus a contact flag.
- An 8-grain blind retest of onsets.
- Every click autosaves atomically to JSON, and an append-only journal records each answer.

**Strengths.** This is the best labelling instrument the project has built.
- It uses only the standard library and has no install friction.
- It refuses to mix a labels file with the wrong movie's cache.
- Views follow each grain's own drift, and out-of-frame pixels are hatched.
- It uses a fixed random sample for held-out labelling, which avoids the bias of taking the first N grains.
- Blind retest is built in, so human noise is measured rather than assumed.
- The vocabulary matches `annotation_schema.GerminationEvent`.
- It is fast. The journal shows Session A took about 50–60 minutes of active clicking: 178 answers, mostly between
  16:28 and 17:12.

**Issues, in order of importance.**
1. **Onset brackets are artificially narrow.** The last-absent bin defaults to first-visible − 1. Session A never
   overrode it, so all 32 brackets are one bin wide, while the retest disagreed by up to 19 bins. Any prediction
   outside a falsely narrow bracket is scored as an error.
   - *Fix now, by how you label:* shift-click an honest last-absent bin when unsure.
   - *Fix in the tool, after movie 2:* require an explicit last-absent click, or offer a "crisp / uncertain" toggle.
2. **Only onsets are retested.** Human length noise is unknown, so nobody can say whether 58/104 is near the ceiling.
   Add a 15-trace blind retest.
3. **The onset criterion is implicit.** The help text says "first tile where a tube is clearly visible growing out".
   The g014 "bulge" (19 bins apart on retest) and the v30 "bump vs tube" debate (H484) show this needs a written rule,
   for example "a protrusion of at least 3 px beyond the rim outline, at the angle the tube later grows". The rule
   should be shown in the tool.
4. **One annotator.** There is no inter-rater agreement. A second person labelling 10 grains of `ld` would
   calibrate the benchmark.
5. **Labels are judged on the same registered bin averages the model reads.** Registration or averaging artefacts are
   therefore shared by truth and prediction. This is acceptable for now; spot-check a few traces on raw keyframes.
6. **Fixed trace times** (40%, 70% and last bin) oversample long, late tubes, which are the ones most likely to touch
   something. On movie 2 (~351 bins) many late traces may be `contact` and drop out of the metric.

### 3.5 The annotation tasks you still need to complete

- **Now (required): movie 2 held-out, `Label_Movie2_Heldout.command`.** A fixed random sample of 30 isolated grains
  away from the edge; labels go to `benchmark/labels/m2_v1.json`.
  - Steps:
    1. Census check: confirm each sampled grain, or exclude it with a reason.
    2. Onset for each grain.
    3. About 3–4 traces per germinated grain, roughly 90–120 in total.
    4. The 8-grain retest at the end.
  - Movie 2 has about twice as many bins as `ld`, so the filmstrips are longer. Expect **about 1–1.5 hours of active
    work**.
  - Before you start:
    - Keep it blind: do not open any SparseTrack output for movie 2 first.
    - Mark an honest last-absent bin whenever the transition is not crisp.
    - Use U (can't tell) and P (partial) rather than guessing.
    - Press T on every trace that touches another tube or grain.
    - Do the retest after a break.
  - When you finish:
    1. Commit `m2_v1.json` and its journal.
    2. Score **0.4.0 once**, then 0.4.2. Record both in `benchmark/reports/`.
- **Soon (cheap, raises the value of both benchmarks):**
  - A 15-trace blind retest on `ld`, about 10 minutes.
  - A written onset rule, applied from now on.
  - A second annotator on 10 `ld` grains.
- **Park:** the v30 napari queues (the 172 "competitive windows", the H474 exit tasks, the 50 stale tasks). They serve
  the frozen app.
- **Later (after prototype v1):** review-mode corrections. Every accept or fix in the review loop is a new label.
  Label a third held-out movie once movie 2 has scored two or three frozen versions. Label dense-field grains (clumps
  and crossings) only when the prototype expands past sparse fields.

---

## 4. Direction

### 4.1 What "working prototype" should mean (acceptance criteria)

- **One step:** drop in a movie, enter the seconds per frame and µm per pixel, and get results in minutes on a laptop.
- **Outputs, per grain:**
  - germination status and onset interval;
  - growth-arrest or burst frame;
  - length and growth-rate curves in µm and minutes;
  - QC flags;
  - per movie: the population germination curve (Turnbull, T50) and CSV plus figures.
- **Review:** flagged grains are shown first, with one-click accept or fix. Corrections are saved as labels.
- **Accuracy:** measured on held-out human labels, and **after review within human test-retest agreement**. Automatic
  accuracy is reported alongside, with intervals.
- **Scope, stated honestly:** isolated grains in v1. Clumps and crossings are counted and excluded, with a per-grain
  density covariate reported, because germination can depend on local pollen density.

### 4.2 Recommended architecture: learned evidence → physics decoder → human review

```
registered bins ──► [1] learned evidence ──► [2] SparseTrack decoder ──► [3] review & correct ──► results
 (bin, before,        per-bin P(tube),        candidate paths, monotone     prefilled in the         CSV, curves,
  after crops)        P(tip), later P(burst)  DP front, onset, censoring   labelling tool           population
```

1. **Learned evidence.** A small U-Net (0.47 M parameters) reads the same three images SparseTrack uses: the bin, the
   "before" reference and the "after" reference. It outputs the probability that each pixel is built tube, plus a tip
   heatmap.
   - It is trained on codec-exact synthetic movies from `sparsetrack synth`: real fields and grains, real measured
     tube cross-sections, contrast maturation, sway, drift, docking particles and the real x264 settings. This gives
     unlimited training data with exact labels.
   - After that it is fine-tuned on the 127 real `ld` traces, and later on review corrections.
2. **SparseTrack's decoder, unchanged.** Candidate paths, rotation and swing, the monotone DP front, onset and contact
   censoring. These carry the biology.
3. **Review and correct.** The labelling tool, opened in a "review" mode on the model's predictions. Grains are ordered
   by flags and path coverage. Each is confirmed or fixed with the same onset-bracket and trace gestures, and every fix
   is a new label.

**Why this, and why now:**
- **It fixes the causes of past learned failures.** Those models had 5–135 labels and were asked to decide ownership
  and selection. Here labels are unlimited and exact (synthetic), the network only supplies local evidence, and the DP
  makes the global, physically constrained decision.
- **It has strong precedent.** Networks trained purely on physically simulated microscopy data, then validated on
  real data, displaced hand-built pipelines in single-molecule localisation (DECODE, Speiser et al., *Nat. Methods*
  2021). Synthetic crossing tubes improved crossing handling in pollen Mask R-CNN work (Zhang et al., *J. Exp. Bot.*
  2023).
- **It reuses what exists.** The generator, the decoder, the benchmark and the tool are already built. The integration
  is one file: a probability "cache" that SparseTrack reads like a movie (`prototypes/learned_evidence/evaluate.py`).
- **It is cheap to falsify.** The real-data test (appendix B) takes about an hour on a Mac. If it does not beat 0.4.x
  on `ld_v1`, you have lost a day, not a month.

### 4.3 Where modern foundation models fit (and where they don't)

| Technology (2025–26) | Use here | Why not more |
|---|---|---|
| SAM 3 / 3.1 (Meta, Nov 2025 / Mar 2026) | none now | Gated weights, 848 M parameters, needs a CUDA GPU; zero-shot is weak on thin, faint, niche structures |
| SAM 2.1, µSAM (MIT) | annotation help: mask drafts for grains | Already failed zero-shot on tubes (H2, H334) |
| Cellpose-SAM / Cellpose 4 (DINOv3 variants) | optional grain census | Grains are already easy (the FRST/rim detector); models are trained on CC-BY-NC data |
| DINOv3 (Aug 2025), frozen features + small head | optional backbone for the evidence net once real fine-tuning starts | Gated and custom-licensed; 16-px patches are coarse for 2–5 px tubes, so it needs full-resolution cues or an upsampler (AnyUp) |
| CoTracker3 / TAPNext++ / AllTracker | rigid grain pose or drift QC | Track material points, not a moving growth front; CoTracker is CC-BY-NC |
| RF-DETR (Apache-2.0), Spotiflow (BSD) | if you ever move to detecting grains and tip points | Ultralytics YOLO is AGPL, including your trained weights |
| Trackastra, motile, laptrack | dense-field linking later | Not needed for per-grain sparse analysis |
| VLMs (Claude, Gemini, GPT) | not for measurement | The ledger shows vision reads misjudged small structures (H373, H375, H480) |

The breakthrough is not a bigger pretrained model. It is **turning your own simulator into a training-data factory**
and letting a verified physical decoder make the decisions.

### 4.4 Acquisition: the cheapest large win for future movies

The current movies are ~14 fps H.264 at a QP-30 quality floor with a keyframe every 12 frames. The dev movie is 45 MB
for 52,583 frames of 1280×1024. SparseTrack must average 25 keyframes per bin to recover SNR, and faint tubes are
attenuated by the codec itself: synth measured a 50% transfer for a 5-grey-level line.

For new experiments:
- **Record time-lapse, not video.** One frame every 10–20 s is plenty for sub-pixel growth per frame.
- **Save lossless 16-bit TIFF** (or ND2/CZI). That is about 1 GB/h at 1280×1024.
- **Turn on hardware autofocus** (e.g. Nikon PFS or Zeiss Definite Focus) against the documented focus drift.
- **Fix exposure and gain.**
- **Store seconds per frame and µm per pixel with every movie.**

If the lab has the raw originals of the current movies, re-export them losslessly. Where a line is available, a
pollen-expressed fluorescent marker (e.g. a LAT52 promoter fusion) turns tube tracking into an easy bright-on-dark
problem.

### 4.5 Roadmap, with gates

| When | Deliverable | Gate |
|---|---|---|
| Now | Finish movie-2 labels; score 0.4.0 once, then 0.4.2 | First held-out numbers, with bootstrap intervals |
| Week 1 | **Prototype v1:** SparseTrack 0.4 plus review mode in the labelling tool, physical units, one-click run/export (macOS and Windows), growth-arrest frame from the DP plateau, burst candidates flagged for review | Lab runs it on a real experiment; corrected results reproduce your manual measurements within retest agreement |
| Weeks 2–3 | **Learned evidence on real data:** appendix B on `ld`; then fine-tune on the `ld` traces; freeze; score once on movie 2 | Beats 0.4.x on `ld_v1` beyond noise (paired bootstrap), then holds on movie 2 |
| Weeks 3–4 | Burst head, trained on synthetic bursts (add to `synth`) plus reviewed bursts | Burst frame within ±2 bins on held-out reviews |
| Weeks 4–8 | Dense fields: instance-aware evidence (which grain owns each tube pixel), learned with synthetic foreign tubes and crossings (v2+ presets); ownership decided by birth time and geodesic reach | Contact-censored fraction halves without losing isolated-grain accuracy |
| Ongoing | Acquisition protocol for new experiments; a new held-out movie every 2–3 frozen versions | — |

### 4.6 Process: how to avoid another 490 hypotheses

1. **Keep one scoreboard.** It holds the dev (`ld_v1`), held-out (movie 2) and synthetic held-out (seeds 3–4) scores,
   each with paired bootstrap intervals. A change counts only if it clears noise on the dev set and does not regress
   the synthetic held-out.
2. **Never tune on held-out data.** Score each frozen version once. Retire a held-out movie after two or three
   versions and label a fresh one (30 grains ≈ 1 hour).
3. **Grow the benchmark rather than the code.** Thirty more labelled grains shrink the intervals more than any
   parameter tweak changes the score.
4. **Scope each AI-agent session to one gated experiment.** Give it a written hypothesis and a benchmark-delta
   acceptance test. It may not add new subsystems without a measured deficit. Keep the ledger to one line per
   experiment.
5. **Freeze v30.** Retire the napari annotator and its queues until dense fields are in scope.

---

## 5. Experiment: does synthetic-trained evidence beat SparseTrack's evidence?

*[Filled in below once the runs finish.]*

---

## 8. Questions only the lab can answer

1. What is the real acquisition interval (seconds per frame) of `test1lowdensjoshua-28c-hz` and movie 2? Were they
   recorded as 14-fps video or as time-lapse exported at 14 fps? What is the pixel size in µm?
2. Do raw or lossless originals exist on the acquisition PC?
3. What density must prototype v1 handle? Is "isolated grains, with clumps counted and excluded" acceptable for the
   first experiments the lab wants to run?
4. Is there GPU access (for example Brown CCV's Oscar) for the fine-tuning step, or is CPU/MPS the budget? The
   experiment here was CPU-only.
5. Who else could label 10 grains, so there is an inter-rater number?

---

## Appendix A: numbers used here

- Dev movie `test1lowdensjoshua-28c-hz.mp4`: 1280×1024, 52,583 frames, 14 fps container, 4,382 keyframes (every 12
  frames), 45 MB. 176 bins of 300 frames. 39 census grains (28 isolated, scored).
- Movie 2 (`Pollen tube movie 2 7-14-26.mp4`): about 105k frames. Census 122 grains (82 isolated). First ~9 bins
  skipped as settling. 30 grains sampled (seed 20260923).
- Session A (`ld_v1`): 39 grains (8 excluded: 5 clump, 1 not a grain, 1 out of focus, 1 other), 32 onsets (all
  `emerged_within`, all brackets one bin), 127 traces (120 full, 6 unsure, 1 no tube). FULL lengths 2.4–109.7 px,
  median 25.1. The journal shows ~50–60 min of active labelling.
- Upstream engine on `sample_movie.avi`: ~23 s, 32 grains, 26 germinated, 114 tip tracks, analysis at 1000×725
  (0.78/0.71 scaling).
- SparseTrack 0.4.2 on `sample_movie.avi`: 37 grains (28 isolated), 114 s including the overlay video.

## Appendix B: run the learned-evidence test on your real dev movie

*[Commands filled in below.]*
