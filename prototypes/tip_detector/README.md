# A learned tip detector, judged leave-one-movie-out (prototype, 3 Oct 2026)

**Question.** Can a small network trained on the human traces find a growing tube's apex far better than the
hand-made evidence (the raw short-interval difference, change from the start), on a movie it never saw?

**Answer (details below): yes for finding tips, not yet for measuring long tubes in crowded fields.**
Held out (version 2), the detector's top peak lies within 4 px of the human apex for 77% of ld traces, 48% of
m2 and 61% of m1, against 34% / 20% / 30% for the best raw short-interval difference and 58% for the shipped tube
network on m1 (the only movie it did not see). It finds young tips (ld 32/34, m2 16/24, m1 25/27) and separates
grains just before onset from young tubes far better than any raw signal (onset AUROC 0.91 / 0.81 / 0.89 against
0.57-0.70), but misses long tubes in crowded fields (m2 3/19, m1 3/17). The quick linking check (straight-line
lengths; written up by the main session, the agent was stopped first) read ld well (96/116 within max(4 px, 10%),
tips within 4 px 94/116, median error 1.2 px), m2 poorly (14/61, median error 30 px) and failed on m1 (no grain
started: grain positions for moving grains in link.py, not investigated). Next: use it for onsets and young tubes
inside the tracker (another agent, 3 Oct).

## Method

Inputs at bin b, registered, local median (~60 px) removed, fixed grey-level scales (common.py):
A = mean of bins b-1..b+1 (appearance, / 20 grey levels); D = that minus the mean of bins b-7..b-5 (the short-interval
difference, / 8); C = A's mean minus the mean of the first (reference) bins (change from the start, / 20). (A movie-level
MAD scale was rejected before training: the backgrounds are so flat it would have put tubes at hundreds of units.)

Samples (build.py, inputs computed once, ~210 MB for all three movies): 192 px extractions at a trace's bin - per FULL
trace one centred on the grain (+/- 24 px), one on the apex (+/- 40 px), one on a random point of the trace (+/- 30
px); PARTIAL traces two (no apex: the tube goes on); no_tube/burst one; five bins before each human onset
(last_absent_bin - 1, 3, 6, 12, 24) on the grain where the labelling tool follows it. Targets: a Gaussian (sigma 2 px)
at every FULL apex in the extraction (any labelled grain traced at that bin), a weak second head for the traced body
(2.5 px), loss only inside a scored region (grain disc r + 25 px, a 25 px band round each trace, disc r + 40 before an
onset; blind 40 px round PARTIAL ends and round UNSURE grains) plus, in version 2, empty background (no |A| > 4 or
|D| > 3 grey levels within 8 px, away from all labels). Training (train.py): U-Net 16-32-64-128 with BatchNorm (a
crop's map = the full frame's), CenterNet focal loss + body BCE, any-angle rotation, flips, gain 0.6-1.6, offset,
noise, blur, 128 px crops, batch 16, AdamW + one-cycle, 2500 steps (~18-24 min on MPS on the shared machine), half
of each batch from each training movie. Nothing of the held-out movie is used (no early stopping).

Evaluation (evaluate.py), per FULL trace of the held-out movie: search region = disc r + L + 25 px round the grain at
that bin (census + the trace's view offset; exit points sit within a few px of radius r of it), inside the frame.
Top-1 / top-3 peaks (local maxima of 9 x 9, >= 4 px apart) within 4 px of the human apex (also max(4 px, 10% L));
the apex's rank among all peaks of the region; a route-given check (window of radius 25 px centred on the human path
15 px behind the apex: only where along the tube the tip is, and what lies just past it). The same protocol for
|D| and |C| (raw, Gaussian sigma 1, 2; best shown) and, on movie 1 only, the existing tube network
(tubes_bn_real_ld_m2, trained on ld and m2 traces; P >= 0.5 pieces reaching the rim, their farthest pixel). Before
the onsets: the best value within r + 25 px at the five pre-onset bins of every grain, against the best value at
young FULL traces (AUROC; share of pre-onset grain-bins above the value that 80% of young traces reach).

## Results

| held out | method | top-1 <= 4 px | young | mid | long | top-1 <= max(4, 10% L) | top-3 <= 4 px | apex rank (median) | route-given window | onset AUROC | false alarms at 80% young |
|---|---|---|---|---|---|---|---|---|---|---|---|
| m2 (61) | det | 29 (48%) | 16/24 | 10/18 | 3/19 | 29 | 41 | 2 | 44 | 0.81 | 0.38 |
| m2 (61) | absD_0 | 12 (20%) | 6/24 | 6/18 | 0/19 | 12 | 18 | 7 | 22 | 0.68 | 0.45 |
| m2 (61) | absD_1 | 11 (18%) | 6/24 | 5/18 | 0/19 | 11 | 18 | 6 | 20 | 0.70 | 0.42 |
| m2 (61) | absD_2 | 8 (13%) | 5/24 | 3/18 | 0/19 | 8 | 16 | 6 | 15 | 0.68 | 0.51 |
| m2 (61) | absD_0_out | 12 (20%) | 6/24 | 6/18 | 0/19 | 12 | 24 | 6 | nan | 0.68 | 0.45 |
| m2 (61) | absC_2 | 7 (11%) | 7/24 | 0/18 | 0/19 | 7 | 12 | 9 | 9 | 0.67 | 0.49 |
| ld (116) | det | 89 (77%) | 32/34 | 43/65 | 14/17 | 91 | 102 | 1 | 103 | 0.91 | 0.06 |
| ld (116) | absD_0 | 40 (34%) | 15/34 | 24/65 | 1/17 | 45 | 53 | 2 | 53 | 0.57 | 0.61 |
| ld (116) | absD_1 | 40 (34%) | 15/34 | 24/65 | 1/17 | 44 | 55 | 2 | 51 | 0.59 | 0.59 |
| ld (116) | absD_2 | 36 (31%) | 14/34 | 21/65 | 1/17 | 38 | 55 | 2 | 49 | 0.61 | 0.61 |
| ld (116) | absD_0_out | 41 (35%) | 13/34 | 25/65 | 3/17 | 49 | 56 | 2 | nan | 0.64 | 0.50 |
| ld (116) | absC_2 | 12 (10%) | 7/34 | 5/65 | 0/17 | 12 | 17 | 5 | 13 | 0.70 | 0.49 |
| m1 (57) | det | 35 (61%) | 25/27 | 7/13 | 3/17 | 36 | 38 | 1 | 41 | 0.89 | 0.22 |
| m1 (57) | absD_0 | 17 (30%) | 14/27 | 3/13 | 0/17 | 18 | 21 | 4 | 23 | 0.61 | 0.55 |
| m1 (57) | absD_1 | 13 (23%) | 11/27 | 2/13 | 0/17 | 14 | 20 | 3 | 20 | 0.61 | 0.52 |
| m1 (57) | absD_2 | 11 (19%) | 10/27 | 1/13 | 0/17 | 12 | 22 | 3 | 20 | 0.59 | 0.55 |
| m1 (57) | absD_0_out | 16 (28%) | 13/27 | 3/13 | 0/17 | 18 | 22 | 3 | nan | 0.65 | 0.43 |
| m1 (57) | absC_2 | 5 (9%) | 4/27 | 1/13 | 0/17 | 5 | 9 | 6 | 9 | 0.66 | 0.55 |
| m1 (57) | tubenet | 33 (58%) | 20/27 | 7/13 | 6/17 | 34 | 33 | nan | 37 | nan | nan |

## Files

- `common.py` inputs, movies, network; `offsets.py` the labelling tool's per-bin grain positions;
  `build.py` samples; `train.py`; `evaluate.py`; `summary.py` / `final_table.py` tables; `sheet.py` contact sheets;
  `link.py` the linking check.
- Outputs in `runs/research/tip_detector/` (git-ignored): `samples_*.npz`, `tip2_{ldm1,m2m1,ldm2}.pt` (version 2,
  named by their training movies), `tip_ldm2.pt` (version 1), `eval_<movie>_<tag>.json`, `link_*.json`, logs.

Reproduce: `python -m prototypes.tip_detector.offsets; python -m prototypes.tip_detector.build;
python -m prototypes.tip_detector.train --train ld m1 --name tip2_ldm1 --iters 2500;
python -m prototypes.tip_detector.evaluate m2 --net runs/research/tip_detector/tip2_ldm1.pt --tag v2` (likewise
m2 m1 -> ld, ld m2 -> m1); `python -m prototypes.tip_detector.final_table v2 v2 v2`.
