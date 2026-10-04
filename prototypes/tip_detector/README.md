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

**Version 3 (4 Oct, section below):** inputs with several lags, all labelled apexes as targets (version 2 kept 4 an
extraction), shifted-bin samples. Long-tube apexes among the top 8 peaks: m2 7 -> 11/19, m1 5 -> 8/17 (ld 16 -> 15),
young tips and top-1 level, onset AUROC m2 0.87, m1 0.94; but no change within the reader's present 9 candidates (a
crowded field holds 20-30 real tips): the reader would need ~20 detector peaks a bin to see the gain (not run).

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

## Version 3 (4 Oct 2026): long tubes in crowded fields

**Question.** The global tip-trajectory reader (prototypes/tip_trajectory, `sparsetrack/tiptraj.py`) takes its tip
candidates from this detector, and on movie 1 only 4 of 11 long traces had any candidate at the apex. Can the
detector find more long-tube tips in crowded fields (top-K recall matters more than top-1) without losing young tips?

**Answer: somewhat, where it was asked for; not a fix.** Held out, long-tube apexes among the detector's 8 strongest
peaks of the search region: m2 7 -> 11 of 19, m1 5 -> 8 of 17 (paired over both crowded movies 8 gained, 1 lost;
sign test p = 0.04), ld 16 -> 15 of 17. Young tips are unchanged (top-8 80 -> 81 of 85 over the three movies), top-1
is level (152 -> 149 of 234), the onset AUROC rises on the crowded movies (m2 0.81 -> 0.87, m1 0.89 -> 0.94; ld 0.92
both). The gain is at K >= 8, not at top-1/top-3 (long top-3 24 -> 25 of 53), and not within the reader's present 9
candidates a bin (long: m2 3 -> 2, m1 3 -> 3). A second training seed on the m1 fold gives the same long-tube top-8
(8/17; young top-8 27/27, top-1 23/27, AUROC 0.91). What limits long tubes now:

1. **Competition from real tips.** A crowded field's 600 px reader region holds 20-30 real tube tips, and the
   strongest peaks are mostly those (contact sheets: m1 g030/g033/g064 b140, g061 b244); the long apexes v3 finds rank
   3rd-23rd there. The top 20 peaks of the reader's region hold 10/19 (m2) and 8/17 (m1) long apexes with v3 (v2: 8,
   5), its 3 + 6 candidates 2 and 3. A detector cannot rank the grain's own tip above its neighbours' tips; that is
   the reader's job (bodies, continuity). For the reader to gain, it should take more peaks (K_DET ~20) or choose them
   by reach along the tube map rather than by value.
2. **The frame edge.** 4 of m1's 17 long apexes (g005 b140/b244, g014 b244/b349: 1-15 px from the frame edge, three
   within 1 px, the tube running into it; the human calls them FULL) and m2 g054 b349 (2 px) get no response from
   either version (no training extraction contains the frame boundary). Two other m1 long apexes 5-14 px from the
   edge (g064, g065 b140) are found by v3 (7th and 8th peaks). A body that reaches the frame edge should itself be a
   candidate in the reader.
3. **No response at all** (map <= 0.05 within 4 px of the apex in both versions): m2 g069 b244, g082 b349, g092 b349
   (the tube barely visible near the apex at the inputs' contrast), g064 b244 (the human apex lies a few px past the
   visible end, against a grain), m1 g009 b244 (ends on a crossing tube) and g028 b244 (beside a grain, the tube
   moving).

### Method (changes from version 2)

- Inputs (`common3.py`): A, D6 and C as version 2, plus D12 and D24, the change over 12 and 24 bins (a slow tip still
  shows; a fast one leaves a streak whose leading end is the tip; long tubes in m2/m1 grow 0.5-3 px a bin, ld's
  mid-length ones ~0.4). Windows before the reference bins are clamped to them. Inputs are clipped at +/- 5.3 units
  (stored as int8, 1/24 unit a step) in training and inference alike.
- **All labelled apexes in an extraction are targets.** Version 2 kept at most 4 (`build.py`), so in 132 of ld's 465
  extractions (m2: 44 of 292, m1: 1) the labelled tips beyond the fourth (243 of ld's 1281 apex targets, m2 64 of
  590) were taught as background.
- Shifted-bin samples (weak supervision, `build3.py`): for a FULL trace at bin b, extractions at bins b - 2 and b - 4
  with every labelled trace shortened along its own path by v x k (v its mean growth since the grain's previous trace
  or its onset; target sigma 2 + v k / 4 px; only where v k <= 12 px; traces without a growth estimate blind): ld
  +193, m2 +94, m1 +69 extractions.
- Long-trace weight 2: extractions holding the apex of a long (>= 60 px) trace of a crowded movie (under 60% of the
  grains labelled: m2 24/122, m1 30/69) are drawn twice as often.
- Unlabelled structure unscored in the crowded movies: pixels with structure (|A| > 4 or |D6| > 3 grey levels within
  2 px) farther than 8 px from every labelled trace and r + 8 px from every labelled grain leave the loss (a
  neighbour's tip there would be taught as background); ~2% of m2's pixels.
- Unchanged: U-Net 16-32-64-128 with BatchNorm (now 5 inputs), focal + body loss, augmentation, 2500 steps of 16,
  half of each batch from each training movie (`train3.py`). Samples: 270 MB for the three movies, memory-mapped;
  a fold trains in 7-14 min on the shared machine.
- Evaluation (`evaluate3.py`, `table3.py`): version 2's protocol (search disc r + L + 25 px round the grain, peaks
  9 x 9 apart >= 4 px, a hit within 4 px of the apex), with top-1/3/8, and the reader's view: its 9 candidates (the 3
  strongest peaks >= 0.05 within r + 30 px of the grain centre, then the 6 strongest from r - 2 to 296 px), the top 20
  peaks of that region, and whether the map reaches 0.2 within 4 px of the apex. v2 and v3 run on identical crops
  (half >= 304 px; v2 moves by at most one trace against the table above). The v3 configuration was fixed before
  its first run (the m1 fold); the ablations on that fold came afterwards and did not change it; the m2 and ld folds
  were each run once.

### Results

| held out (traces: young, mid, long) | detector | top-1 within 4 px: all (young, mid, long) | top-3 | top-8 | reader's 9 candidates | top-20 of the reader's region | map >= 0.2 at the apex | onset AUROC (false alarms at 80% young) |
|---|---|---|---|---|---|---|---|---|
| m2 (61: 24, 18, 19) | v2 | 29 (16, 10, 3) | 41 (21, 15, 5) | 44 (21, 16, 7) | 32 (19, 10, 3) | 38 (16, 14, 8) | 39 (13, 16, 10) | 0.81 (0.38) |
| m2 (61: 24, 18, 19) | v3 | 28 (15, 12, 1) | 38 (21, 13, 4) | 47 (22, 14, 11) | 32 (19, 11, 2) | 38 (15, 13, 10) | 39 (13, 13, 13) | 0.87 (0.27) |
| ld (116: 34, 65, 17) | v2 | 88 (31, 43, 14) | 102 (34, 53, 15) | 105 (34, 55, 16) | 98 (33, 51, 14) | 104 (34, 54, 16) | 101 (31, 55, 15) | 0.92 (0.06) |
| ld (116: 34, 65, 17) | v3 | 87 (30, 44, 13) | 103 (33, 55, 15) | 104 (33, 56, 15) | 100 (32, 53, 15) | 102 (33, 54, 15) | 104 (31, 57, 16) | 0.92 (0.10) |
| m1 (57: 27, 13, 17) | v2 | 35 (25, 7, 3) | 38 (25, 9, 4) | 40 (25, 10, 5) | 36 (25, 8, 3) | 37 (23, 9, 5) | 36 (24, 7, 5) | 0.89 (0.22) |
| m1 (57: 27, 13, 17) | v3 | 34 (25, 6, 3) | 41 (26, 9, 6) | 43 (26, 9, 8) | 35 (25, 7, 3) | 41 (24, 9, 8) | 43 (25, 9, 9) | 0.94 (0.11) |

Paired over m2 and m1 (v3 vs v2, traces gained / lost): long top-8 +8 / -1, long in the reader region's top 20
+7 / -2, long top-3 +2 / -1, mid top-8 +0 / -3, young top-8 +3 / -1. On m2, v3 raises the map at visible long and
mid tips (g016 b244 0.03 -> 0.21, g052 b244 0.09 -> 0.29, g005 b349 0.26 -> 0.42, g060 b244 0.21 -> 0.55) but drops
three mid ones (g043 b244, g082 b140, g106 b244).

Ablations on the m1 fold (trained on ld + m2; 17 long traces):

| variant | top-1 (young, mid, long) | top-8 (young, mid, long) | long in the reader region's top 20 | long apexes with map >= 0.2 | onset AUROC |
|---|---|---|---|---|---|
| v2 | 35 (25, 7, 3) | 40 (25, 10, 5) | 5 | 5 | 0.89 |
| all apexes; v2 inputs and sampling | 33 (25, 6, 2) | 41 (26, 9, 6) | 6 | 7 | 0.96 |
| the same, seed 1 | 36 (26, 8, 2) | 43 (27, 10, 6) | 6 | 6 | 0.95 |
| all apexes + D12, D24 | 31 (24, 5, 2) | 42 (26, 9, 7) | 9 | 6 | 0.92 |
| all apexes + shifted samples, long weight, unscored structure; v2 inputs | 35 (24, 8, 3) | 42 (26, 9, 7) | 7 | 9 | 0.93 |
| **v3** (all of the above) | 34 (25, 6, 3) | 43 (26, 9, 8) | 8 | 9 | 0.94 |
| v3, seed 1 | 32 (23, 6, 3) | 44 (27, 9, 8) | 8 | 8 | 0.91 |
| v3 with a 5-level U-Net (reach ~90 px) | 33 (24, 6, 3) | 42 (27, 9, 6) | 7 | 6 | 0.90 |
| v3, maps averaged over 4 flips | 35 (25, 7, 3) | 42 (25, 9, 8) | 8 | 8 | 0.94 |

Fixing the apex cap alone gives +1 long tip on m1 (6/17 in both seeds) and the best m1 onset AUROC (0.95-0.96);
the new inputs and the sample changes each add about one more, and together +2 (8/17 in both seeds), at a small cost
in m1's onset AUROC (0.91-0.94). On the m2 fold the apex fix alone does nothing (long top-8 8/19, reader region
top-20 7, map >= 0.2 at 9 long apexes, AUROC 0.82, against v2's 7, 8, 10, 0.81 and v3's 11, 10, 13, 0.87): there
the gain comes from the new inputs and samples. A larger reach (5 levels) and test-time flips did not help. The
reader's 9 candidates hold 1-3 of the 17 long apexes for every variant.

### Use in the reader

`sparsetrack/tiptraj.py` loads only version 2 checkpoints (inputs A, D, C). For v3, build the maps with
`python -m prototypes.tip_detector.maps3 <movie> --net runs/research/tip_detector/tip3_<fold>.pt` (held out: ld <-
`tip3_m2m1`, m2 <- `tip3_ldm1`, m1 <- `tip3_ldm2`; a new movie <- `tip3_all`) and pass the directory
(`runs/research/tip_detector/det3/<net>/<movie>`) as `Params.tiptraj_det`: the same format as `det_cache`
(`b<bin>.npz` with `tip`, uint8 P x 250; `--body` adds the body head). Checked: the full-frame inputs equal the
evaluated crops' exactly, the maps agree to 1e-5, and `tiptraj.DetMaps` reads them (m1 detector scale 0.62). Built,
each movie with its held-out fold model (1.4-4.4 s a bin on the shared machine; 80 MB in all): `det3/tip3_ldm2/m1`,
`det3/tip3_ldm1/m2`, `det3/tip3_m2m1/ld` under runs/research/tip_detector. `tip3_all.pt` (ld + m2 + m1, the same
configuration and schedule, a third of each batch from each movie) is for production use only; it cannot be judged.

Recommendation: use v3 in the reader only together with more detector candidates a bin (K_DET ~20 instead of 6, or
peaks chosen by reach along the tube map) - with the present 3 + 6 candidates it changes nothing on the long tubes;
and add candidates where a body meets the frame edge. Its better onset separation (AUROC 0.87 / 0.94 on the crowded
movies) may help the reader's germination calls by itself.

## Files

- `common.py` inputs, movies, network; `offsets.py` the labelling tool's per-bin grain positions;
  `build.py` samples; `train.py`; `evaluate.py`; `summary.py` / `final_table.py` tables; `sheet.py` contact sheets;
  `link.py` the linking check.
- Version 3: `common3.py` inputs with several lags; `build3.py` samples (all apexes, shifted bins); `train3.py`;
  `evaluate3.py` (several detectors on identical crops, the reader's view); `table3.py`; `maps3.py` full-frame maps
  in the reader's format.
- Outputs in `runs/research/tip_detector/` (git-ignored): `samples_*.npz`, `tip2_{ldm1,m2m1,ldm2}.pt` (version 2,
  named by their training movies), `tip_ldm2.pt` (version 1), `eval_<movie>_<tag>.json`, `link_*.json`, logs;
  version 3: `samples3_<movie>.{npy,npz}`, `tip3_{ldm1,m2m1,ldm2}.pt` (fold models), `tip3_all.pt` (all three
  movies, production only), `abl*_*.pt` / `tip3s1_ldm2.pt` (ablations, seed 1), `eval3_<movie>_<tag>.json`,
  `train3_*.log`.

Reproduce: `python -m prototypes.tip_detector.offsets; python -m prototypes.tip_detector.build;
python -m prototypes.tip_detector.train --train ld m1 --name tip2_ldm1 --iters 2500;
python -m prototypes.tip_detector.evaluate m2 --net runs/research/tip_detector/tip2_ldm1.pt --tag v2` (likewise
m2 m1 -> ld, ld m2 -> m1); `python -m prototypes.tip_detector.final_table v2 v2 v2`.

Reproduce version 3: `python -m prototypes.tip_detector.build3 ld m2 m1;
python -m prototypes.tip_detector.train3 --train ld m2 --name tip3_ldm2 --shift --long-w 2 --protect` (likewise
ld m1 -> tip3_ldm1, m2 m1 -> tip3_m2m1, ld m2 m1 -> tip3_all; ablations: `--channels A,D6,C`, no `--shift` etc.,
`--widths 16,32,64,128,128`, `--seed 1`); `python -m prototypes.tip_detector.evaluate3 m1 --nets
v2=runs/research/tip_detector/tip2_ldm2.pt tip3_ldm2=runs/research/tip_detector/tip3_ldm2.pt --tag v2v3a` (append
`+tta` to a checkpoint for flip averaging); `python -m prototypes.tip_detector.table3 eval3_m1_v2v3a.json [--md]`.
