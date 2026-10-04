# A global tip-trajectory reader (research prototype, 4 Oct 2026)

**Question.** Per grain, choose the tube's tip at every bin in one whole-movie optimisation (Viterbi over bins), each
candidate tip with its tube body (the cheapest route on that bin's tube map from the grain's rim), transitions
rewarding bounded growth and a body that agrees with the previous bin's (so that jumping onto another tube is
expensive). Does it read the crowded movies (m2, m1) better than SparseTrack 0.8.8? (Literature: Li, Shen & Huang,
IPMI 2011, global tip DP with per-frame body search; RootNav 2.0, tip heatmaps driving path search.)

**Answer: yes on the crowded movies, offline.** Tuned on ld alone and applied unchanged, it reads movie 1 (held out for
the weights, the design and the tube maps) far better than 0.8.8: lengths 25 vs 15 of 50 (+10, 95% CI +2 to +18),
length-and-tip 22 vs 15, onsets 10 vs 8; movie 2 lengths 32 vs 27 of 54 (+5, -3 to +13), length-and-tip 31 vs 23 (+8,
CI 0 to +16); ld 77 vs 75 of 104, length-and-tip 75 vs 67. Tuned on movie 2 instead: movie 1 28/50 (+13, CI +4 to +22),
movie 2 37/54 (in sample), ld onsets fall (7 vs 15/26) unless only flooded grains are re-read. Jumps onto other tubes
nearly disappear on movie 2 (off-route misses 10 -> 1); movie 1's floods that never started are read. What still limits
it: where a body leaves the grain (tip right, length off), and long tubes on moving movie-1 grains. Implemented as
`Params.tiptraj` (off by default, sparsetrack/tiptraj.py); end-to-end check below.

## Method

- `detmaps.py`: the tip detector (prototypes/tip_detector, version 2) run on the whole registered frame of every bin
  with each movie's held-out fold (ld: trained on m2+m1; m2: ld+m1; m1: ld+m2), four overlapping tiles, stored
  compressed (both heads, ~120 KB per bin; ~100 MB for the three movies).
- `cands.py`: per scored grain and bin, where 0.8.8 put the grain (census + its drift; label-free). On a crop of the
  shipped tube map P (tubes_bn_real_ld_m2; trained on ld and m2 traces, so only m1 is honest for the maps) round it:
  passable within 4 px of P >= 0.2 or within 25 px beyond the rim, never inside the grain; cost 1 / (P + 0.1); one
  multi-source minimal-path search (skimage MCP) from the rim ring gives every candidate's body. Candidates: the
  detector's 3 strongest peaks within 30 px of the rim and its 6 strongest anywhere (>= 0.05; in a crowded field the
  grown tips round a grain outshine its young tip), the far ends of map pieces reached from the rim (local maxima of
  path cost on P >= 0.5, the 6 farthest), and the fresh candidates of the last 8 bins carried with the grain
  ("hold"); merged within 3 px, at most 32. Per candidate: body (smoothed, 1 px), its arc length from the rim, exit
  angle and the grain's visible edge there (`analyze.exit_edge`), map support along the body and its longest unmarked
  gap, detector value at the tip, map ahead of the tip and how far the map's band runs on past it, radial gain over
  the first 8 px. Per pair of candidates at consecutive bins: distance from the shorter body's tip to the longer body,
  and mean distance of the shorter body to the longer (grain frame).
- `dp.py`: states per bin = not germinated / a candidate / held (the tube keeps its last reading: no candidate
  explains the bin). Unary: theta - (w_det x detector (optionally divided by the movie's detector scale, the 99th
  percentile of the full-frame map's peaks, label-free) + w_sup x support - w_gap x gap - w_ahead x map ahead -
  w_tan x tangential start - w_carry x carried). Transitions: germination once (w_on; start length soft-capped),
  never back; tube -> tube with length change in [-shrink, vmax a bin], the body distances capped (cap_tip,
  cap_share) and penalised; held -> tube at w_reacq if the length change and the held tip/body agree (cap_reacq).
  Length = arc length - visible-edge offset - tip offset c + w_ext x band beyond the tip (- c_end for map ends),
  optionally non-decreasing from the onset.
- `tune.py`: random search + local refinement on one movie for lengths + length-and-tip + onsets (vmax a generous
  fixed cap, 3-8 px/bin), applied unchanged to the others; `summary.py`: tables (paired bootstrap over grains vs
  0.8.8, by length class, failure classes from `failures.py`). Predictions are 0.8.8's with the scored grains'
  status, onset, lengths and tips replaced (drift kept). `bench.py`: the integrated reader end to end on one movie.

## Results

### Offline (4 Oct 2026)

Predictions = 0.8.8's with the scored grains' status, onset, lengths and tips replaced, scored with
`sparsetrack.evaluate.score`; paired change against 0.8.8 over grains with a 95% bootstrap interval. "flood grains":
only the grains 0.8.8 reads with the flood are re-read (the hybrid deployment, `Params.tiptraj="flood"`). Young < 15 px,
long >= 60 px (length hits of FULL traces, 0.8.8 in brackets).

Tuned on ld (random search 250 + refinement 200 on lengths + length-and-tip + onsets), applied unchanged (setting
`tune_ld_c`, now `sparsetrack.tiptraj.WEIGHTS`):

| movie | grains read | onsets | lengths | length+tip | young | mid | long |
|---|---|---|---|---|---|---|---|
| ld | all | 16/28 (15/26) +1 [-6, +7] | 77/104 (75) +2 [-9, +12] | 75 (67) +8 [-2, +19] | 19/30 (19) | 42/57 (40) | 16/17 (16) |
| ld | flood grains | 15/26 (15/26) +0 | 77/104 (75) +2 [0, +5] | 70 (67) +3 [0, +8] | 20/30 (19) | 41/57 (40) | 16/17 (16) |
| m2 | all | 8/19 (7/18) +1 [-3, +5] | 32/54 (27) +5 [-3, +13] | 31 (23) +8 [0, +16] | 14/25 (11) | 11/18 (10) | 7/11 (6) |
| m2 | flood grains | 7/18 (7/18) +0 [-4, +4] | 30/54 (27) +3 [-4, +11] | 29 (23) +6 [-1, +14] | 13/25 (11) | 10/18 (10) | 7/11 (6) |
| m1 | all (= flood grains) | 10/28 (8/26) +2 [-2, +6] | 25/50 (15) +10 [+2, +18] | 22 (15) +7 [-1, +15] | 16/28 (10) | 4/11 (4) | 5/11 (1) |

T50 (annotator / 0.8.8 / new, frames): ld 7383 / 7559 / 7208; m2 24049 / 23874 / 23523; m1 8602 / 9304 / 8251. Growth
rates within tolerance: ld 24/27 (0.8.8 25), m2 15/18 r 0.91 (11/18, r 0.73), m1 8/18 r 0.46 (8/18, r 0.60).

Tuned on m2 (150 + 150), applied to ld and m1 (setting `tune_m2_c`):

| movie | grains read | onsets | lengths | length+tip | young | mid | long |
|---|---|---|---|---|---|---|---|
| m2 (in sample) | all | 9/19 (7/18) +2 [-2, +6] | 37/54 (27) +10 [+2, +18] | 33 (23) +10 [+2, +18] | 18/25 (11) | 12/18 (10) | 7/11 (6) |
| ld | all | 7/28 (15/26) -8 [-15, -1] | 69/104 (75) -6 [-17, +4] | 68 (67) +1 [-9, +11] | 19/30 (19) | 35/57 (40) | 15/17 (16) |
| ld | flood grains | 14/26 (15/26) -1 [-3, 0] | 75/104 (75) +0 [-3, +3] | 68 (67) +1 [0, +3] | 20/30 (19) | 40/57 (40) | 15/17 (16) |
| m1 | all | 7/28 (8/26) -1 [-7, +5] | 28/50 (15) +13 [+4, +22] | 24 (15) +9 [+1, +18] | 19/28 (10) | 5/11 (4) | 4/11 (1) |

Why traces are missed (`failures.py`, ld-tuned; 0.8.8 in brackets): tip within max(5 px, 10%) but length off - ld 16
(18), m2 14 (13), m1 9 (13); tip elsewhere (another tube or branch, a moved tube) - ld 4 (5), m2 1 (10), m1 10 (9, 6 of
them on grains that had moved > 10 px); nothing read - ld 4 (2), m2 2 (3), m1 5 (13); short/long along the right
tube - ld 3 (4), m2 5 (1), m1 1 (0).

Candidate ceiling (`dp.py oracle`, c = 0): a candidate within 4 px of the apex at ld 100/104, m2 47/54, m1 38/50
traces (m1 long tubes 4/11); one with both the human length and apex at 68 / 27 / 26.

What limits it, with examples:
- **Where the body leaves the grain.** The cheapest route from the rim ring leaves the grain where the tube parts from
  it, not where the annotator starts: on tubes that hug their grain the start lies 6-9 px from the human exit click
  (ld g002 b18: 5.0 vs 11.9 px with the tip 1.1 px from the apex; g004 b25, g011 b70), and tubes that wrap round their
  grain are cut short (ld g031 b122 18 vs 35 px). Adding the arc along the rim from the first exit made ld worse
  (72 -> 67-69): grains turn. The visible-edge offset is trusted only to +/-3.3 px (without: ld 72 vs 74).
- **Long tubes on moving movie-1 grains.** g028 (270 px at b244, read 23), g003 b140 (91 px, read 28, tip 65 px off),
  g064 b140 (153 px, read 93): the candidates do not reach the apex (m1 long tubes: a candidate within 4 px at 4 of 11
  traces) or the chain breaks where the grain's drift or the map jumps.
- **Young tubes against early onsets.** A lower threshold reads more young tubes on movie 2 but calls ld's tubes early
  (m2-tuned: ld onsets 15 -> 7/26); the detector answers more weakly on movie 2 (90th percentile of candidate values
  0.36 vs ld 0.63), partly taken out by dividing it by the movie's own detector scale (label-free).

Development notes (honesty): the hold state, the re-acquisition cap and the visible-edge clip were found on ld; the
near-rim detector peaks and the movie detector scale were added after looking at movie-2 young-tube misses, so movie 2
is a development movie here (as it is for 0.8.x); movie 1 was not looked at before the ld-tuned setting was fixed.
The tube maps saw ld and m2 traces in training, the detector folds did not see the movie they read.

### Integrated reader

`sparsetrack/tiptraj.py` reproduces the prototype exactly (ld g008: same candidates in all 176 bins, same choices and
lengths; the rounded default weights give the tuned scores on all three movies). `Params.tiptraj` ("off" default,
"flood", "all"), `tiptraj_model` (detector checkpoint: maps built once per movie by `tiptraj.det_cache`, ~1 s a bin on
MPS, ~40 MB compressed for 351 bins), `tiptraj_det` (precomputed maps), `tiptraj_weights`, `tiptraj_half` (300).
Tests: `tests/test_tiptraj.py`. Reading costs ~20-30 s per grain over 351 bins (crop operations on 600 px crops and the
minimal-path search), on top of the existing readers.

End to end (`bench.py`: SparseTrack 0.8.8 defaults + `tiptraj=flood` with the default (ld-tuned) weights and each
movie's held-out detector maps, reading the scored grains only - every grain is read independently and the speed-cap
probe still samples the whole census, so these are the readings a whole-movie run gives; paired against the 0.8.8
baseline over grains):

| movie | onsets | lengths | length+tip |
|---|---|---|---|
| ld (`tiptraj_half=200`) | 15/26 (15/26) +0 | 77/104 (75) +2 [0, +5] | 70 (67) +3 [0, +8] |
| m2 | 7/18 (7/18) +0 [-4, +4] | 30/54 (27) +3 [-4, +11] | 29 (23) +6 [-1, +14] |
| m1 | 10/28 (8/26) +2 [-2, +6] | 25/50 (15) +10 [+2, +18] | 22 (15) +7 [-1, +15] |

They equal the offline "flood grains" rows exactly on all three movies (movie 1: every scored grain is flooded).

Secondary variant: movie 1 read on the label-free self-trained m1 maps (prototypes/self_train `m1_r1_sb`, built with
`altmaps.py`) instead of the shipped ones: with the ld-tuned weights onsets 15/28 (+7 vs 0.8.8; +5, CI -1 to +11 vs the
shipped maps), lengths 27/50 (+12, CI +4 to +20; +2, 0 to +5 vs shipped), length-and-tip 24 (+9, +1 to +17); with the
m2-tuned weights 27/50, no gain over the shipped maps. Better maps help this reader a little, mostly its onsets.

### Recommendation

On the held-out movie 1 the gain over 0.8.8 has an interval above zero (+10 lengths of 50 with the ld-tuned weights,
+13 with the m2-tuned ones; the earlier label-free attempts this week, tip detector in the readers and map adaptation,
gave +2 to +7, none significant), the crowded dev movie gains in length-and-tip, and the sparse movie stays level when
only the flooded grains are re-read. Candidate for the default reader of flooded
grains (`tiptraj="flood"`), after (1) a tip detector trained on all three labelled movies is frozen for new movies (the
fold models are for evaluation only), (2) its cost is cut (~20-30 s a grain on top of the other readers; most of it is
full-crop array work on 600 px crops), and (3) one more labelled movie confirms it. The next levers are where a body
leaves the grain (start bodies where the tube emerged instead of anywhere on the rim) and candidates for long tubes on
moving grains.

Files: `common.py`, `detmaps.py`, `cands.py`, `dp.py`, `tune.py`, `failures.py`, `summary.py`, `bench.py`, `altmaps.py`.
Outputs (scratch, not kept in git): detector maps (~100 MB for the three movies), candidates (`cands_<movie>.pkl`),
tuning (`tune_ld_c.json`, `tune_m2_c.json`), predictions (`pred_ld_c_<movie>.json`, `pred_m2_c_<movie>.json`).
Reproduce: `python -m prototypes.tip_trajectory.detmaps ld m2 m1; python -m prototypes.tip_trajectory.cands ld m2 m1;
python -m prototypes.tip_trajectory.tune --tune ld --apply m2 m1 --n 250 --refine 200 --tag ld_c --start '{"c": -1.0,
"w_ext": 0.5, "w_l0": 0.3, "theta": 0.5, "det_norm": 2, "vmax": 4.0}'` (seed 0).
