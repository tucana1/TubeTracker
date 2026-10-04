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
`Params.tiptraj` (off by default, sparsetrack/tiptraj.py); end-to-end check below. Round 2 (below): starting
bodies where the tube emerged did not work (three variants, all below the first reader on ld); lengths along the
middle of the tube help a little everywhere (`tiptraj_mid`: ld +2, m2 +1, m1 +1); a guided second pass is neutral;
on movie 1 the reader on the label-free self-trained maps with `tiptraj_mid` reads 29/50 lengths, 15/28 onsets
(0.8.8: 15/50, 8/26). Round 3: the version 3 tip detector helps once the weights are re-tuned on ld for it (m2 +6
lengths, CI 0 to +13; m1 equal, +2 onsets; with `tiptraj_mid` m1 27/50, m2 37/54); more detector peaks and
frame-edge candidates raise the candidates' ceiling for long tubes but are not taken (not adopted).

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

## Round 2 (4 Oct, after the merge): where the body starts, long tubes on moving grains, movie 1 combination

All measured as round 1: weights tuned on ld (`tune_ld_c`, unchanged unless said), each movie's held-out detector
fold, paired over grains (95% bootstrap) against 0.8.8 and against the merged reader (round 1, "R1").

### 1. Where the body starts (negative)

"Tip right, length off" misses (ld 16, m2 13-14, m1 9) are mostly short on ld (10 of 16) and m2 (9 of 13). Two parts:
the start ANGLE (tangential tubes: the annotator starts where the tube emerged, 20-40 deg from where the cheapest
route leaves the rim; ld g002, g030, m2 g009, g052, g060) and the start RADIUS (the annotator's exit lies 3-11 px
inside or 4-7 px outside the census circle: ld g004 -7, g011 -8, m2 g038 -11, g085 +6.5; the visible-edge offset is
clipped at +/-3.3 px). Tried, all with the ld-tuned weights:

| variant | ld lengths / l&t (R1 77 / 75) |
|---|---|
| bodies start within +/-25 deg of the reading's emergence angle (circular mean of its first young tips' angles; second pass, `guide.py` + `cands.py --exit-arc 25`) | 74 / 72 (-3); with re-tuned offsets c / edge clip / w_ext still 74 / 72 |
| plus the map's band along the rim behind the body's start (`backrim.py`, half weight) | 41 / 40 (the band is the tube's own width and rim marks almost everywhere) |
| lengths measured from the visible edge along the body's own first direction (`dp.py` len_mode "ray") | best 75 / 73 |

The emergence angle from the young tips is the tip's side, not the base's, for tangential tubes (g002: tips at
9 deg, the annotator's exit at -29 deg). On the radius, no label-free edge estimate transfers: against the
annotator's exit radius the visible-edge offset (clipped) is better than the census circle on ld (median error 1.03
vs 1.71 px), equal on m2 (1.65 vs 1.66) and worse on m1 (2.09 vs 1.34); smoothing it round the grain or trusting it
only where the radial step is strong does not fix m1 under any one rule (`edges.py`, step-strength gate). Left as it
is.

A related fix that did help is where the body RUNS: the cheapest route hugs the inside of a curving tube, the
annotator traces its middle (about half the tube's width x the turn, ~11 px for a U-turn). Moving each chosen body
onto the middle of the map's band (`centred.py`, `Params.tiptraj_mid`, weight chosen on ld) helps a little on every
movie and never hurts:

| movie | R1: lengths, l&t | + mid | + mid vs R1 (lengths; l&t) | by class | as `tiptraj='flood'` runs it (flooded grains only): R1 -> + mid |
|---|---|---|---|---|---|
| ld | 77/104, 75 | 79/104, 77 | +2 [-2, +6]; +2 [-2, +6] | mid 42 -> 44 of 57 | 77 -> 78, l&t 70 -> 71 (+1 [0, +3]) |
| m2 | 32/54, 31 | 33/54, 32 | +1 [0, +3]; +1 [0, +3] | long 7 -> 8 of 11 | 30 -> 31, l&t 29 -> 30 (+1 [0, +3]) |
| m1 | 25/50, 22 | 26/50, 23 | +1 [0, +3]; +1 [0, +3] | long 5 -> 6 of 11 | all scored grains flooded; end to end (`bench.py`, `tiptraj_mid=true`): 25 -> 26, l&t 22 -> 23 (+1 [0, +3]), vs 0.8.8 +11 [+3, +19] |

The integrated option reproduces the prototype (ld grains: at most 0.4 px apart, same final lengths; movie 1 end to
end gives exactly the offline numbers).

### 2. Long tubes on moving grains

Why long m1 tubes get no candidate at their apex (`longdiag.py`, 11 traces >= 60 px): ranking (the apex is a map
piece end or a detector peak but not among the 6 kept: g014 b244 end 42nd of 47, g030 b140 29th of 29), the map
blind to the tube (g003 b140 0% of 91 px marked, g007 8%), gaps in the map (g028 b244 104 px, g064 b140 30 px). The
detector's body head marks 19% of these tubes (no help). And one stall with a consistent candidate chain all the way
(g028 b120-140: the detector's peak at the apex every bin, 0-1 px between bins) that the ld-tuned weights do not take:
they weigh map support along the body 13 x the detector (w_sup 2.0 vs w_det 0.16), and the chain's body is less
marked than a stub's on movie 1. The m2-tuned weights, which lean more on the detector, do not take it either (g028
at b244: 32 px of 270; ld-tuned 23).

Guided second pass (`guide.py`, `cands.py --ext --corridor`, `Params.tiptraj_guided`): the first reading's chosen
tip at the latest bin as a candidate, points along the map beyond it (cheapest route ahead within 70 deg, sampled
at 6-80 px and its far end) and a corridor along its body through gaps in the map:

| movie | R1 (first pass): lengths, l&t, onsets | guided: lengths, l&t, onsets | guided vs R1 (lengths; l&t; onsets) | guided vs 0.8.8 (lengths; l&t) | long tubes (>= 60 px) within tolerance: 0.8.8 / R1 / guided |
|---|---|---|---|---|---|
| ld | 77/104, 75, 16/28 | 76/104, 75, 16/28 | -1 [-3, 0]; 0; 0 | +1 [-9, +11]; +8 [-2, +19] | 16 / 16 / 16 of 17 |
| m2 | 32/54, 31, 8/19 | 33/54, 33, 9/19 | +1 [0, +3]; +2 [0, +5]; +1 [0, +3] | +6 [-2, +14]; +10 [+3, +17] | 6 / 7 / 8 of 11 |
| m1 | 25/50, 22, 10/28 | 25/50, 22, 10/28 | 0; 0; 0 | +10 [+2, +18]; +7 [-1, +15] | 1 / 5 / 5 of 11 |

Neutral to slightly positive (m2 one more long tube), at twice the reading time: `Params.tiptraj_guided`, off. With
`tiptraj_mid` (section 3) on top the two add up a little: ld 79/104, l&t 78; m2 34/54, l&t 34, long 9/11 (vs R1 +2
lengths [0, +6], +3 l&t [0, +8]); movie 1's guided readings are R1's.

DIS-flow carrying of the guide's tip was not tried: the diagnosis shows the long misses are ranking, map blindness and
gaps, not tubes moving while held.

### 3. Movie 1 combination

Movie 1 only (the honest test: maps, detector fold and weights never saw it), offline (`combo.py`): the reader on the
shipped maps or on the label-free self-trained m1 maps (prototypes/self_train `m1_r1_sb`, `altmaps.py`), with the
lengths along the middle of the tube (`centred.py`, `Params.tiptraj_mid`) and with the tip detector's young-tube
lengths on top (`tipdet.apply` as on main, `Params.tipdet_young`, read from the stored full-frame maps). Paired over
the 30 grains; R1 = the merged reader on the shipped maps.

| m1, ld-tuned weights | onsets | lengths | length+tip | young / mid / long (of 28 / 11 / 11) | vs 0.8.8 (lengths; l&t; onsets) | vs R1 (lengths; l&t; onsets) |
|---|---|---|---|---|---|---|
| 0.8.8 | 8/26 | 15/50 | 15 | 10 / 4 / 1 | | |
| R1: reader, shipped maps | 10/28 | 25/50 | 22 | 16 / 4 / 5 | +10 [+2, +18]; +7 [-1, +15]; +2 | |
| + mid | 10/28 | 26/50 | 23 | 16 / 4 / 6 | +11 [+3, +19]; +8 [0, +16]; +2 | +1 [0, +3]; +1 [0, +3]; 0 |
| + tipdet_young | 10/28 | 24/50 | 22 | 15 / 4 / 5 | +9; +7; +2 | -1 [-4, +2]; 0; 0 |
| reader, self-trained maps | 15/28 | 27/50 | 24 | 17 / 5 / 5 | +12 [+4, +20]; +9 [+1, +17]; +7 [+1, +13] | +2 [0, +5]; +2 [0, +5]; +5 [-1, +11] |
| **self-trained maps + mid** | **15/28** | **29/50** | **26** | **17 / 6 / 6** | **+14 [+6, +22]; +11 [+4, +19]; +7 [+1, +13]** | **+4 [+1, +8]; +4 [+1, +8]; +5 [-1, +11]** |
| self-trained maps + mid + tipdet_young | 15/28 | 27/50 | 24 | 15 / 6 / 6 | +12; +9; +7 | +2 [-3, +7]; +2 [-3, +7]; +5 |

The young-tube lengths from the detector do not help on top of this reader (it already takes young tips from the
detector: ld -1, m2 -1, m1 -1 to -2, and on movie 1 the loss is in the young class itself, 16 -> 15 and 17 -> 15).
The best movie-1 combination is the self-trained maps with lengths along the middle of the tube: 29/50 lengths and
15/28 onsets, against 15/50 and 8/26 for 0.8.8; its gain over R1 is in the mid and long classes (4 -> 6, 5 -> 6)
and the onsets.

With the m2-tuned weights (secondary): reader 28/50 (onsets 7/28), self-trained maps + mid 30/50, length-and-tip 27
(+15 lengths vs 0.8.8, CI +7 to +23) but onsets 6/28: those weights call movie 1's onsets late.

Weights tuned on both development movies at once (ld + m2, 150 + 150; `tune.py --tune ld m2`, setting `tune_ldm2_c`)
fit them (ld 76/104, l&t 73; m2 37/54, l&t 36) but transfer no better to movie 1: 24/50 lengths, l&t 21, onsets
10/28 (ld-tuned: 25, 22, 10). The ld-tuned weights stay the defaults. No tuning reads movie 1's long stalls: at
the last labelled bin g028 is read 23 / 32 / 31 px of 270 with the ld / m2 / joint weights, g064 93 / 80 / 77 of
153, g003 28 / 19 / 27 of 91 (the map blind or gapped there, section 2).

The same combinations with the joint weights (`combo.py --setting tune_ldm2_c.json`) keep the order:

| m1, ld + m2 weights | onsets | lengths | length+tip | vs 0.8.8 (lengths; l&t; onsets) | vs the first row (lengths; l&t) |
|---|---|---|---|---|---|
| reader, shipped maps | 10/28 | 24/50 | 21 | +9 [+1, +17]; +6 [-1, +14]; +2 | |
| + mid | 10/28 | 28/50 | 23 | +13 [+4, +22]; +8 [0, +16]; +2 | +4 [+1, +8]; +2 [0, +5] |
| + tipdet_young | 10/28 | 24/50 | 22 | +9; +7; +2 | 0 [-4, +4]; +1 |
| reader, self-trained maps | 13/28 | 28/50 | 25 | +13 [+5, +21]; +10 [+2, +17]; +5 [0, +11] | +4 [0, +9]; +4 [0, +9] |
| **self-trained maps + mid** | **13/28** | **30/50** | **27** | **+15 [+7, +23]; +12 [+5, +19]; +5 [0, +11]** | **+6 [+1, +11]; +6 [+1, +11]** |
| self-trained maps + mid + tipdet_young | 13/28 | 28/50 | 25 | +13; +10; +5 | +4 [-2, +10]; +4 [-2, +11] |

Self-trained maps with lengths along the middle are again the best (30/50, length-and-tip 27, onsets 13/28; with
the ld-tuned weights 29, 26, 15/28), the middle lengths again never hurt (+4 on the shipped maps here, +2 on the
self-trained ones), and the detector's young-tube lengths again cost 2 on top.

### Round-2 recommendation

Turn on `tiptraj_mid` together with the reader (small, consistent: +2 / +1 / +1 lengths, +2 / +1 / +1 length-and-tip
on ld / m2 / m1, never negative; cheap: one band profile per chosen reading). On a new movie, adapt the maps first
(label-free self-training): with them and `tiptraj_mid` movie 1 reads 29/50 lengths and 15/28 onsets, +4 lengths
(CI +1 to +8) and +5 onsets over round 1 (with weights tuned on ld + m2: 30/50, 13/28). Leave `tiptraj_guided` off
(neutral, twice the time) and do not add `tipdet_young` on top (-1 to -2). Where the body starts stays open: the
annotator's exit is a property of the grain's boundary that no label-free estimate here measures well on movie 1.
Long stalls on movie 1 (g028, g064, g003) stay open too: no weights read them, the map is blind or gapped there.

## Round 3 (4 Oct): the version 3 tip detector

The v3 detector (prototypes/tip_detector, version 3: inputs with 12- and 24-bin changes) as each movie's held-out
fold (ld tip3_m2m1, m2 tip3_ldm1, m1 tip3_ldm2; maps in runs/research/tip_detector/det3), in place of v2, with the
round-1 candidate settings ("6 peaks") or with its author's two suggestions ("20 + edge": 20 detector peaks a bin,
48 candidates with the carried ones, and a candidate wherever the map meets the frame's border, `cands.py --det v3
--k-det 20 --max-c 48 --edge`: per stretch of map along the border its point farthest along, the cheapest route per
px of reach first, 2 a bin). R1 = the merged reader (v2 maps, 6 peaks, `tune_ld_c`); paired over grains.

Candidates at the apex (within 4 px among a bin's candidates; `combo.py --oracle`): the suggestions raise the
ceiling for long tubes as expected.

| candidates | long traces (>= 60 px): ld / m2 / m1 | all traces: ld / m2 / m1 |
|---|---|---|
| v2, 6 peaks (R1) | 17/17, 8/11, 4/11 | 100/104, 47/54, 38/50 |
| v3, 6 peaks | 16/17, 8/11, 6/11 | 98/104, 47/54, 39/50 |
| v2, 20 + edge | 17/17, 10/11, 5/11 | 100/104, 50/54, 40/50 |
| v3, 20 + edge | 16/17, 11/11, 7/11 | 98/104, 51/54, 40/50 |

The reader with the SAME weights (`tune_ld_c`; lengths, length-and-tip, onsets; vs R1 lengths):

| candidates | ld | m2 | m1 | long tubes read: m2 / m1 |
|---|---|---|---|---|
| v2, 6 peaks (R1) | 77/104, 75, 16/28 | 32/54, 31, 8/19 | 25/50, 22, 10/28 | 7 / 5 |
| v3, 6 peaks | 74, 74, 18 (-3 [-7, 0]) | 34, 33, 7 (+2 [-2, +6]) | 23, 19, 13 (-2 [-6, +2]) | 7 / 4 |
| v2, 20 + edge | 77, 75, 16 (0) | 31, 30, 8 (-1 [-4, +2]) | 22, 19, 10 (-3 [-7, +1]) | 7 / 3 |
| v3, 20 + edge | 74, 74, 18 (-3 [-7, 0]) | 35, 34, 7 (+3 [-1, +7]) | 23, 20, 13 (-2 [-7, +3]) | 7 / 3 |

With `tiptraj_mid` the four rows read ld 79 / 76 / 79 / 76, m2 33 / 36 / 31 / 36, m1 26 / 25 / 24 / 25. The
added long-tube candidates are not taken (long tubes read stay 7/11 on m2 and fall on m1), and the frame-edge
candidates cost about one reading where they occur (dropped at the Viterbi: m1 v3 23 -> 24, v2 22 -> 23; m2 v2
31 -> 32). With these weights the v3 maps help m2, cost ld 3 (three mid-length traces of 23-25 px, missed by at most 1.3 px
beyond the tolerance) and m1 2 lengths, and add onsets on ld and m1 (+2, +3).

Each candidate set with weights RE-TUNED on ld (same protocol, search and start as `tune_ld_c`; applied unchanged):

| candidates (weights) | ld (in sample) | m2 | m1 |
|---|---|---|---|
| v2, 6 peaks (R1, `tune_ld_c`) | 77/104, 75, 16/28 | 32/54, 31, 8/19 | 25/50, 22, 10/28 |
| v3, 6 peaks (`tune_ld_v3`) | 78, 78, 20 (+1 [-3, +6]) | 38, 36, 8 (+6 [0, +13]) | 25, 22, 12 (0 [-5, +6]) |
| v2, 20 + edge (`tune_ld_v2k20e`) | 77, 76, 17 (0) | 31, 30, 10 (-1 [-5, +3]) | 23, 19, 12 (-2 [-7, +3]) |
| v3, 20 + edge (`tune_ld_v3k20e`, = `tune_ld_v3`) | 78, 78, 20 (+1 [-3, +6]) | 36, 35, 8 (+4 [-4, +12]) | 22, 21, 12 (-3 [-8, +2]) |
| R1 + mid | 79, 77, 16 | 33, 32, 8 | 26, 23, 10 |
| **v3, 6 peaks + mid** | **77, 77, 20 (0 [-7, +7])** | **37, 35, 8 (+5 [0, +11])** | **27, 24, 12 (+2 [-3, +7])** |

As `tiptraj='flood'` runs it (only the flooded grains re-read): ld 78/104 (l&t 72) vs R1 + mid 78 (71); m2 35/54
(33) vs 31 (30); m1, all flooded, 27/50 (24), onsets 12/28, vs 26 (23), 10/28 (0.8.8: ld 75, m2 27, m1 15/50 and
8/26). End to end on movie 1 (`bench.py`, v3 fold maps, `weights_ld_v3.json`, `tiptraj_mid`): 27/50, l&t 24,
onsets 12/28, exactly the offline numbers; vs 0.8.8 +12 lengths [+3, +21], vs round 2's end-to-end reader (v2 + mid,
26/50) +1 [-2, +4], onsets +2 [-3, +8]; by class young 16, mid 5, long 6 (round 2: 16, 4, 6).

The re-tuned weights trust the v3 detector nine times more than the v2 ones did (w_det 1.42 vs 0.16; map support
unchanged at 2.0, the top of its range; w_gap at the top, vmax and w_ahead at the bottom of theirs). On ld the 20 +
edge candidates differ too little to change the tuning (same weights found).

On movie 1's label-free self-trained maps (round 2's best combination) the v3 maps add to it: with `tiptraj_mid`,
v3 + its weights read 32/50 lengths, length-and-tip 29, onsets 15/28 (young 20, mid 6, long 6), against v2 + its
weights 29, 26, 15/28: +3 lengths [-1, +7], +3 l&t [-1, +7]; vs 0.8.8 +17 [+9, +25], vs R1 +7 [+1, +13].

The production loader (`tiptraj.det_cache`, `load_detector`) now builds maps from a version 3 checkpoint directly;
for ld with tip3_m2m1.pt all 168 bins equal the precomputed det3 maps (0 of 220 M pixels differ; inputs and float
maps of the two code paths identical on three bins).

### Round-3 recommendation

For the next blind movie: v3 detector (`tiptraj_model` = runs/research/tip_detector/tip3_all.pt), its ld-tuned
weights (`tiptraj_weights` = prototypes/tip_trajectory/weights_ld_v3.json), lengths along the middle
(`tiptraj_mid=True`), `tiptraj='flood'`, candidates at the round-1 settings; where the label-free self-trained
maps can be made first, read on them (movie 1: 32/50 vs 27/50 on the shipped maps). Do not raise the peaks or add
frame-edge candidates: they raise the ceiling but this reader does not use it; what limits long tubes now is the
choice among candidates (m1: 7 of 11 long apexes have a candidate, 3-4 are read), so the next lever is the unary
for long bodies (it weighs a well-marked stub against a long body whose map has gaps), not more candidates.

Files: `common.py`, `detmaps.py`, `cands.py`, `dp.py`, `tune.py`, `failures.py`, `summary.py`, `bench.py`, `altmaps.py`;
round 2: `guide.py`, `longdiag.py`, `backrim.py`, `edges.py`, `centred.py`, `combo.py`; round 3: `cands.py --det v3
--k-det --max-c --edge`, `combo.py --oracle --drop-src --settings`, `weights_ld_v3.json`.
Outputs (scratch, not kept in git): detector maps (~300 MB for the three movies), candidates (`cands_<movie>.pkl`,
~180 MB), tuning (`tune_ld_c.json`, `tune_m2_c.json`, `tune_ldm2_c.json`), predictions (`pred_ld_c_<movie>.json`,
`pred_m2_c_<movie>.json`); round 2 also self-trained m1 maps (~440 MB) and second-pass candidates, deleted after use.
Reproduce: `python -m prototypes.tip_trajectory.detmaps ld m2 m1; python -m prototypes.tip_trajectory.cands ld m2 m1;
python -m prototypes.tip_trajectory.tune --tune ld --apply m2 m1 --n 250 --refine 200 --tag ld_c --start '{"c": -1.0,
"w_ext": 0.5, "w_l0": 0.3, "theta": 0.5, "det_norm": 2, "vmax": 4.0}'` (seed 0). Round 2: guided pass `guide ld m2
m1`, `cands ld m2 m1 --guide "guide_{movie}.pkl" --ext --corridor --tag _g`, `combo <movie> --cands cands_<movie>.pkl
cands_<movie>_g.pkl`; movie 1 combination `altmaps m1 runs/research/self_train/models/m1_r1_sb.pt`,
`TT_GT_PROB=<maps> cands m1 --tag _st`, `combo m1 --cands cands_m1.pkl cands_m1_st.pkl --young --mid --prob <maps>`
(add `--setting tune_ldm2_c.json` for the joint weights; `tune --tune ld m2 --apply m1 --n 150 --refine 150 --tag
ldm2_c` with the same start). Round 3: `cands ld m2 m1 --det v3 --tag _v3` (and `--k-det 20 --max-c 48 --edge
--tag _v3k20e`, without `--det v3` for `_v2k20e`); `tune --tune ld --apply m2 m1 --cands "cands_{movie}_v3.pkl" --n
250 --refine 200 --tag ld_v3` with the same start (its `best` is `weights_ld_v3.json`); `combo <movie> --cands
cands_<movie>.pkl cands_<movie>_v3.pkl --settings tune_ld_c.json tune_ld_v3.json --mid [--oracle] [--only-flood]`;
self-trained maps `TT_GT_PROB=<maps> cands m1 --det v3 --tag _st_v3`.
