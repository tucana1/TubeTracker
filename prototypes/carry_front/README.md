# carry_front: final state first, then backward (oracle test, 3 Oct 2026)

Question: if each tube's route is decided where it is most visible (its latest human trace), carried backward
through the movie with composed DIS optical flow, and the tip read off the learned tube maps along it with a
globally optimal monotone front, do the crowded movies (m2, m1) read far better than SparseTrack 0.8.8? If so, the
remaining problem would only be finding the final route automatically.

## Method

- `carry.py`: crops round each route (bounding box + 60 px), 3-bin means of the registered frames, OpenCV DIS
  (PRESET_MEDIUM) on uint8 over the crop's 1-99 percentile window, composed in 10-bin steps, linear in between.
  Routes are carried as polylines, never images. `carry_inext` carries a route as a tube: per step the flow's median
  over the route's first 15 px (grain and base) plus the sideways (normal) part of the rest (running median 9,
  Gaussian 5 px along the tube), material spacing kept from the base; only points passing a forward-backward flow
  check (1.5 px) count; at most 12 px per step. The plain per-point carry (`carry`) is kept as a diagnostic.
- `build.py`: per scored grain with a FULL trace, the route = the latest full human trace (its exit first), carried
  on straight 40 px past the apex (so the front is not capped at the answer). Variants: `a` fixed in reference
  coordinates; `a2` moved rigidly with 0.8.8's grain drift; `braw` plain composed flow; `b` tube carry backward and
  forward from the route's bin; `c` as b, but between two human traces the earlier trace, carried forward from its
  bin, replaces the route up to its length (at a scored trace the route comes from the previous trace, never from
  that trace itself). K[bin, s] = tube map (tubes_bn_real_ld_m2, the 0.8.8 maps) maxed over +/- w px across the
  route.
- `o1.py`: front = monotone non-decreasing DP (as `sparsetrack.analyze.dp_front`, with bounds) on K - theta, speed
  cap vmax px/bin (or `--cap auto`: a factor x each movie's own label-free growth speed), the first `skip` px
  neutral; length = material arc length to the front minus a tip offset c; onset = first bin with a length.
  "forced": the human onset imposed (no tube through the last absent bin, >= 1 px from the first visible bin).
  `--arrival`: ownership by arrival (`carry.dp_arrival`: passing a route point costs mu per bin by which the map had
  marked it more than delta bins earlier).
  theta, w, vmax, skip, c tuned on one movie (ld; reverse: m2), applied unchanged to the others. Predictions = the
  0.8.8 baseline's with labelled grains' length, tip (reference coordinates, drift zero) and onset replaced, scored
  with `sparsetrack.evaluate.score`, paired bootstrap over grains as `scripts/compare_predictions.py`.
- `o2.py`: why 0.8.8 misses: no tube read (onset), its drawn route at that bin (`report.turned_path` + drift, cut to
  the shorter length) > 3 px from the human polyline (route), else front short/long.
- `mapcheck.py`: does the map mark the human tube at the traced bin and stop at its apex (no carrying involved).
- `o3.py` + `followup.py` (with helpers in `o3eval.py`): O3, 0.8.8's own routes through the same carry and front,
  and the soft recency term (`carry.dp_recency`); see "Follow-ups". `apply.py` re-applies tuned settings to other
  kymographs (carry sensitivity); `show_rows.py` prints per-trace rows.

Outputs (git-ignored): `runs/research/carry_front/` (kymographs per grain, predictions, summaries, logs).

## Results (3 Oct 2026; 0.8.8 baseline = the main session's scratchpad bt/base088/{ld,m2,m1}_real_0, scored with
evaluate.score, isolated grains)

Verdict: **negative for the crowded movies.** Even with the human's own final route, carrying it backward and reading
a globally optimal monotone front on the tube maps reads movie 2 only slightly better than 0.8.8 (+5/+6 lengths, CIs
include 0) and movie 1 not at all. Finding the final route is not the bottleneck. With 0.8.8's own routes (O3,
run afterwards at the main session's request, below) the same reader loses to 0.8.8 on all three movies.

O1, settings tuned on ld and applied unchanged (lengths within max(2 px, 10%) / length-and-tip; paired against 0.8.8
over grains, 95% bootstrap CI):

| | ld | m2 | m1 |
|---|---|---|---|
| 0.8.8 | 75/104, 67 | 27/54, 23 | 15/50, 15 |
| a: route fixed | 76, 70 (+1, -8..+9) | 30, 27 (+3, -9..+15) | 14, 13 (-1, -10..+8) |
| a2: route moved with 0.8.8's grain drift | 79, 74 | 29, 24 | 15, 14 |
| braw: plain composed flow per point | 35, 33 | 13, 12 | 4, 3 |
| **b: tube carry** | **82, 81 (+7, -3..+17; l&t +14, +2..+26)** | **32, 29 (+5, -6..+15; l&t +6, -2..+14)** | **14, 14 (-1, -11..+8)** |
| c: b, re-anchored on every earlier human trace | 80, 70 (+5) | 33, 29 (+6, -4..+16) | 24, 18 (+9, +2..+16) |
| b, human onset forced* | 83, 80 (+8, -2..+18) | 31, 29 (+4, -7..+14) | 15, 15 (+1) |
| c, human onset forced* | 81, 71 (+6) | 36, 31 (+9, -1..+19) | 14, 9 (-1) |
| b, per-movie speed cap (1.0 x own growth: 1/3/2 px per bin) | 82, 81 | 32, 30 (+5, -5..+15) | 13, 13 |
| b + arrival ownership (strong) | 83, 81 | 28, 26 (+1) | 13, 12 |
| reverse, tuned on m2: b | 77, 76 | 37, 34 (+10, -1..+21) | 10, 10 (-5) |
| reverse, tuned on m2: c | 74, 66 | 38, 28 (+11, +2..+20) | 25, 13 (+10, +4..+16) |

b (ld-tuned: band +/-1 px, P >= 0.3, 1 px/bin, offset 1.5 px): median |error| / bias ld 1.52 / -1.53 px (0.8.8 2.31 /
-0.48), m2 2.67 / -0.38 (2.91 / -5.88), m1 4.80 / -12.2 (4.13 / -10.2). Its misses: ld 22 = route 8, onset 1, short 9,
long 4; m2 22 = route 6, onset 2, short 5, long 9; m1 36 = route 17, onset 6, short 8, long 5 (route: the route at
that bin > 3 px from the human polyline over the human length). Onsets from the front are worse than 0.8.8's (ld 9/27
vs 15/26); forcing the human onset changes lengths by 0-1, so onset is not what limits lengths.

- The carry works where it is given a chance: b's route lies within 3 px of the human trace at 89% (ld), 80% (m2),
  35% (m1) of earlier traces (fixed route 72%, 44%, 15%). Carried naively per point, composed flow plays growth in
  reverse and piles the route beyond the tip onto the tube (braw: fronts 10-13 px long), the 29 Sep dense-warp failure
  again; dropping the flow along the tube (inextensible carry) is essential. m1 with 5-bin steps and a 30 px step cap:
  b 13/50, c 21/50 (unchanged).
- Map check (no carrying; the map along the human polyline at its own bin, +/-2 px): >= 90% of the traced tube
  marked for 90% (ld), 89% (m2), 70% (m1) of traces; past the apex (3-12 px) clear for 100/94/98%. The network was
  fine-tuned on ld and m2 traces, so those two are in-sample; on m1 (unseen), 12% of traces have unmarked runs > 5 px
  (g003 b140: 0% of a 91 px tube marked; g007 b140 9%; g028 b244 gap 105 px).
- What b still misses on m2 (per-movie cap): 6 stubs <= 8 px read 2.5-3.8 px long (the young tube's blob runs past
  its apex; tolerance 2 px); 5 young tubes read as none or on a moved base (g043, g082, g106, g064 b109, g069 b70);
  6 over-reads of 9-24 px (g052 b140 jumps a 24 px gap onto a tube on its future route that was there since bins 79-124;
  g011 b349 and g016 b244 run past the human's final apex); 4 short; g016 b140 +3.6 px on 24 px. A speed cap trades g005 (257 px: 186 at 1 px/bin,
  258 at 3 px/bin) against g016 b244 (117 -> 147 vs 123). Ownership by arrival (refuse material that was there before
  the tip came) fixes the gap jump in a toy but blocks tubes that grow across older ones (g005 b349 -> 95 px, g038
  b349 157 -> 110): m2 32 -> 28.

O2, why 0.8.8 misses (`o2.py`; its drawn route on hits: median 0.74 / 0.90 / 0.85 px from the trace):

| | misses | route off (> 3 px) | front short | front long | no tube read |
|---|---|---|---|---|---|
| ld | 29/104 | 9 | 8 | 10 | 2 |
| m2 | 27/54 | 5 | 9 | 8 | 5 (2 late, 3 zero) |
| m1 | 35/50 | 10 | 6 | 5 | 14 (10 late, 3 never, 1 zero) |

On m2, 17 of 0.8.8's 27 misses are front errors on a correctly drawn route; on m1 onsets (four grains whose flood never
started: g005, g033, g047, g065) and routes dominate.

*Forced rows re-run after a bug fix (`followup.py`): forcing a tube from the first visible bin made the step from the
forced 0 infeasible when vmax = 1 px/bin, so the first runs' forced grids silently lost every vmax = 1 setting (they
had read b 82/33/14 and c 76/29/15); the cap on the forced minimum is now <= vmax. The free (unforced) runs were not
affected.

## Follow-ups (3 Oct, later): O3 with 0.8.8's own routes, and a soft recency term

`o3.py` builds O3 kymographs (`runs/research/carry_front/o3/<movie>/`); `followup.py` scores everything in one
process (results `followup.json`; O3 predictions `o3/pred/O3{b,c}_<config>_<movie>.json`, the 0.8.8 baseline's
with the labelled grains' length, tip and, in the "front onsets" configs, onset replaced).

- O3b: 0.8.8's final route (its drawn route at the last bin, `report.turned_path` + drift: `res["path"]`, centred),
  carried on 40 px straight and carried backward like b. O3c: re-anchored on 0.8.8's per-bin routes (`route_at` via
  `turned_path`, + drift) every 25 bins and wherever `path_by_bin` switches, each carried forward to the next anchor
  (as c). Grains 0.8.8 drew no route for keep its prediction (ld g015; m2 g068, g100; m1 g005, g029, g047).
- How close those routes lie to the human traces (all scored FULL traces, within 3 px / median): O3b ld 88% / 1.0 px,
  m2 61% / 2.0 px, m1 2% / 24 px; O3c 80%, 72%, 47% / 3.7 px (O1's b from the human route: 89% / 80% / 35% of
  earlier traces).

| O3 vs 0.8.8 (lengths, len&tip; paired change, 95% CI) | ld | m2 | m1 |
|---|---|---|---|
| 0.8.8 | 75, 67 | 27, 23 | 15, 15 |
| O3b, O1's ld settings | 64, 61 (-11, -20..-1) | 10, 9 (-17, -26..-8) | 1, 1 (-14) |
| O3b, re-tuned on ld | 69, 65 (-6, -15..+3) | 15, 13 (-12, -22..-2) | 0, 0 (-15, -23..-8) |
| O3b, re-tuned on ld, 0.8.8's onsets | 68, 65 (-7) | 16, 14 (-11, -21..-1) | 1, 0 (-14) |
| O3b, tuned on m2, 0.8.8's onsets | 46, 40 (-29) | 23, 22 (-4, -13..+6) | 2, 1 (-13) |
| O3c, O1's ld settings | 59, 55 (-16) | 14, 13 (-13) | 14, 10 (-1) |
| O3c, re-tuned on ld | 66, 58 (-9, -19..+1) | 15, 14 (-12, -19..-6) | 14, 13 (-1, -8..+6) |
| O3c, re-tuned on ld, 0.8.8's onsets | 65, 56 (-10) | 15, 14 (-12) | 13, 12 (-2) |
| O3c, tuned on m2, 0.8.8's onsets | 55, 42 (-20) | 28, 21 (+1, -8..+10) | 15, 11 (+0) |

Onsets: the front's are worse than 0.8.8's (O3b ld 12/27, m2 5/17, m1 0/12 vs 15/26, 7/18, 8/26), so the "0.8.8's
onsets" rows keep 0.8.8's status and onset (the front held at 0 before it); lengths move by 0-1.

Why O3 loses even on ld, where 0.8.8's routes are good: O1's routes started at the annotator's exit click, an oracle
worth about 10 ld hits on its own (29 Sep: model started at the human exit, ld 62 -> 71-73); O3 starts at 0.8.8's
exit, and its extra misses are lengths off by a few px (ld O3b: short 16, long 6, route 10, vs O1 b's 9, 4, 8). On
m2, 0.8.8's FINAL route is often not the traced tube (route off at 20 of 54 traces; g052's ends 32 px from the human's
final trace), so carrying it backward misplaces every earlier bin; re-anchoring on its per-bin routes (O3c) helps the
routes (15 off) but ld's front settings then read m2 long (13 long). On m1 the end routes lie tens of px from where
the tubes were (grains knocked and carried). O2 had found 0.8.8's drawn route right for 22 of m2's 27 misses at the
bin of the miss; O3 shows the end route alone is not that route.

Soft recency (`carry.dp_recency`): advancing at bin t through a route point whose map mark arrived more than delta bins
before costs mu x (bins left) x min(1, (t - arrival - delta) / delta), waived when fresh material (arrived from t -
delta to t + wait) lies within 4 points beyond the end of that old run (a tube growing across an older one). Toy
check: alone it is dodged (the front races ahead through unmarked route to pass the old material as it arrives, and
passes it before it is old); with a void cost (void x bins left for advancing through an unmarked run > 3 points at
that bin, also at the first bin) it follows both a g052-like jump and a tube crossing an older one with a map hole.
Grid mu 0.1/0.25/0.5, delta 10/20, wait 10/20, void 0/0.1/0.3; front settings fixed.

| soft recency (lengths, len&tip) | ld | m2 | m1 |
|---|---|---|---|
| O1 b, without | 82, 81 | 32, 29 | 14, 14 |
| O1 b, ld-tuned (mu 0.1, no void: nearly inert, ld has almost no such jumps) | 82, 81 | 32, 28 | 15, 13 |
| O1 b, m2-tuned (mu 0.1, void 0.1) | 81, 80 | 33, 30 (+6 vs 0.8.8, -4..+16) | 13, 13 |
| O1 c, without / ld-tuned / m2-tuned | 80 / 81 / 80 | 33 / 33 / 35 | 24 / 23 / 24 |
| O3b, without / ld-tuned / m2-tuned | 69 / 69 / 67 | 15 / 16 / 17 | 0 / 0 / 0 |
| O3c, without / ld-tuned / m2-tuned (mu 0.5, void 0.3) | 66 / 66 / 65 | 15 / 15 / 20 | 14 / 13 / 12 |

On the grains it was meant for (O1 b, m2-tuned / stronger mu 0.5, void 0.3): g052 b140 52 -> 38 / 33 px (human 20),
g011 b349 44 -> 34 (35, now a hit); but g038 b140 35 -> 16 (26) and, stronger, g005 b349 186 -> 118 (257: a long tube
crossing older ones) and g016 b244 117 -> 108 (124). It trades fixes for breaks: +0 to +2 on m2 tuned on m2 itself,
nothing when tuned on ld. Not a lever.

Files: `runs/research/carry_front/o1/<tag>/summary.json` (per-trace rows, params, paired CIs) and
`<movie>_<variant>.json` predictions; `o2.json`; `mapcheck_*.json`; logs in `logs/`.
