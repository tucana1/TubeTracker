# review_fill: a person's trace fills in the grain's growth curve from the image (3 Oct 2026)

Question: when a person traces (or confirms) a tube at one or two times per grain, can the grain's whole growth
curve be read from the image along their trace, instead of rescaling the model's curve between their lengths (the
current review: `sparsetrack.review.reviewed_curve`, used by the app)?

**Verdict: negative on the honest movie; not built into the review.** With settings tuned on ld and applied
unchanged, the image fill (M2) beats the app's rescaled curve (M1) on movie 2 with one late trace (+8 lengths,
95% CI +0 to +17; length-and-tip +15, +8 to +22) but not on movie 1, the movie the tube network never saw (-3, -7 to
+1). Two findings worth acting on instead:

- **The current rescaling hurts.** With one late trace (P1), 0.8.8's own readings (M0) beat the rescaled curve (M1)
  on all three movies: lengths +5 / +6 / +4 (ld / m2 / m1), length-and-tip +10 (+2..+18) / +11 (+5..+17) / +7
  (0..+15). Rescaling the whole curve to one late length inflates or shrinks every earlier length when the model's
  curve has the wrong shape (m2 g005: 0.8.8 stalled at 96 px, the person traced 257 px at bin 349, so bin 244 became
  204 px against the person's 76; 0.8.8 itself read 76). Acted on: prototypes/review_curve, now the default reviewed
  curve (M1 below is the old one, frozen as `prototypes.review_curve.curves.rescaled_curve`).
- **The image fill's tips are much better where it reads the right tube** (length-and-tip M2 - M1: ld P1 +12, P2
  +13, m2 P1 +15): its tip lies on the carried tube, not on a fixed route moved by the grain's drift.

The implementation (review mode + app switch + tests) is kept, unmerged, on the local branch `review-fill-wiring`.

## Results

Lengths within max(2 px, 10%) / length-and-tip (median |error| px); targets = the grain's other full traces not in
contact. Paired differences summed over grains, 95% bootstrap CI. Settings tuned on ld (`final`): M2 band +/- 2 px,
P >= 0.5, first 3 px neutral, tip offset 2 px, speed cap 2 x the traces' mean speed (>= 1 px/bin), onset read from
the image; M3 = M2 within 20 bins of a trace, else 0.8.8 (ld preferred 0.8.8 almost everywhere).

| movie | protocol | targets (grains) | M0 0.8.8 | M1 app rescaled | M2 image fill | M3 | ML lines | M2 - M1 | M2 - M1 len & tip | M2 - M0 |
|---|---|---|---|---|---|---|---|---|---|---|
| ld | P1 | 76 (27) | 53 / 48 (1.8) | 48 / 38 (1.8) | 51 / 50 (1.8) | 53 / 48 | 30 / 22 | +3 [-6, +12] | +12 [+2, +22] | -2 [-9, +5] |
| ld | P2 | 74 (26) | 59 / 54 (2.3) | 52 / 35 (2.1) | 50 / 48 (2.5) | 59 / 54 | 25 / 17 | -2 [-11, +7] | +13 [+3, +24] | -9 [-19, +0] |
| ld | P3 | 48 (25) | 37 / 35 (1.6) | 32 / 30 (1.8) | 34 / 29 (1.9) | 37 / 35 | 23 / 15 | +2 [-6, +9] | -1 [-9, +7] | -3 [-9, +2] |
| m2 | P1 | 41 (18) | 21 / 17 (2.4) | 15 / 6 (3.2) | 23 / 21 (2.2) | 21 / 17 | 8 / 3 | +8 [+0, +17] | +15 [+8, +22] | +2 [-7, +10] |
| m2 | P2 | 30 (14) | 16 / 13 (3.7) | 16 / 11 (2.6) | 16 / 16 (3.1) | 16 / 13 | 1 / 1 | +0 [-6, +6] | +5 [-2, +12] | +0 [-7, +8] |
| m2 | P3 | 21 (14) | 12 / 9 (3.4) | 9 / 8 (3.0) | 13 / 13 (2.9) | 12 / 9 | 3 / 2 | +4 [-3, +11] | +5 [-2, +12] | +1 [-6, +8] |
| **m1** | **P1** | 26 (18) | 9 / 9 (3.0) | 5 / 2 (4.8) | **2 / 2 (5.2)** | 9 / 9 | 4 / 2 | **-3 [-7, +1]** | +0 [-3, +3] | -7 [-15, +0] |
| m1 | P2 | 14 (8) | 4 / 4 (4.2) | 6 / 6 (4.1) | 6 / 5 (3.4) | 4 / 4 | 5 / 5 | +0 [-5, +5] | -1 [-5, +3] | +2 [-3, +7] |
| m1 | P3 | 10 (8) | 4 / 4 (3.1) | 6 / 6 (1.8) | 5 / 5 (2.1) | 4 / 4 | 5 / 5 | -1 [-5, +3] | -1 [-5, +3] | +1 [-3, +5] |

With the person also giving every onset (`final_human`, same settings): M2 - M1 m2 P1 +10 (+2..+18), P3 +5 (-1..+12);
m1 P1 -2 (-7..+4), P3 +3 (-1..+7). The rescaled curve loses from the person's onset (ld P1 48 -> 41: 0.8.8's curve
shape belongs with 0.8.8's onset).

Why M2 misses (`why.py`; route = its route at that bin > 3 px from the human trace over the human length; blind = the
route is right but the map marks < half of the traced tube):

| | targets (<= 8 px) | hits | route | blind | long | short |
|---|---|---|---|---|---|---|
| ld P1 | 76 (20) | 51 | 4 | 5 | 5 | 11 |
| m2 P1 | 41 (20) | 23 | 7 | 2 | 2 | 7 |
| m1 P1 | 26 (18) | 2 | 18 | 5 | 0 | 1 |

Movie 1's P1 targets are mostly young tubes (18 of 26 are <= 8 px, traced 6 bins after onset) read from a trace
100-300 bins later: the carried route has drifted off them (the carry over long gaps, as carry_front found: 35% of
its routes within 3 px on m1), and where it has not, the network (which never saw m1) often does not mark young tubes.

Examples:
- Carry lost (moving grain, long gap): m1 g061 b140, route 25 px off the person's trace, M2 reads 0 against 33 px
  (0.8.8 32); m2 g069 b140, route 21 px off, 6 px against 57.5.
- Map blind spot on young tubes (m1): g030 b48 and g011 b25, route within 0.8 px but the map marks none of the 4.5 /
  5.4 px tube; M2 reads 0.
- Material ahead of the tip (a crossing or older tube on the route's future part): m1 g014 b140, 55% of the 20 px
  past the apex marked, M2 44 px against 33.5; m1 g009 b43, 12 px against 3.8.
- Where M2 wins: m2 g005 b244 (M1 204, M2 75, person 76), m2 g054 b112 (0.8.8 and M1 0, M2 8 against 6),
  m2 g066 b95/b140 (M2 5 and 15 against 4.1 and 15.1; 0.8.8 8.5 and 20.4).

Also tried on ld: the onset imposed or clipped at the review's onset (no better than read from the image); M2 pulled
towards M1 by a cost per px (best lambda 0.5: 140 ld hits over the three protocols, against 135 for M2 and 149 for
the chosen M3; m1 14 against M1's 17); P2's route carried on straight past the early trace instead of along 0.8.8's
per-bin route (P2 ld 40 vs 50, m2 12 vs 16, m1 5 vs 6). Post hoc, not chosen by the ld tuning (ld prefers 0.8.8
everywhere): M2 where the map marks >= 80% of what it read, else 0.8.8, scored m2 P1 24 (vs M1 +9, +1..+18) and m1 P1
9 (+4, 0..+8), mostly by not rescaling (vs 0.8.8: +3 and 0). A hypothesis for the next labelled movie, not a result.

Timing of the review mode (`timing.py`, branch `review-fill-wiring`, movie 2, single thread, machine load ~19): the
first trace of a grain 2-11 s (carrying it through 351 bins; longest for a 257 px tube), each later reading of that
grain 0.3-0.5 s (the carry is kept); the app runs it in a background thread and shows the rescaled curve meanwhile.

## Method

- Simulated reviewers, per scored grain (isolated, not excluded) with >= 2 FULL human traces: **P1** traces once at
  its latest full trace; **P2** once at its earliest full trace with a tube >= 8 px (grains where that is not the
  latest); **P3** both (same grains as P2). Contact traces may be traced, never scored.
- Onset (the same review state for every method): 0.8.8's, unless it comes after the person's first trace or 0.8.8
  read no germination; then the person gives theirs (the app makes them set it).
- **M0** 0.8.8 alone (length; tip + drift). **M1** the app's curve then (`rescaled_curve` through the person's lengths;
  tip on the app's route, `overlay.route_at`). **M2** (`fill.py`): each trace becomes a route (the trace, on along
  0.8.8's route at that bin if its apex lies on it, then straight 40 px), carried through the movie with composed DIS
  flow as a tube (`carry_front`'s `carry_inext`, flows on a 10-bin grid so a grain's traces share them); one route
  per bin (before the first trace its route; between two traces the later one's beyond the earlier length, joined
  onto the earlier trace's carried apex; after the last trace its carried apex on along 0.8.8's route at each bin);
  K = the tube map maxed over +/- w px across the route; length = the best monotone front (DP) through every traced
  length with per-stretch speed caps. **M3** M2 near a trace, a fallback farther away (distance rule; also tried:
  a cost towards M1, and the map-support rule above). **ML** (reference) straight lines from the onset through the
  person's lengths.
- Settings tuned on ld only (staged grid over all protocols together, `evaluate.py`), applied unchanged to m2 and m1.

Files: `build.py` (carried routes and kymographs per grain, `runs/research/review_fill/<movie>/`, 8 MB in all;
that folder is in this worktree, git-ignored),
`evaluate.py` (protocols x methods, tuning, `runs/research/review_fill/eval/<tag>/summary.json`), `table.py`,
`show.py` (per-target rows), `why.py` (miss causes), `fastdp.py` (the DP compiled with numba for the grids; same
fronts, `fastdp.check`).

    python -m prototypes.review_fill.build ld        # then m2, m1
    python -m prototypes.review_fill.evaluate --tune ld --apply ld m2 m1 --tag final
    python -m prototypes.review_fill.evaluate --apply ld m2 m1 --onset human --tag final_human --settings-from final
    python -m prototypes.review_fill.table final; python -m prototypes.review_fill.show final m1 P1 --miss M2
