# review_curve: a reviewed growth curve that does not distort far-away bins (3 Oct 2026)

Question (from prototypes/review_fill): after one late trace, the app's reviewed curve (the model's whole curve
rescaled to the person's lengths) had fewer earlier lengths in tolerance than 0.8.8's own readings on all three
labelled movies (P1: 0.8.8 +5 / +6 / +4 on ld / m2 / m1). Can a curve use the person's answers without that?

**Result: yes; now the default `sparsetrack.review.reviewed_curve` (the app and the reviewed exports use it).**
The model's own readings, corrected only near the person's lengths: at each checked length the model's error there
ramps in linearly over the 20 bins before it and fades out over the 20 bins after; where the model read shorter
than the person by more than the scoring tolerance (max(2 px, 10%)) and had stopped growing (grown <= 2 px since bin
s: it lost the tip, or never saw the tube), the missed growth is spread from s instead; between two checked lengths
the curve stays between them; it passes through them, never shrinks and is zero before the onset. Its lengths at
the traced bins equal 0.8.8's or better on every movie and protocol, and beat the old curve on ld and under P1 on
all three movies. Exception: movie 1 P2/P3, one target (g014 b87: old 8.7, new 8.9, person 6.8, tolerance 2) -1.

## Results

Simulated reviews of prototypes/review_fill (`evaluate.py`): per grain with >= 2 full traces, P1 = the person traces
its latest full trace, P2 its earliest >= 8 px, P3 both; scored at its other full traces not in contact. Cells:
lengths in tolerance / length-and-tip (median |error| px); differences paired over grains, 95% bootstrap CI. Review
onset = 0.8.8's unless it comes after the person's first trace or 0.8.8 read none (as in the app).

| movie | protocol | targets | M0 0.8.8 | M1 rescaled (old app) | new | new - M1 lengths | new - M1 len & tip | new - M0 lengths |
|---|---|---|---|---|---|---|---|---|
| ld | P1 | 76 | 53 / 48 (1.81) | 48 / 38 (1.84) | 53 / 39 (1.81) | +5 [+0, +11] | +1 [-4, +6] | +0 [+0, +0] |
| ld | P2 | 74 | 59 / 54 (2.31) | 52 / 35 (2.13) | 59 / 35 (2.31) | +7 [+2, +13] | +0 [-4, +4] | +0 [+0, +0] |
| ld | P3 | 48 | 37 / 35 (1.64) | 32 / 30 (1.77) | 37 / 30 (1.64) | +5 [+0, +10] | +0 [-4, +4] | +0 [+0, +0] |
| m2 | P1 | 41 | 21 / 17 (2.42) | 15 / 6 (3.17) | 21 / 9 (2.42) | +6 [-2, +15] | +3 [-3, +11] | +0 [+0, +0] |
| m2 | P2 | 30 | 16 / 13 (3.67) | 16 / 11 (2.62) | 16 / 10 (3.67) | +0 [-4, +4] | -1 [-3, +0] | +0 [+0, +0] |
| m2 | P3 | 21 | 12 / 9 (3.35) | 9 / 8 (2.98) | 12 / 7 (3.35) | +3 [-2, +8] | -1 [-4, +2] | +0 [+0, +0] |
| m1 | P1 | 26 | 9 / 9 (3.0) | 5 / 2 (4.79) | 10 / 2 (2.79) | +5 [+0, +11] | +0 [-3, +3] | +1 [+0, +3] |
| m1 | P2 | 14 | 4 / 4 (4.24) | 6 / 6 (4.08) | 5 / 5 (4.22) | -1 [-3, +0] | -1 [-3, +0] | +1 [+0, +3] |
| m1 | P3 | 10 | 4 / 4 (3.08) | 6 / 6 (1.81) | 5 / 5 (1.89) | -1 [-3, +0] | -1 [-3, +0] | +1 [+0, +3] |

With the person giving every onset (`pick_human`): new - M1 lengths ld +12 [+4, +20] / +11 / +9, m2 +7 [-1, +15] /
+1 / +4, m1 +5 [+2, +9] / +3 [+1, +6] / +3 [+1, +6]; new - M0 again 0 or +1 everywhere. (The old curve loses more
with the person's onset: its first stretch subtracts what the model had read by the person's onset.)

How it was chosen (all on ld; m2 and m1 only checked; `runs/research/review_curve/<tag>/summary.json`):
- Round 1 (`v1_review`, 41 candidates): carrying a correction on past its checked length (P2, P3) and windows of 40
  bins or more cost on ld (carry: P2 53 vs 59); stretching the model's growth inside a window was worse than
  adding a correction; moving the model's curve in time to the review's onset cost (ld 133-141 vs 147-149).
- Round 2 (`r2_review`, `r2_human`): the pure band ("clip": the model's readings, only clipped between the checked
  lengths) and the ramp-and-fade corrections with windows of 10 or 20 bins tie on ld (149 lengths over P1-P3, as
  0.8.8); a stall rule on any shortfall costs one ld trace (g032: a 2.8 px shortfall within tolerance moved b122 out
  of it), so it acts only beyond tolerance ("tol") or where the model read nothing ("zero"), both tied on ld.
  Chosen among the ld ties on design grounds, not on m2 / m1: 20 bins (continuous; 40 lost 2 on ld in review mode),
  fade out rather than drop (no jump back to an over-read after a checked length), the "tol" stall rule (it also
  handles a model that lost the tip: m2 g005, 0.8.8 at 96 px from bin 263, the person 257 px at bin 349; the old
  curve made bin 244 204 px against the person's 76 and 0.8.8's 76; the new curve keeps 75.8 there).
- Round 3 (`r3_*`): the onset handled when the review's differs from the model's: zero before it (chosen), the
  model's curve moved down to start at zero there (ld 124 vs 149 with the person's onsets: the model's lengths after
  a slightly different onset are right as they are), or moved in time (133 vs 149).

What M0 and the new curve still miss is mostly 0.8.8's own errors far from any checked length (young tubes, onsets
far off), which no curve built from the model's readings can fix; image evidence did not either
(prototypes/review_fill).

The new curve costs 0.14 ms per grain (351 bins, two checked lengths).

## Where the tube is drawn after review (`routes.py`, 3 Oct 2026, now `tubetracker.app.overlay.route_at`)

With the same lengths, the app's tip (then on the person's traced route at every bin) was in tolerance less often
than a tip on 0.8.8's own route at that bin cut to the same length (`tips.py`: length-and-tip ld P1 39 vs 45, P2 35
vs 49; m2 P1 9 vs 18; m1 P1 2 vs 9): the person's later route does not follow the tube's sway at earlier bins.

**Now:** within 20 bins of a bin a person traced (`NEAR_BINS`), their route, as before; farther, the model's own
route at that bin (bent / the flood's own route then, turned), carried on along the person's route where the tube
is longer than the model's route reaches (its end within 10 px of theirs, `JOIN_PX`; up to 5 px short it is carried
on straight, as `to_length` always did); the person's route where the model had read no tube by then (its route is
that of a tube it saw later) or the join fails. A tip click reads along the route drawn at that bin
(`corrections.tip` uses the same `route_at`), and Confirm saves the route drawn there where it draws a length itself.

Length-and-tip on the simulated reviews, the app's own code drawing the tube (`model.RunData` on 0.8.8's
predictions, the reviewed curve, `overlay.tube_at`); lengths are the same for both, paired over grains, 95% CI:

| movie | protocol | targets | lengths | person's route (old) | new | new - old | median tip error old / new (px) |
|---|---|---|---|---|---|---|---|
| ld | P1 | 76 | 53 | 39 | 45 | +6 [-1, +14] | 3.7 / 2.5 |
| ld | P2 | 74 | 59 | 35 | 48 | +13 [+5, +23] | 4.6 / 2.7 |
| ld | P3 | 48 | 37 | 30 | 31 | +1 [-4, +6] | 3.3 / 2.4 |
| m2 | P1 | 41 | 21 | 9 | 18 | +9 [+3, +15] | 7.4 / 2.8 |
| m2 | P2 | 30 | 16 | 10 | 14 | +4 [+1, +7] | 8.6 / 3.1 |
| m2 | P3 | 21 | 12 | 7 | 10 | +3 [+0, +6] | 4.8 / 2.9 |
| m1 | P1 | 26 | 10 | 2 | 10 | +8 [+3, +14] | 17.7 / 1.8 |
| m1 | P2 | 14 | 5 | 5 | 5 | +0 | 3.5 / 2.9 |
| m1 | P3 | 10 | 5 | 5 | 5 | +0 | 2.2 / 1.6 |

Over the three protocols: ld 104 -> 124, m2 26 -> 42, m1 12 -> 20 (0.8.8 alone: 137, 39, 17; its own tips, e.g.
from tip continuation, are still better on ld).

Variants (`--grid`; ld totals over P1-P3, then m2, m1): the window round a person's traces makes no difference at
0-20 bins (no scored trace lies that close), 40 bins costs ld 4. A check that the model's route leaves the grain where
the person's does (mean distance over the first 10 px within 3-12 px, at the bin drawn or at the person's) gains at
most ld 2 (126 vs 124) and costs m2 3-15 and m1 6-7 (late in the movie, where a person traces, the model's route is
often off while its earlier routes are right); not kept. Using the model's route only where it had read a tube
changes ld nothing and keeps m1 g033 b32 (its route then was that of a tube it saw 290 bins later): m1 19 -> 20.
Costs 0.16 ms per call (0.12 before).

Files: `curves.py` (candidates; `rescaled_curve` = the old app curve, frozen, M1 here and in review_fill),
`evaluate.py` (protocols x candidates, ranking on the tuning movie), `show.py` (per-target rows), `table.py`,
`tips.py` (the first tip comparison), `routes.py` (the route policies, drawn by the app's own code).

    python -m prototypes.review_curve.evaluate --tune ld --apply ld m2 m1 --round r2 --tag r2_review
    python -m prototypes.review_curve.evaluate --tune ld --apply ld m2 m1 --round r2 --pick decay-stalltol-20 --tag pick_review
    python -m prototypes.review_curve.table pick_review
    python -m prototypes.review_curve.show m2 P1 clip decay-stalltol-20 --diff
    python -m prototypes.review_curve.tips      # the app's tips against 0.8.8's route at the same lengths
    python -m prototypes.review_curve.routes --grid  # route policies: the old one against overlay.route_at
