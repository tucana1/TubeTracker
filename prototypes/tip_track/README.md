# The learned tip detector in the tracker: onsets and young tubes (research record, 3-4 Oct 2026)

**Question.** The tip detector (`prototypes/tip_detector`: a U-Net trained on human traces; held out, its top peak is
within 4 px of the apex on 77 / 48 / 61% of traces on ld / m2 / m1, against 34 / 20 / 30% for the raw short-interval
difference) is strongest where SparseTrack is weakest: germination onsets and young tubes. Does it help there, end to
end, leave one movie out?

**Answer: a little on every movie, nothing significant.** End to end (SparseTrack 0.8.8 + `tipdet_onset="later"`,
`tipdet_young=True`, each movie with the detector trained on the other two): lengths +3 / +3 / +2 (ld / m2 / m1),
length + tip +5 / +3 / +2, onsets +2 / 0 / +2; only m1's onsets (95% CI 0 to +5) and ld's length + tip (0 to +11)
have intervals that do not go below zero, and both touch it. The detector's onset alone beats 0.8.8's on two movies (ld, m1) and loses on movie 2, where its rim response is
weak and confounded; used only where the reading's onset is more than 10 bins later it is level on movie 2. Most of
the length gain comes from young tubes' lengths read from the detector's tip. Implemented behind `Params.tipdet_*`
(off by default, `sparsetrack/tipdet.py`, tests `tests/test_tipdet.py`).

## Method

Maps (`maps.py`, `sparsetrack/tipdet.py: GrainTipMaps`): the detector's tip map round every labelled grain (128 px,
the network run on 224 px so the kept part has its full context), every bin, where the 0.8.8 reading put the grain
then (census + drift), with that movie's held-out fold model (ld: trained on m2+m1; m2: ld+m1; m1: ld+m2). These crop
maps equal the prototype's within 0.002. Census debris (excluded by the annotator as not a grain) too. ~7 s per grain
on ld (176 bins), ~16 s on m2/m1 (351 bins) on the shared machine.

Rim response (`rim.py`): per bin, the strongest local maximum within r - 2 .. r + 25 px of the grain.

Onsets (`onsets.py`, `fixed.py`): first bin from which the rim response stays >= thr for W bins, alone or combined with
the reading's onset. Hit = within 600 source frames (2 bins) of the human bracket, over every human 'emerged within'
isolated grain (a grain called never counts as a miss; ld 28, m2 19, m1 28). Settings chosen on two movies and
applied to the third; paired bootstrap over grains (95%).

Young tubes (`young.py`, `replay.py`): where the reading is shorter than 20 px (or reads nothing after a moved
onset), the strongest peak (>= 0.2) within r - 2 .. r + 25 px is the tip; length = its distance from the grain's
visible edge in that direction (`analyze.exit_edge`, `edges.py`) + k (2 px). The route-based version asked for first
(peak nearest the reading's tip, length = arc length along `routes.route_at` to its projection) was worse (ld 70 vs
75/104 with each of the three settings tried): the detector's peak is right (median 1.5 px from the apex on ld young traces, 1.9 on m2,
1.4 on m1) but young lengths are decided by where the trace starts, and route starts are 2-5 px off the annotator's
exit on many young tubes (some routes are wrong altogether, e.g. ld g039: apex 21 px off the route). The peak also
sits ~1.5-2 px behind the apex along the tube. Lengths are made non-decreasing again (pool adjacent violators from the
onset on); every variant is scored with `sparsetrack.evaluate.score`.

## Results offline (on 0.8.8's stored readings)

Onsets, hits / human 'emerged within' grains (0.8.8: ld 15/28, m2 7/19, m1 8/28); change and 95% CI:

| rule | ld | m2 | m1 |
|---|---|---|---|
| detector alone, level, chosen on the other two | 14 (-1, -8..+7) | 0 (-7, -11..-3) | 13 (+5, -2..+12) |
| detector alone, relative to the grain's own plateau (0.5 x its 98th pct), chosen on the other two | 18 (+3, -4..+10) | 5 (-2, -7..+3) | 15 (+7, 0..+13) |
| detector where the reading's is later by > N bins or missing, chosen on the other two (first grid) | 12 (-3, -9..+3) | 7 (0) | 10 (+2, 0..+5) |
| the same, second grid (no hysteresis/angle options) | 16 (+1, -2..+4) | 7 (0) | 10 (+2, 0..+5) |
| detector only where the reading has no tube | 16 (+1, 0..+3) | 7 (0) | 8 (0) |
| **implemented `later`**: thr 0.3, 3 bins, reading later by > 10 bins (fixed) | 17 (+2, -2..+6) | 7 (0, -3..+3) | 10 (+2, 0..+5) |
| `later_or_missing` (also where the reading has no tube) | 18 (+3, -1..+7) | 7 (0, -3..+3) | 10 (+2, 0..+5) |

- The detector's onsets err early (a rim response before the annotator sees a tube), the readers' late. On movie 2
  the rim response is weak or confounded: median 0.22 in the six bins after the onset (ld 0.46, m1 0.40), while
  neighbours' tubes and debris reach 0.3-0.6 at the rim; a threshold chosen on the other movies does not transfer
  (a per-movie scale normalisation and an exit-sector restriction did not fix it).
- m1's floods that never start properly (g005 never, g065 252 bins late, g033 297 late, g047 never): `later` fixes
  g033 (on time) and moves g065 to 6 bins early (still a miss); g005 and g047 (no tube read) stay as they were;
  `later_or_missing` gives them onsets 12 and 11 bins early (misses, but counted germinated, as the annotator has
  them).
- False early onsets added by `later`: ld g034, m2 g069, m1 g030 and g065 (2 of them were late misses before).
  Grains that never germinated (m2 g100, m1 g029) and debris are left alone by `later` (no tube read); with
  `later_or_missing` both never-germinated grains are called germinated, and the detector's onset fires on 4/4
  movie-2 and 1/4 movie-1 debris objects (0/1 on ld): in a whole-field run it would turn debris into germinated
  grains. Hence `later` is the implemented default of the option.

Young lengths (FULL traces: all / young = human < 15 px), length hits and length + tip vs 0.8.8 (`replay.py` runs the
tracker's own `tipdet.apply` on the stored readings and the stored uint8 maps; it reproduces the end-to-end run up to
the maps' storage precision: ld identical, m2 one length hit fewer):

| setting | ld (104 / 30) | m2 (54 / 25) | m1 (50 / 28) |
|---|---|---|---|
| chosen on the other two, no onset change | -2 (-8..+3); l+t +1 | +2 (-4..+9); l+t +2 | +1 (-4..+6); l+t +1 |
| chosen on the other two, with the detector's onsets | -3 (-8..+2); l+t 0 | +1 (-7..+8); l+t +2 | +5 (-1..+11); l+t +5 |
| fixed: young only (no onset change) | +3 (-2..+8); l+t +5 (0..+11) | +3 (-3..+9); l+t +3 | +1 (-4..+6); l+t +1 |
| fixed: `later` only (no young lengths) | +1 (0..+3); l+t 0 | 0; l+t 0 | +1 (0..+3); l+t 0 |
| **fixed: `later` + young (implemented setting)** | +3 (-2..+8); l+t +5 (0..+11) | +2 (-5..+8); l+t +2 | +2 (-3..+7); l+t +2 |
| fixed: `later_or_missing` + young | +3; l+t +5 | +2; l+t +2 | +4 (-2..+10); l+t +4 |

The per-fold choices disagree (k 1.5-2.5, young 15-25 px, radial or radial from the edge), so the leave-one-out
rows are the honest estimate of a setting chosen this way. The fixed setting (k 2, 20 px, what the m1 fold chose) was
set after looking at all three movies; only the detector networks are held out in it. Sensitivity of young-only
(change in length hits, ld / m2 / m1): k 1.5: +2/+1/+2; k 2: +3/+3/+1; k 2.5: 0/+5/+3; k 3: -2/+5/+2 (young < 20 px);
summed over the movies it is positive for every k in 1-3 and young limit 15-25 px (+1 to +8 lengths, +4 to +10 l+t),
but ld gains only for k 1.5-2 (down to -2/-3 at k <= 1 or >= 2.5), and movie 2 would like k 2.5.

## End to end

`scripts/synth_bench.py --real <movie> --no-synth --no-legacy --set tipdet_model=<held-out fold> tipdet_onset=later
tipdet_young=true --baseline <0.8.8 dump>` (0.8.8: ld 15/26 onsets, 75/104 lengths, 67 l+t; m2 7/18, 27/54, 23; m1
8/26, 15/50, 15). Readings the detector did not touch are identical to 0.8.8's.

| movie | onsets (scorer, timed) | onsets, all emerged grains | lengths | length + tip | young lengths (< 15 px) |
|---|---|---|---|---|---|
| ld | 17/26 | 17/28 (+2, -2..+6) | 78/104 (+3, -2..+8) | 72 (+5, 0..+11) | 21/30 (+2) |
| m2 | 7/18 | 7/19 (0, -3..+3) | 30/54 (+3, -4..+9) | 26 (+3, -2..+8) | 14/25 (+3) |
| m1 | 10/26 | 10/28 (+2, 0..+5) | 17/50 (+2, -3..+7) | 17 (+2, -3..+7) | 12/28 (+2) |

Germination curve (Turnbull, `evaluate.germination_curves`), T50 in frames (human / 0.8.8 / new) and the largest gap
between the human and model curves (0.8.8 -> new):

- ld: T50 7383 / 7559 / 7120 (0.6 bins late -> 0.9 early); gap 0.14 -> 0.18
- m2: T50 24049 / 23874 / 23523; gap 0.09 -> 0.14 (worse: 4 onsets moved earlier, g048 now right, g069 now wrong)
- m1: T50 8602 / 9304 / 7899 (as far off as before, now early); gap 0.23 -> 0.10

Length bias and median |error| (px): ld -0.48 -> -0.22, 2.31 -> 2.20; m2 -5.88 -> -5.51, 2.91 -> 2.41; m1 -10.19 -> -9.21, 4.13 -> 3.28.

## Recommendation

Keep it off by default. The gains are consistent in sign but small (+2 to +3 length hits per movie on 50-104 traces)
and none is significant; the onset rule is level on the crowded movie, where the detector's rim response does not
separate the grain's own young tube from neighbours' tubes and debris, and it moves T50 earlier on every movie. If the
detector is used, `tipdet_young` (young lengths from the tip, measured from the grain's visible edge) is the part that
pays on all three movies; it needs one detector network for new movies (a model trained on all three labelled movies;
the fold models here never saw the movie they read). What would make the onsets work is a rim response that knows
which tube belongs to the grain (e.g. the detector run on the grain's own route region or trained with neighbours'
tips as negatives), not a different threshold.

## Files

`common.py` (movies, fold models, 0.8.8 baseline), `maps.py`, `rim.py`, `edges.py`, `onsets.py`, `fixed.py`,
`young.py`, `replay.py` (the tracker's `tipdet.apply` on stored readings and maps), `e2e.py` (end-to-end scoring).
Maps and intermediate JSON were written to the session scratchpad (`TT_TIPTRACK_OUT`) and deleted afterwards (maps are
rebuilt by `maps.py` in ~25 min for the three movies, ~430 MB as uint8). The 0.8.8 baseline readings
(`TT_BASE088`/<movie>_real_0/predictions.json) are a `synth_bench.py --real --dump-real` run of 0.8.8 defaults.
Order: `maps`, `rim`, `edges`, then `onsets` / `fixed` / `young` / `replay`.
