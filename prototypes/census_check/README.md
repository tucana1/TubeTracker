# Census check: census "grains" that are not grains (prototype, 30 Sep 2026)

**Status: measured; kept as an option, off by default** (`Params.census_check`, `sparsetrack/census_check.py`). See
"Decision" at the end.

The census (`sparsetrack/grains.py`: Hough circles with a dark rim or body on the reference image) also finds things
an annotator does not count. Across the three labelled movies the annotator judged 110 census grains and excluded 25
(`benchmark/labels/*_v1.json`: clump 9, not a grain 9, out of focus 3, other 4; "not_sampled" grains were never
judged). The product's population statistics (germinated share, T50) use the census-isolated grains away from the
edge, which is also where movie 2's and movie 1's random samples were drawn from: 103 judged grains there, 21
excluded (ld 32/4, m2 30/6, m1 41/11). The dev movie's four other clump exclusions (g018-g020, g023) are census
clumps, already outside the population.

## Features (`sparsetrack.census_check.features`; no labels)

On the census's own reference image (flat-fielded as the census was), contrasts divided by the movie's median ring
contrast over its whole census (so they mean the same at other exposures): ring and body contrast; `disc_dark` (the
centre's darkness: debris and dead matter are dark all through, grains have a dark rim), `fill` (share of the disc
darker than the background), rim coverage and its variation round the circle, `outer_dark` (dark pixels 3-10 px
outside the rim: attached bits, a touching neighbour), `sharp` (steepest radial step at the rim: focus), the dark mass
the circle sits on (area, extent, offset from the centre), nearest census neighbour / r; and from the first bins after
the reference (bins 12-17 after it): `stay` (correlation of the grain's crop then with the reference: a passing or
smeared thing is gone) and `move` (its phase-correlation shift). `python -m prototypes.census_check.data` writes them
for every census grain of the three movies (runs/census_check/features.json).

Single features, AUROC for "excluded" among the judged grains (ld / m2 / m1): `disc_dark` 0.83 / 0.84 / 0.71,
`body_rel` 0.70 / 0.74 / 0.68, `mass_area` 0.86 / 0.71 / 0.62, `outer_dark` 0.88 / 0.58 / 0.57; the rest are weak or
reverse between movies. Visually (crops of every judged grain): "not a grain" is mostly dark filled blobs (m1 g004,
g032, g048; m2 g019, g096, g117) or faint smears (ld g016, m1 g050); "clump" a grain with dark bits attached;
"out of focus" and "other" look like ordinary grains at the census time (ld g017, a smear, is the exception).

## Leave one movie out (`loo.py`)

Fitted on two movies' judgements, tested on the third. Six variants, all fixed before any held-out result was seen
and all reported: logistic regression (L2, C = 1, balanced class weights) on ALL 14 features, on SMALL (`disc_dark`,
`stay`, `mass_area`, `outer_dark`, `ring_rel`: one feature for each kind of false grain seen) or on TWO (`disc_dark`,
`stay`), each with the features standardised over the training grains ("global") or per movie over its whole census
(median / IQR, no labels: "movie"). The flagging threshold maximises F1 on the two training movies.

| variant | ld (32 grains, 4 excluded): AUROC, flags (right), precision / recall | m2 (30, 6) | m1 (41, 11) | pooled precision / recall |
|---|---|---|---|---|
| ALL, global | 0.94, 8 (3), 0.38 / 0.75 | 0.82, 8 (4), 0.50 / 0.67 | 0.74, 6 (4), 0.67 / 0.36 | 0.50 / 0.52 |
| ALL, movie | 0.88, 5 (2), 0.40 / 0.50 | 0.76, 3 (2), 0.67 / 0.33 | 0.76, 6 (4), 0.67 / 0.36 | 0.57 / 0.38 |
| **SMALL, global** | 0.80, 3 (3), **1.00 / 0.75** | 0.87, 4 (4), **1.00 / 0.67** | 0.70, 5 (4), **0.80 / 0.36** | **0.92 / 0.52** |
| SMALL, movie | 0.85, 7 (3), 0.43 / 0.75 | 0.81, 3 (2), 0.67 / 0.33 | 0.68, 7 (6), 0.86 / 0.55 | 0.65 / 0.52 |
| TWO, global | 0.88, 4 (3), 0.75 / 0.75 | 0.84, 6 (4), 0.67 / 0.67 | 0.73, 5 (4), 0.80 / 0.36 | 0.73 / 0.52 |
| TWO, movie | 0.81, 2 (1), 0.50 / 0.25 | 0.82, 4 (3), 0.75 / 0.50 | 0.74, 6 (5), 0.83 / 0.45 | 0.75 / 0.43 |

SMALL / global (the one kept; chosen among the six on these results, so read its numbers as slightly optimistic)
flags 12 grains over the three held-out movies, 11 of them excluded by the annotator: not a grain 8/9 found, clump
2/5, out of focus 1/3, other 0/4. Its one wrong flag is m1 g029, a dark grain that never germinated (the annotator
kept it). What it finds is dark debris and smears; clumps whose neighbours are faint, and grains excluded for being
out of focus or "other", look like grains at the census time and are not found. `fit.py` fits it on all 103 grains
(`sparsetrack.census_check.MODEL`; in sample 13 flags, 12 right); on the full censuses it flags 3/32 of ld's
isolated grains away from the edge (all judged), 17/82 of m2's (4/30 judged, all right; 13/52 unjudged) and 8/46 of
m1's.

## What leaving flagged grains out does to the germination statistics (`germ.py`, `effect.py`)

Each movie read as the product reads it (SparseTrack 0.8.0 defaults on the movie's own census: every census grain a
grain; `germ.py predict`), over its judged census-isolated grains away from the edge; flags from the classifier that
did not see the movie (SMALL / global, leave one movie out). Germinated share by the end and T50 (Turnbull; bins):

| movie | annotator (grains kept) | model, all census grains (now) | model, flagged grains left out | model, the annotator's grains (a perfect check) | change in the model's error, flagged out (95% CI over grains) |
|---|---|---|---|---|---|
| ld | 28 grains: 100%, T50 24.6 | 31: 97%, 25.2 | 29 (g005, g016, g017 out): 97%, 25.5 | 28: 96%, 25.2 | share +0.2 pts (-0.0 to +1.0), T50 +0.3 bins (-3.8 to +4.7) |
| m2 | 21: 95%, 80.2 | 30: 80%, 114.1 | 26 (g019, g096, g103, g117 out): 87%, 113.5 | 24: 85%, 105.9 | share -6.1 pts (-16.2 to +0.7), T50 -0.6 bins (-30.5 to +12.9) |
| m1 | 30: 97%, 28.7 | 41: 52%, 166.2 | 36 (g004, g029, g032, g049, g050 out): 51%, 148.6 | 30: 47%, - | share +1.3 pts (-4.3 to +7.9), T50 -17.6 bins (-89.5 to +38.3) |

(ld: g016 is read unobservable, so it was never in the model's population. m1: SparseTrack 0.8.0's default dispatch
reads 16 of its 28 germinated grains as never germinating - the hybrid's germination veto, `Params.hybrid_onset` - so
its share is near 50% and its T50 swings with any grain.) The false grains are not mostly read as "never germinated":
on ld the other two flagged grains (g005, g017) are read as germinated; on m2 three of the four flagged are read as
never germinating (g019, g096, g103) but g117, dark debris, is read as a 151 px tube present from the start, and the
excluded clump g045 and "other" g080 as germinated; on m1 7 of the 11 excluded "grains" are read as germinated (g058, a
clump, as a 131 px tube). So leaving them out moves the share only where debris is read as empty, and a perfect census
check would leave most of each movie's error in place (ld 96% vs 100%, m2 85% vs 95%, m1 47% vs 97%): the model's own
misses on real grains dominate. The other variants give the same picture (runs/census_check/effect.json; e.g. TWO /
global: share error ld +0.3 pts, m2 -9.5, m1 -1.6).

## Decision

Off by default (`Params.census_check = False`): it does not clearly help on all three movies. It finds dark debris well
(not a grain 8/9, with one wrong flag in 12 leave-one-out flags), but the dev movie's statistics do not move (a
perfect check would not move them either), and no movie's T50 moves measurably. Where a movie has much debris read as
empty (movie 2), `census_check=True` brings the germinated share closer to the annotator's; the flags are worth
showing in the review either way (`python -m sparsetrack.census_check CACHE` lists them; `--write` stores each grain's
P in the census). Clumps whose attached bits are faint, and grains excluded as out of focus or "other", are not found
from the census image: a larger labelled set (more movies' judgements) is what would move this.

## Files

`data.py` (features, runs/census_check/features.json), `loo.py` (leave one movie out; runs/census_check/loo.json),
`fit.py` (the model in `sparsetrack/census_check.py`), `germ.py` (predictions of every judged census grain,
runs/census_check/pred_<movie>/), `effect.py` (the table above; runs/census_check/effect.json). Tests:
tests/test_census_check.py.
