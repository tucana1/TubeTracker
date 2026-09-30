# Review confidence on a third movie (30 Sep 2026)

**Result: the confidence that orders the review transfers to movie 1 unchanged; refitting it on three movies changes
nothing that matters. Keep it.**

`sparsetrack.review.trace_confidence` orders the review (least sure first): z = -0.41 - 0.63 log1p(stall) +
0.50 log1p(L), where L is the model's length at the traced bin and stall the bins since its reading last grew. It was
fitted on the dev movie and movie 2 on 29 Sep (runs/review_triage). Movie 1 never took part in fitting it, so its
movie-1 result is an honest check. `conf3.py` measures, per movie, over every scored FULL trace: the AUROC of the
confidence against the reading being within tolerance, and the review effort (the share of traces a reviewer checks,
least sure first, before 79% / 90% are within tolerance; checked ones count as right) against a random order, with 95%
bootstrap intervals over grains; then the same two features refitted on two movies and applied to the third.

Readings: each movie read by a network that never saw its traces - movie 1 by SparseTrack 0.8.0 as frozen
(runs/sparsetrack/m1_frozen_0.7.0_bn: ld + m2 traces, default dispatch), the dev movie by tn_bn_r3v6_m2 and movie 2 by
tn_bn_r3v6_ld (prototypes/tube_net, the two-movie folds; their bench predictions). Record: runs/tube_net/conf3_loo.json.

| movie | traces (within tolerance) | AUROC, fixed | effort to 79%: least sure first / random (paired, 95% CI) | to 90% | AUROC, refitted on the other two |
|---|---|---|---|---|---|
| ld | 104 (73) | 0.71 | 12% / 30% (-18 pts, -25 to +0) | 48% / 67% (-19, -35 to +0) | 0.69 |
| m2 | 54 (24) | 0.74 | 44% / 63% (-19, -25 to -2) | 70% / 83% (-13, -29 to +2) | 0.74 |
| **m1** | 50 (8) | **0.91** | 64% / 76% (**-12, -21 to -4**) | 76% / 90% (-14, -23 to +2) | 0.92 |

Refitted on each pair of movies the weights stay close to the fixed ones (stall -0.58..-0.76, length +0.31..+0.49), and
the held-out review efforts are the fixed function's within 2 points (95% CIs within -6 to +11). Pooled over the three:
z = +0.04 - 0.66 log1p(stall) + 0.43 log1p(L): stall weighted a little more against length (1.53 against 1.26) and a
higher baseline; the held-out refits show that this makes no difference to the review. Not changed. (A first run with 0.7.0's readings of ld and m2 gave the same: AUROC 0.71 / 0.75 / 0.91.)

On movie 1 most readings are wrong (8 of 50 within tolerance, mostly grains read as never germinating: the hybrid's
germination veto, `Params.hybrid_onset`), so any order must check most traces; the function still puts the wrong ones
first (AUROC 0.91). Worth re-running once movie 1's dispatch is settled.
