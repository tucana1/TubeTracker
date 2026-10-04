# A global tip-trajectory reader (research prototype, 4 Oct 2026)

**Question.** Per grain, choose the tube's tip at every bin in one whole-movie optimisation (Viterbi over bins), each
candidate tip with its tube body (the cheapest route on that bin's tube map from the grain's rim), transitions
rewarding bounded growth and a body that agrees with the previous bin's (so that jumping onto another tube is
expensive). Does it read the crowded movies (m2, m1) better than SparseTrack 0.8.8? (Literature: Li, Shen & Huang,
IPMI 2011, global tip DP with per-frame body search; RootNav 2.0, tip heatmaps driving path search.)

**Answer:** RESULTS_PENDING

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

RESULTS_PENDING
