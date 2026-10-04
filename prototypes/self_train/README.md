# Label-free per-movie adaptation: self-training the tube network on the tracker's own confident readings (prototype, 3 Oct 2026)

**Question.** Can the tube-probability network adapt to a new movie with no human labels, by fine-tuning on the
tracker's own confident readings of that movie (pseudo-labels)? Movie 1 (m1) is the clean test: the shipped network
(`sparsetrack/models/tubes_bn_real_ld_m2.pt`) never saw it, and no m1 label is used to train or to choose anything.
Compared with `prototypes/tube_adapt` (fine-tuning on 15 human-traced m1 grains) on the same grains.

**Answer: the maps adapt about as much as with 15 corrected grains (traced tube marked 75% -> 90%, CI above 0), but
end to end the gains are small and not significant (lengths +2..+6 of 50, onsets +3..+5, length-and-tip -3..+4;
training noise alone moves lengths by 3); a second round adds nothing and starts to drift; movie 2 shows the same
pattern. Not worth building in as a default yet.** Details in "Results".

## Method

- **Pseudo-labels** (`select.py`) from 0.8.8's m1 readings of all 30 census grains (the analysis census is the
  labelled sample; no other grain of m1 is analysed). Rules fixed before any result:
  - grains: germinated, no unsafe flag (`touches:`, `drift_rejected`, `shared_change_split`, `rotates:`,
    `onset_at_focus_change`); a grain lost partway only up to 10 bins before it was lost;
  - confident bins: length >= 8 px, `review.trace_confidence` >= 0.7 (in practice: the reading grew within the last
    0-3 bins and is >= 12 px), the drawn tube (as the app draws it) on the map (P >= 0.5) for >= 80% of its length
    beyond the rim (`learned.drawn_check`'s measure at every bin), no point within 10 px of the frame edge;
  - pseudo-traces: every 3rd confident bin, at most 40 per grain; state `partial`, the route cut 3 px short of the
    reading's apex, nothing within 14 px of the apex scored; no background band where > 5% of it is marked on the map;
  - grains read as never germinated: used as negatives only if their surroundings stay clean on the map.
- **Two changes after the pseudo-label diagnosis, before any training** (the diagnosis uses the human labels, so
  these are disclosed as informed by them):
  - the pre-registered **negatives** (bins >= 10 before the earlier of the tracker's onset and the emergence
    extrapolated back from its first confident reading) were half wrong (135 of 270 bins had a traced tube; g033 and
    g065 alone 113: the tracker's onsets there are 250-300 bins late, so "before the onset" is not "no tube");
    pseudo-negatives are **off**;
  - the drawn routes ran a median 1.7 px (up to 3.7 px) off the human routes, along one wall where the flood's own
    per-bin route or an un-bent final route is drawn; each pseudo-route is **re-centred** on the middle of its band on
    the map (bins b-1..b+1, `learned.centre_route`'s band-middle rule): 0.8 px from the human routes.
- **Crops** (`pcrops.py`): as `prototypes/tube_net/realdata.py` v3 does for traces (4 crops along, 1 at the exit,
  the same route at b +/- 1 where the grain did not move), plus (`--prop`) `prototypes/tube_adapt/propagate.py`'s
  between/after crops from the pseudo-traces (the tube across non-confident gaps, and after the last confident
  reading while DIS flow carries it reliably), with the tracker's own drift as grain offsets.
- **Fine-tuning** (`train.py`): `tube_adapt/finetune.py`'s recipe from the starting checkpoint (1500 steps, batch 32,
  lr 5e-4 one-cycle, BN statistics frozen), half of each batch replay of the starting network's own training data.
  `sb` (primary, fixed beforehand) = pseudo-traced 25% + propagated 25%; `sa` = pseudo-traced 50%.
- **Judged** (`evaluate.py`, `table.py`) on all 30 labelled m1 grains (no fold split: no label was used), maps in
  memory: the pixel check and end to end with 0.8.8's defaults and only the network changed, paired over grains
  against 0.8.8 and against tube_adapt's human-label fine-tunes pooled over its folds (each grain judged by the
  network that did not see its traces), 95% bootstrap intervals over grains.

## Results (3 Oct 2026)

**Pseudo-labels, movie 1** (round 1; human labels used only for this diagnosis): 30 grains read, 3 skipped
(`drift_rejected`), the 2 grains read as never germinated had marked surroundings (no clean negatives); 25 germinated
grains, 1063 confident bins -> 359 pseudo-traces -> 2495 traced crops + 985 propagated (301 between, 684 after).
- At the 8 human FULL traces that fall on confident bins (all at bin 140): length within tolerance 4/8 (median
  -6 px: the tracker reads short), apex within tolerance 7/8; at the 42 on non-confident bins 11/42 and 18/42, so the
  rule does pick the better readings. Routes: median 1.75 px off the human route as drawn, 0.82 px after centring.
- Crops within 1 bin of a human trace: 90-92% of their tube pixels on the human route, 1.9% of their scored
  background on it; propagated "between" 92-98%, "after" 59% (6 crops; g007's reading follows another structure).
- The pre-registered negatives would have been 135/270 wrong. Without them, self-training found the tubes of g033 and
  g065, the two grains they would have suppressed most (g033 at bin 140: -66.6 -> -4.5 px; g065's onset 252 bins late
  -> a hit).
- Round 2 (re-read with round 1's network): 26 grains, 1088 bins -> 371 pseudo-traces; 4/7 lengths, 6/7 apices.

**Movie 1, all 30 labelled grains (50 FULL traces)**, against 0.8.8 (shipped network) on the same grains, paired 95%
bootstrap intervals over grains (`table.py m1 --nets m1_r1_sb m1_r1_sb_s1 m1_r1_sa m1_r2_sb --base-e2e e2e_m1_base088`):

| network | traced tube marked | beside | rim before onset | lengths in tolerance | length and tip | onsets |
|---|---|---|---|---|---|---|
| shipped (0.8.8) | 75% | 3.5% | 30.6% | 15/50 | 15 | 8/26 |
| self r1 sb (primary) | 90% (+14.5, +4.1..+27.5) | 7.2% | 27.6% | 17 (+2, -3..+7) | 13 (-2, -7..+3) | 11/27 (+3, -3..+9) |
| self r1 sb, seed 1 | 89% (+13.9, +3.6..+28.6) | 6.5% | 26.3% | 20 (+5, -1..+11) | 16 (+1, -5..+7) | 11/26 (+3, -3..+9) |
| self r1 sa (traced only) | 79% (+3.8, +0.0..+8.7) | 3.5% | 25.7% | 21 (+6, -1..+13) | 19 (+4, -3..+11) | 12/26 (+4, +1..+8) |
| self r2 sb | 90% (+14.8, +3.7..+25.3) | 15.3% | 36.8% | 16 (+1, -4..+6) | 12 (-3, -9..+3) | 13/26 (+5, -1..+11) |
| human a (15 grains, held-out) | 90% (+15.0) | 11.0% | 27.6% | 22 (+7, -1..+15) | 17 (+2) | 14/27 (+6, +2..+11) |
| human b | 93% (+17.3) | 17.8% | 25.9% | 18 (+3) | 15 (+0) | 17/26 (+9, +4..+14) |
| human c | 96% (+21.1) | 21.6% | 30.6% | 21 (+6) | 17 (+2) | 14/27 (+6, +1..+11) |

- Self minus human, same grains: sb's pixel gain equals human a's (-0.4 pts, -3.7..+2.6); end to end sb seed 0 vs a:
  lengths -5 (-11..+0), onsets -3 (-7..+1); human b's onsets are not recovered (sb -6, -10..-2; sa -5, -9..-1).
- The self-trained network was judged on the grains whose readings trained it (transductive: no label was used);
  the human fine-tunes on grains they never saw. sa was not the pre-registered variant and was run on m1 only.
- Round 2 against round 1: onsets +2 (-2..+6), lengths -1 (-6..+4), pixels +0.3 pts; marks beside tubes 7 -> 15% and
  rim marks before onset 28 -> 37% (above the shipped network's 31%): the network starts to confirm its own marks.
- Note on tube_adapt's README table: its "rim marks before onset" column (3.5 -> 11-22%) is the "beside" measure
  (marks 8-14 px beside traced tubes); its rim marks before onset are 30.6% -> 27.6 / 25.9 / 30.6%.

**Movie 2 check** with `runs/tube_net/tn3_ldm1.pt` (ld + m1 traces; never saw movie 2's), same procedure (replay ld +
m1 crops): 24 grains read, 14 germinated + 1 clean grain used (its human verdict: no emergence), 847 confident bins ->
261 pseudo-traces; accepted human FULL traces 6/9 lengths, 7/9 apices (non-confident 11/45, 18/45); the
pre-registered negatives would have been 23/255 wrong.
- Pixels (24 grains, 54 FULL traces): traced marked 90% -> 96% (+5.8, +2.8..+8.3), young stubs 11 -> 13/20, beside
  5.0 -> 8.0%, rim before onset 14.8 -> 15.6%.
- End to end vs tn3_ldm1 under 0.8.8's defaults: lengths 17 -> 21/54 (+4, -2..+10), length and tip 14 -> 16 (+2,
  -4..+8), onsets 3 -> 6/18 (+3, +0..+6).

**Reading.** Label-free self-training on the tracker's confident readings makes the maps mark much more of the traced
tubes on a new movie (as much as 15 corrected grains' traced crops), but as with the human fine-tune the flood turns
little of it into lengths or onsets, and the end-to-end effects are within training noise (one seed to the next: 3
lengths). Pseudo-negatives before the tracker's onset are unsafe (half wrong on m1); a second round drifts. The
pixel check and end to end disagree on the better variant (sb pixels, sa end to end), and nothing label-free can
choose between them. Not a default step; worth revisiting after the flood's start and stop rules are re-tuned for
adapted maps, then judged on a held-out movie with sa and sb fixed beforehand.

Cost per movie on the laptop: one extra read (~8 min), selection (~1 min with maps in memory), crops (~1 min), a
fine-tune of 2.5-6 min. Shards were deleted (the disk is full); rebuild them from the kept pseudo-labels with
`python -m prototypes.self_train.pcrops runs/research/self_train/pseudo/<name>.json --movie <m1|m2> --prop`.
