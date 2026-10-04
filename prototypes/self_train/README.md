# Label-free per-movie adaptation: self-training the tube network on the tracker's own confident readings (prototype, 3 Oct 2026)

**Question.** Can the tube-probability network adapt to a new movie with no human labels, by fine-tuning on the
tracker's own confident readings of that movie (pseudo-labels)? Movie 1 (m1) is the clean test: the shipped network
(`sparsetrack/models/tubes_bn_real_ld_m2.pt`) never saw it, and no m1 label is used to train or to choose anything.
Compared with `prototypes/tube_adapt` (fine-tuning on 15 human-traced m1 grains) on the same grains.

**Answer: the maps adapt about as much as with 15 corrected grains (traced tube marked 75% -> 90%, CI above 0), but
end to end the gains are small and not significant (lengths +2..+6 of 50, onsets +3..+5, length-and-tip -3..+4;
training noise alone moves lengths by 3); a second round adds nothing and starts to drift; movie 2 shows the same
pattern. Not worth building in as a default yet.** Details in "Results".

**Update, 4 Oct: the step.** Made movie-agnostic as `sparsetrack/selftrain.py` (`python -m sparsetrack selftrain`;
the cache's own census, no labels) and judged under the tip-trajectory reader: movie 1 28/50 lengths with either of
two seeds (27/50 on the shipped maps; onsets 15-17/28 vs 12/28); movie 2, from a network that never saw it, 24, 31
and 31/54 with seeds 0-2 (28/54 on that network's own maps). Frozen as a second blind candidate, one run, seed 0.
See "The step" at the end.

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

## The step: `sparsetrack/selftrain.py` (4 Oct 2026)

**What it is.** The prototype as one label-free step that runs on any movie from its cache alone:
`python -m sparsetrack selftrain CACHE --out NET.pt [--predictions PRED.json] [--start NET0.pt] [--replay SHARD...]
[--steps 1500] [--seed 0]`. It (1) reads every grain of the cache's own census (`grains.json`) with SparseTrack's
defaults and the starting network, unless given those readings; (2) selects pseudo-traces with `select.py`'s rules
(unchanged: confident bins, drawn tube on the map, centred on its band, every 3rd, at most 40 per grain; clean
never-germinated grains as background; no negatives before the tracker's onset); (3) builds the traced and the
propagated crops in memory (no shards on disk); (4) fine-tunes once from the starting network (1500 steps, batch 32,
lr 5e-4 one-cycle, BatchNorm statistics frozen; each batch a quarter pseudo-traced, a quarter propagated, a quarter
replayed trace crops and a quarter replayed synthetic crops of the starting network's own training data). It writes
the network and a record beside it (`NET.json`: settings, sha1s of the starting network, the replay shards and the
readings, pseudo-label and crop counts, timings); with nothing confident to learn from it keeps the starting network.
The adapted network is then the movie's `Params.model` (the hybrid's flood, the route centring and the
tip-trajectory reader all read its maps).

**Differences from the prototype.** The census is the cache's `grains.json` (no labels file: neither the human
sample nor the human exclusions), and every census grain is read, not only the labelled sample; nothing else
changed. Port check on the prototype's own input (0.8.8's readings of m1's 30 labelled grains, shipped maps): the
same 359 pseudo-traces (every route identical) and the same crops (2495 traced, 985 propagated, counts by kind
identical).

**Variant, fixed before the step was run: sb (propagated crops on).** Evidence from the record above (no new run was
used to choose): its maps mark more of the traced tubes than sa's on m1 (90% vs 79%) and on m2 (sb only: 90 -> 96%);
the one test of the tip-trajectory reader on self-trained maps (m1, 32/50) used sb; sa's end-to-end lead under
0.8.8's flood (+1 to +4 lengths) is within the seed noise (3), and the reader leans on map support along a body
(its weight at the top of its range), so the maps that mark more of the tubes are its better input. sa was not re-run.

**Judged** (`stepeval.py`): the tip-trajectory reader as the 0.9.0 candidate runs it (`tiptraj="flood"`,
`weights_ld_v3.json`, `tiptraj_mid`), with each movie's held-out detector fold, read end to end on the movie's scored
grains as `scripts/score_heldout.sh` reads a new movie (census = the labels file), only `Params.model` changed;
scored on all labelled grains, paired 95% bootstrap intervals over grains (onsets; lengths in tolerance; length and
tip). The self-trained networks never saw a label: they are judged on the grains whose readings trained them
(transductive, as on a new movie). The reader on the shipped maps, re-run with this code, gives the same 30
length series as the round-3 bench run it is compared with (prototypes/tip_trajectory, 27/50).

**Movie 1** (30 grains, 50 FULL traces; starting network the shipped one, which never saw m1; detector fold
`tip3_ldm2`; pseudo-labels from all 69 census grains: 35 used, 1375 confident bins -> 471 pseudo-traces -> 3277
traced + 1319 propagated crops):

| reading | onsets | lengths | length and tip | vs 0.8.8 | vs the reader on the shipped maps |
|---|---|---|---|---|---|
| 0.8.8 | 8/26 | 15/50 | 15 | | -4 [-9, +1]; -12 [-21, -3]; -9 [-17, -1] |
| reader, shipped maps (0.9.0 candidate) | 12/28 | 27/50 | 24 | +4 [-1, +9]; +12 [+3, +21]; +9 [+1, +17] | |
| **reader, self-trained maps, seed 0** | **17/28** | **28/50** | **26** | +9 [+3, +14]; +13 [+5, +20]; +11 [+4, +18] | **+5 [0, +11]; +1 [-4, +6]; +2 [-3, +7]** |
| reader, self-trained maps, seed 1 | 15/28 | 28/50 | 25 | +7 [+2, +12]; +13 [+5, +20]; +10 [+3, +17] | +3 [-3, +9]; +1 [-4, +6]; +1 [-3, +5] |
| reader, soup of seeds 0-2 (`--runs 3`) | 16/28 | 27/50 | 25 | | +4 [-1, +9]; 0 [-4, +4]; +1 [-2, +5] |
| reader, the prototype's `m1_r1_sb` (labelled sample's readings) | 15/28 | 31/50 | 28 | | +3 [-3, +9]; +4 [0, +9]; +4 [0, +9] |
| reader, the prototype's `m1_r1_sb_s1` | 15/28 | 31/50 | 28 | | +3 [-2, +8]; +4 [0, +8]; +4 [+1, +8] |

- Seed sensitivity (crop jitter and training seed; the selection is deterministic): seed 1 vs seed 0 onsets -2
  [-5, 0], lengths 0 [-3, +3], length and tip -1 [-4, +2]; 4 grains differ, the length classes not at all (young
  18/28, mid 5/11, long 5/11 for both; the shipped maps 16, 5, 6).
- Against the shipped maps the gain is in onsets (7 grains gain one, seed 0) and young tubes (+2); long tubes -1.
- The prototype's own network (`m1_r1_sb`: pseudo-labels from 0.8.8's readings of the 30 labelled grains, the
  census and its exclusions from the labels file) reads 31/50 end to end (offline it read 32): integration is not
  what separates it from the step's 28/50; the pseudo-labels are. Against the step's seed 0: lengths +3 [-1, +7].
  The step reads the cache's 69 census grains instead (11 unlabelled grains add pseudo-traces, and the labelled
  grains are read in another context: 25 of 30 read differently, through the obstacles and the speed-cap probe).
  Its seed-1 network reads 31/50 too, and the step's two seeds 28 and 28: the 3 lengths are the pseudo-labels'
  source, not training noise (prototype seed 1 vs the step's seed 1: +3 [-1, +7]). The prototype's census was
  the labelled sample (the "slight label leak": the person's sample and exclusions); on movie 2 the same
  comparison shows no gap (the prototype's `m2_r1_sb` 28/54, the step's seeds 24-31, mean 28.7). A label-free way
  to recover it (e.g. reading only the census' isolated grains, as the person sampled) was not tried.
- Pseudo-label check (human labels used only for this report, `stepdiag.py`): at the 8 human FULL traces on
  confident bins the readings' lengths are within tolerance 4/8 (median -6 px: they read short), their centred apices
  8/8; at the other 42, 11/42 lengths - as in the prototype (4/8, 7/8, 11/42).

**Movie 2** (24 grains, 54 FULL traces; nothing here saw movie 2: starting network `runs/tube_net/tn3_ldm1.pt` (ld +
m1 traces), replay its own training data (ld + m1 trace crops `real3_ld`, `real3_m1`, and the same two synthetic
shards), detector fold `tip3_ldm1`; pseudo-labels from all 122 census grains: 49 germinated + 7 clean used, 3059
confident bins -> 937 pseudo-traces -> 6846 traced (301 of them clean-grain background) + 2160 propagated crops):

| reading | onsets | lengths | length and tip | vs the reader on tn3_ldm1's maps |
|---|---|---|---|---|
| 0.8.8, tn3_ldm1's maps (context) | 3/18 | 17/54 | 14 | +1 [0, +3]; -11 [-19, -3]; -13 [-21, -5] |
| reader, tn3_ldm1's maps | 2/18 | 28/54 | 27 | |
| **reader, self-trained from tn3_ldm1, seed 0** | **3/18** | **24/54** | **23** | **+1 [0, +3]; -4 [-9, +1]; -4 [-9, +1]** |
| reader, self-trained from tn3_ldm1, seed 1 | 4/18 | 31/54 | 29 | +2 [0, +5]; +3 [-2, +9]; +2 [-3, +8] |
| reader, self-trained from tn3_ldm1, seed 2 | 2/18 | 31/54 | 31 | 0 [0, 0]; +3 [-5, +12]; +4 [-4, +12] |
| reader, soup of seeds 0-2 (`--runs 3`) | 2/18 | 28/54 | 27 | 0 [0, 0]; 0 [-7, +8]; 0 [-7, +8] |
| reader, the prototype's `m2_r1_sb` (labelled sample's readings) | 4/18 | 28/54 | 28 | +2 [0, +5]; 0 [-8, +9]; +1 [-7, +10] |
| (0.8.8 on the shipped maps, which saw m2's traces) | 7/18 | 27/54 | 23 | |

- The training seed matters on movie 2: seed 1 vs seed 0 lengths +7 [+3, +11], seed 2 vs seed 0 +7 [0, +14]
  (length and tip +6 [+2, +10], +8 [+2, +15]); seeds 0-2 average 28.7 lengths, 27.7 length and tip, 3 onsets,
  against 28, 27, 2 on tn3_ldm1's own maps. Seed 0 loses the mid-length tubes (12/18 -> 7/18) and gains long ones
  (5 -> 7/11); seed 1 gains young (11 -> 15/25) and long (5 -> 7) and loses mid (12 -> 9).
- Pseudo-label check (reporting only): at the 9 human FULL traces on confident bins lengths 6/9, centred apices 7/9;
  at the other 45, 11/45 - as in the prototype's movie-2 check (6/9, 7/9, 11/45).

**Seed noise on movie 2, and a soup** (written before the soup's results were seen). Seed 1 against seed 0 on m2:
lengths +7 [+3, +11], length and tip +6 [+2, +10]: a single frozen run on a blind movie would carry that much luck.
`--runs K` averages K fine-tunes (seeds S..S+K-1, each on its own crops) into one network (a uniform "model soup":
same start, same frozen BatchNorm statistics). Rule fixed before its evaluation: the candidate uses `--runs 3` if the
soup of seeds 0-2 reads at least the mean of the single seeds' lengths on both movies (those read: m1 seeds 0 and
1, m2 seeds 0-2), else `--runs 1`; the candidate is added if, for that recipe, m1 reads >= 27/50 (the reader on the
shipped maps) and m2 >= 28/54 (the reader on tn3_ldm1's maps), single-seed recipes judged by their seed mean.

**Soup result and decision.** The soup of seeds 0-2 read 27/50 on m1 (single seeds 28, 28) and 28/54 on m2 (single
seeds 24, 31, 31; mean 28.7): below the single seeds' mean on both, so by the rule above the candidate keeps one run
(`--runs 1`, seed 0). Its estimates by the seed means: m1 28/50 >= 27 (the reader on the shipped maps), m2 28.7/54
>= 28 (the reader on tn3_ldm1's maps): **added to `scripts/score_heldout.sh` as "0.9.0-tiptraj-selftrained"**
(the selftrain step on the new movie with these settings, the starting network's and replay shards' sha1s checked,
then the reader with the adapted network as `Params.model`; to be tagged `sparsetrack-0.9.0-selftrained-candidate`).
What to expect blind: about the reader's lengths on the shipped maps, more onsets on movie 1 (+3 to +5) and movie 2
(+0 to +2), and a run-to-run spread of up to 7 lengths of 54 on a crowded movie.

**Runtime** (this Mac, MPS; one heavy process): movie 1 (69 census grains) 11.6 min = the census read 551 s (its
maps were on disk) + selection 3 s + crops 23 s + fine-tune 121 s; movie 2 (122 grains) 21.2 min = the starting
network's maps 72 s + census read 1046 s + selection 4 s + crops 33 s + fine-tune 119 s. The census read is ~8 s per
grain, so a crowded census dominates; the adapted network's own maps (~1-2.5 min, ~440 MB) are built by the analysis
that uses it. Memory: the crops are kept in RAM (m2: ~9000 pseudo crops ~0.65 GB, the replay ~0.5 GB).

Files: `sparsetrack/selftrain.py` (`python -m sparsetrack selftrain`), `tests/test_selftrain.py`; `stepeval.py` (the
reader end to end on a labelled movie with a given network; tables with paired intervals), `stepdiag.py` (the
pseudo-label check). Outputs (scratch, not kept): networks `models/{m1,m2}_s{0,1,2}.pt` and soups, their records
(`.json`), the census readings, the readers' predictions; probability movies deleted after use.
