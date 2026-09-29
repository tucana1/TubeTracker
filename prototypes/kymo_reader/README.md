# KymoReader: a learned space-time reader of the growth front (prototype, 29 Sep 2026)

Question: SparseTrack's length errors come mostly from *reading the growth front along a path*,
not from choosing the path (oracle-route tests, 23-27 Sep). Can a small network that reads the
whole kymograph along the path (arc x time, with temporal context) read the front better than
SparseTrack 0.5.3's hand-made change evidence + DP?

Everything here is a prototype: nothing in `sparsetrack/` was changed. Outputs are in
`runs/kymo_reader/` (git-ignored).

## Design

| file | what |
|---|---|
| `kymo.py` | kymograph of one grain along one path, every observed bin, 17 lateral offsets |
| `strot.py` | SparseTrack's own rotation track (and front) along any route (the `oracle_route.py` trick) |
| `synth_data.py` | synthetic training kymographs with exact truth (codec-exact v5 movies, 1 cache at a time) |
| `real_data.py` | kymographs of the labelled real grains + partial labels from the human traces |
| `flood_paths.py` | centrelines for st053's flood-read grains (the frozen files store their tube pixels, not a line) |
| `features.py`, `model.py` | network inputs; KymoNet, a dilated residual CNN over (bin, arc) |
| `crf.py` | optional loss through the decoder: a CRF over monotone fronts (forward algorithm) |
| `train.py` | masked BCE (+ CRF) on synthetic (+ one real movie's labels) |
| `decode.py` | monotone DP front, length start, calls, tips |
| `evaluate.py` | reads the real movies, writes prediction copies, scores, paired bootstrap vs st053 |
| `protocol.py` | the A / B / C protocol (choices never from the test movie) |
| `cv.py` | within-movie cross-validation (labels from the other half of the movie's grains) |
| `tables.py`, `compare.py`, `show.py` | result tables, per-grain comparison, diagnostic sheets |


### Kymographs (`kymo.py`)

- The path (SparseTrack's `path`, or a human trace's `path_xy_view`) is resampled at 1 px, extended
  20 px straight past its end and **8 px straight back into the grain** (arc s < 0): annotators start a
  trace where the tube leaves the grain's *visible* edge, a median 1-2 px inside the census circle on
  ld, so the reader must see that region.
- Per observed bin t, 17 lateral offsets w = -8..8 px: `I_t - B` on the path **turned by SparseTrack's
  rotation track** about the exit (the model's own `rotation_deg`; along a human route, SparseTrack's
  track along that route from `strot.read_route`) and the same on the unturned path. Static per arc:
  `E - B`, `B` (17 offsets each), distance from the grain rim, distance to the nearest other grain.
  B = mean of the first 3 observed bins, E = mean of the last 3 full bins (SparseTrack's early/late).
- The view frame follows the grain exactly as the labelling tool does (`Bench.follow`; `kymo.follow_offsets`
  is the same computation, checked equal on ld grains), then `Renderer.crop` adds the field registration.
- Why the rotation track: earlier human traces often do not lie on the latest route (apex-to-route distance
  ld median 1.9 px, p90 8.3 px; m2 median 4.7 px, p90 15.6 px). On an unturned route the tube leaves the
  kymograph for dozens of bins (e.g. ld g034); on SparseTrack's turned path it stays in view.

### Targets

"Covered" Y(t, s) = arc s is grain body or tube by bin t; the front F(t) is monotone. Lengths are
F(t) minus the arc where the length starts.

- Synthetic (exact): Y = [s < L(t)], L the true length from the census circle at the bin centre. Where the
  grain edge is (s in [-4, 2]) nothing is asserted while L < 2 px: where a length starts is the annotators'
  convention, which only real labels can teach.
- Real (partial), along the human route or SparseTrack's path: each trace's own first click is projected
  onto the path (its start o_k); FULL trace (t_k, L_k): covered for s < o_k + L_k - m at t >= t_k, not
  covered for s > o_k + L_k + m at t <= t_k, m = max(2 px, 10%); PARTIAL: the first half only; "no tube"
  and the onset bracket: nothing beyond the start + 2 px up to the last absent bin, covered up to start + 1
  px from the first visible bin; the grain (s < start - 2 px) always covered; conflicting cells unlabelled.
  On SparseTrack's path a trace is used only if its apex lies within max(4 px, 10%) of the (turned) path.

### Data

| set | source | samples |
|---|---|---|
| synthetic train | v5 on ld's field, seeds 0, 1, 2, 5, 6, 7 (exact + perturbed route per tube, + controls) | ~68 per movie |
| synthetic train | v5 on m2's field (recipe.M2), seed 10 (60 tubes, one route each, + controls) | 70 |
| synthetic validation | ld field seed 3, m2 field seed 20 | 70 + 70 |
| real ld | 31 included grains: 30 on SparseTrack's path (4 flood-read), 31 human routes | 61 |
| real m2 | 24 included grains: 23 on SparseTrack's path (21 flood-read), 19 human routes | 42 |

Perturbed synthetic routes: bent sideways by a smooth offset of up to 2.5 px (zero at the exit), end cut or
extended by up to 8 px. Synthetic caches were built one at a time and deleted after extraction.

The frozen st053 files store a flood grain's `path` as its tube pixels in rim-distance order (the
centreline fix de251b0 landed while they were being made), so `flood_paths.py` re-ran the 0.5.3 flood
(st053's own parameters, every option added since switched off) and kept its centreline; all 27 flood
grains reproduced the frozen length series exactly. Two grains (ld g008, m2 g100) have no usable path:
they keep st053's answer.

### Model, loss, decoder

- KymoNet (`model.py`): dynamic input (35 channels per bin and arc) and static input (36 channels per
  arc) through their own first layers, summed; 10 residual blocks (3x3 conv, dilations up to 32 bins in
  time and 8 px along the arc; receptive field ~157 bins x ~59 px), per-position LayerNorm, width 32,
  0.12 M parameters. Output: logit of "arc s covered by bin t".
- Loss (`train.py`, `crf.py`): masked BCE per cell (mean per sample) **plus a front loss that trains
  through the decoder**: per bin, a softmax over front positions f with score sum_{i<f} logit(t, i) (the
  decoder's own objective), and -log P(f in the range the labels allow). The full CRF over monotone
  fronts (forward algorithm, `crf.loss`, checked against brute force) is implemented too but its
  176-step sequential loop cost ~2 s per batch on the shared GPU, so the per-bin version is used; the
  decoder's DP supplies the coupling between bins.
- Training: random crops of 160 bins x 96 px (the arc crop starts at the exit half of the time), lateral
  mirroring, contrast gain; AdamW, one-cycle LR (2e-3 for the pilot, 1e-3 for the fine-tunes), batch 8,
  every batch padded to one fixed shape; checkpoint kept by synthetic validation.
- Decoder (`decode.py`): the monotone DP of SparseTrack (front advance 0..4 px per bin) on the logits;
  F(t) = arc of the front. Length = F(t) - start, start = the annotator's click (human route), or on
  SparseTrack's paths `edge` (steepest step of the before image along the path, the idea of
  sparsetrack.analyze.exit_edge), `rest` (the reader's own front in the first bins) or `census` (the
  path's first point). Onset = first bin with length >= onset_px; no germination if the final length
  < min_len; tips = path point at F(t), turned like the path was read.

### Protocol (`protocol.py`)

- **A** (synthetic only): onset_px, min_len chosen on the synthetic validation movies (ld field s3, m2
  field s20); length start fixed in advance to `edge` (annotators start at the visible edge); onsets from
  the reader. Scored on ld and m2. A secondary row, also fixed in advance, keeps st053's own germination
  calls and onsets and replaces only the lengths.
- **B** (the pilot fine-tuned like A, with ld's labels added): start, onset source (reader or st053) and
  thresholds chosen on ld's labels; scored on m2 only. **C**: the same with m2's labels, scored on ld only.
- Every row is compared with st053 on the same grains by a paired bootstrap over grains (4000
  resamples, as scripts/synth_bench.py); the oracle-route rows (reader on the human's latest route,
  SparseTrack's rotation along it) are also compared with SparseTrack's change reader on the same routes.

(Seeds 11 and 13 of the m2 field were planned too; with the laptop at load ~48 from other jobs, building
one m2-field cache took 17 min, so they were dropped to keep A, B and C on one synthetic pool.)

## Runs

### Synthetic validation (held-out movies: ld field s3, m2 field s20; 140 kymographs, lengths at every 12th bin)

Pilot (trained on ld-field s0-s2 only, 2000 steps): lengths within max(2 px, 10%) **1411/1691** vs
SparseTrack's change reader along the same routes **873/1691**; onsets 105/126 vs 68/126 (SparseTrack's
synthetic onsets here are its front onsets, not its matched stub); absences 1293/1319. Per field: ld
field 498/582 vs 320, m2 field 913/1109 vs 553 (no m2-field movie in its training).

Ablations, same data, 1000 steps each (one seed each, so differences of a few dozen are indicative):

| variant | lengths /1691 | onsets /126 | absences /1319 |
|---|---|---|---|
| BCE + per-bin front loss, full temporal context | **1403** | **106** | 1275 |
| BCE only | 1327 | 102 | 1256 |
| BCE + front loss, no temporal dilation (receptive field ~21 bins) | 1356 | 96 | 1254 |

So on synthetic data both the loss through the decoder and the long temporal context help.

Final models: A, B and C are the pilot fine-tuned on the same synthetic pool (ld field s0, s1, s2, s5,
s6, s7; m2 field s10) with the same schedule (one-cycle LR 1e-3 planned over 2500 steps), B adding ld's
partial labels and C m2's (35% of each batch real). A's run was stopped by an outside interruption after
~900 steps; its kept checkpoint is the step-500 one (the only synthetic validation it reached: lengths
1472/1691, onsets 111/126). To keep the three comparable, B and C were then trained with exactly that
schedule and stopped after step 500 (`--stop-after 500`): synthetic validation B 1416/1691 and 108/126,
C 1395/1691 and 110/126. The real test movies were never looked at while training or choosing.


## Results on the real movies (the answer: no)

**KymoReader does not beat SparseTrack 0.5.3 on either real movie, in any configuration.** It reads
held-out synthetic movies far better than SparseTrack (lengths 87% vs 52% within tolerance on the same
routes), but that does not transfer: on real footage it loses 7-12 length hits and several onsets, and
real labels from the other movie (B, C) do not close the gap. Rows: the reader on SparseTrack's own path
and rotation track (the deployable variant), on the human's latest route (oracle route: path choice
removed), and SparseTrack's change reader on the same human routes. Paired differences against st053 over
grains, 95% bootstrap intervals in brackets. FULL traces: ld 104 (28 isolated grains), m2 54 (24 grains).

| model | test | reader / path | onsets | lengths | length + tip | vs st053: onsets | lengths | length + tip |
|---|---|---|---|---|---|---|---|---|
| - | ld | SparseTrack 0.5.3 (st053) | 15/26 | 62/104 | 51/104 | | | |
| A | ld | KymoReader on st053's path | 3/22 | 50/104 | 42/104 | -12 [-18, -6] | -12 [-22, -2] | -9 [-19, +1] |
| A | ld | KymoReader lengths, st053's calls | 15/26 | 50/104 | 42/104 | 0 | -12 [-22, -2] | -9 [-19, +1] |
| A | ld | KymoReader on the human route (oracle) | 4/22 | 53/104 | 42/104 | -11 [-17, -4] | -9 [-19, +1] | -9 [-22, +4] |
| C | ld | KymoReader on st053's path | 8/20 | 50/104 | 41/104 | -7 [-12, -2] | -12 [-23, -2] | -10 [-20, +0] |
| C | ld | KymoReader on the human route (oracle) | 5/19 | 52/104 | 44/104 | -10 [-17, -3] | -10 [-21, +2] | -7 [-21, +7] |
| - | ld | SparseTrack on the human route (oracle) | 7/27 | 59/104 | 53/104 | -8 [-13, -3] | -3 [-14, +8] | +2 [-11, +15] |
| - | m2 | SparseTrack 0.5.3 (st053) | 5/19 | 23/54 | 18/54 | | | |
| A | m2 | KymoReader on st053's path | 2/16 | 13/54 | 9/54 | -3 [-8, +2] | -10 [-20, +0] | -9 [-17, -2] |
| A | m2 | KymoReader lengths, st053's calls | 5/19 | 10/54 | 8/54 | 0 | -13 [-23, -4] | -10 [-18, -3] |
| A | m2 | KymoReader on the human route (oracle) | 1/16 | 8/54 | 6/54 | -4 [-8, -1] | -15 [-24, -6] | -12 [-20, -4] |
| B | m2 | KymoReader on st053's path | 3/18 | 16/54 | 13/54 | -2 [-7, +3] | -7 [-17, +4] | -5 [-13, +4] |
| B | m2 | KymoReader on the human route (oracle) | 3/19 | 15/54 | 14/54 | -2 [-7, +3] | -8 [-19, +3] | -4 [-15, +7] |
| - | m2 | SparseTrack on the human route (oracle) | 4/19 | 11/54 | 11/54 | -1 [-5, +3] | -12 [-24, +1] | -7 [-18, +5] |

Choices (never from the test movie): A (synthetic validation) onset_px 1.5, min_len 6, start `edge`
(fixed in advance), onsets from the reader; B (ld labels) start `census`, onsets from the reader,
onset_px 1.5, min_len 8; C (m2 labels) start `edge`, onsets from the reader, onset_px 2, min_len 8. The
"st053's calls" rows (fixed in advance, A only) keep st053's germination calls and onsets and replace only
the lengths.

Reading quality alone (both readers on the same human routes, paired over grains):

| model | test | lengths | onsets | length + tip |
|---|---|---|---|---|
| A | ld | -6 [-17, +5] | -3 [-9, +3] | -11 [-21, -1] |
| A | m2 | -3 [-10, +3] | -3 [-7, +0] | -5 [-12, +1] |
| B | m2 | +4 [-2, +11] | -1 [-5, +3] | +3 [-3, +9] |
| C | ld | -7 [-18, +5] | -2 [-8, +4] | -9 [-19, +2] |

Only B on m2's routes reads slightly better than SparseTrack's change reader (15 vs 11 of 54), and not
significantly. Even on the movie whose labels they were trained on (settings chosen there, so optimistic)
B and C stay below st053: B on ld 51/104 lengths and 16 onsets (st053 62, 15), C on m2 22/54 and 5 (st053
23, 5).

### Where it fails (post hoc, frozen models)

| movie | prediction | tubes < 15 px | tubes >= 15 px (bias) | onsets: early / late (> 600 frames) |
|---|---|---|---|---|
| ld | st053 | 18/30 | 44/74 (-4.8 px) | 5 / 6 of 26 |
| ld | A on st053's path | 9/30 | 41/74 (-3.2 px) | 16 / 3 of 22 |
| ld | C on st053's path | 11/30 | 39/74 (-3.2 px) | 8 / 4 of 20 |
| m2 | st053 | 5/25 | 18/29 (-7.3 px) | 9 / 5 of 19 |
| m2 | A on st053's path | 8/25 | 5/29 (-30.5 px) | 7 / 7 of 16 |
| m2 | B on st053's path | 9/25 | 7/29 (-21.2 px) | 10 / 5 of 18 |

- **ld: the rim zone.** Long ld tubes are read about as well as st053 (many end-of-movie lengths within
  1 px: g012 91 vs 92.5, g035 82 vs 83.5, g037 83 vs 83.5, g038 73 vs 73.5), but short tubes and onsets
  fail: the synthetic-only reader's front leaves the grain edge bins too early (onsets a median 1500
  frames early, 5 grains called "emerged at start"). Real grains change at their rim before a tube
  emerges (halo, swelling, focus); the synthetic field is a static real frame, so the reader never saw
  that. m2 labels (C) halve the early onsets but do not fix the short lengths.
- **m2: long tubes stop early.** A reads long m2 tubes 30 px short on average (e.g. g038 157 px read as
  7.5, g092 153 as 23.5, g054 113 as 11.5); ld labels (B) reduce but do not remove it (-21 px). On m2's
  short tubes the reader is better than st053 (8-9/25 vs 5/25).
- The synthetic validation gain (87% vs 52%) is therefore not evidence for real use: the synthetic movies
  lack exactly the structures the real failures come from (rim change before emergence on ld; whatever
  interrupts long m2 tubes). This repeats what 0.3 (synthetic length gains) and the learned maps
  (27 Sep) showed.

### Within-movie cross-validation (ld): are labels from the same movie enough?

`cv.py ld`: ld's 31 included grains split at random into two halves; each half read by the pilot fine-tuned
like B/C on the synthetic pool plus the OTHER half's labels (settings chosen on that other half), so every
grain is read once by a model that never saw its labels.

| reader | onsets | lengths | length + tip | vs st053: onsets | lengths | length + tip |
|---|---|---|---|---|---|---|
| st053 | 15/26 | 62/104 | 51/104 | | | |
| KymoReader on st053's path, ld labels of the other half | 13/26 | 46/104 | 39/104 | -2 [-6, +2] | -16 [-26, -6] | -12 [-22, -2] |
| same, on the human route (oracle) | 5/26 | 45/104 | 36/104 | -10 [-16, -4] | -17 [-29, -5] | -15 [-28, -2] |
| SparseTrack on the human route (oracle) | 7/27 | 59/104 | 53/104 | -8 [-13, -3] | -3 [-14, +8] | +2 [-11, +15] |

On identical human routes the cross-validated reader is behind SparseTrack's change reader by 14 lengths
(95% CI -25 to -3). Labels from the same movie did not help either (46/104, against 50/104 for the
synthetic-only A); fold 0 chose st053's onsets, fold 1 the reader's. So the gap is not only that the other
movie looks different: with ~15 labelled grains and this training budget the learned reader reads real
fronts worse than SparseTrack's hand-made evidence.

## What failed, in short

- Synthetic-only reader (A): best synthetic validation of all variants, worst real onsets (early at the
  rim), lengths -12 (ld) and -10 (m2) against st053.
- Real labels from the other movie (B, C): small gains over A on the test movie (m2 16 vs 13, ld 50 vs 50
  lengths; C halves A's early onsets on ld), never reaching st053.
- Real labels from the same movie (2-fold CV on ld): no gain over A.
- Keeping st053's calls and onsets (A): onsets as st053 by construction, lengths unchanged (ld 50/104,
  m2 10/54): the length deficit is not an onset artefact.
- Oracle route: the reader on the human's own route is not better than on SparseTrack's path, and not better
  than SparseTrack on the same route (except B on m2, +4, not significant). Reading the front is still the
  bottleneck, and this reader does not remove it.
- Engineering notes: the full CRF over monotone fronts was implemented and verified but too slow on the
  shared GPU (sequential 160-step loop); the per-bin front loss is a cheap stand-in that helped on synthetic
  data. MPS recompiles for every new tensor shape: batches must be padded to one fixed shape (4-5x faster).
  The frozen st053 files carry point-cloud paths for flood grains (see Data).

## Recommendation

Do **not** integrate KymoReader into SparseTrack. On both labelled movies it is worse than 0.5.3 (ld -12
lengths, CI -22 to -2; m2 -7 to -10, CIs reaching 0), onsets are worse, and neither real labels from the
other movie nor from the same movie changed that. The synthetic benchmark says the opposite (87% vs 52%),
which is the third time on this project that synthetic gains did not transfer; synthetic validation alone
should not be used to accept a reader.

What it would need before it is worth another try:

1. Synthetic movies with the real failure structures: grains whose rim changes before a tube emerges (halo,
   swelling, focus drift) - the cause of the early onsets and short-tube errors on ld - and long tubes
   interrupted like m2's (crossings, tubes leaving focus, maturing look), the cause of the -20 to -30 px
   under-reads there.
2. More labelled movies, and more labelled grains per movie: 15-30 grains per movie were not enough for the
   network to learn the real rim and tip conventions.
3. If it is retried: keep SparseTrack's onset detector (the learned onsets were the weakest part), read only
   lengths, and gate it (e.g. long tubes on sparse movies, where it matched st053 on ld) - and accept it only
   on real held-out labels, with the paired comparison used here.

Integration mechanics, if it ever earns it: a `reader="kymo"` option applied after SparseTrack has chosen
the path and rotation track (the `st` variant here): `kymo.extract` along that path (~0.1-1 s per grain),
one forward pass of a 0.12 M-parameter model (torch, already needed for the flood's maps), the monotone DP
already in `analyze.dp_front`, and a length start convention (`edge`, now also SparseTrack's default).

## Files and numbers

- `runs/kymo_reader/results/`: for every row above the predictions (`{A,B,C}_{st,route}_{movie}.json`,
  `A_stonset_st_*.json`, `cv_{st,route}_ld.json`, `st053_route_*.json`: copies of st053's files with the
  read grains replaced, flagged `reader:kymo`), per-trace CSVs (`*_traces.csv`), per-model summaries
  (`protocol_{A,B,C}.json`, `cv_ld_summary.json`), and `all_traces.csv` (every FULL trace of every
  prediction, st053 included) for re-scoring against newer SparseTrack defaults.
- `runs/kymo_reader/models/`: `pilot_front.pt`, `A.pt`, `B.pt`, `C.pt`, `cv_ld_f{0,1}.pt`, ablations.
- `runs/kymo_reader/data/`: the kymograph datasets (200 MB) and the flood centrelines.
- Commands: `python -m prototypes.kymo_reader.synth_data NAME...`, `...real_data ld m2`,
  `...flood_paths ld m2`, `...train --out ... --init runs/kymo_reader/models/pilot_front.pt --synth ... --val ...
  --steps 2500 --stop-after 500 --lr 1e-3 --front 1.0 [--real ld --p-real 0.35]`,
  `...protocol --model M --tag A|B|C --train-movie none|ld|m2`, `...cv ld`, `...tables A B C`.
