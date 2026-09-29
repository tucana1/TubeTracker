# Synthetic v6: young tubes as bulbs, and a grain body that changes (prototype, 29 Sep 2026)

**Status: done (29 Sep 2026). Recommendation: keep the generator options; do not replace
`tubes_synth_v1.pt` with any network trained here** - they see young bulbs and tips much better on the pixel level,
but end to end none is better than the shipped network on either movie (see the end of this file).

## Why

On movie 2, 18 of SparseTrack's 31 length misses are young stubs (<= 8 px) that the tube network
(`sparsetrack/models/tubes_synth_v1.pt`, trained only on synthetic v5 movies) never marks. Young real tubes
(`runs/lab_checks_2026-09-29/young_stubs_m2.png`, and `runs/synth_v6/real_m2_a.png`, `real_m2_b.png`,
`real_ld_a.png`) differ from synthetic v5 ones in two ways:

1. A young tube emerges as a rounded **bulb** 5-8 px across (dark, or light-cored with a dark outline); v5 drew a
   short thin line. Measured across real young bulbs in the change image: dark bulbs 5-8 px FWHM, peak -25 to -55
   grey levels; light-cored ones a bright centre with dark walls ~3-4 px off-axis; along the tube, dark signal falls
   to half about 0.5 px beyond the clicked apex, and a light-cored tip's dark wall wraps round ~1 px beyond it.
2. Near the rim the change is dominated by the **grain itself**. Measured on both labelled movies
   (`grainstats.py`, bins before each grain's onset, relative to the "before" image the network sees):
   - grains shift a median 0.35-0.42 px against the registered field (p90 2-3 px);
   - the focus changes field-wide: movie 2 is ~1 px blurrier than its reference until bin ~60, then ~1.5 px sharper;
     the dev movie's early bins are ~0.5 px sharper, later ones up to ~1.25 px blurrier;
   - the rim's own change is 4-5 grey levels RMS (p90 13-18), ~10x the background noise in a bin;
   - interiors change by a median 5-10% of their contrast around germination (p90 27-77%).
   v5 pasted a static grain (only 12% of grains drift, rigidly).

## What changed in the generator (`sparsetrack/synth.py`, options off by default)

New `SynthConfig` fields (their defaults switch them off; v1-v5 reproduce exactly - checked frame by frame at 8
times, noise included, and the truth files, for v1-v5 on the dev field and v5 sized like movie 2 on movie 2's field):

| field | v6 value | what |
|---|---|---|
| `tip_bulb`, `tip_bulb_px` | (0.5, 1.2), (2, 4) px | width grows towards the apex: w(s) = w (1 + a exp(-(L - s)/l)), capped at 2.6 (dark) / 1.25 (light-cored) profile widths so wide tubes and fat stubs are not blown up |
| `tip_round` | True | a closed, rounded apex: beyond the dome's centre the cross-section is evaluated at the distance to that centre (dark: centred on the apex; light-cored: 2.5 widths behind it, so the wall wraps ~1 px beyond the apex) |
| `body_change`, `body_bins`, `body_lead_bins`, `p_body`, `p_body_idle` | +/-(0.10, 0.25), (10, 30) bins, (-30, +4) bins, 0.8, 0.3 | a germinating grain's interior contrast changes gradually, starting up to 30 bins before its onset (real rims change before a tube emerges: the kymograph reader's onsets on the dev movie run early for that reason, `prototypes/kymo_reader/README.md`) or up to 4 after, stronger towards the pore; 30% of the other grains change at a random time. (Round 1: -2 to +4 bins.) |
| `focus_px`, `focus_bins`, `focus_max_px` | 3.0, 4 bins, 3.0 | field-wide focus drift (blur sigma, either way: sharper is an unsharp mask; beyond 2 px the halo change is extrapolated linearly), a wander smoothed over `focus_bins` plus, in half the movies, a refocus step |
| `focus_grain_px` | 1.2 | + each grain's own focus / halo drift |
| `jitter_px` | (0.3, 4.0) px | each grain's own slow wander, largest excursion log-uniform in this range (real per-grain maxima span ~0.2-4 px; grains that drift far are v2's `p_move`) |

Also: a v6 tube is not drawn over its own grain (it fades out between 2.5 and 1 px inside the census circle, the
grain's visible edge); every grain is a sprite (sub-pixel placement with cubic interpolation; the holes are
inpainted); v6 random draws come from their own generator, so a v6 scene has the same tubes as the v5 scene of the
same seed. The truth rasters for v6 are `truth6.py` (body widened at the tip, rounded, not over the grain).

## Look comparison and calibration of the grain's own change

`runs/synth_v6/compare_young_tubes.png`: real young tubes (movie 2: g052, g033, g005, g060; dev movie: g031, g038)
next to synthetic v6 ones, as the network sees them (registered bin / change from the "before" image, +/-30 grey
levels) at -3, 0, +3, +6, +10, +20 bins from the first visible bin. The bulbs match well (e.g. real g060 / synthetic
s007: a dark bulb at the rim; real g005, ld g031 / synthetic s005, s008: light-cored rings).
(`runs/synth_v6/preview_r1/` and the first version of the sheet show the round-1 settings; the sheet was redrawn with
the final ones.)

The grain's own change was then measured, not just looked at (`crescent.py`, `graindyn.py`): RMS of the change inside
r + 3 px of the grain centre (bins before its onset, and never-germinating grains; field-wide brightness removed),
against the contrast of a young tube (largest |change| on its first trace, <= 8 px):

| | grain change, median by bins since the reference (0-30 / 30-60 / 60-90 / 90-120 / 120-150 / 150-180) | young tube contrast (median) |
|---|---|---|
| real, dev movie (ld) | 5.0 / 8.2 / 9.8 / 17.2 / - / - | 33.6 |
| real, movie 2 (m2) | 2.3 / 4.2 / 6.4 / 9.0 / 10.0 / 12.2 (21.1 at 180-210) | 26.6 |
| synthetic, round-1 settings (m2 field, seed 40, noise-free) | 1.6 / 3.1 / 3.8 / 4.3 / 5.4 / 6.3 | - |
| synthetic, final v6 (same scene, per-grain focus 1.0 in this check; final 1.2) | 2.0 / 4.1 / 5.0 / 6.5 / 7.8 / 7.4 | - |

In the round-1 training crops themselves the grain change inside r + 3 px was a median 2.5 grey levels (p90 8.0; v5
crops 1.8, p90 21.9 from its few drifting grains), against 5.6-5.9 (p90 17-21) in the real movies: about half of
real at matched times. The wander (median ~0.4 px) was already right against the measured displacements; what was
missing was focus / halo change (real movie 2 is 1.25-1.75 px out of focus for most of the movie). Round 2 and the
final `v6` preset therefore use a larger, faster focus drift (`focus_px` 3, `focus_max_px` 3 - beyond the last blur
level the halo change grows linearly - `focus_bins` 4, per-grain 1.2) and a wider wander (0.3-4 px). Round 1's
settings stay available as overrides (`recipe.R1`).

After the change (final settings): round-2 training crops on movie 2's field, grain change median 3.1 (p75 5.7, p90
10.3; round 1: 2.5 / 4.4 / 8.0). On the final previews, measured exactly like the real movies (grain change inside
r + 3 px vs young-tube contrast, median):

| | grain change p50 / p75 / p90 | young tube contrast | ratio |
|---|---|---|---|
| real ld | 5.6 / 9.8 / 16.9 | 33.6 | 0.17 |
| synthetic, dev field (preview seed 2) | 3.8 / 10.9 / 16.5 | 25.4 | 0.15 |
| real m2 | 5.9 / 11.9 / 21.0 | 26.6 | 0.22 |
| synthetic, movie 2's field, light-cored look (preview seed 1) | 2.5 / 6.4 / 13.1 | 11.8 | 0.21 |

So the grain's change relative to a young tube is now about real, but on movie 2's field both are about half as
strong as in the real movie: young synthetic light-cored tubes are too faint there (the 45-bin previews also compress
the wander in time). Not corrected here.

## Training (no human labels)

Shards: `recipe.py` (`shards.py` renders only the bins a shard uses: 90 of 351 bins on movie 2's and movie 1's
fields, 60 of 176 on the dev field; 16 crops per bin, 15% of them centred on a young tube). Round 1: movie 2's field
seeds 40 (movie-2 sizing) and 41 (+ light-cored, wider look), dev field seed 30, validation dev seed 33 (30 bins);
round 2 (final v6 settings): movie 2's field seeds 42 (look) and 43, dev field 31, movie 1's field 50.

Both networks were fine-tuned from `tubes_synth_v1.pt` with `train6.py` (3000 steps, lr 1e-3, batch 32, half of each
batch from the v5 shards `tubes_synth_v1` was trained on - `runs/learned_flood/shards/train_v5s[0-2]`,
`train_v5m2s1[0-2]` - so the v5 scenes are not forgotten):

- **A** `runs/synth_v6/tubes_v6r1_ft.pt`: round-1 shards (3,840 crops);
- **B** `runs/synth_v6/tubes_v6_ft.pt`: rounds 1 + 2 (8,640 crops);
- **control** `runs/synth_v6/tubes_v5_ft_control.pt`: the same fine-tune on the v5 shards only (no v6 data), to
  separate what the v6 data does from what 3000 more training steps do;
- **C** `runs/synth_v6/tubes_synth_v6.pt`: trained from scratch like tubes_synth_v1 (8000 steps, lr 2e-3), on the
  seven v6 shards only (8,640 crops; tubes_synth_v1 had 7,200 v5 crops).

Synthetic validation (480 crops each; total loss / body recall / false marks at P >= 0.5):

| shard | tubes_synth_v1 | A | B | control | C |
|---|---|---|---|---|---|
| v6 dev field seed 33 | 0.185 / 89.8% / 0.39% | 0.149 / 88.6% / 0.20% | 0.155 / 89.1% / 0.23% | 0.178 / 89.5% / 0.36% | 0.129 / 91.2% / 0.20% |
| v5 dev field seed 3 | 0.203 / 83.0% / 0.17% | 0.248 / 74.8% / 0.14% | 0.228 / 79.8% / 0.17% | 0.182 / 83.9% / 0.14% | 0.312 / 69.5% / 0.19% |

## Results on the human traces (both movies are DEVELOPMENT sets)

Both labelled movies have been used to develop SparseTrack (m2 since 27 Sep), and I looked at movie-2 (and dev-movie)
young tubes to design the v6 generator. The generator and the networks never saw a label, but these are not held-out
scores; an honest score needs a movie labelled after this work (movie 1 is the candidate).

### Pixel check (`recall.py`)

Every FULL trace not touching another tube or grain, from the grain's visible edge; `realpix` is the earlier measure
(points beyond r + 4 px), for comparison with the 78% / 68% quoted for tubes_synth_v1. "stub" = the trace has >= 1 px
called tube beyond the edge; "tip" = P >= 0.5 within 2.5 px of the traced apex; "beside" = points 8-14 px either side.

| movie | model | all: realpix | all: stub | all: tip | all: beside | first trace per grain: points / stub / tip | young (<= 8 px): points / stub / tip |
|---|---|---|---|---|---|---|---|
| ld | tubes_synth_v1 (shipped) | 78% | 86/116 | 63/116 | 5.2% | 40% / 12/31 / 11/31 | 48% / 11/22 / 10/22 |
| ld | A: v6 round 1, fine-tuned | 89% | 101/116 | 90/116 | 6.3% | 55% / 19/31 / 22/31 | 60% / 14/22 / 16/22 |
| ld | B: v6 rounds 1+2, fine-tuned | 90% | 105/116 | 99/116 | 7.3% | 61% / 22/31 / 23/31 | 68% / 16/22 / 18/22 |
| m2 | tubes_synth_v1 (shipped) | 68% | 43/54 | 31/54 | 5.2% | 60% / 10/18 / 8/18 | 55% / 11/20 / 9/20 |
| m2 | A: v6 round 1, fine-tuned | 54% | 38/54 | 29/54 | 3.5% | 45% / 8/18 / 7/18 | 36% / 9/20 / 9/20 |
| m2 | B: v6 rounds 1+2, fine-tuned | 55% | 41/54 | 32/54 | 3.6% | 62% / 11/18 / 9/18 | 57% / 13/20 / 11/20 |
| ld | control: v5 only, fine-tuned | 79% | 92/116 | 71/116 | 6.6% | 43% / 15/31 / 15/31 | 51% / 13/22 / 14/22 |
| m2 | control: v5 only, fine-tuned | 64% | 46/54 | 33/54 | 5.4% | 63% / 12/18 / 10/18 | 60% / 14/20 / 12/20 |
| ld | C: v6 only, from scratch | 94% | 106/116 | 99/116 | 5.1% | 74% / 24/31 / 25/31 | 85% / 19/22 / 20/22 |
| m2 | C: v6 only, from scratch | 53% | 42/54 | 35/54 | 3.6% | 65% / 12/18 / 11/18 | 62% / 14/20 / 13/20 |

Rim marked before any tube exists (grain-bins, up to 2 bins before the human's first visible bin, with >= 6 px of
P >= 0.5 within r to r + 7 px of the grain centre: a chance for the flood to start on the grain itself):
ld 22.2% (tubes_synth_v1) / 31.0% (A) / 32.5% (B) / 48.0% (C); m2 27.0% / 14.8% / 23.7% / 8.5%.

What the maps show (`runs/synth_v6/maps_m2_v6r1.png`, `maps_m2_v6ft.png`: bin, change, P of each model, human trace
in green): several of tubes_synth_v1's movie-2 "young-stub hits" are rim artefacts overlapping the trace (a whole
ring painted on g066, a rim crescent on g085); the v6 networks paint much less of the rim and find bulbs the shipped
model misses (g005, g082, g060, g106), but they lose some long, faint light-cored movie-2 tubes (g054, g043 at bin
244) - hence the lower all-trace recall on movie 2.

### End to end (`scripts/synth_bench.py --real`, SparseTrack 0.6.0 defaults, only the network changed)

`--set model=runs/synth_v6/<model>.pt --baseline runs/lab_checks_2026-09-29/final.json` (0.6.0 with tubes_synth_v1).
A and B ran at commit 1ef0481, the control and C at 111eaed. Between the 0.6.0 tag and these only default-off
options were removed and the change reader's background level was cached; the baseline re-run at 62f8106 (analysis
code as at 111eaed) reproduced final.json exactly (every paired difference 0), so the runs read grains identically
apart from the network. 95% intervals: paired bootstrap over grains.
The network only matters for grains the hybrid reader floods (crowded or noisy ones: most of movie 2, few on ld).

| movie | model | onsets | lengths (FULL traces) | length and tip | paired vs baseline: onsets / lengths / length and tip |
|---|---|---|---|---|---|
| ld | baseline (tubes_synth_v1) | 15/26 | 69/104 | 56 | - |
| ld | A (round 1) | 15/27 | 69/104 | 56 | +0 (-3 to +3) / +0 (-3 to +3) / +0 (-3 to +3) |
| ld | B (rounds 1+2) | 14/26 | 69/104 | 58 | -1 (-4 to +2) / +0 (-6 to +6) / +2 (-2 to +7) |
| m2 | baseline (tubes_synth_v1) | 5/19 | 25/54 | 18 | - |
| m2 | A (round 1) | 5/19 | 16/54 | 13 | +0 (-3 to +3) / **-9 (-16 to -1)** / -5 (-10 to +0) |
| m2 | B (rounds 1+2) | 5/19 | 24/54 | 16 | +0 (+0 to +0) / -1 (-6 to +5) / -2 (-6 to +2) |
| ld | control (v5 only) | 15/26 | 67/104 | 55 | +0 (-3 to +3) / -2 (-7 to +2) / -1 (-3 to +0) |
| m2 | control (v5 only) | 5/19 | 22/54 | 16 | +0 (+0 to +0) / -3 (-9 to +3) / -2 (-7 to +2) |
| ld | **B vs control** (what the v6 data adds) | | | | -1 (-3 to +0) / +2 (+0 to +5) / +3 (+0 to +8) |
| m2 | **B vs control** | | | | +0 (+0 to +0) / +2 (-3 to +8) / +0 (-5 to +6) |
| ld | C (v6 only, from scratch) | 14/26 | 65/104 | 53 | -1 (-4 to +2) / -4 (-13 to +2) / -3 (-10 to +2) |
| m2 | C (v6 only, from scratch) | 4/18 | 20/54 | 17 | -1 (-3 to +0) / -5 (-11 to +0) / -1 (-5 to +2) |

Young traces (human length <= 8 px; `young_e2e.py`, same scorer and tolerances, paired over grains):

| movie | model | lengths | length and tip |
|---|---|---|---|
| ld (20 traces) | baseline | 15 | 11 |
| ld | A | 14 (-1, -3 to +0) | 11 (+0) |
| ld | B | 15 (+0, -3 to +3) | 12 (+1, +0 to +3) |
| m2 (20 traces) | baseline | 7 | 4 |
| m2 | A | 5 (-2, -7 to +4) | 4 (+0, -3 to +3) |
| m2 | B | 9 (+2, +0 to +6) | 4 (+0, -3 to +3) |
| ld | control | 14 (-1, -5 to +2) | 10 (-1, -3 to +0) |
| m2 | control | 7 (+0, -4 to +5) | 5 (+1, +0 to +3) |
| ld | B vs control | +1 (+0 to +3) | +2 (+0 to +5) |
| m2 | B vs control | +2 (+0 to +5) | -1 (-3 to +0) |
| ld | C | 13 (-2, -5 to +0) | 10 (-1, -3 to +0) |
| m2 | C | 5 (-2, -5 to +0) | 5 (+1, +0 to +3) |

The control run (its bench ran at commit 111eaed; the baseline re-run above confirms that code reads grains as 0.6.0
did) shows that 3000 more fine-tuning steps on
v5 alone already find more young stubs on the pixel level (movie 2 young stubs 11 -> 14 of 20) - including g005, whose
young traces it also turns into hits - and cost a little end to end. Against that control, the v6 data adds a little
(dev movie +2 lengths, +3 length and tip; movie 2 +2 lengths, young +2), all at the edge of significance.

Which grains changed with B: movie 2 - g005's two young traces become hits (its bulb is now seen: +2), while three
long tubes at bin 244 are read shorter (g043 55 px: -17 px, a faint grey tube the network now paints only in part;
g016 and g060 just outside tolerance: -16.4 of 123 px, -5.5 of 51 px); dev movie - crowded g008 gains 2 lengths, 2
tips and its onset, g032 loses 2 lengths and its onset.

## Files

| file | what |
|---|---|
| `sheet.py` | contact sheets (bin / change from "before") around germination, real or synthetic |
| `grainstats.py` | the grain's own change per bin: displacement, signed blur, interior, rim |
| `preview.py` | short synthetic movies with early onsets, encoded and binned like the real ones |
| `compare.py` | the real-vs-synthetic comparison sheet |
| `truth6.py` | truth rasters for v6 scenes |
| `shards.py` | training shards rendering only the bins a shard uses (x264-encoded, binned, registered) |
| `recipe.py` | the shards that were built |
| `train6.py` | the trainer (float16 shards in memory; optional mixing of a second shard set) |
| `recall.py` | pixel check against the human traces, including young traces (from the grain's visible edge); rim marks before onset |
| `evalmodel.py` | a model's probability caches on both movies (next to their caches) + the pixel check |
| `valloss.py` | synthetic validation losses, body recall and false marks |
| `mapsheet.py` | two models' probability maps side by side on real traces |
| `crescent.py`, `graindyn.py` | the grain's own change inside r + 3 px, real vs synthetic (crops, previews, noise-free by component) |
| `young_e2e.py` | end-to-end hits on young traces, paired over grains |

Outputs in `runs/synth_v6/`: `compare_young_tubes.png` (the comparison sheet), `real_*.png` (real young tubes),
`preview/` (final v6 previews) and `preview_r1/` (round-1 settings), `shards/` (training shards), the networks
(`tubes_v6r1_ft.pt` A, `tubes_v6_ft.pt` B, `tubes_v5_ft_control.pt`, `tubes_synth_v6.pt` C), `recall_*.json`,
`e2e_*.json` (`synth_bench --dump-real`), `pred_*.json` (the benches' predictions, kept for rescoring), `maps_*.png`,
`logs/`. The networks' probability caches (0.66 GB each; `runs/sparsetrack/{ld,m2}/prob_<model>`) were deleted to
save disk: `python -m prototypes.synth_v6.evalmodel runs/synth_v6/<model>.pt` (or the first bench run) rebuilds them
in about 3 minutes.

## What it means, and the recommendation

- **The generator (keep):** v6's young tubes look like real ones (bulbs, rounded and light-cored tips), and the
  grain's own change is now measured and, relative to a young tube, about as strong as in the real movies (both are
  still weaker than real on movie 2's field). v1-v5 are unchanged. Networks trained on it find young tubes and tips
  far better on the pixel level: from scratch (C), young traces with the tip seen go from 10/22 to 20/22 on the dev
  movie and from 9/20 to 13/20 on movie 2, and movie 2's rim is marked before any tube exists in 8.5% of grain-bins
  instead of 27%.
- **The networks (do not adopt):** none of them reads more lengths than the shipped tubes_synth_v1 end to end.
  Best is B (v6 fine-tune): dev movie lengths +0 (-6 to +6), length and tip +2 (-2 to +7); movie 2 lengths -1 (-6 to
  +5), length and tip -2 (-6 to +2), young traces +2 (+0 to +6). A and C are worse (movie 2 -9 and -5). Against a
  control fine-tuned on v5 alone, the v6 data adds a little (dev movie +2 lengths, +3 length and tip; movie 2 +2),
  at the edge of significance; much of B's movie-2 young-stub gain (g005) comes from the extra training alone.
- **Why better maps did not give better lengths:** (1) all v6 networks lose faint, long, light-cored movie-2 tubes
  (g043, g054; movie-2 length bias -8 -> -10 to -15 px), so late lengths are read short; synthetic light-cored tubes on
  movie 2's field are about half as bright as real ones (measured above) - the next thing to fix in the generator.
  (2) They mark the dev movie's rims more before the human's first visible bin (22% -> 31-48% of grain-bins), which
  gives the flood early starts. (3) The flood's rules (start band, bridges, P >= 0.5, look-back) were settled on
  tubes_synth_v1's maps while looking at movie-2 grains; a different network is read by rules tuned to the old one.
- **Honesty:** both labelled movies are development sets, and I looked at movie-2 and dev-movie young tubes while
  designing v6 (no labels were used for training). Nothing here is a held-out result; movie 1 (unlabelled; one v6
  training shard used its field, no labels) is the candidate for an honest test.
- **Next:** brighter light-cored young tubes on movie 2's field (the measured contrast gap), then retrain C-style and
  re-check the maps on g043/g054 first; if the pixel gains then survive end to end, re-tune the flood's start rules on
  the new maps (on one movie, checked on the other) before scoring on a freshly labelled movie.

