# Tube network for the flood reader, judged leave-one-movie-out (prototype, 30 Sep 2026)

**Status: done. The maps got much better; the flood as tuned reads them about as well as the existing candidate
recipe's, not clearly better.** A network trained with one labelled movie's traces and judged on the other marks
traced tubes far better than the existing recipe in both directions (movie 2: 84% of traced points vs 67%, long
tubes 84% vs 65%, young stubs 16/20 vs 14/20, rim marks before onset 8.7% of grain-bins vs 36.5%; dev movie: 98% vs
82%, marks ending within 2 px of the apex 63/104 vs 12/104). End to end (SparseTrack 0.7.0, only the network
changed), against the existing recipe (unet_d) over both directions: lengths +1, length and tip +5, onsets -3 (one-bin
shifts); no difference is significant. The same recipe trained on both movies' traces is fixed as a movie-1
candidate (`prototypes/learned_flood/models/tubes_bn_real_ld_m2.pt`, below), as a marginal one.

## Three things wrong with the maps (pixel level; no labels needed to find them)

1. **The input was normalised by the frame's median, the training crops by their own.** The network sees
   (image - median of the "before" image) / 20 grey levels. Each training crop (96 px) used its own median; the
   product uses one median for the whole 1280 x 1024 frame. Movie 2's illumination falls off across the frame: its
   local background (median over ~96 px) departs from the frame median by -77 to +38 grey levels, while training
   offsets spanned +/-0.2 units (+/-4 grey levels). g082 sits at +1.85 units: its faint tube is marked on a 260 px
   crop and not on the full frame, by either normalisation layer. Fix (label-free): inputs relative to the before
   image's local median over 96 px (`learned.local_background`, checkpoint field `bg_px`). Local minus frame median,
   in input units (`illum.py`; movie 1's field only, no labels):

   | movie | percentiles 1 / 10 / 50 / 90 / 99 | share beyond +/-0.2 | beyond +/-0.5 |
   |---|---|---|---|
   | dev movie (ld) | -0.56 / -0.36 / -0.01 / +0.19 / +0.29 | 38% | 3% |
   | movie 2 | -3.85 / -3.34 / +0.11 / +1.69 / +1.91 | 94% | 82% |
   | movie 1 | -1.01 / -0.67 / +0.03 / +0.37 / +0.48 | 59% | 17% |
2. **GroupNorm normalises every feature over the whole input.** Full-frame maps are not the maps of the crops the
   network was trained on; its sensitivity follows how busy the frame is (sparse early frames amplified - rim marks,
   young stubs "seen"; crowded late movie-2 frames damped - faint long tubes lost). Fix: BatchNorm (fixed statistics
   at inference; checkpoint field `norm`), trained with tubes_synth_v1's exact recipe otherwise.
3. **The trace targets had round caps past the apex.** Body = within 3 px of the traced polyline includes a half-disc
   of 3 px beyond the clicked apex, while the synthetic truth ends the body flat at the tube's length (the image
   blurs on past it, as real fronts run ~2.9 px past the clicked apex). Networks fine-tuned on those targets marked
   ~2 px past the apex, and the flood (which subtracts nothing from its reach) read short and mid-length tubes 2-6 px
   long. Fix: flat caps (`realdata.py --flat-cap`: beyond the apex's plane, the cap within 6 px is background).

Also fixed in the trace crops (`realdata.py`, "v2"/"v3" below; v1 = `prototypes/learned_flood/realdata.py`): the
grain's inside at the grain's place at that bin (movie 2's grains move; v1 used the census place), nothing scored
within 18 px beyond the end of PARTIAL traces (the tube goes on there), no background band on traces flagged as
touching (the band could hold the other tube), 2 exit-centred crops per trace, the trace at bins +/-1, +/-2 (tip zone
unscored), negatives at 0, 4 and 12 bins before the onset bracket (half of them at the exit).

## Harness (all in memory: no probability movie is written)

- `pixels.py`: the pixel check of `prototypes/synth_v6/recall.py` on maps computed in memory for just the bins it
  reads (reproduces the recorded tubes_synth_v1 numbers exactly; ~1 min for both movies), plus marking of long
  traces (>= 50 px) and **where the marks end relative to the traced apex** (`tip_extent`: > 0 = marked past it, which
  the flood reads as length).
- `bench.py`: `scripts/synth_bench.py --real` with the probability movie held in memory (`learned.prob_cache`
  replaced by the same computation); reproduces `runs/lab_checks_2026-09-29/final_070.json` exactly (every paired
  difference 0, both movies). Same dump format; paired bootstrap over grains.
- `train.py` (shard sets at fixed shares of each batch; `--norm`, `--bg-px`), `loo.py` (one recipe, both
  directions), `realdata.py` (trace crops v2/v3), `classes.py` (hits by trace length), `onsets.py` (onset errors in
  bins), `falsestart.py`, `versus.py` (paired, any two dumps), `table.py` (the tables below), `illum.py` (the
  illumination table above), `sheet.py`, `youngsheet.py` (maps over time round one trace).
- `sparsetrack/learned.py`: checkpoint options `norm` ("group" default / "batch") and `bg_px` (0 default); a
  checkpoint without them gives bit-identical maps (checked against the stored probability movies of both movies;
  `tests/test_learned_options.py`).

## Label-free networks (no traces used: valid on both movies)

Pixel check, dev movie (116 FULL traces; young <= 8 px: 22; rim: grain-bins before onset with >= 6 px marked within
r..r+7; "marks end": median (IQR) of where P >= 0.5 ends along the tube relative to the apex, and traces within 2 px):

| network | traced | long | stubs, tips seen | young stubs, tips | beside | rim before onset | marks end vs apex: median (IQR), within 2 px |
|---|---|---|---|---|---|---|---|
| tubes_synth_v1 (shipped), frame median | 75% | 86% | 86/116, 63 | 11/22, 10 | 5.2% | 22.2% | -0.5 (-9.2..+3.0), 20/104 |
| tubes_synth_v1, 256 px tiles | 77% | 88% | 88/116, 66 | 9/22, 10 | 6.9% | 11.1% | - |
| tubes_synth_v1, local background | 72% | 83% | 89/116, 61 | 10/22, 9 | 7.4% | 26.6% | - |
| BatchNorm, frame median | 87% | 95% | 91/116, 77 | 10/22, 10 | 7.6% | 8.7% | - |
| BatchNorm, local background | 87% | 96% | 93/116, 77 | 11/22, 10 | 8.2% | 7.5% | +1.5 (-5.6..+3.0), 25/104 |

Movie 2 (54 FULL traces; young: 20):

| network | traced | long | stubs, tips seen | young stubs, tips | beside | rim before onset | marks end vs apex: median (IQR), within 2 px |
|---|---|---|---|---|---|---|---|
| tubes_synth_v1 (shipped), frame median | 67% | 66% | 43/54, 31 | 11/20, 9 | 5.2% | 27.0% | +0.0 (-4.1..+1.0), 19/40 |
| tubes_synth_v1, 256 px tiles | 74% | 75% | 41/54, 29 | 8/20, 7 | 4.6% | 25.0% | - |
| tubes_synth_v1, local background | 72% | 72% | 36/54, 31 | 5/20, 5 | 4.7% | 13.4% | - |
| BatchNorm, frame median | 68% | 68% | 40/54, 31 | 9/20, 8 | 4.4% | 23.2% | - |
| BatchNorm, local background | 78% | 81% | 38/54, 32 | 6/20, 5 | 5.3% | 13.6% | +2.0 (-3.4..+3.6), 11/40 |

(BatchNorm = `runs/tube_net/tn_bn_syn.pt`: tubes_synth_v1's data and schedule - 6 v5 shards, 8000 steps, lr 2e-3,
64 px crops - with BatchNorm; synthetic validation loss v5 0.161 vs 0.203, v6 0.161 vs 0.185.) Movie 2's long
failures: g082@244 9% -> 100%, g092@244 25% -> 100%, g038@244 70% -> 100% (BatchNorm, local background).

End to end, BatchNorm + local background (0.7.0, only the network): dev movie 73/104 lengths (+1, 95% CI -3 to +6),
length and tip 60 (+3, +0 to +8), onsets 14/26 (-1); movie 2 18/54 (-5, -15 to +5), length and tip 15 (-2), onsets
3/18 (-1). Movie 2's long tubes gain (g064@104850 -63.5 -> -11.2 px, g092@73350 -62.1 -> +13.4), but short and
mid-length tubes are read 2-6 px long (item 3) and some young stubs are no longer "seen" (see "Why" below).

## Leave one movie out

Each recipe trains two networks: one with the dev movie's (ld) traces, judged on movie 2, and one with movie 2's
traces, judged on ld. All fine-tunes: 3000 steps, lr 1e-3, batch 32 (half synthetic, half trace crops), 64 px crops.
Recipes:

- **unet_d** (the existing candidate's recipe; `runs/learned_flood/unet_d_{ld,m2}.pt`): GroupNorm, from
  tubes_synth_v1, v5 shards + trace crops v1, frame median.
- **tn_gn_r3**: the same with trace crops v3 (flat caps and the other fixes above).
- **tn_bn_r2** / **tn_bn_r3**: BatchNorm base (`tn_bn_syn`), v5 shards + trace crops v2 (round caps) / v3 (flat
  caps), local background.
- **tn_bn_r3v6**: as tn_bn_r3, with the synthetic half split between the v5 and the v6 shards
  (`runs/synth_v6/shards/train_v6*`: bulbs, rounded tips, the grain's own change; 7 shards, 9,120 crops, incl. one on
  movie 1's field - no labels).

Pixel check on the movie each network did NOT see - movie 2 (networks trained with ld's traces):

| network | traced | long | stubs, tips seen | young stubs, tips | beside | rim before onset | marks end vs apex: median (IQR), within 2 px |
|---|---|---|---|---|---|---|---|
| unet_d_ld (ld traces; existing recipe), frame median | 67% | 65% | 46/54, 36 | 14/20, 13 | 5.7% | 36.5% | +0.0 (-6.4..+2.0), 19/40 |
| unet_d_ld, local background | 77% | 80% | 38/54, 32 | 5/20, 5 | 5.7% | 4.0% | - |
| tn_gn_r3_ld (existing recipe, new trace targets), frame median | 66% | 63% | 46/54, 33 | 14/20, 12 | 5.2% | 35.7% | -0.5 (-6.7..+1.5), 17/40 |
| tn_bn_r2_ld (BatchNorm, traces v2), frame median | 78% | 78% | 47/54, 39 | 13/20, 12 | 5.0% | 31.3% | - |
| tn_bn_r2_ld, local background | 87% | 87% | 48/54, 45 | 14/20, 14 | 3.9% | 11.3% | +2.0 (+0.5..+3.6), 19/40 |
| tn_bn_r3_ld (+ flat caps), local background | 86% | 87% | 45/54, 39 | 12/20, 11 | 4.1% | 9.0% | +0.2 (-1.1..+2.5), 21/40 |
| **tn_bn_r3v6_ld (+ v6 shards), local background** | 84% | 84% | **49/54, 42** | **16/20, 13** | 4.0% | **8.7%** | **+0.0 (-1.1..+1.6), 27/40** |

Dev movie (networks trained with movie 2's traces):

| network | traced | long | stubs, tips seen | young stubs, tips | beside | rim before onset | marks end vs apex: median (IQR), within 2 px |
|---|---|---|---|---|---|---|---|
| unet_d_m2 (m2 traces; existing recipe), frame median | 82% | 92% | 93/116, 84 | 14/22, 15 | 6.4% | 24.6% | +3.0 (-7.0..+4.5), 12/104 |
| unet_d_m2, local background | 79% | 90% | 96/116, 82 | 14/22, 15 | 7.4% | 32.5% | - |
| tn_gn_r3_m2 (existing recipe, new trace targets), frame median | 83% | 90% | 93/116, 84 | 12/22, 13 | 5.1% | 11.1% | +1.5 (-2.2..+2.5), 42/104 |
| tn_bn_r2_m2 (BatchNorm, traces v2), frame median | 97% | 99% | 110/116, 107 | 20/22, 21 | 5.8% | 19.4% | - |
| tn_bn_r3_m2 (+ flat caps), local background | 97% | 99% | 107/116, 103 | 18/22, 19 | 5.6% | 11.9% | +1.5 (+0.4..+3.0), 54/104 |
| **tn_bn_r3v6_m2 (+ v6 shards), local background** | **98%** | **100%** | 109/116, 106 | **20/22, 21** | 5.4% | 11.1% | +1.5 (+0.5..+3.0), **63/104** |

End to end on the movie each network did not see (0.7.0 defaults, only the network; paired against 0.7.0 with
tubes_synth_v1, `runs/lab_checks_2026-09-29/final_070.json`; 95% bootstrap intervals over grains) - movie 2:

| network | lengths | onsets | length and tip | paired: lengths | onsets | length and tip |
|---|---|---|---|---|---|---|
| (0.7.0: tubes_synth_v1) | 23/54 | 4/18 | 17 | | | |
| unet_d_ld (existing recipe) | 25/54 | 5/18 | 18 | +2 (-5 to +9) | +1 (+0 to +3) | +1 (-4 to +6) |
| tn_gn_r3_ld (existing recipe, new trace targets) | 23/54 | 4/18 | 17 | +0 (-5 to +6) | +0 (+0 to +0) | +0 (-3 to +3) |
| tn_bn_syn, local background (no traces) | 18/54 | 3/18 | 15 | -5 (-15 to +5) | -1 (-3 to +0) | -2 (-10 to +6) |
| tn_bn_r2_ld | 18/54 | 5/18 | 16 | -5 (-13 to +3) | +1 (+0 to +3) | -1 (-7 to +5) |
| tn_bn_r3_ld (flat caps) | 19/54 | 4/18 | 15 | -4 (-11 to +4) | +0 (+0 to +0) | -2 (-8 to +4) |
| **tn_bn_r3v6_ld (flat caps + v6)** | **24/54** | 4/18 | **22** | +1 (-6 to +8) | +0 (+0 to +0) | **+5 (-2 to +12)** |

Dev movie:

| network | lengths | onsets | length and tip | paired: lengths | onsets | length and tip |
|---|---|---|---|---|---|---|
| (0.7.0: tubes_synth_v1) | 72/104 | 15/25 | 57 | | | |
| unet_d_m2 (existing recipe) | 71/104 | 16/26 | 57 | -1 (-6 to +3) | +1 (+0 to +3) | +0 (-3 to +3) |
| tn_gn_r3_m2 (existing recipe, new trace targets) | 72/104 | 15/25 | 58 | +0 (+0 to +0) | +0 (+0 to +0) | +1 (+0 to +3) |
| tn_bn_syn, local background (no traces) | 73/104 | 14/26 | 60 | +1 (-3 to +6) | -1 (-4 to +2) | +3 (+0 to +8) |
| tn_bn_r2_m2 | 73/104 | 16/26 | 60 | +1 (-3 to +6) | +1 (+0 to +3) | +3 (+0 to +8) |
| tn_bn_r3_m2 (flat caps) | 72/104 | 14/26 | 58 | +0 (-4 to +5) | -1 (-4 to +2) | +1 (+0 to +3) |
| **tn_bn_r3v6_m2 (flat caps + v6)** | **73/104** | 14/26 | 58 | +1 (-3 to +6) | -1 (-4 to +2) | +1 (+0 to +3) |

(The intervals of the label-free network, benched on both movies at once, can differ in the last digit from the
bench's own print: the bootstrap draws one random stream per run.)

**tn_bn_r3v6 against the existing recipe directly** (`versus.py`, paired over grains): movie 2 lengths -1 (-8 to +5),
length and tip +4 (-2 to +10), onsets -1 (-3 to +0); dev movie lengths +2 (+0 to +5), length and tip +1 (+0 to
+3), onsets -2 (-5 to +0). Summed over both directions: lengths +1, length and tip +5, onsets -3.

Finer onsets (`onsets.py`: model onset bin minus the human's first visible bin, timed grains): movie 2 within 10 bins
5/19 (0.7.0) / 8 (unet_d_ld) / 9 (tn_bn_r3v6_ld); starts more than 10 bins early 7 / 5 / 4; median |error| 19 / 15 /
12 bins. Dev movie within 2 bins 12/31 / 13 / 12, within 10 bins 22 / 25 / 25, median 4 / 3 / 3. The +/-2-bin
onset differences are one-bin shifts on three flood-read dev-movie grains (g029 +3 vs +1 bins, g032 -7 vs -2) and
one movie-2 grain.

By trace length (`classes.py`), movie 2: young (<= 8 px, 20 traces) 6 (0.7.0) / 8 (unet_d_ld) / 7 (tn_bn_r3v6_ld),
8-60 px (23) 12 / 13 / 11, > 60 px (11) 5 / 4 / 6. The flat-cap BatchNorm network without v6 had 3 / 9 / 7: v6's bulbs
brought the young traces back. Dev movie: only its 3 flood-read grains (g008, g029, g032; 12 FULL traces) can
change; tn_bn_r3v6_m2 young 15, 8-60 px 42, > 60 px 16/17 (+1).

## Why the better maps are not read much better on movie 2

- **Many of 0.7.0's movie-2 hits on short tubes are coincidences of false starts.** The flood starts a tube on rim
  marks long before the tube exists and stalls at a few px, which then matches a young trace: g085 starts at bin 31
  (the human's first visible bin is 188) and reads 5.0 px at bin 194 (human 4.9); g066 at 54 (89), g038 at 43 (62),
  g106 at 15 (80). 8 of 0.7.0's 23 movie-2 length hits are on the 7 grains it starts > 10 bins early; unet_d_ld 5 of
  25 on 5 grains; tn_bn_r3v6_ld 4 of 24 on 4 grains (`falsestart.py`). On the other grains: 15 / 20 / 20. The
  shipped maps mark movie 2's rims before onset in 27% of grain-bins (36.5% for unet_d_ld), the new ones in 8.7%:
  fewer false starts, fewer coincidental hits.
- **The flood's rim rules were settled on the old maps.** Nothing inside r + 3 px is claimed, a tube starts within
  r + 3..7 px, the onset look-back walks back while P at the exit stays >= 0.25, and nothing is subtracted from the
  reach. A young stub (the annotator traces it from ~r - 1 px) is only read once the map marks it beyond r + 3 px, so
  young-trace lengths depend on how far the marks bleed outwards (`youngsheet.py`: g016@52 marked by both networks
  from bin 44; the shipped map's blob reaches further out and reads 4.7 px, tn_bn_r3_ld's compact one 3.3; human
  5.6).
- Movie 2's remaining long misses (g005@349 257 px, g038@244/349) are moving and crossing tubes: a reader problem.

## Not adopted

- Tiled inference of the shipped network (256 px tiles, Hann blend; `pixels.py --tile`): movie 2 traced 67 -> 74%,
  dev-movie rim marks halved, but young stubs 11 -> 8/20; superseded by BatchNorm + local background (not benched).
- The shipped network with the local background: movie 2 72% but young stubs 5/20, dev movie worse (72%, rim 26.6%).
- Round-capped trace targets (tn_bn_r2): movie 2 18/54 (-5), short tubes read 2-6 px long.
- The existing recipe with the new trace targets (tn_gn_r3): 0.7.0's scores (+0 on both movies).
- Brighter light-cored synthetic tubes on movie 2's field (the v6 README's measured gap) were not built: the long
  light-cored misses (g082, g092) were the normalisation, not the tube contrast.

## The movie-1 candidate (fixed 30 Sep 2026, before movie 1's tube labels)

`prototypes/learned_flood/models/tubes_bn_real_ld_m2.pt`: the tn_bn_r3v6 recipe trained once on both labelled
movies' trace crops (v3) - 25% v5 shards, 25% v6 shards, 50% trace crops of ld and m2; 3000 steps from
`runs/tube_net/tn_bn_syn.pt`; BatchNorm; `bg_px` 96 in the checkpoint. Exact commands:
`prototypes/learned_flood/models/tubes_bn_real_ld_m2.recipe.sh`. sha1 and scoring command: see
"Candidate record" at the end.

It is a marginal candidate: in leave-one-out it is level with the existing candidate's recipe on lengths (+1 over
both movies), ahead on length and tip (+5) and on the finer movie-2 onsets, behind by three one-bin onset shifts,
none significant; far ahead on every pixel measure. It needs `learned.py`'s `norm`/`bg_px` options, which SparseTrack
0.7.0 as tagged does not have (it refuses the file): score it from the 0.7.0 tag with the `learned.py` commit of this
branch cherry-picked.

## Honesty

- Both labelled movies are development sets: the flood's rules were settled while looking at movie-2 grains with the
  shipped maps, and the v6 generator was designed looking at both movies' young tubes (no labels used in training
  it). Leave-one-out here means a network never saw the traces of the movie it is judged on; the recipes themselves
  (4 of them) were chosen in sequence while looking at these leave-one-out results, which favours the last one a
  little. The honest test is movie 1.
- The local-background scale (96 px) was set to the training crop size, not tuned.

## Files

`runs/tube_net/`: networks (`tn_*.pt`), pixel checks (`pix_*.json`, `pix2_*.json` with the tip-extent metric),
bench dumps (`e2e_*.json`; their `pred` fields point to predictions in the session scratchpad), trace shards
(`shards/real2_*`, `real3_*`), logs.

## Candidate record

- File: `prototypes/learned_flood/models/tubes_bn_real_ld_m2.pt`, **sha1 91cb95715eae5122d3f70eb89aa7246ce5b34aaa**,
  trained 30 Sep 2026 by `tubes_bn_real_ld_m2.recipe.sh` (base `runs/tube_net/tn_bn_syn.pt`; log
  `runs/tube_net/logs/train_tubes_bn_real_ld_m2.log`; synthetic validation at the end: v5 loss 0.231, recall 77.4%,
  false marks 0.14%; v6 0.155, 87.8%, 0.18%). Checkpoint fields: `norm` "batch", `bg_px` 96. Not to be changed.
- Sanity only (it saw both movies' traces, so these are not scores): it loads with BatchNorm and `bg_px` 96 from the
  file alone, a GroupNorm build (SparseTrack 0.7.0) refuses it, and its maps mark 99% / 98% of the traced points of
  the dev movie / movie 2.
- Scoring on movie 1, once, after its blind labels exist (0.7.0 plus the `learned.py` commit "learned: checkpoint
  options for BatchNorm networks and inputs relative to the local background", ba89c96 on this branch; it changes
  nothing else and applies cleanly on the tag):

```bash
git worktree add ../tt-0.7.0-bn sparsetrack-0.7.0
cd ../tt-0.7.0-bn && git cherry-pick ba89c96
shasum ../TubeTracker/prototypes/learned_flood/models/tubes_bn_real_ld_m2.pt   # 91cb95715eae5122d3f70eb89aa7246ce5b34aaa
../TubeTracker/.venv/bin/python -c "
from sparsetrack.analyze import Params, analyze
analyze('../TubeTracker/runs/sparsetrack/m1', '../TubeTracker/runs/sparsetrack/m1_frozen_0.7.0_bn',
        grains_path='../TubeTracker/benchmark/labels/m1_v1.json',
        params=Params(model='../TubeTracker/prototypes/learned_flood/models/tubes_bn_real_ld_m2.pt'))"
cd ../TubeTracker && .venv/bin/python -m sparsetrack eval --labels benchmark/labels/m1_v1.json \
    --pred runs/sparsetrack/m1_frozen_0.7.0_bn/predictions.json --out benchmark/reports/m1_v1_scores_0.7.0_bn.md
```
