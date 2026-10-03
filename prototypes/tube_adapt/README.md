# Adapting the tube network to a new movie from a few corrected grains (prototype, 3 Oct 2026)

**Question.** Can the tube-probability network adapt to a new movie from a few of that movie's own human traces (as a
lab would correct a few grains in the app), especially if each trace is propagated through time to the bins around it?
Movie 1 (m1) is the clean test: the shipped network (`sparsetrack/models/tubes_bn_real_ld_m2.pt`, ld + m2 traces) never
saw it. **Answer: the maps adapt strongly and onsets improve significantly; lengths improve a little (not significant) and length-and-tip not at all, while rim marks before onset rise.** Details in "Results".

## Method

- **Folds** (`common.folds`): movie 1's 30 labelled grains split by grain, balanced on label counts only (best of
  20,000 random splits): 2 folds of 15 grains, each with 25 scored FULL traces, 7 long (>= 50 px), 13-14 young
  (<= 8 px), 43 traced tubes. Fold `k2f0` holds out g003 g005 g006 g007 g014 g015 g023 g029 g030 g033 g036 g044 g060
  g061 g066 (trains on the other 15); `k2f1` the reverse.
- **Fine-tuning** (`finetune.py`) from the shipped checkpoint: 1500 steps, batch 32, AdamW lr 5e-4 one-cycle, 64 px
  random crops of the 96 px samples, 0.8.0's losses and augmentation, BatchNorm statistics frozen (full-frame inference
  keeps the shipped normalisation; only weights adapt), checkpoint options (`norm` batch, `bg_px` 96) kept. Half of
  every batch is replay of the shipped network's own training data: ld + m2 trace crops v3 25%, one v5 synthetic shard
  (m2 field) 12.5%, one v6 shard (m1 field, label-free) 12.5% (one shard each: the laptop has 8 GB).
- **(a) traced bins only**: the training grains' crops exactly as `prototypes/tube_net/realdata.py` v3 builds movie 1's
  (`--flat-cap --over-grain 0.6 --follow`: trace, exits, the trace at +/- 1-2 bins with the tip zone unscored,
  negatives 0/4/12 bins before onset), 50% of each batch.
- **(b) with temporal propagation** (`propagate.py`): traced crops 20%, propagated tube crops 20%, propagated negatives
  10%. Between consecutive traces b1 < b2 of a grain, the b1 tube is tube at every bin in between: its route is carried
  forward by DIS optical flow (PRESET_MEDIUM, 3-bin-mean registered frames, 1-99 percentile uint8 window, composed in
  10-bin steps; `runs/lab_checks_2026-09-29/short_interval/flowcheck.py`) and the first L(b1) px of the b2 route are
  carried back; where both carries are reliable (forward-backward flow error <= 1.5 px per step) they must agree
  within 3 px (median closest-point distance) and are blended by time; where only one is, it is used within 30 bins
  of its own trace. The rest of the b2 route (where the tube grows in between) is unscored within 8 px, the carried
  apex within 10 px; body = within 2 px of the route, unscored 2-8 px, background 8-16 px (not on traces touching
  anything). Intervals where the tube got shorter (by > max(5 px, 15%): g065, g064, g005 late) are skipped. After the
  last trace the tube stays (carried forward while the flow is reliable; at most 35 bins when a later bin was answered
  "unsure"). Before the onset bracket (every 2nd bin up to `last_absent_bin` - 1) the grain's first route, carried
  back with the grain, is background ("no tube at the exit"). The grain's inside at that bin is background (unscored
  for traces from face-on pores), with the grain's centre carried by the same flow.
- **(c)** as (b) plus skeleton recall (Kirchhoff et al. 2024, arXiv 2404.03010): 1 - sum(P S) / sum(S) over the tubed
  skeleton of the labelled body, weight 1.
- **Judged on the held-out grains only**, maps in memory: the pixel check of `prototypes/tube_net/pixels.py`
  (`evaluate.py pix`) and end to end with SparseTrack 0.8.8's defaults and only the network changed
  (`evaluate.py e2e`: `bench.py`'s in-memory maps, `analyze(..., only=held-out grains)`), paired over grains against
  the 0.8.8 predictions on the same grains (95% bootstrap intervals over grains). Pooled = each grain judged once, by
  the network of the fold that held it out. Forgetting: the pixel check on all of movie 2 (in the shipped network's
  training, so its 98% is in-sample).

## Results (written up by the main session from `table.py k2f0 k2f1 --variants a b c`; the agent was stopped first)

Held-out movie-1 grains, each judged once by the network of the fold that did not see it (30 grains, 50 full traces),
against the shipped network on the same grains; paired 95% bootstrap intervals over grains.

| variant | traced tube marked | long tubes marked | rim marks before onset | lengths in tolerance | length and tip | onsets |
|---|---|---|---|---|---|---|
| shipped network | 75% | 71% | 3.5% | 15/50 | 15 | 8/27 |
| (a) traced bins only | 90% (+15 pts, +3..+30) | 89% | 11.0% | 22/50 (+7, -1..+15) | 17 (+2, -6..+11) | 14/27 (+6, +2..+11) |
| (b) + temporal propagation | 93% (+17, +4..+35) | 92% | 17.8% | 18/50 (+3, -4..+11) | 15 (+0, -6..+6) | 17/26 (+9, +4..+14) |
| (c) + skeleton recall | 96% (+21, +6..+39) | 97% | 21.6% | 21/50 (+6, -1..+14) | 17 (+2, -6..+11) | 14/27 (+6, +1..+11) |

Per fold the end-to-end differences vary (fold k2f0: lengths +0..+2; fold k2f1: +1..+6), so 15 corrected grains are
not yet a reliable length gain. Forgetting on movie 2 (in the shipped network's training): pixel marks change by
-0.8 to +0.3 points; negligible.

Reading: per-movie fine-tuning from 15 corrected grains makes the maps mark nearly all of the held-out grains' tubes
and fixes many onsets, but the flood (whose start and stop rules were tuned on the shipped maps) does not turn the
better maps into better lengths or tips, and the adapted maps mark more of the rim before germination (false starts).
Next: re-tune or replace the flood's start rule for adapted maps (e.g. onsets from the learned tip detector,
prototypes/tip_detector), then re-measure; a learning curve (5/10/15 grains) was not run.
