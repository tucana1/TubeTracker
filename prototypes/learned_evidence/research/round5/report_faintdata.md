# Round 5, agent "faintdata": faint synthetic tubes in the evidence network's training

> The agent's report of 27 Sep 2026, as delivered. Paths such as `SCR/agents/faintdata/` are the cloud session's; the
> code, the training records (`records.jsonl`), logs, scores and the best model (M1c,
> `models/unet_fd_scratch_cal.pt`) are in `faintdata/` beside this file. Development numbers only: the sealed held-out
> set was never opened.

**Answer: no.** Adding faint and intermediate synthetic movies to the evidence network's training:
- raises faint lengths by +7.5 to +8.3 pt, with the 95% interval above 0;
- costs thin lengths (−2.5 pt over the 9 thin movies, interval below 0), thin onsets (−3.4 to −4.9 pt) and wide
  lengths (about −3.5 pt);
- on the real movie, gives more gross breaks than fixes, mostly new early onsets.

The pooled thin+faint change is about zero. **Freeze: none.**

## Training data

- **v2's 10 shards:** v5 s0, 1, 2, 9, 10, 11, v2 s0, 1, v3 s0, 2 (11,200 samples).
- **6 new movies, seeds 50–55 only,** rendered on `runs/sparsetrack/sample_movie/cache`:
  - v5faint s50–53 = `preset("v5", amplitude=(0.35, 0.7))`;
  - v5mid s54, s55 = `preset("v5", amplitude=(0.5, 1.0))`.
- **Preparation:** `stack.prepare(frames_per_bin=25, ref_bins=3, ref_start=0)` + `write_census(cache, 3, False)`.
  Shards come from `data.build` with its defaults.
- 1,120 samples per new shard, 17,920 in total (25% faint, 12.5% intermediate). Every seed and its full SynthConfig
  is in `records.jsonl`.
- The v5w shards (development seeds) were not used, and no held-out seed was touched.

## Models

All three use `train5.py`: train.py's UNet, augmentation, losses, AdamW (weight decay 1e-4), OneCycle schedule and
batch sampling, plus checkpoints and `--init`. Batch 32, crop 64, widths 16-32-64-128, seed 0.

- **M1 `fds`, from scratch** (the primary test): 6000 steps, lr 2e-3. sha1 de7e7e2c84f7c6b64ab3b9bb2988e121765e807a,
  fingerprint 96799d79e2dee936. It ran partly at 1 thread from saved checkpoints, with every random stream restored.
- **M2 `fdft`, continued from v2** on the same 16 shards: 2000 steps, OneCycle peak lr 5e-4. sha1
  229526a96f426f6de8936b0468a153edca5651e1.
- **M1c `fdsc`: M1 with −0.696 added to the tube-channel logit.** `calib.py` sets the shift so that the model marks
  as many pixels at P > 0.5 as v2 does on v2's own shards (M1 marked 4.6% more). The shift is equivalent to
  thresholding M1 at P > 0.667. sha1 5c988c6bd14f04684305ff05f3245e76ccb42fd9 (kept here).
- **Training checks:**
  - M1's final training loss was bce 0.039, dice 0.130 (v2 reached 0.021 / 0.083 on its own data).
  - On shard crops at P > 0.5, M2 raised faint recall from 0.40–0.69 to 0.55–0.75. Precision on v2's shards fell
    from 0.82–0.92 to 0.72–0.87.
- A wider network was not tried: no compute left.

## Development results, all 15 movies, paired 95% intervals against the default

Each cell: change in counts within tolerance [interval] / total = percentage points.

### M1c (best pooled)

| group | orig lengths | orig onsets | human_t2 lengths | human_t2 onsets |
|---|---|---|---|---|
| thin (9 movies) | −45 [−85, −5] / 1807 = −2.5 pt [−4.7, −0.3] | −7 [−16, +2] / 203 = −3.4 pt | −14 [−34, +6] / 642 = −2.2 pt | −10 [−23, +3] |
| faint | +32 [+13, +54] / 385 = +8.3 pt [+3.4, +14.0] | +3 [0, +7] / 44 | +15 [+6, +24] / 137 = +10.9 pt | 0 [−4, +4] |
| thin+faint pooled | −13 [−59, +34] / 2192 = −0.6 pt [−2.7, +1.6] | −4 [−14, +5] | +1 [−21, +23] / 779 = +0.1 pt | −10 [−24, +3] |
| thick | +7 [−7, +22] / 419 = +1.7 pt | −2 [−5, 0] | 0 [−9, +9] / 147 | −1 |
| wide | −15 [−32, +2] / 445 = −3.4 pt [−7.2, +0.4] | −3 [−8, +2] | −2 [−10, +6] / 159 | +2 |

Rule preview: pooled lower bound FAIL, thin onsets FAIL, thick PASS, wide FAIL.

Per-movie length changes (orig):

| s3 | s4 | s5 | s6 | s7 | s8 | s13 | s14 | s15 | faint30 | faint31 | thick30 | thick31 | wide26 | wide27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| −13 | −8 | −6 | −1 | +5 | +10 | −6 | −8 | −18 | +11 | +21 | +6 | +1 | −18 | +2 |

### M1

| group | orig lengths | orig onsets | human_t2 lengths | human_t2 onsets |
|---|---|---|---|---|
| thin (9 movies) | −45 [−86, −5] / 1807 = −2.5 pt [−4.8, −0.3] | −10 [−19, −1] / 203 = −4.9 pt | −19 [−39, 0] / 642 = −3.0 pt | −13 [−27, 0] |
| faint | +29 [+8, +52] / 385 = +7.5 pt [+2.1, +13.5] | +4 [+1, +8] / 44 | +8 [−5, +20] / 137 = +5.8 pt | +1 |
| thin+faint pooled | −16 [−63, +33] / 2192 = −0.7 pt [−2.9, +1.5] | −6 [−16, +4] | −11 [−34, +12] / 779 = −1.4 pt | −12 [−26, +2] |
| thick | −1 [−24, +19] / 419 = −0.2 pt | −1 | 0 [−9, +9] / 147 | 0 |
| wide | −17 [−37, +5] / 445 = −3.8 pt [−8.3, +1.1] | −3 | −5 [−14, +3] / 159 = −3.1 pt | −2 |

Rule preview: all four checks FAIL.

### M2 (first look only: faint s30, s31, thin s3–s6, thick s30)

| group | orig lengths | orig onsets | human_t2 lengths | human_t2 onsets |
|---|---|---|---|---|
| thin | −13 [−40, +12] / 793 = −1.6 pt | +2 [−3, +7] / 89 | −2 [−12, +7] / 282 | −5 [−12, +1] |
| faint | +8 [−28, +39] / 385 = +2.1 pt | +2 | +4 [−12, +18] / 137 | 0 |
| thin+faint pooled | −5 [−49, +37] / 1178 = −0.4 pt | +4 | +2 [−16, +20] / 419 | −5 |
| thick s30 | −4 [−15, +7] / 211 | 0 | −4 [−10, +1] / 74 | −1 |

## Why it fails

- **The faint gains come from fewer under-reads:** faint traces more than 10% short fall from 46% to 29–36%.
- **The thin losses sit in young tubes.** On s3–s6 with M1c, tubes under 20 px truth go from 282 to 256 within
  tolerance (of 414); tubes of 20 px or more go from 294 to 291 (of 379).
- **The cause: the new networks mark small dark blobs at grain rims as tube.** They produce more pixels above 0.5
  (M2: 15–30% more) and more separate components (40–50% more). For example:
  - v5s3 s002: a particle docks on the rim at bin 59, and M2 gives an onset at bin 59 against a truth bracket of
    76–82. The blob sits nearer the grain centre than the tube's base, so the decoder starts its reading there and
    reads 1–2 px, and the growth fit collapses.
  - v5s3 s006: a rim blob appears from bin about 49; the tube emerges at bin 68.
  - M1's early onsets on s3 go from 0 to 4.
- The calibration in M1c removed some early onsets (thin onsets −10 → −7) but none of the thin length loss.

## Real-movie audit

Against the default, with the round 3/4 fix/break definitions.

- **M1c: 2 fix-only grains, 4 break-only grains,** so rule 3 fails. Marks: onsets 29 → 27 of 35, b64 23 → 18,
  end 20 → 19 of 34; onsets more than 6 bins early 4 → 6.
  - Fixes: g001 end 134 → 115 (audit about 95); g022 onset 28 → 42 (about 40).
  - Breaks, early onsets: g007 24 → 12 (about 22); g008 30 → 12 (about 28); g033 33 → 20 (about 27).
  - Break, foreign tube: g031 b64 80 → 106 (about 80).
  - Newly short: g012 b64 72 → 18 with onset 36 → 50; g014 b64 112 → 83; g023 b64 60 → 44 and end 94 → 74; g032 end
    210 → 115; g035 b64 31 → 16.
- **M1: 2 fix-only, 6 break-only.** Marks: onsets 29 → 26, b64 23 → 19, end 20 → 17.
- **M2: 1 fix, 1 break, 5 newly short.** b64 marks 23 → 19.

## Time

- **Analysis time is unchanged:** the architecture is v2's (0.49 M parameters).
- One-off costs: 8.3–9.5 min to render, prepare and shard each movie; M1 training about 1 h at 1 thread; M2 33 min.

## Caveats

- **There is no seed control.** v2's data retrained from scratch with another seed was not run. M2, which keeps v2's
  weights, points the same way (thin −1.6 pt, faint only +2.1 pt), and the rim-blob examples explain the thin losses
  directly.
- The M1c shift idea came from looking at development results.

**Recommendation: none.** Adding faint positives to the base network trades thin young-tube accuracy and real-movie
onsets for faint lengths.
