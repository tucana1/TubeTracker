# Round 5, agent "tta": test-time augmentation of the evidence networks

> The agent's report of 27 Sep 2026, as delivered. Paths such as `scratchpad/agents/tta/` are the cloud session's; the
> code, logs and scores are in `tta/` beside this file. Development numbers only: the sealed held-out set was never
> opened, and nothing was trained.

**Answer: no.** Dihedral TTA does not raise faint or thin lengths under the current default decoder:
- Every variant stays within about ±2 pt, and every interval spans 0.
- TTA on v2 alone slightly costs thick and wide tubes.
- On the real movie, every variant breaks more grains than it fixes.
- Only the 2-transform variants fit rule 4, and they show nothing. 4 and 8 transforms are no better and cost 1.7–4.2×
  the analysis time.

**Candidate to freeze: none.**

## Method

- **One pass** is the repository's own `model.predict`, run on the (bin, before, after) input transformed as a whole;
  the output is transformed back.
  - The input comes from `evaluate.registered_frame` and `data.normalise`, the same code path as `evaluate.prob_cache`
    (and the kit's `sparse.build`).
  - Each transform is stored sparse as `sparse/<model>_x<T>_<movie>.npz`.
  - The identity term is the shared plain cache that the default was decoded from.
- **Variants** are averaged from the stored maps (`tta.combine`):
  - t2 = identity + 180° rotation; t4 = the four flips; t4r = the four rotations; t8 = all eight.
  - Averaging: p = mean probability; l = mean logit (P clipped to [0.001, 1−1e-6]); m = max (union), exploratory only.
  - Network scope: v2 only, or both v2 and B3.
- **Decode and score:** decoded with the kit's `decode` (vmax 4, real 16); scored with `score` against the tag
  `default`.
- **Exactness checks:**
  - The identity pass on v5faints30 at 1 thread is bitwise equal to the stored v2 map (idx/val/ptr, all 176 bins).
  - At 2 threads, one pixel differed by 1 float16 ulp, so every pass ran at 1 thread.
  - Decoding the plain maps through the agent's path reproduces `default` exactly.
- **Passes run (45, 4.8 CPU hours):**
  - v2 180° rotation on all 15 development movies and the real movie.
  - v2 horizontal flip, vertical flip, 90°, 270°, transpose and anti-transpose on faint s30 and s31.
  - B3 180° rotation on all 15 development movies and the real movie.

## Development results vs the default

Paired 95% bootstrap over grains; pt = percent of the group's traces. Trace counts, orig / human_t2: thin 1807 / 642,
faint 385 / 137, thick 419 / 147, wide 445 / 159.

Each cell: lengths difference [CI] = points [CI], then the onset difference (orig only).

### tta_v2t2p (t2, mean probability, v2 only)

| group | orig | human_t2 |
|---|---|---|
| thin | −1 [−25, +23] = −0.1 [−1.4, +1.3]; onsets +0 | −2 [−17, +13] = −0.3 [−2.6, +2.0] |
| faint | −1 [−30, +19] = −0.3 [−7.8, +4.9]; onsets +1 | −2 [−15, +9] = −1.5 [−10.9, +6.6] |
| thin+faint pooled | −2 [−40, +31] = −0.1 [−1.8, +1.4]; onsets +1 | −4 [−23, +15] = −0.5 [−3.0, +1.9] |
| thick | −6 [−17, +3] = −1.4 [−4.1, +0.7]; onsets 0 | −3 [−10, +3] = −2.0 [−6.8, +2.0] |
| wide | −5 [−16, +6] = −1.1 [−3.6, +1.3]; onsets −2 | +3 [−3, +9] = +1.9 [−1.9, +5.7] |

### tta_v2t2l (t2, mean logit, v2 only)

| group | orig | human_t2 |
|---|---|---|
| thin | +1 [−23, +25] = +0.1 [−1.3, +1.4]; onsets 0 | +3 [−12, +18] = +0.5 [−1.9, +2.8] |
| faint | +0 [−28, +21] = 0.0 [−7.3, +5.5]; onsets +1 | +2 [−6, +10] = +1.5 [−4.4, +7.3] |
| thin+faint pooled | +1 [−35, +34] = 0.0 [−1.6, +1.6]; onsets +1 | +5 [−12, +22] = +0.6 [−1.5, +2.8] |
| thick | −5 [−16, +3] = −1.2 [−3.8, +0.7]; onsets 0 | −2 [−9, +5] = −1.4 [−6.1, +3.4] |
| wide | −8 [−20, +4] = −1.8 [−4.5, +0.9]; onsets −2 | +1 [−5, +8] = +0.6 [−3.2, +5.0] |

### tta_botht2p (t2, mean probability, v2 and B3)

| group | orig | human_t2 |
|---|---|---|
| thin | +6 [−20, +32] = +0.3 [−1.1, +1.8]; onsets 0 | −10 [−26, +6] = −1.6 [−4.0, +0.9] |
| faint | −4 [−33, +16] = −1.0 [−8.6, +4.2]; onsets +1 | −4 [−17, +7] = −2.9 [−12.4, +5.1] |
| thin+faint pooled | +2 [−37, +37] = +0.1 [−1.7, +1.7]; onsets +1 | −14 [−34, +5] = −1.8 [−4.4, +0.6] |
| thick | +4 [−5, +13] = +1.0 [−1.2, +3.1]; onsets 0 | 0 [−8, +8] = 0.0 [−5.4, +5.4] |
| wide | +4 [−9, +18] = +0.9 [−2.0, +4.0]; onsets −2 | +6 [0, +13] = +3.8 [0.0, +8.2] |

### tta_botht2l (t2, mean logit, v2 and B3)

| group | orig | human_t2 |
|---|---|---|
| thin | −3 [−29, +22] = −0.2 [−1.6, +1.2]; onsets −1 | −10 [−26, +6] = −1.6 [−4.0, +0.9] |
| faint | −6 [−33, +14] = −1.6 [−8.6, +3.6]; onsets +1 | +1 [−7, +9] = +0.7 [−5.1, +6.6] |
| thin+faint pooled | −9 [−45, +25] = −0.4 [−2.1, +1.1]; onsets 0 | −9 [−26, +8] = −1.2 [−3.3, +1.0] |
| thick | +7 [−3, +17] = +1.7 [−0.7, +4.1]; onsets 0 | +3 [−6, +12] = +2.0 [−4.1, +8.2] |
| wide | +6 [−7, +19] = +1.3 [−1.6, +4.3]; onsets −2 | +3 [−5, +11] = +1.9 [−3.1, +6.9] |

**Rule preview (orig truth):**
- All four variants fail rule 1: the pooled lower bound is between −1.6 and −2.1 pt.
- The two v2-only variants also fail rule 2 on thick and wide (lower bounds −3.6 to −4.5 pt).
- The two both-network variants pass the rule 2 preview.

### 4 and 8 transforms, faint s30 + s31, v2 only

| variant | orig lengths [CI] | human_t2 lengths [CI] | orig onsets, of 44 |
|---|---|---|---|
| t4p | −2 [−33, +21] | +0 [−12, +10] | −2 |
| t4l | +1 [−30, +24] | −3 [−17, +8] | −2 |
| t4rp | −3 [−33, +19] | −5 [−16, +4] | −2 |
| t4rl | +7 [−6, +22] | +0 [−10, +9] | −1 |
| t8p | −5 [−35, +17] | −2 [−13, +8] | −3 |
| t8l | −6 [−37, +18] | −7 [−19, +3] | −2 |

More transforms never helped, and faint onsets drop by 2–4 with 4 or 8 transforms.

**Max (union), exploratory, t2m, all groups:** thin lengths −21 [−51, +7], faint +8 [−4, +19], pooled −13
[−44, +17]; onsets contradict between the truth documents (orig pooled +8 [0, +16], human_t2 −11 [−22, −1]); real movie
3 fixes, 4 breaks. Not a candidate.

## Thresholded behaviour (measured)

- **With 2 transforms, probability and logit averaging threshold identically.** For two maps, p1 + p2 > 1 exactly when
  logit1 + logit2 > 0.
  - On faint s30, only 63 of 105,651 above-threshold pixels differ (float16 rounding), although P differs by 0.04 on
    average.
  - So the t2p against t2l difference in decoded counts (pooled −2 against +1 orig, −4 against +5 human_t2) comes only
    from the decoder resampling P. It measures the noise floor of these comparisons.
- **Faint tube core recall** (points within 0.6 × width of the centreline with P > 0.5, every 8th bin):

  | map | faint s30 | faint s31 |
  |---|---|---|
  | identity | 70.6% | 77.0% |
  | t2p | 69.5% | 76.4% |
  | t4p | 69.8% | 77.6% |
  | union of 4 flips | 73.5% | 81.4% |

  - 19–27% of the faint core is below 0.5 in every orientation, so no average can recover it.
  - Averaging moves core recall by at most 1.1 pt, either way.
  - Across the whole visible tube width on s30, t2p pushes more tube pixels below 0.5 (1,170) than above it (456).
- **No orientation is systematically best.** On faint s30 the identity pass has the best single-pass recall (57.7%
  against 54.5–57.1%); on faint s31 the flips do better (68.6% and 68.9% against 67.9%); on thin s3–s5 all are within
  ±0.5 pt.

## Real-movie audit

Against `default` (vmax 16), which scores onsets 29/35, mid-movie (b64) 23/35 and end 20/34.

| variant | onsets | b64 | end | grains whose readings change | fixes | breaks |
|---|---|---|---|---|---|---|
| tta_v2t2p | 27/35 | 22/35 | 19/34 | 33 | 0 | 4 |
| tta_v2t2l | 26/35 | 21/35 | 18/34 | 32 | 0 | 7 |
| tta_botht2p | 27/35 | 21/35 | 19/34 | 34 | 3 | 8 |
| tta_botht2l | 27/35 | 21/35 | 17/34 | 34 | 2 | 9 |

- **tta_v2t2p breaks:** g007 onset 24 → 12 (audit about 22); g023 b64 59.5 → 40 (audit 60–100); g032 end
  209.5 → 255 (audit about 200); g033 onset 33 → 20 (audit about 27).
- **Why so many real grains change:** the identity and 180° v2 masks overlap much less on real footage (Jaccard 0.74)
  than on synthetic movies (thin 0.93, faint 0.84, thick 0.89).
- Every variant fails rule 3. Grain-by-grain listings: `tta/logs/audit_<tag>.txt`.

## Timing

CPU seconds at 1 thread on the cloud machine.
- **Network passes:** v2 median 384 s per synthetic movie (176 bins), B3 379 s; on the real movie (129 bins) v2 278 s,
  B3 284 s.
- **The rest of an analysis:** faint s30 127 s (64 s with `--only-perbin`); real movie 574 s (340 s).
- The networks are 86% of a synthetic analysis and 49% of the real one.

| recipe | synthetic | real |
|---|---|---|
| t2, v2 only | 1.43× | 1.24× |
| t2, both | 1.86× | 1.49× |
| t4, v2 only | 2.29× | 1.73× |
| t4, both | 3.57× | 2.48× |
| t8, v2 only | 4.02× | 2.71× |

## Recommendation

**None.** No TTA recipe clears rule 1 on development; all fail rule 3 on the real movie; 4 or 8 transforms also break
rule 4.
