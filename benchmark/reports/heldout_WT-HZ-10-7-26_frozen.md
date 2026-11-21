# heldout_WT-HZ-10-7-26_v1.json: 0.8.8, 0.9.0-tiptraj, 0.9.0-tiptraj-selftrained (paired against 0.8.8)

| predictions | onsets | lengths | length and tip | T50 (model vs annotator, frames) | growth rates within tolerance |
|---|---|---|---|---|---|
| 0.8.8 | 7/28 | 23/83 | 16 | 6845 vs 10366 | 11/29 (r 0.58) |
| 0.9.0-tiptraj | 10/28 | 30/83 | 23 | 7236 vs 10366 | 12/29 (r 0.34) |
| 0.9.0-tiptraj-selftrained | 8/28 | 33/83 | 24 | 6845 vs 10366 | 12/29 (r 0.39) |

- 0.9.0-tiptraj vs 0.8.8: onsets +3 (95% CI -1 to +7); lengths +7 (95% CI -2 to +17); length and tip +7 (95% CI -1 to +15) over 30 grains
- 0.9.0-tiptraj-selftrained vs 0.8.8: onsets +1 (95% CI -4 to +6); lengths +10 (95% CI +1 to +20); length and tip +8 (95% CI +0 to +16) over 30 grains
