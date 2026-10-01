# ld_v1.json: key300, all300, key150, key150scaled, key100 (paired against key300)

| predictions | onsets | lengths | length and tip | T50 (model vs annotator, frames) | growth rates within tolerance |
|---|---|---|---|---|---|
| key300 | 15/26 | 75/104 | 67 | 7559 vs 7383 | 25/27 (r 0.91) |
| all300 | 14/26 | 70/104 | 62 | 7999 vs 7383 | 25/27 (r 0.92) |
| key150 | 13/27 | 69/104 | 58 | 8790 vs 7383 | 24/27 (r 0.89) |
| key150scaled | 11/27 | 70/104 | 63 | 7735 vs 7383 | 24/27 (r 0.86) |
| key100 | 12/27 | 70/104 | 58 | 8526 vs 7383 | 24/27 (r 0.85) |

- all300 vs key300: onsets -1 (95% CI -3 to +0); lengths -5 (95% CI -9 to -1); length and tip -5 (95% CI -10 to -1) over 28 grains
- key150 vs key300: onsets -2 (95% CI -7 to +3); lengths -6 (95% CI -12 to -2); length and tip -9 (95% CI -16 to -3) over 28 grains
- key150scaled vs key300: onsets -4 (95% CI -10 to +1); lengths -5 (95% CI -10 to +0); length and tip -4 (95% CI -10 to +2) over 28 grains
- key100 vs key300: onsets -3 (95% CI -8 to +2); lengths -5 (95% CI -11 to +1); length and tip -9 (95% CI -17 to -2) over 28 grains
