# Benchmark

Human labels that every method is scored on. Methods are compared with
`python -m sparsetrack eval --labels LABELS --pred PRED [PRED ...]`.

| File | What it is |
|---|---|
| `labels/ld_v1.json` | The sparse-movie benchmark, written by the labelling tool (`Label_Sparse_Benchmark.command`): grain census, onset bracket per grain, exit-to-apex traces at fixed times, blind retest. Judged on registered 300-frame averages. |
| `labels/ld_v1.journal.jsonl` | Append-only log of every answer (recovery and audit). |
| `labels/m2_v1.json` | Held-out labels for *Pollen tube movie 2 7-14-26* (`Label_Movie2_Heldout.command`): grains isolated at germination, onset + traces up to first contact. **Never used for tuning**; scored once per frozen SparseTrack version. |
| `labels/legacy_v0.json` | The 7 pre-reset human onset brackets and 20 traces / 16 absences, converted by `scripts/build_legacy_benchmark.py`. Judged on single compressed frames, and the older methods were tuned on them: a development reference, not the benchmark. |
| `reports/` | Generated score reports (never retype numbers from them by hand). |

Scoring (see `sparsetrack/evaluate.py`): onset error is the signed distance of the
predicted onset frame from the human bracket (tolerance ±600 frames); FULL traces are
scored within max(2 px, 10%); "no tube" traces count as absences; isolated grains only
unless `--subset all`. Traces flagged `contact` (touching another tube or grain) are counted
apart; `burst` marks a tube that has burst by that trace time (recorded, not yet scored).
