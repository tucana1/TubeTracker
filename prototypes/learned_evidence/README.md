# Learned evidence for SparseTrack (experiment)

**Question:** can a small network trained only on SparseTrack's codec-exact synthetic movies give
the unchanged SparseTrack decoder better evidence than its hand-built change maps?

The network reads the same three registered images SparseTrack uses: the bin being read, the
"before" reference (first 3 bins) and the "after" reference (last full bins). It outputs, per pixel,
the probability that tube has already been built there, plus a tip heatmap. Nothing in
`sparsetrack/` changes.
- The per-bin probability, ×16, is written as a SparseTrack cache and analysed like a movie.
- At that scale SparseTrack's grey-level thresholds sit at P ≈ 0.3 (tube map), and its front evidence
  runs from −1 at P = 0 to +1 at P ≥ 0.63.
- Per-grain registration is measured on the image cache.

| File | What it does |
|---|---|
| `truth.py` | Exact per-frame truth rasters (built tube body, instances, tips) from a `sparsetrack.synth.Scene`, reusing its geometry: rotation, drift, substrate anchoring, sway |
| `data.py` | Training crops (bin, before, after) plus targets from a synthetic movie's cache and scene |
| `model.py` | 0.49 M-parameter U-Net (BatchNorm, so tiled inference does not depend on tile size); tiled prediction |
| `train.py` | Training: dihedral, gain, offset and noise augmentation; BCE + Dice for the body; weighted BCE for tips. CPU, MPS or CUDA |
| `evaluate.py` | Probability caches; end-to-end SparseTrack runs (baseline, learned, and "perfect" = exact truth masks as evidence); oracle-path fronts; paired bootstrap over grains |
| `pipeline.py` | One command for a real movie: synthetic movies on its field → shards → training → probability cache → both SparseTrack runs scored on its human labels |

## Run it on the dev movie (on the machine that has the movies)

```bash
.venv/bin/pip install torch==2.13.0          # the project's `cnn` extra
.venv/bin/python -m prototypes.learned_evidence.pipeline \
    --field runs/sparsetrack/ld --labels benchmark/labels/ld_v1.json --work runs/learned_evidence/ld
```

About an hour on a laptop. That is five synthetic movies at about 5 min each, training (about 25 min
on 4 CPU cores, less on an Apple GPU) and a probability cache for the real movie. It prints both
SparseTrack runs scored on `ld_v1` and a paired bootstrap of the difference.

**Movie 2 stays held out.**
- Never pass its cache as `--field` or `--train-field` for training.
- Score it once, with a model trained on the dev field:

```bash
.venv/bin/python -m prototypes.learned_evidence.pipeline --field runs/sparsetrack/m2 \
    --labels benchmark/labels/m2_v1.json --model runs/learned_evidence/ld/unet.pt \
    --work runs/learned_evidence/m2 --heldout-once
```

## Results so far

Full tables are in `docs/assessment-2026-09-24.md`, section 5. All numbers come from synthetic movies built on the field of
`sample_movie.avi`, the only movie in the repository. The real test is the command above.

- **Evidence only** (true per-bin geometry, SparseTrack's `dp_front`; held-out seed 3):
  - learned: 157/173 lengths in tolerance (91%), median error 1.12 px;
  - SparseTrack's `union` evidence: 124/173 (72%), median 1.78 px.
- **End to end** (unchanged SparseTrack decoder; held-out v5 seeds 3, 4, 6, 7 and 8, 135 grains):

  | Evidence | Lengths in tolerance | Median error | Onsets |
  |---|---|---|---|
  | Learned | 699/1014 (69%) | 1.62 px | 82/112 |
  | SparseTrack's own | 623/1024 (61%) | 2.12 px | 79/112 |
  | Perfect evidence (ceiling) | 760/1014 (75%) | — | 102/112 |

  - Paired over grains, lengths gain +76 traces (95% CI −16 to +165).
  - Bright-cored tubes go from 41% to 70%.
  - Weak spots: drifting grains (67% → 58%) and 12 missed germinations against 6.
  - Development seed 5: learned 144/192 (75%) against 102/192 (53%).
- **Two integration choices**, made on the development seed only:
  - Onset comes from the growth front (`onset_source="front"`). SparseTrack's matched stub filter z-scores against
    control angles that are exactly zero on probability maps, and called grains "emerged at start" (3/22 onsets).
  - Tip offset is 0 px: a 2 px offset scored 136/192 against 144/192.
- **The decoder's own ceiling:**
  - With perfect evidence (exact truth masks), SparseTrack's decoder reaches only 71–79% of lengths in tolerance.
    Rotating, drifting and curling tubes remain its losses.
  - `reach.py`, a naive per-bin decoder (geodesic reach through P > 0.5, then a monotone L1 fit), ties it on the
    development seed (147/192 against 152/192 with perfect evidence) but calls onsets ~3 bins late. Decoder v2 needs
    real design work (exit handling, skeleton length); it is a starting point, not a result.
- **Real footage** (qualitative, `show.py`): tube-specific, near zero on grain bodies, but misses wide, dark-walled
  tubes whose profile is outside the synthetic range.
