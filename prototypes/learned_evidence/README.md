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

See `docs/assessment-2026-09-24.md`, section 5. Those numbers come from synthetic movies built on the
field of `sample_movie.avi`, the only movie in the repository; the real test is the command above.
