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
| `evaluate.py` | Probability caches; end-to-end SparseTrack runs (baseline, learned, and "perfect" = exact truth masks as evidence); the adaptive crop (below); oracle-path fronts; paired bootstrap over grains |
| `pipeline.py` | One command for a real movie: synthetic movies on its field → shards → training → probability cache → three runs scored on its human labels (SparseTrack as it is; learned evidence through SparseTrack's decoder; learned evidence through `reach.py`) |
| `reach.py` | Decoder v2, the per-bin decoder: in every bin, the medial-axis length of the region with P > 0.5 attached to the grain, then a monotone fit over bins |
| `show.py` | Side-by-side panels (registered bin, SparseTrack's evidence, learned probability) for real footage |
| `models/unet_v2_sample_field.pt` | The trained v2 model (ten synthetic movies on the sample movie's field), for a quick first look |

## Run it on the dev movie (on the machine that has the movies)

```bash
.venv/bin/pip install torch==2.13.0          # the project's `cnn` extra
.venv/bin/python -m prototypes.learned_evidence.pipeline \
    --field runs/sparsetrack/ld --labels benchmark/labels/ld_v1.json --work runs/learned_evidence/ld
```

About 1.5 hours on a laptop. That is ten synthetic movies at about 5 min each, training (about 30 min on 4 CPU
cores, less on an Apple GPU) and a probability cache for the real movie. It prints both SparseTrack runs
scored on `ld_v1` and a paired bootstrap of the difference.

**Quick first look (about 10 minutes).** Add `--model prototypes/learned_evidence/models/unet_v2_sample_field.pt`
to skip the synthetic movies and training. That model (2 MB) is v2 below: trained on ten synthetic movies built on
the field of `sample_movie.avi`, the repository's only movie. The full run builds them on your own `ld` field, so it
remains the proper test.

**Long tubes.** SparseTrack reads each grain in a ±150 px crop, and a tube that leaves it stops at the edge with no
flag. Movie 2's longest tubes pass ±128 px, so both runs use `evaluate.adaptive_crop`:
- A grain whose path ends at the crop edge is read again at ±300 px.
- The matched kymograph is read in angle groups, because OpenCV's `remap` crashes on paths longer than ~268 px.
- On `ld` no path reaches the edge, so nothing changes. `--fixed-crop` turns it off.
- On a synthetic movie of movie 2's length it takes the traces that leave the crop from 9/20 to 20/20 in tolerance.
  The other 33 grains are bit-identical.

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

  | Evidence | Lengths in tolerance | Median error | Onsets | Germinations missed |
  |---|---|---|---|---|
  | Learned v2 (10 synthetic movies, the default) | 734/1014 (72%) | 1.60 px | 86/112 | 7 |
  | Learned v1 (5 synthetic movies) | 699/1014 (69%) | 1.62 px | 82/112 | 12 |
  | SparseTrack's own | 623/1024 (61%) | 2.12 px | 79/112 | 6 |
  | Perfect evidence (ceiling) | 760/1014 (75%) | 1.33 px | 102/112 | 8 |

  - Paired over grains, lengths: v1 +76 traces over SparseTrack's evidence (95% CI −16 to +165), v2 +111 (+25 to
    +195), v2 over v1 +35 (+6 to +70).
  - Bright-cored tubes go from 41% to 70–71%.
  - Weak spot left: drifting grains (67% with SparseTrack's evidence, 63% with v2).
  - Development seed 5: v1 144/192, v2 141/192, SparseTrack's evidence 102/192.
- **Replication** (three fresh movies, v5 seeds 13–15, rendered after v2 was chosen; then four more, 22–25, below):
  - v2 398/601 (66%) against SparseTrack's evidence 389/612 (64%): +9 traces (95% CI −47 to +64), within noise.
  - Pooled over all eight held-out movies (216 grains): 1132/1615 (70%) against 1012/1636 (62%), +120 traces
    (+17 to +226). Perfect evidence: 1176/1615 (73%).
  - The gain per movie runs from −3 to +43 traces. It follows the mix of tubes: bright-cored tubes and sways gain
    most, and drifting grains lose.
- **Two integration choices**, made on the development seed only:
  - Onset comes from the growth front (`onset_source="front"`). SparseTrack's matched stub filter z-scores against
    control angles that are exactly zero on probability maps, and called grains "emerged at start" (3/22 onsets).
  - Tip offset is 0 px: a 2 px offset scored 136/192 against 144/192.
- **The decoder's own ceiling:**
  - With perfect evidence (exact truth masks), SparseTrack's decoder reaches only 73% of lengths in tolerance over
    the eight held-out movies (58–89% per movie). Rotating, drifting and curling tubes remain its losses.
- **Decoder v2: `reach.py`, the per-bin decoder.** It reads the region with P > 0.5 attached to the grain in every
  bin, measures its length along the medial axis (+1 px), then fits a monotone L1 curve. Settings were frozen on the
  development seed. With learned v2 evidence:
  - First eight held-out movies: a tie with SparseTrack's decoder, 1144 against 1132 of 1615 (+12, 95% CI −67 to
    +88).
  - Seven new development movies (seeds 5, 16–21): 1073 against 989 of 1399 (+84, +32 to +139), better on every
    tube type.
    - No label-free per-grain switch between the two decoders beat using `reach.py` everywhere. The switches
      tried were width, path coverage, drift, rotation and the longer reading, each scored leave-one-movie-out.
  - Four untouched test movies (seeds 22–25), frozen beforehand: 664 against 620 of 860 (+44, −6 to +98), onsets 74
    against 67. SparseTrack as it is gets 504 there.
  - Pooled over the twelve held-out and test movies: 1808/2475 (73%) against 1516/2496 (61%) for SparseTrack as it
    is (+292, +170 to +413). It is ahead on all twelve movies, and drifting grains no longer lose.
  - `pipeline.py` scores it as a third run (`perbin/`), so `ld_v1` decides on real footage.
- **Real footage** (qualitative, `show.py`): tube-specific, near zero on grain bodies, but misses wide, dark-walled
  tubes whose profile is outside the synthetic range.
