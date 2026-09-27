# Learned tube evidence (prototype, 27 Sep 2026)

A small U-Net reads SparseTrack's three images (the bin, the "before" and the "after" reference)
and outputs, per pixel, the probability that tube body is there (and a tip heatmap). The idea
(from the 24 Sep assessment on branch `claude/magical-maxwell-i5tpeh`, whose scaffold this is):
exact labels from the codec-exact synthetic movies, fine-tuned on the human traces, feeding
SparseTrack's decoder. **Result: in its first round (below) not better than change evidence
on either benchmark. In the second round, synthetic movies built on movie 2's own field made
the maps good enough that an arrival flood beats change evidence on crowded movie 2. That model and
reader now live in `sparsetrack/learned.py` (reader `hybrid`, SparseTrack 0.5.0).** This folder
keeps the training and evaluation code and the record of every experiment.

| file | what |
|---|---|
| `truth.py` | exact per-frame tube-body and apex rasters from a synthetic scene |
| `data.py` | training crops from a synthetic movie (`python -m prototypes.learned_evidence.data ...`) |
| `realdata.py` | training crops from the human traces: body = centreline +/- 3 px, background 6-14 px off it, the same crops before the onset bracket as negatives; everything else unscored |
| `model.py`, `train.py` | U-Net (0.49 M parameters); training on MPS; `--real` mixes in trace crops, `--init` fine-tunes |
| `realpix.py` | pixel check against the traces: share of traced centreline called tube, and of points 8-14 px beside it |
| `real.py` | probability movies for ld/m2 and SparseTrack run on them, scored like `synth_bench --real` |
| `evaluate.py` | the scaffold's synthetic end-to-end and oracle-path comparisons |
| `flood.py` | a different reader: arrival-time flood from the rim (below) |

## What was measured

Training: 2,880 crops from synthetic v5 seeds 0-2 (seed 3 for validation), 6,000 steps, 8 min on
the M1 GPU; synthetic validation Dice about 0.75. Fine-tuning: + real trace crops at 50% of each
batch, 3,000 steps. Each fine-tuned model was tested on the movie whose traces it did not see.

Pixel level (traced tube points called tube / points beside the tube called tube):

| model | dev movie (ld) | movie 2 (m2) |
|---|---|---|
| synthetic only | 54% / 2.6% | 43% / 2.5% |
| + ld traces | 87% / 3.1% (trained on) | **49% / 4.4%** (held out) |
| + m2 traces | **69% / 1.1%** (held out) | 45% / 4.4% (trained on) |

End to end, FULL traces within max(2 px, 10%) (SparseTrack 0.4.3 with change evidence: ld 61/104, m2 9/54):

| reader | ld | m2 |
|---|---|---|
| SparseTrack on the probability movie (synthetic-only model) | 18/104 | 2/54 |
| same, model fine-tuned on ld (ld in-sample) | 42/104 | - |
| SparseTrack evidence gated by P >= 0.5 (held-out models) | 52/104 | 3/54 |
| SparseTrack evidence replaced by P (held-out models) | 37/104 | 4/54 |
| flood on ridge-filtered change | 19/104 | 8/53 |
| flood on P, model fine-tuned on ld | 54/104 (in-sample; +2 px length offset) | 12/53 (held out; 11/53 with the offset) |
| flood on P, model fine-tuned on m2 | 41/104 (held out; +2 px offset) | - |

(The flood's length offset was applied afterwards to its printed lengths; `flood.py` does not
apply one.)

The flood (`flood.py`): each pixel's arrival bin (first bin from which it stays present);
the grain's tube grows in arrival order, a newly arrived piece joining only if it touches the
tube's most recently joined pixels (its tip), or the rim before the tube exists. Material that
was already there (a foreign tube) or appears beside old tube (sway) never joins. Where the map
is good it is precise (many ld grains within 1-5 px at every trace time); it fails where the map
has gaps near the rim (5-7 grains per movie never start) and on raw or ridge-filtered change
(focus/illumination blobs and rim halos start it on noise).

## Second round: synthetic movies on movie 2's field

The generator (`sparsetrack synth`) draws tubes over any prepared movie's real pre-germination
field and grains. Three movies on movie 2's field, sized like it (351 bins, onsets to bin 250,
tubes to 280 px, seeds 10-12; 11-12 with 80% light-cored, 1-1.6x wider tubes like movie 2's),
were added to the three dev-field movies: 7,200 synthetic crops, 8,000 steps (19 min, MPS),
synthetic validation Dice about 0.83. No human labels.

| model | traced tube found: ld | m2 |
|---|---|---|
| synthetic, both fields (`sparsetrack/models/tubes_synth_v1.pt`) | 78% (beside 6.0%) | 68% (beside 5.3%) |
| + ld traces | - | 67% (held out) |
| + m2 traces | 86% (held out) | - |

The flood on the synthetic-only maps (lengths in tolerance; 0.4.3: ld 61/104, m2 9/54):
ld 39/104, **m2 22/54 (paired +13, 95% CI +4 to +23)**. Crowding decides which reader wins:
using the flood only where a grain's change region touches a neighbour gives ld 61/104
(unchanged), m2 19/54 (+10, CI +2 to +19), onsets ld 13/26, m2 6/19. That is SparseTrack
0.5.0's default. Caveat: the flood's rules were settled while looking at movie-2 grains; an
honest score needs a movie labelled after 0.5.0.

Also tried in this round, not kept: a wider start band for the flood (ld +5, m2 -2); an
arrival-order front along SparseTrack's path (robust monotone fit of per-point arrival bins:
neutral on model paths, worse on human routes, where crossings are too dense to be outliers).

## Conclusions (first round)

- The network sees real tubes cleanly where it fires (no blobs, halos, banding or vignetting),
  but on a movie it was not trained on it finds only half to two thirds of the traced tube.
  Each movie's traces helped the other: synthetic data alone does not capture real tube looks.
- With maps at that recall, every reader tried loses to change-based evidence on ld and is at
  best level on m2. The flood's ownership logic is the right shape for crowded fields.
- What would change this: traced tubes from more movies (different fields, densities, optics)
  for a detector that transfers, and a rim-gap-tolerant start for the flood.
