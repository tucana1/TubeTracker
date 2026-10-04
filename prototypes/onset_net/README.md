# A grain-aware onset network, judged leave one movie out (research record, 4 Oct 2026)

**Question.** Can a learned model that is told WHICH grain it judges (a crop centred on the grain where it is at
each bin, plus a channel marking the grain's own disc) time germination onsets much better than SparseTrack 0.8.8 on
a movie it never saw? The tip detector (`prototypes/tip_detector`, `prototypes/tip_track`) finds young tips well but
its rim response cannot tell the grain's own young tube from neighbours' tubes and debris on movie 2.

**Answer: no.** Held out, the network's onsets are below 0.8.8 on every movie with the pre-registered decoder (ld
11/28 vs 15, m2 6/19 vs 7, m1 7/28 vs 8) and at best level with it with the one fix tried (the decoder's switch level
chosen on the other two movies: ld 9, m2 5, m1 8). No difference from 0.8.8 is significant (ld with the chosen level,
-6, CI -12..0, touches zero) and none is positive. It is also below 0.8.8 + the tip detector's `later` onsets (17 / 7
/ 10), and below the detector's rim rules alone on ld and m1 (by 3-8 hits; the plateau-relative rule's 15 on m1
significantly); only on the crowded movie 2, where those rules collapse (0-5 hits; the two fixed rules fire on all four debris objects), is
it above them (+1 to +6; significant only against the level rule chosen on the other two movies, which gets 0 there),
and it leaves all four m2 debris alone. Its per-bin probabilities separate the bins before an onset from young tubes
about as well as the tip detector (AUROC 0.95 / 0.85 / 0.83 vs the detector's 0.91 / 0.81 / 0.89: better on ld and m2,
worse on m1), but that is not what timing needs: on ld and m2 it becomes confident only once the tube is several px
long (late by 3-25 bins, m2 g043 by 62), on m1 it fires early on the grain body's own rim/interior changes,
neighbours' tubes and debris, and no single switch level suits all three movies. Not for the tracker. The disc channel
did not teach ownership from ~50 training grains.

## Method

Data (`build.py`, 28 s, 337 MB float16 memmaps): for every labelled grain (not excluded, verdict emerged within /
at start / never) and every census object the annotator excluded as 'not a grain' (debris), an 80 px registered crop
of EVERY bin centred on where the grain is at that bin (census + the labelling tool's own per-bin following,
`prototypes/tip_detector/offsets.py`; debris at its census place). Grain-centred crops keep the grain still, so its own
drift does not draw crescents.

Inputs at bin b (`common.py`, as the tip detector but in the grain's frame; M(k) = mean of bins k-1..k+1):
A = M(b), D = M(b) - M(b-6), C = M(b) - mean(bins rs..rs+2), each less the crop's median, / 20, 8, 20 grey levels;
G = the grain's disc (census radius) at the crop centre. Network input 72 px (4 channels).

Labels per bin: 0 at bins <= last_absent_bin, 1 at bins >= first_visible_bin (unlabelled between: usually nothing,
brackets are one bin wide), never-germinated grains and debris 0 at every bin, emerged-at-start grains 1 at every bin.
ld 32 objects (31 grains, all emerged within), m2 25 (19 + 1 at start + 1 never + 4 debris), m1 34 (28 + 1 + 1 + 4).

Network and training (`train.py`): a per-bin classifier, conv blocks 16-32-64-64 with BatchNorm and max-pooling,
global mean + max pooling, dropout 0.3, one logit. 3000 steps of 64 (half from each training movie, half positive;
grain drawn uniformly, then one of its bins, bins within 15 of the onset 3x as likely), BCE, AdamW (2e-3, wd 1e-2),
one-cycle, fixed schedule (nothing of the held-out movie looked at). Augmentation on the raw grain-centred crops
BEFORE A, D, C are formed: a focus step (p 0.3: blur sigma 0.6-2 of every bin before, or from, a random bin, as m2's
refocus at bin 63) and a sub-pixel motion step (p 0.3: N(0, 0.5 px) shift from a random bin); then centre jitter +/-3
px (the disc stays at the crop centre, as when the tracker's grain position is off), the 8 rotations/flips, gain
0.6-1.6 (x 0.85-1.15 per channel), noise, blur, disc radius x 0.9-1.1. ~6 min per fold on MPS when the machine is
quiet (22 min under load). Fold models named by their training movies: `on_m2m1` (ld held out), `on_ldm1` (m2),
`on_ldm2` (m1).

Evaluation (`evaluate.py`): per-bin probabilities for every scored grain and debris of the held-out movie, every bin,
mean logit over the 8 rotations/flips, with the grain where the 0.8.8 reading put it (census + its drift: what the
tracker would give; primary) and, as a check, where the labelling tool followed it (the same hits on all three movies
with tau 0.5, within one with the chosen tau). Onset = the single switch 0 -> 1 that best explains the probabilities
(likelihood change point: maximise sum over b >= t of logit p_b; first valid bin = at start; no switch = never). Hit =
within 600 frames of the human bracket (`sparsetrack.evaluate` rule) over the scorer's grains (isolated, not excluded)
the annotator calls 'emerged within'; a grain called never or at start counts as a miss (ld 28, m2 19, m1 28 grains).
Paired bootstrap over grains (10 000 resamples, 95%) against 0.8.8 (`scratchpad/bt/base088`), 0.8.8 + tip detector
`later` + young (end-to-end predictions of `prototypes/tip_track`) and the tip detector's rim rules recomputed for the
same grains (`rim_series.py`: the fold detector's maps where the 0.8.8 reading put the grain, reduced to the peaks
within r + 60 px as `prototypes/tip_track/rim.py`; rules from `tip_track`: fixed level thr 0.3 for 3 bins, relative to
the grain's own plateau, and the level rule chosen on the other two movies). T50 and the largest curve gap from
`sparsetrack.evaluate.score` on 0.8.8's readings with the onsets replaced. AUROC: (a) as the tip detector's README
(bins last_absent - 1, 3, 6, 12, 24 vs bins of young FULL traces, human length < 15 px); (b) the 13 bins up to
last_absent vs the 7 bins from first_visible.

The one fix (`decode.py`): the held-out m2 probabilities rose at the human onset but stayed at 0.2-0.5 for 10-25
bins, so the decoder got a switch level tau (maximise sum over b >= t of logit p_b - logit tau), chosen for each
held-out movie on the other two movies' held-out probabilities (most hits; ties to tau nearest 0.5).

## Results (held out; grains position from the 0.8.8 reading)

Onset hits / human 'emerged within' grains; change vs the network (network minus other) and 95% CI:

| held out | network, tau 0.5 (pre-registered) | network, tau chosen on the other two | 0.8.8 | 0.8.8 + tipdet `later` (e2e) | tipdet rim: level 0.3 x 3 | tipdet rim: relative | tipdet rim: level chosen on the other two |
|---|---|---|---|---|---|---|---|
| ld (28) | 11 | 9 (tau 0.35) | 15 (-4, -11..+3) | 17 (-6, -13..+1) | 17 (-6, -14..+2) | 18 (-7, -14..0) | 14 (-3, -11..+4) |
| m2 (19) | 6 | 5 (tau 0.1) | 7 (-1, -5..+3) | 7 (-1, -5..+3) | 4 (+2, -3..+7) | 5 (+1, -5..+7) | 0 (+6, +2..+10) |
| m1 (28) | 7 | 8 (tau 0.1) | 8 (-1, -7..+6) | 10 (-3, -10..+4) | 13 (-6, -13..+1) | 15 (-8, -15..-1) | 13 (-6, -13..+1) |

(The CIs in the 0.8.8 / `later` / rim columns pair them with the tau-0.5 network. The recomputed rim rules reproduce
`prototypes/tip_track`'s offline numbers exactly: level chosen on the other two 14 / 0 / 13, relative 18 / 5 / 15.)
With tau chosen on the other two movies: vs 0.8.8 ld -6 (-12..0), m2 -2 (-7..+3), m1 0 (-6..+6); vs `later` -8
(-15..-1), -2, -2; vs the relative rim rule -9 (-16..-2), 0, -7 (-14..0). Debris objects given an onset: network 0/1,
0/4, 3/4; rim level rule 0/1, 4/4, 1/4; relative 0/1, 4/4, 4/4; level chosen on the other two 0/1, 1/4, 1/4. Hits by
tau on each movie's own held-out probabilities (for the record, not a choice): ld 16 at tau 0.05-0.1 down to 9-11 at
0.3-0.5; m1 4 at 0.05, 10 at 0.25-0.4, 7 at 0.5; m2 4-6 at every tau from 0.05 to 0.7. The movies want different
levels.

Error direction (tau 0.5, network vs 0.8.8): ld early 4 / late 13 (0.8.8 5 / 6); m2 3 / 10 (8 / 3); m1 12 / 9 (10 / 8).

| held out | AUROC (a) pre-onset vs young traces | AUROC (b) 13 before vs 7 after | tip detector (a), from its README |
|---|---|---|---|
| ld | 0.952 | 0.856 | 0.91 |
| m2 | 0.849 | 0.703 | 0.81 |
| m1 | 0.830 | 0.745 | 0.89 |

Germination curve, T50 in frames (human / 0.8.8 / network tau 0.5 / network tau chosen / `later`) and largest gap
(0.8.8 -> network tau 0.5):
- ld: 7383 / 7559 / 7823 / 7735 / 7120; gap 0.14 -> 0.18
- m2: 24049 / 23874 / 28613 / 26156 / 23523; gap 0.09 -> 0.19 (15 bins late)
- m1: 8602 / 9304 / 8777 / 7373 / 7899; gap 0.23 -> 0.23

Grains that never germinated: both called germinated (m2 g100 at bin 325, p creeping to ~0.5 as its interior and
rim change; m1 g029 at bin 17: interior brightening ring, later a neighbour's tube along its rim). Emerged at start
(m2 g082, m1 g056): both called 'emerged within' (C is the change from the start, so a tube already there is
cancelled; the network only fires once it grows: g082 at bin 96). Debris given an onset: ld 0/1, m2 0/4, m1 3/4
(appearance change inside the object plus neighbours' tubes crossing the crop).

## Failures (examples; crops checked by eye)

- **Late on the youngest tubes (ld, m2; the main loss).** The probability steps up 3-25 bins (up to 62) after the
  annotator's first visible bin, once the tube is several px long: ld g022 (first visible 11, network 27), g027 (5 ->
  20), g014 (17 -> 35), g011 (37 -> 46); m2 g043 (63 -> 125: a tiny bump below the grain at 63-80 is ignored until it
  is a tube), g009 (53 -> 74), g060 (63 -> 89: a clear bulb at 63, p 0.26-0.48 for 25 bins), g016 (46 -> 61).
- **The grain body's own changes (m1).** Rim crescents and interior brightening in the early bins (hydration,
  sub-pixel motion) read as a tube: m1 g007 (25 -> 8), g060 (26 -> 4, a dark mark on the rim from bin 4 the annotator
  did not count), g006, g011, g053 (9-11 bins early), the never-germinated g029 (interior ring from bin 17).
- **Rim bulbs or particles the annotator did not count.** m2 g005: a light-cored bulb on top of the grain from bin 56;
  the annotator's onset (124) is the tube that leaves the other side (0.8.8 and the tip detector are early on it too).
- **Neighbours' tubes and large objects.** m2 g069: a large dark object drifting through the crop at bins 44-64
  (network 48, annotator 64); m2 g064: a brief spike to p ~1 at bins ~38-45 (ignored by the decoder); m1 g013 (265 ->
  23, p 0.4-0.8 from bin 23 on); m1 debris g032/g048/g050 when tubes cross the crop.
- **Focus change (m2 bin 63).** Not a source of false onsets here (trained with the focus-step augmentation; whether
  that was needed was not tested): at bins 60-66 the nine grains germinating later, the never-germinated grain and the
  debris stay at p <= 0.15, except one bin of g066 (0.92 at bin 62, ignored by the decoder). Of the four annotator
  onsets at bins 62-64 ('visible by'), g038 was called 5 bins early from the blurred bulb, g069 15 early (the dark
  object above), g043 and g060 late.
- **Grain position.** The 0.8.8 drift and the labelling tool's following gave the same hits, but where they differ
  the onset can move a lot: ld g007 (not isolated, not scored; the two positions up to 5 px apart) is called at bin 2
  from the 0.8.8 position and ~30 from the labelling tool's.

## Recommendation

Do not put this network into the tracker, as an override or as a vote: it does not beat 0.8.8 on any held-out movie,
let alone two, and its errors go opposite ways on different movies (late on ld/m2, early on m1), so no calibration
chosen on two movies transfers. The ownership problem is not solved by telling the network where the grain is when
it has ~50 grains of two movies to learn from: what separates the grain's own first bulb (2-5 px) from the grain's
rim changes, a neighbour's tube or a passing particle is rare in the training data (one onset per grain). If onsets
are worked on again, the evidence here says the bottleneck is the first 1-5 bins of a tube (where both this network
and the readers are blind or confused), so: (1) more labelled onsets from more movies (each movie has its own
nuisance: m1's early grain changes, m2's crowding and refocus) before any learned onset model; (2) a model of the
change ITSELF over a window of bins (a bulb appearing and growing at one rim point) rather than a per-bin 'tube
visible' state; (3) keep the readers' onsets (0.8.8, optionally the detector's `later`) meanwhile.

## Files

`common.py` (inputs, network, decoder, scoring), `build.py` (crops), `train.py`, `evaluate.py`, `decode.py` (the
switch level chosen leave one movie out), `rim_series.py` (tip detector rim response for the same grains),
`compare.py` (the final table), `curves.py` (per-grain probability plots). Outputs in `runs/research/onset_net/`
(git-ignored): `crops_*.npy` + `index_*.json` (337 MB), `on_*.pt` fold models, `eval_<movie>_v1.json` (tau 0.5) and
`eval_<movie>_v1tau.json` (tau chosen), `tiptrack/rim_*.json`, `compare_*.json`, `curves_*.png`, logs.

Reproduce: `python -m prototypes.onset_net.build ld m2 m1`; per fold `python -m prototypes.onset_net.train --train ld
m1 --name on_ldm1` then `python -m prototypes.onset_net.evaluate m2 --net runs/research/onset_net/on_ldm1.pt --tag v1`
(likewise m2 m1 -> ld, ld m2 -> m1); `python -m prototypes.onset_net.decode --tags ld=v1 m2=v1 m1=v1 --out v1tau`;
`python -m prototypes.onset_net.rim_series ld m2 m1`; `python -m prototypes.onset_net.compare --tags ld=v1 m2=v1
m1=v1`.
