# Real-footage audit: per-bin vs prefix decoder (sample_movie, 37 grains)

Judged grain by grain by eye from panels (no labels exist; visual estimates ±10–20%). Per-grain verdicts, estimates
("seen" onset / b64 / end) and reasons: `verdicts.csv` in this folder; panels `panel_gXXX*.png`, films
`work/film_gXXX.png`, telling cases `telling_*.png`. PB = per-bin decoder (runs/learned_evidence/sample_movie/perbin),
PF = prefix decoder (agents/redesign/out/sample/refauto.json at the time).

## Counts
- per-bin right / prefix wrong: 8; prefix right / per-bin wrong: 4; both ok: 1; both wrong: 23; can't tell: 1.
- 7 of the prefix decoder's 8 "tubeless" calls are wrong (g005, g014, g016, g018, g019, g020, g022; g030 unclear).
- Onset within 6 bins of what was seen (33 grains with a visible onset): per-bin 26, prefix 12.
- Wrong tube: prefix end-state path is another grain's tube or none on 11 grains; per-bin final length comes from a
  foreign tube on 9.
- Does the grain have a tube? yes 33, no 2 (g024, g025 are out-of-focus ghost discs), unclear 2 (g004, g030).

## Failure modes shared by both decoders (the larger gains are here)
- The U-Net (trained only on synthetic movies) misses thick, dark, wide or defocused tubes: g005, g019, g020, g022,
  g002 and g011 onsets, g036 for PB. Coils read as ~10–16 px (g017, g023, g035).
- Frame-edge mask too broad: a pixel whose source leaves the frame in ANY bin is blocked in ALL bins, hiding tubes of
  grains that drift toward an edge (g016, g018, g037, probably g030).
- Census ghost discs (g024, g025): out-of-focus discs counted as grains.
- Abrupt frame-to-frame changes at b16→17 and b30→31 (and b28–32): field-wide frame difference doubles, like gaps in
  the recording (g013, g015, g026, g034, g035 affected).
- Grains dragged away / jumping / drifting tens of px (up to ~90 px): per-bin tracker wanders onto other grains'
  tubes; prefix path lands on whatever passes the census spot; template tracker loses grains that jump (g007, g013,
  g026, g018).

## Per-bin decoder failure modes
1. L1 monotone fit dragged down by long runs of failed readings (regions split at crossings or neighbours' rings, or
   the network half-misses the tube): g014 raw 120 px vs fit 18; g036 raw ~100 vs fit 51; g027 stuck at 47; g006 cut
   at g003's ring by geodesic ownership.
2. Reads foreign tubes after its tracker jumps or when a crossing tube joins the rim region (g021 351, g031 235, g002).
3. "Burst" misfires on grain jumps and scene changes (g034, g016, g017, g026, g037).
4. Late onsets where the network misses young dark tubes (g002, g011, g012, g018).

## Prefix decoder failure modes
1. False "tubeless" on clear tubes (end-state region at the rim too short, evidence starting off the rim, own stub
   rejected by the ownership test, frame-edge mask).
2. End state anchored on a foreign tube (crossing/passing tubes; pairs swapped: g003/g006).
3. Final length = length of the traced end-state path; tracing errors are never corrected by earlier evidence.
4. Blind to growth history outside its deformation models (rotations > 45°, big drift/jumps): g028, g033, g009, g032.
5. Holding after an anchor bin freezes tubes that kept growing (g012, g027, g010, g011, g036).

## Judgment
The prefix decoder's synthetic lead (84% vs 71%) does not carry over to this real movie; the per-bin decoder degrades
more gracefully. On 23 of 37 grains both are wrong, so the per-bin decoder is not validated on real footage either.
The sample movie is real but unlike the lab's own movies: thick double-walled tubes, grains moving tens of px, one
frame per bin, fast growth. Recommendation: keep the per-bin decoder; the larger gains are in shared inputs
(evidence for thick/dark/wide tubes, a per-bin frame-edge mask, trackers that survive jumps, removing census ghosts).
