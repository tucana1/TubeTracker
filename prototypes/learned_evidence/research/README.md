# Research records and kit

What the assessment's sections 5 and 7 rest on, kept so that the work can be checked and carried on from a fresh
checkout. The prototype itself is the folder above; nothing here is needed to run it on a movie.

## Records

| File | What it is |
|---|---|
| `heldout_rules.txt` | Every adoption rule, written before its round's results, and each round's outcome: rounds 2–4 (rules A–F) and round 5 (G, H). Read it before changing a default. |
| `audit/verdicts.csv`, `audit/report.md` | A visual audit of the real sample movie's 35 grains (25 Sep 2026): onset, mid-movie (bin 64) and end length estimates, good to about 10–20%. It is the only real-footage check here. |
| `round5/` | Round 5's brief and, once it reports, its agents' reports and code. |

## The protocol

- **Development movies** (`common.DEV`, all synthetic, on the sample movie's field):
  - thin v5 seeds 3–8 and 13–15 (7, 8 and 13–15 were held out until 27 Sep, and were retired after four looks);
  - faint v5faint s30 and s31, thick v5thick s30 and s31, wide v5w s26 and s27.
  - Use them freely. Never train on them.
- **The sealed held-out set** (`common.FRESH`): v5 s40–44, v5faint s40, v5thick s40, v5w s40. It is rendered only
  for the one look at a frozen candidate, with the rule written before the look. The kit refuses these movies
  unless a look asks for them.
- **Training data so far:** v2 on v5 s0, 1, 2, 9, 10, 11, v2 s0, 1 and v3 s0, 2; B3 on round 3's synthetic negatives.
  New training movies take seeds 50 and up.
- **Movie 2 of the lab** is the real held-out benchmark: never train, calibrate or tune on it, and score each frozen
  version once (`pipeline.py --heldout-once`).
- **Every comparison is paired, under one decoder,** with a 95% bootstrap over grains. It uses the original truth and
  the human-style documents ("human_t2", `humanstyle.py`). The real movie is judged by the audit (`real_check.py`)
  as a gross-failure check only.

## The kit

Everything lives under `$LE_RESEARCH` (default `runs/learned_evidence/research`, which git ignores).
Run each module from the repository root with `.venv/bin/python -m prototypes.learned_evidence.research.<module>`.

| Module | What it does |
|---|---|
| `common` | Movie names, groups, the seal, paths. |
| `regen` | Rebuilds the data in a new container, in this order: `regen field` (the real sample movie's cache, the synthetic movies' field), `regen movies`, `regen maps`, `regen baselines`. That is hours of CPU and about 0.5 GB per prepared movie. |
| `sparse` | Networks' probability maps stored sparse (P < 0.001 dropped, the rest exact): `sparse MODEL MOVIE ...`. |
| `decode` | Fuses a base map with B3's and decodes with the repository's `reach.analyze` plus continuity, exactly as `pipeline.py` does: `decode TAG BASE THICK MOVIE,...`. The current default is `decode default v2 B3 ...`. |
| `score` | `score BASE_TAG NEW_TAG`: every group, orig and human_t2, paired intervals, and round 5's rule preview. |
| `real_check` | `real_check BASE_TAG NEW_TAG`: the audit's counts, and the gross fixes and breaks grain by grain. |

Checked on 27 Sep 2026 against the cloud session's own data: the kit's decode reproduces the default's predictions
exactly, and its scores and audit counts match the assessment's.

### Development numbers of the current default

Fused v2 + B3 with continuity; lengths within max(2 px, 10%):

| Group | Lengths in tolerance | Share |
|---|---|---|
| Thin (9 movies) | 1310 / 1807 | 72.5% |
| Faint | 209 / 385 | 54.3% |
| Thick | 88 / 419 | 21.0% |
| Wide | 311 / 445 | 69.9% |

Onsets within 2 bins: thin 150 / 203, faint 16 / 44.

Real sample movie (audit):

| Measure | Result |
|---|---|
| Onsets within 6 bins | 29 / 35 |
| Mid-movie lengths within 25% | 23 / 35 |
| End lengths within 25% | 20 / 34 |
