# Handover

The work was done in a cloud session that had none of the lab's movies. It paused on 27 Sep 2026. It is written up in
`docs/assessment-2026-09-24.md` (sections 5 and 7; appendices B and C) and in this folder's `README.md`.

- **Part A** is for a person or a local coding agent in the lab's checkout, where the movies, the prepared caches and
  the labels live: how to run the prototype on them, and score movie 2 once.
- **Part B** is for whoever carries the research on: the state at the pause, where everything is, the protocol, the
  open threads, and how to resume in a new session.

# Part A: running it on the lab's own movies

## What the branch adds

Branch `claude/magical-maxwell-i5tpeh` adds only its own files:
- `prototypes/learned_evidence/` (with `models/`);
- `docs/assessment-2026-09-24.md`;
- `tests/test_learned_evidence_*.py`;
- three launchers at the root: `Adapt_Learned_To_Dev_Movie.command`, `Analyze_Movie_Learned.command` and
  `Review_Movie_Learned.command`.

It does not change `sparsetrack/`, `benchmark/labels/` or the labelling tool. So merging it into the lab's branch
should not conflict.

## Rules that must hold

1. **Movie 2 is the held-out benchmark** (`benchmark/labels/m2_v1.json`, `runs/sparsetrack/m2`).
   - Never train, fine-tune or calibrate on it, and never choose a setting by looking at it. The scripts refuse
     labels whose name contains `m2`, except `pipeline.py --heldout-once`.
   - Decide everything on the dev movie first, and write it down (step 4).
   - Score each frozen version once. Never score it again with other settings after seeing the result.
2. **Leave `sparsetrack/`, `benchmark/labels/` and the labelling tool alone** for this work.
3. **Keep `runs/` out of git.** Its probability caches are about 0.5 GB per network per movie.

## Steps

### 1. Merge and test (about 5 minutes)

Start from the lab's checkout, on its own branch, with its work committed.

```bash
git fetch origin
git merge --no-ff origin/claude/magical-maxwell-i5tpeh
.venv/bin/pip install torch==2.13.0      # once, if torch is missing
.venv/bin/python -m pytest -q tests      # every test must pass
```

If the merge conflicts, stop and ask: the branch adds only its own files.

### 2. Adapt to the dev movie (about 2.5–3 hours the first time)

This step needs the prepared dev movie (`runs/sparsetrack/ld`) and the finished dev labels
(`benchmark/labels/ld_v1.json`).

```bash
.venv/bin/python -m prototypes.learned_evidence.adapt --field runs/sparsetrack/ld \
    --labels benchmark/labels/ld_v1.json
```

Double-clicking `Adapt_Learned_To_Dev_Movie.command` does the same. It runs four steps, and skips each one once it
is done:
1. the dev test: it trains a model on the dev movie's field (about 1.5 hours) and scores it on the labels;
2. calibration of the decoder;
3. fine-tuning;
4. trace-once.

If it says the labels have changed, or that a step read the movie otherwise than now, run it again with `--redo`
(the launcher offers this).

### 3. Read `runs/learned_evidence/SUMMARY.md`

- **Section 1, the dev test.** Under the report, one line says whether your labels find the default reading (fused
  evidence with continuity) worse than the reading from before 27 Sep. The rule: worse means a 95% interval below
  zero for lengths or for onsets.
  - If it says **worse**, switch back: `python -m prototypes.learned_evidence.adapt --reading plain --redo` (the Adapt
    launcher offers this). Every step then reads the old way, and the choice is kept for later runs, the Analyze
    launcher and the movie-2 command.
  - Otherwise keep the default.
  - Decide this now, on the dev movie.
- **Sections 2 and 3.** They say whether calibration (`ld_cal/decoder.json`) and fine-tuning (`ld_ft/unet_ft.pt`)
  were adopted. "What is used now" names the model and decoder settings that result.
- **Section 4,** trace-once, is for information.

### 4. Freeze before movie 2

Write `runs/learned_evidence/FROZEN.md` with:
- the commit (`git rev-parse HEAD`);
- the model file and its `shasum`;
- the decoder file (or "none") and its `shasum`;
- the reading (`runs/learned_evidence/reading.json`: fused or plain);
- the date.

### 5. Score movie 2, once

1. First SparseTrack, 0.4.0 and 0.4.3, as appendix C of the assessment says.
2. Then the learned pipeline, with the frozen model, decoder and reading: the command under "Movie 2, once" in
   `SUMMARY.md`, as it stands (it carries `--no-thick-model --no-continuity` if the plain reading was chosen). It
   looks like this:

```bash
.venv/bin/python -m prototypes.learned_evidence.pipeline --field runs/sparsetrack/m2 \
    --labels benchmark/labels/m2_v1.json --model <the frozen model> \
    --decoder runs/learned_evidence/ld_cal/decoder.json \
    --work runs/learned_evidence/m2 --prefix --heldout-once
```

Leave out `--decoder` if calibration was not adopted. This takes about 30–60 minutes.

### 6. Report back

Send `SUMMARY.md`, `FROZEN.md` and `runs/learned_evidence/m2/report.txt` to whoever continues the work, as text.
Tune nothing after this.

## Also available

- **Any movie:** `Analyze_Movie_Learned.command` analyses a chosen movie with the adapted model (about 15–25
  minutes). `Review_Movie_Learned.command` then opens the labelling tool pre-filled with the model's answers, and
  exports the reviewed results.
- **Help:** each script's `--help`, and the README's module table.

## If something fails

- **"no prepared cache":** open `Label_Sparse_Benchmark.command` once on the movie, or run
  `python -m sparsetrack prepare MOVIE --out runs/sparsetrack/ld`.
- **"torch is needed":** `.venv/bin/pip install torch==2.13.0`.
- **Out of disk:** delete the `prob_*` folders under `runs/learned_evidence/*_old/`. Steps build their caches again
  when needed.
- **A step failed half-way:** run the same command again. Finished steps are skipped, and a step's report is written
  last.

# Part B: carrying the research on

## State at the pause (27 Sep 2026)

- **The default pipeline** has been in place since 27 Sep:
  - evidence from the synthetic-trained network v2;
  - fused with the thick-tube network B3 where B3 marks a structure about 7 px wide or more;
  - read by the per-bin decoder with tip-growth continuity.
  - It was adopted under rule F. On held-out synthetic movies, thin tubes were not hurt: +2 lengths of 1014, 95% CI
    −12 to +18. On the real sample movie, mid-movie lengths went from 17 to 23 and end lengths from 15 to 20 of
    about 35, but those numbers are in-sample.
- **Nothing has been measured on the lab's own footage yet.** The dev test on `ld_v1` (Part A, step 2) is the first
  real test; movie 2 is the held-out one.
- **Checks:**
  - the test suite (`pytest -q tests`);
  - an independent code review of the fusion default, with every finding fixed (commit 65ea175);
  - end-to-end runs of Adapt, Analyze and Review on a synthetic dev movie (assessment, appendix B).
- **Round 5** looks for label-free gains on faint tubes, the regime closest to the lab's movies. It was running when
  the work paused. Its rule is in `research/heldout_rules.txt`; its brief, reports and any candidate are in
  `research/round5/`.
- **Nothing runs now.** The daily routine that checked for movie-2 labels is disabled.

## Where everything is

| What | Where |
|---|---|
| Every module and how to run it | this folder's `README.md` |
| Findings and numbers, round by round | `docs/assessment-2026-09-24.md`, section 7 |
| Adoption rules and outcomes, written before each round's results | `research/heldout_rules.txt` |
| The real sample movie's visual audit | `research/audit/` |
| The kit: rebuild the synthetic data, decode, score, audit check | `research/` (its README) |
| A summary for the lab (private) | https://claude.ai/artifact/CeHnAbWFmhry5ykjoNci1n |

## The protocol

See `research/README.md` for the details. In short:
- choose on development movies;
- write the adoption rule before any result;
- give each frozen candidate one look at the sealed held-out set (v5 s40–44, v5faint s40, v5thick s40, v5w s40);
- judge the real sample movie by its audit, as a gross-failure check only;
- never train, calibrate or tune on movie 2, and score each frozen version on it once.

## Open threads, most important first

1. **The lab's dev labels, then Adapt** (Part A, steps 2 and 3). Adapt decides whether the fused reading stays (the
   dev test's verdict line), calibrates the decoder's end offset, and tries fine-tuning.
2. **Movie 2, once** (Part A, steps 4 and 5).
3. **Round 5's outcome** (`research/round5/`): a candidate frozen on development gets one look at the sealed set
   under rule G/H, then the real-movie audit, before it can become a default.
4. **Known weak spots,** from the development numbers in `research/README.md` and the audit:
   - faint tubes: about 54% of lengths in tolerance;
   - thick tubes read long: 21% raw, though the end-offset calibration corrects much of this per movie;
   - 8 real grains where the fused evidence bridges onto a neighbour's tube;
   - early onsets from dark structures at grain rims.

## Resuming in a new cloud session

```bash
git checkout claude/magical-maxwell-i5tpeh
# .venv as the repository sets it up (pyproject.toml), with the extras "cnn" (torch 2.13.0) and "dev" (pytest)
.venv/bin/python -m pytest -q tests
export LE_RESEARCH=runs/learned_evidence/research            # the kit's data (git ignores runs/)
.venv/bin/python -m prototypes.learned_evidence.research.regen field
.venv/bin/python -m prototypes.learned_evidence.research.regen movies      # hours; about 7 GB
.venv/bin/python -m prototypes.learned_evidence.research.regen maps
.venv/bin/python -m prototypes.learned_evidence.research.regen baselines
.venv/bin/python -m prototypes.learned_evidence.research.score default default   # zero change: the kit works
```

The earlier session kept to these rules:
- it committed only to this branch;
- it touched only `prototypes/learned_evidence/`, `docs/assessment-2026-09-24.md`, `tests/test_learned_evidence_*.py`
  and the three launchers;
- it never touched `sparsetrack/`, `benchmark/labels/` or the lab's branch `sparse-reset-2026-09-23`.

If the lab's movie-2 labels land, rebase this branch onto the lab's branch; its commits touch only its own files. Then
follow Part A.
