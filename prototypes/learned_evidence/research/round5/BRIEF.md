> Round 5's brief as the agents received it on 27 Sep 2026. Its paths (SCR/..., SCR/round5/...) are the cloud session's scratchpad; the kit in `../` (common, sparse, decode, score, real_check) replaces SCR/round5's tools one for one.

# Shared brief for round-5 agents (read fully before starting)

SCR = /tmp/claude-0/-home-user-TubeTracker/21fce780-cdc3-5467-a04a-73e34cbe3068/scratchpad
R5 = SCR/round5 (shared tools, read-only for you; write only in your own folder SCR/agents/<your name>/)

## The problem

Pollen-tube time-lapse microscopy:
- Grains are dark discs, radius about 10–15 px, on a substrate. Some germinate and grow a tube out of their rim.
- Growth is at the tip only: every earlier tube is a prefix of the later one, and lengths never shrink (bursts aside).
- Per grain we measure the onset (germination bin) and the tube length at every bin, exit to apex along the centreline.
- The lab's own movies are NOT in this container. Their tubes are thin and FAINT (2–5 px wide), 300 frames per bin,
  about 176 bins, low density, H.264 compressed. **Faint and thin synthetic tubes are the regime closest to theirs.**
  A change that costs there costs the lab; a change that gains there is what this round is for.

## The current pipeline (the default since 27 Sep)

1. **Learned evidence.** A 0.49 M-parameter U-Net (`prototypes/learned_evidence/model.py`; shipped model
   `models/unet_v2_sample_field.pt`, called "v2") reads three registered images (the bin, a "before" and an "after"
   reference) and outputs per pixel P(tube built here). It was trained only on codec-exact synthetic movies from
   `sparsetrack synth` on the sample movie's field: presets v5 (seeds 0, 1, 2, 9, 10, 11), v2 (0, 1), v3 (0, 2), with
   `prototypes/learned_evidence/pipeline.py`'s recipe (`data.build` shards, `train.main`, 6000 steps). Shards:
   SCR/le/shards/train_*.npz (and wide-tube v5w shards in SCR/le/shards_wide/). **No faint-amplitude movie was in
   training**: the faint development movies (tube amplitude 0.35–0.7 against v5's 0.9–2.0) are outside its
   training range.
2. **Fused evidence** (`prototypes/learned_evidence/fuse.py`): v2's P everywhere, except inside `wide_gate(P_B3)` — B3
   (`models/unet_thick_b3.pt`, a network taught thick hollow tubes) marks a structure ≥ about 7 px wide — where the
   larger of the two is taken.
3. **The per-bin decoder** (`reach.py`, `reach.analyze(pcache, image_cache, big=300, burst=True, vmax=4.0,
   continuity="path")` on synthetic movies; vmax 16 on the real movie): in each bin the region attached to the grain
   (P > 0.5), its medial-axis length, then a monotone growth fit; tip-growth continuity keeps foreign tubes out.

Measured so far (development, lengths within max(2 px, 10%)): thin about 73%, faint about 52%, thick about 25%,
wide about 70%. Faint tubes are the weakest regime that matters to the lab.

## Shared tools (use them; one implementation for everyone)

- `R5/common5.py`: movie lists and paths. Development movies:
  - thin v5 seeds 3, 4, 5, 6, 7, 8, 13, 14, 15 (7, 8, 13–15 were held out until 27 Sep and are now retired to
    development);
  - faint v5faints30, v5faints31; thick v5thicks30, v5thicks31; wide v5ws26, v5ws27;
  - "real": the real sample movie (runs/sparsetrack/sample_movie/cache), for the audit only.
  - Image caches: SCR/synth/<movie>_cache. Truth: see `truth_path`.
- **The FRESH held-out set** — v5 seeds 40–44, v5faint s40, v5thick s40, v5w s40 — is rendered by me, sealed, and
  scored once per frozen candidate by me. **Never render, open, score or train on these seeds. Never train on any
  development seed either (3–8, 13–31).** Train only on seeds ≥ 50 (or the existing training shards).
- Sparse probability caches (P x 16 as float16, values under P = 0.001 dropped; exact otherwise):
  `<model>_<movie>.npz` in R5/sparse/ and SCR/agents/fusion/sparse/ (`common5.sparse_path`). Load with
  `from sparse import Sparse` (SCR/agents/fusion/sparse.py; `Sparse(path).bin(b)` gives one dense bin). v2 and B3
  caches exist for every development movie (thick s31 and seeds 7, 8, 13–15 are being added now; check).
  `R5/sparse5.py build <model> <movie>` shows how a network pass is stored sparse (same code path as
  `evaluate.prob_cache`); copy and adapt it in your folder for your own models.
- `R5/decode5.py`: fuses a base sparse cache with a thick one using the REPOSITORY's `fuse.fuse`, and decodes with the
  repository's `reach.analyze` at the current defaults plus continuity — exactly what `pipeline.py` does. From Python:
  `decode(base_npz, thick_npz, movie, out_json)`. It reproduces round 4's predictions exactly.
- **Baselines** (the current default): R5/preds/<movie>/default.json, being written now by R5/baselines.sh (log:
  R5/logs/baselines.log). Wait for a movie's baseline before comparing on it; do not rewrite them.
- `R5/score5.py BASE NEW`: paired comparison over the groups (thin, faint, thin+faint pooled, thick, wide), original
  truth and human-style documents ("human_t2"), with the round-5 rule's development preview. It reads
  R5/preds/<movie>/<TAG>.json: write your predictions to YOUR folder and either symlink them under R5/preds/<movie>/
  with a tag prefixed by your agent name (e.g. `tta_v2x4.json`), or import `score5.compare` after pointing its PREDS
  at your folder.
- The real-movie audit: SCR/agents/audit/verdicts.csv (per grain est_onset, est_b64, est_end, visual estimates ±10–20%)
  and SCR/agents/simreal/real_check.py (onsets within 6 bins, lengths within ±25%; "≥X" is a lower bound). The current
  default's real predictions: R5/preds/real/default.json (and SCR/agents/fusion/preds/real/repo_w3f_B3_on.json).

## The round-5 rule (written before any result; SCR/heldout_rules_round2.txt)

A candidate becomes the default only if ALL hold on the fresh held-out set (one look, by me):
1. thin+faint pooled lengths: 95% lower bound > 0 (a clear gain where the lab's movies are);
2. thin onsets ≥ −1.0 pt; thick and wide lengths ≥ −1.0 pt with lower bound ≥ −3.0 pt;
3. the real-movie audit (judged before the look) shows no more gross breaks than gross fixes;
4. analysis time on a laptop at most doubles.
You choose your candidate on development only, and say which one to freeze.

## Rules

- **Repository** /home/user/TubeTracker: read and import only; do NOT modify anything there. Python:
  /home/user/TubeTracker/.venv/bin/python. Lint code you propose for the repo with /root/.local/bin/ruff.
- **Your folder only:** SCR/agents/<your name>/. Absolute paths; don't `cd` in commands. zsh aborts on unmatched
  globs: wrap globs in `bash -c '...'`. Foreground `sleep` is blocked: run long jobs with run_in_background and poll
  with short checks. Never `pkill -f` a pattern that appears in your own command line.
- **Shared machine:** 4 CPU cores, shared with another agent and a long regression run. Use at most 2 threads
  (`torch.set_num_threads(2)`, `OMP_NUM_THREADS=2`, `cv2.setNumThreads(1)`). About 6 GB of free disk for everyone:
  keep your folder under 1.5 GB, store probability maps sparse, delete dense caches and image caches you create as
  soon as they are used. The container can restart: write results as you go.
- **Honesty:** report failures as plainly as successes; say which numbers are development numbers; don't overfit to
  the one real movie.
- **Deliverable:** your final message IS the report: what you tried, exact commands, file paths of any model or code,
  development numbers (every group, orig and human_t2, paired 95% intervals) against R5 default, the real-movie
  audit grain by grain for grains that change, the time cost, and a clear recommendation with the one candidate to
  freeze (or none).
