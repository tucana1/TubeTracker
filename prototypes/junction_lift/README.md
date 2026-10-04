# Junction takeovers and bending-penalised minimal paths (3 Oct 2026)

Where a pollen tube meets, touches or crosses another, the flood's reading can follow the other tube (movie 2:
5-7 of its 15 worst traces). This folder asks whether bending-penalised (orientation-lifted) minimal paths choose
the right branch, and measures `Params.flood_lift_px` (`sparsetrack/learned.py`: `lifted_path`, `_lift_join`;
off by default) end to end against SparseTrack 0.8.8. Findings and numbers: `results.txt`.

Short answer: on the tube map, the right branch is the straight one, and minimal paths choose it with or without a
bending penalty. The flood's takeovers come from what it claims and when (an older crossing tube it cannot bridge,
another grain's disc on the tube's way, a passing tube that reaches a stalled tip, a moved tube), not from its
choice among claimed branches. A lift across older crossing tubes along a bending-penalised path changes almost
nothing end to end; it stays off.

Scripts (run from the repo root with the venv python; they read the main checkout's labels and probability caches
and the 0.8.8 baseline predictions named in `common.py`, and write to the session scratchpad):

| script | what it does |
|---|---|
| `find_cases.py` | 0.8.8's missed traces whose drawn route leaves the human route onto tube material, and traces passing within ~6 px of another tube (`cases.json`) |
| `reflood.py` | re-runs the flood for one grain (same result as 0.8.8) and keeps its arrival map and claimed tube |
| `viz.py`, `along.py`, `claim_cov.py`, `blockers.py` | what the flood claimed and when, along each human route; routes through other grains' discs |
| `lifted.py` | plain (x, y) and lifted (x, y, direction) minimal paths on a P map (16 or 32 grid directions, oriented line evidence, w per radian turned) with scipy's dijkstra |
| `route_test.py`, `route_dec.py`, `angles.py`, `summarise.py`, `viz_routes.py` | the offline route test: exit-to-apex routes, the junction decision, turn angles |
| `eval_flood.py` | scores a flood option against 0.8.8 by re-reading only the flood-read grains (the hybrid's choice of reader does not depend on flood options) |
| `trace_diff.py`, `viz_lift.py`, `lift_check.py` | which traces an option changed, and how |
| `bench.sh` | the full `scripts/synth_bench.py` benchmark, one movie at a time, waiting for a quiet machine |

Follow-up (4 Oct, `ownership.txt`): passage through another grain's disc (`Params.flood_disc_pass`) and a start on
probation against tubes passing the rim (`Params.flood_pass_bins`), both off. Scripts: `disc_diag.py` and
`disc_open_all.py` (what the map shows inside the discs on the traced routes), `start_diag.py`, `start_evo.py` and
`wrong_starts.py` (what starts look like, at and after their bin), `opt_check.py` (an option on chosen grains).

**Note (main session, 3 Oct):** the Params.flood_lift_px option itself (commit d309bb6, with tests) was not merged (no score change on any movie); it lives on branch worktree-agent-acd9ac5f4741507c2. Scripts that use it need that branch.

**Note (main session, 4 Oct):** the ownership options (flood_disc_pass, flood_pass_bins; commit fdbfb86 with tests) were not merged (no length change on any movie); they live on branch junction-ownership. See ownership.txt.
