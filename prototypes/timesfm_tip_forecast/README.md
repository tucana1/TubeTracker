# TimesFM-3 forecast-guided tip analysis (research prototype)

Zero-shot multivariate forecasting of pollen-tube tip trajectories with
Google's TimesFM-3 (`timesfm-3.0-pytorch`, 330M), fused with v29
mechanistic tracing into per-owner detection → acceptance → video.
No TimesFM training or fine-tuning; the v29 pipeline's measured tip
tracks are the context, the model's learned temporal prior does the rest.

> Scope: research prototype; validate against blinded manual centerlines
> before quantitative use. TimesFM-3.0 weights are non-commercial licensed.

## Spec (all inputs ablated, H144)

Joint targets `tip_x, tip_y, tube_length` (cross-series attention) +
trust-only past (`accepted`) + true sampling-schedule dynamic
(`time_minutes`) + 9 quantiles. Tip-speed / length-gain covariates were
ablated to zero effect and removed; joint-vs-marginal kept on
cost/benefit; window24-context for the burst flag, full-context for
everything else.

## Instruments (see `prototypes/LEDGER.md` H135–H156)

| # | instrument | catches | blind to |
|---|---|---|---|
| 1 | closed-loop veto (`guided_replay.py`) | extension flicker (P131: 10 faults, trips v29.36 gate) | smooth motion |
| 2 | HOLD + recovery filter (`guided_tip_filter.py`) | departures freeze; flicker-backs pass (dense 11→6, lowdens 3→1) | burst-labeled rejoins |
| 3 | open-loop rollout (24-step, triage) | unpredictable-from-emergence ranking (P61: 102) | late-window behavior; detonates past ~40 steps |
| 4 | switch-cut truncation (`switch_cut.py`, triage) | mid-course gap switches (cut onset = veto onset) | real-tube valleys (regresses — experimental) |
| 5 | root-zone validator (`normalized_root_support`, triage) | smooth background projections incl. P55 (passes every TimesFM test) | 0.099-vs-0.105 margin; grain-edge FPs |
| 6 | tip CNN + forecast gate (`guided_redetect.py`) | veto-blind phantoms via UNSUPPORTED holds; CNN proposes, TimesFM assigns | faint apexes (training) |

Product gate: v29.36 `--forecast-veto-dir` demotes tripped owners to
`forecast_stability_review` (tripped == {131} on both specs; purely
subtractive). Everything else is triage — thresholds too thin to gate.

## Honest negatives (selection)

- Residual magnitude alone can't split bursts from faults (gain needed).
- Surprise marks bursts, not faults; intervals don't mark bursts.
- Step-size ≈ veto on faults; TimesFM's edge is bursts/gates/rollouts.
- Kink, chord direction, longitudinal sensing, ridge extension,
  direction consensus, apex brightness: all dead with mechanisms.
- Corrected-series truncation REGRESSES on real-tube valleys (autopsy).
- Recursive rollout detonates past ~40 steps (end err 319k on clean P58).

## Layout

- `tip_tracks.py` — measurements.csv → tip trajectories
- `forecast.py` — TimesFM-3 wrapper (trust-only past)
- `backtest.py` / `run_backtest.py` — rolling backtest, baselines, variants
- `run_ablation.py` — single-factor input ablations
- `guided_replay.py` / `run_guided_replay.py` — veto + rollout
- `guided_tip_filter.py` — HOLD + recovery acceptance sim
- `guided_redetect.py` — CNN-confirm + forecast-gate re-detection
- `switch_cut.py` — support-dip truncation, root validator, ridge utils
- `mine_tip_dataset.py` — weak-label mining (instruments as filters)
- `eval_tip_cnn.py` / `eval_redetect_probe.py` — CNN + integration probes
- `render_guided_overlay.py` (`--guided`), `render_corrected.py`
  (experimental), `render_forecast_overlay.py` — videos (libx264 yuv420p)

## Field results

- Dense (48 owners): veto trips {131}; filter holds 6; root flags 10
  owners (10/11 eye precision); rollout queues P61/P93/P64/…
- Lowdens (35 owners): veto flags P22 tail only (no trip — rate gate);
  filter holds 1; P22/P3 rollout agreement ≠ biology (replays holds).
- Tip CNN (1,539 weak labels, 8/12 eye-good): bright apexes 4–6px,
  ghost/background silent, faint apexes missed (training).
