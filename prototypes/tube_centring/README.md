# Where the drawn tube lies (1 Oct 2026, SparseTrack 0.8.3-0.8.8)

The measurements behind `learned.centre_route`, the per-bin bend, `drawn_check` and the flood's per-bin routes
(`docs/status-2026-09-29.md`, 1 Oct). Research scripts: their input and output paths point at the session's scratch
folder where they were run; pass your own (most take them as arguments).

- `centre_profiles.py MOVIE PRED.json OUT.npz`: every px of every full trace beyond the grain, the image and the tube
  network's map across the trace, and where the drawn tube (as the app draws it) crosses the trace's normal.
  `prof_plot.py` draws the mean cross-section (`prof_all.png`: two dark walls ~7 px apart, the traces on the clear
  middle, the old routes on the darker wall).
- `perbin_eval.py`: the stored route against the traces, against the route re-centred on each bin's own map, and
  against a rigid per-bin shift (per-bin centring won).
- `ontube.py`, `ontube_eval.py`: the share of the drawn tube on the network's map (the `drawn_off_tube` check).
- `apply_centre.py MOVIE IN OUT [key=value]`: the pipeline's centring applied to saved predictions.
- `before_after.py`, `centre_look.py`, `centre_time.py`: close-ups with the traces (`before_after_ld.png`).
- `apptry.py`: drives the TubeTracker app on a scratch copy of a run folder and saves screenshots.
- `m2_routes/`: a look at movie 2's 15 worst-drawn traces, case by case (`sheet.png`, `numbers.txt`).
