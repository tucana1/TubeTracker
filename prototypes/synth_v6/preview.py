"""Short synthetic movies for looking at young tubes: onsets forced into the first bins, encoded and
binned exactly like the real movies (x264 intra frames, 25 frames per bin), then drawn as contact sheets
with ``sheet.py`` next to real young tubes.

    python -m prototypes.synth_v6.preview m2 --preset v6 --seed 1 --look
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/synth_v6/preview"
FIELDS = {"ld": "runs/sparsetrack/ld", "m2": "runs/sparsetrack/m2", "m1": "runs/sparsetrack/m1"}
M2 = dict(rate_px_per_bin=(0.15, 1.5), max_length=280.0)
LOOK = dict(p_bright_core=0.8, width=(1.0, 1.6))


def make(field: str, preset_name: str, seed: int, n_bins: int = 45, onset=(8.0, 16.0), look: bool = False,
         synth_module=None, **over) -> Path:
    from sparsetrack import stack
    from sparsetrack.cli import write_census
    S = synth_module or __import__("sparsetrack.synth", fromlist=["x"])
    name = f"pv_{field}_{preset_name}_s{seed}{'_look' if look else ''}"
    cache = OUT / f"{name}_cache"
    if (cache / "grains.json").exists():
        return cache
    extra = {**(M2 if field != "ld" else {}), **(LOOK if look else {}), **over}
    cfg = S.preset(preset_name, seed=seed, n_frames=n_bins * 25, onset_bins=onset, **extra)
    movie = S.make_movie(REPO / FIELDS[field], OUT, cfg, name=name)
    with contextlib.redirect_stdout(io.StringIO()):
        stack.prepare(movie, cache, frames_per_bin=25, ref_bins=3, ref_start=0, log=lambda *a: None)
        write_census(cache, 3, False)
    shutil.copy(OUT / f"{name}_truth.json", cache / "truth.json")
    return cache


if __name__ == "__main__":
    import argparse
    from .sheet import View, sheet, synth_onsets
    ap = argparse.ArgumentParser()
    ap.add_argument("field", choices=tuple(FIELDS))
    ap.add_argument("--preset", default="v6")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--look", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--dev", help="import the generator from this file instead of sparsetrack.synth")
    a = ap.parse_args()
    mod = None
    if a.dev:
        import importlib.util
        spec = importlib.util.spec_from_file_location("synth_dev", a.dev)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["synth_dev"] = mod
        spec.loader.exec_module(mod)
    cache = make(a.field, a.preset, a.seed, look=a.look, synth_module=mod)
    ons = synth_onsets(cache / "truth.json")
    T = json.loads((cache / "truth.json").read_text())
    gids = sorted(ons)[:a.n]
    out = OUT / f"{cache.name.replace('_cache', '')}.png"
    sheet(View(cache), ons, gids, (-3, 0, 3, 6, 10, 20), out, tag="syn ")
    for g in gids:
        t = T["labels"][g]["truth"]
        print(g, {k: t.get(k) for k in ("bright_core", "width", "bulb", "body_change", "evolves", "rate_px_per_bin",
                                          "amplitude", "stub")})
    print(out)
