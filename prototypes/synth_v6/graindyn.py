"""Noise-free size of the grain's own change in a v6 scene, and what makes it: every grain sprite rendered at
random frames (wander + focus + interior, or one at a time) minus its reference look, RMS inside r + 3 px.

    python -m prototypes.synth_v6.graindyn m2 40 [key=value ...]
"""

from __future__ import annotations

import sys

import numpy as np

from sparsetrack.synth import Scene, _focus_delta, _paste, preset

from .shards import LOOK, M2

FIELDS = {"ld": "runs/sparsetrack/ld", "m2": "runs/sparsetrack/m2", "m1": "runs/sparsetrack/m1"}


def rms_parts(scene: Scene, n_frames: int = 40, seed: int = 0) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    ks = rng.integers(3 * 25, scene.cfg.n_frames, n_frames)
    out = {"all": [], "wander": [], "focus": [], "interior": []}
    census = {g["id"]: g for g in scene._census}
    for gid, (S, al, x0, y0) in scene.dsprites.items():
        g = census[gid]
        d = scene.dyn[gid]
        h, w = S.shape
        yy, xx = np.mgrid[0:h + 1, 0:w + 1]
        disc = np.hypot(xx + x0 + 0.5 - g["x"], yy + y0 + 0.5 - g["y"]) < g["r"] + 3

        def draw(sp, jx, jy):
            canvas = np.zeros((h + 1, w + 1), np.float32)
            import cv2
            _paste(canvas, sp, al, float(jx), float(jy), cv2.INTER_CUBIC)
            return canvas
        ref = draw(S, 0.0, 0.0)
        for k in ks:
            jx, jy = (d["jit"][k] if d["jit"] is not None else (0.0, 0.0))
            f = (scene.focus[k] if scene.focus is not None else 0.0) + (d["focus"][k] if d["focus"] is not None else 0.0)
            foc = _focus_delta(d["D"], f) if d["D"] is not None else 0.0
            inn = d["c"][k] * d["M"] if d["c"] is not None else 0.0
            for name, sp, j in (("all", S + foc + inn, (jx, jy)), ("wander", S, (jx, jy)), ("focus", S + foc, (0, 0)),
                                ("interior", S + inn, (0, 0))):
                out[name].append(float(np.sqrt(np.mean((draw(np.asarray(sp, np.float32), *j) - ref)[disc] ** 2))))
    return out


if __name__ == "__main__":
    field, seed = sys.argv[1], int(sys.argv[2])
    over = {**(M2 if field != "ld" else {}), **(LOOK if field != "ld" and seed % 2 else {})}
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        over[k] = tuple(float(x) for x in v.split(",")) if "," in v else float(v)
    import sparsetrack.synth as S
    orig = S.Scene._grain_dynamics

    def keep_census(self, rng, census, original):  # remember the census for the measurement
        self._census = census
        return orig(self, rng, census, original)
    S.Scene._grain_dynamics = keep_census
    sc = Scene(FIELDS[field], preset("v6", seed=seed, **over))
    f = np.abs(sc.focus) if sc.focus is not None else np.zeros(1)
    print(f"{field} seed {seed} {over if len(sys.argv) > 3 else ''}: field focus |sigma| p50 {np.median(f):.2f} max {f.max():.2f}")
    parts = rms_parts(sc)
    for k, v in parts.items():
        print(f"  {k:9s} RMS inside r+3 p25/50/75/90 " + " ".join(f"{x:.1f}" for x in np.percentile(v, [25, 50, 75, 90])))
