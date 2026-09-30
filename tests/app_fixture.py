"""A tiny synthetic analysis folder for the app's tests: a 40-bin cache whose reference starts at bin 2 (as a
movie that settled first), four grains, and SparseTrack-shaped predictions.

- g001: germinates at bin 10, its tube grows 2 px per bin straight along +x from its exit; touches g002.
- g002: never germinates.
- g003: germinates at bin 12, grows 1 px per bin, drifts +0.5 px per bin in x, and is lost after bin 25.
- g004: in a clump (not isolated); germinated before the start, grows to 20 px by bin 12 and stops.
The movie's focus changes at bin 20.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FPB, N_BINS, REF_START, W, H = 300, 40, 2, 200, 160
CENTRES = {"g001": (60.0, 50.0), "g002": (150.0, 50.0), "g003": (60.0, 115.0), "g004": (150.0, 115.0)}


def frame(b: int) -> int:
    return b * FPB + FPB // 2


def _series(by_bin: list) -> list:
    return list(by_bin[REF_START:])


def predictions() -> dict:
    frames = [frame(b) for b in range(REF_START, N_BINS)]
    L1 = [0.0 if b < 10 else 2.0 * (b - 9) for b in range(N_BINS)]
    L3 = [0.0 if b < 12 else float(min(b, 25) - 11) for b in range(N_BINS)]
    L4 = [min(20.0, 4.0 + 2.0 * b) for b in range(N_BINS)]
    x1, y1 = CENTRES["g001"]
    x3, y3 = CENTRES["g003"]
    x4, y4 = CENTRES["g004"]
    drift3 = [[0.5 * min(b, 25), 0.0] for b in range(N_BINS)]
    g1 = {"id": "g001", "x": x1, "y": y1, "r": 8.0, "flags": ["reader:flood", "touches:g002"],
          "status": "emerged_within", "onset_frame": frame(10), "onset_interval": [frame(9), frame(10)],
          "length": {"frames": frames, "px": _series(L1)}, "rotation_deg": [0.0] * len(frames),
          "path": [[x1 + 8.0 + k, y1] for k in range(0, 90, 2)], "exit_xy": [x1 + 8.0, y1],
          "final_length_px": L1[-1]}
    g2 = {"id": "g002", "x": CENTRES["g002"][0], "y": CENTRES["g002"][1], "r": 8.0, "flags": [],
          "status": "no_emergence_by_end", "onset_frame": None, "onset_interval": None,
          "length": {"frames": frames, "px": [0.0] * len(frames)}, "path": []}
    g3 = {"id": "g003", "x": x3, "y": y3, "r": 8.0, "flags": ["onset_at_focus_change", f"grain_lost_after:{frame(25)}"],
          "status": "emerged_within", "onset_frame": frame(12), "onset_interval": [frame(11), frame(12)],
          "length": {"frames": frames, "px": _series(L3)}, "drift": {"frames": frames, "xy": _series(drift3)},
          "path": [[x3 + 8.0 + k, y3] for k in range(0, 40, 2)], "exit_xy": [x3 + 8.0, y3],
          "observed_until_frame": frame(25), "lost_reason": "gap", "final_length_px": L3[-1]}
    g4 = {"id": "g004", "x": x4, "y": y4, "r": 8.0, "flags": [], "status": "emerged_at_start",
          "onset_frame": frames[0], "onset_interval": None, "length": {"frames": frames, "px": _series(L4)},
          "path": [[x4 - 8.0 - k, y4] for k in range(0, 40, 2)], "exit_xy": [x4 - 8.0, y4], "final_length_px": 20.0}
    return {"schema": "sparsetrack.pred.v1", "method": "sparsetrack-v1 test", "frames_per_bin": FPB,
            "params": {"rot_pivot": "exit"}, "focus_changes": [{"bin": 20, "frame": frame(20), "ratio": 3.0}],
            "grains": [g1, g2, g3, g4]}


def make_run(root: Path, setup: dict | None = None) -> Path:
    """Write the run folder (cache, analysis, and a setup if given) under ``root``; returns it."""
    root = Path(root)
    cache = root / "cache"
    cache.mkdir(parents=True)
    rng = np.random.default_rng(0)
    bins = 150.0 + 3.0 * rng.standard_normal((N_BINS, H, W))
    yy, xx = np.mgrid[0:H, 0:W]
    for (cx, cy) in CENTRES.values():
        bins[:, (xx - cx) ** 2 + (yy - cy) ** 2 <= 64] -= 60.0
    x1, y1 = CENTRES["g001"]
    for b in range(10, N_BINS):  # g001's tube: a bright line growing along +x
        bins[b, int(y1) - 1:int(y1) + 2, int(x1 + 8):int(x1 + 8 + 2 * (b - 9))] += 40.0
    np.save(cache / "bins.npy", bins.astype(np.float16))
    (cache / "meta.json").write_text(json.dumps({
        "schema": "sparsetrack.cache.v1", "frames_per_bin": FPB, "n_bins": N_BINS, "ref_start": REF_START,
        "shifts": [[0.0, 0.0]] * N_BINS,
        "movie": {"name": "tiny.mp4", "path": str(root / "tiny.mp4"), "size_bytes": 1, "n_frames": FPB * N_BINS,
                  "width": W, "height": H}}))
    census = [{"id": gid, "x": x, "y": y, "r": 8.0, "isolated": gid != "g004", "border": False,
               "clump_size": 2 if gid == "g004" else 1} for gid, (x, y) in CENTRES.items()]
    (cache / "grains.json").write_text(json.dumps({"grains": census}))
    (root / "analysis").mkdir()
    (root / "analysis" / "predictions.json").write_text(json.dumps(predictions()))
    if setup is not None:
        (root / "setup.json").write_text(json.dumps({"setup_done": True, **setup}))
    return root
