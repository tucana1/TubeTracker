"""Tip-track loader: v29 measurements.csv -> per-owner tip trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TRACK_COLUMNS = [
    "sample_index",
    "source_frame",
    "time_minutes",
    "tip_x",
    "tip_y",
    "length",
    "accepted",
]


def load_tip_tracks(
    measurements_csv: str | Path,
    owner_ids: list[int],
    min_points: int = 24,
    accepted_only: bool = True,
) -> dict[int, pd.DataFrame]:
    """Load one tip-trajectory frame per owner.

    By default only ``accepted == 1`` rows are kept: the forecasting task
    runs on the pipeline's actual measurements, and diagnostic rows are
    explicitly not measurements. The ``accepted`` covariate is then
    constant, but the loader keeps the column so ablations can re-enable
    diagnostic context deliberately.
    """
    m = pd.read_csv(measurements_csv)
    tracks: dict[int, pd.DataFrame] = {}
    for pid in owner_ids:
        d = m[(m.pollen_id == pid) & (m.tip_x_px.notna())].copy()
        if accepted_only:
            d = d[d.accepted == 1]
        if len(d) < min_points:
            continue
        d = d.sort_values("sample_index").reset_index(drop=True)
        tracks[int(pid)] = pd.DataFrame(
            {
                "sample_index": d.sample_index.to_numpy(),
                "source_frame": d.source_frame.to_numpy(),
                "time_minutes": d.time_minutes.to_numpy(dtype=float),
                "tip_x": d.tip_x_px.to_numpy(dtype=float),
                "tip_y": d.tip_y_px.to_numpy(dtype=float),
                "length": d.tube_length_px.to_numpy(dtype=float),
                "accepted": d.accepted.to_numpy(dtype=float),
            }
        )
    return tracks


def tip_speed(track: pd.DataFrame) -> np.ndarray:
    """Per-step tip displacement magnitude (source px), first step = 0."""
    xy = track[["tip_x", "tip_y"]].to_numpy(dtype=float)
    dsp = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], dsp]).astype(np.float32)
