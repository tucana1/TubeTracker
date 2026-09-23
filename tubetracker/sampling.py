"""Frame schedules shared by long-video analysis pipelines."""

from __future__ import annotations

import numpy as np


def plan_tracking_and_output_indices(available_count, tracking_count, output_count):
    """Plan dense tracking observations and a nested sparse output schedule."""
    available_count = int(available_count)
    tracking_count = int(tracking_count)
    output_count = int(output_count)
    if available_count <= 0:
        raise ValueError("available_count must be positive")
    if tracking_count <= 0:
        raise ValueError("tracking_count must be positive")
    if output_count <= 0:
        raise ValueError("output_count must be positive")
    tracking_size = min(available_count, tracking_count)
    tracking_indices = np.unique(
        np.rint(np.linspace(0, available_count - 1, tracking_size)).astype(int)
    )
    output_size = min(len(tracking_indices), output_count)
    output_positions = np.unique(
        np.rint(np.linspace(0, len(tracking_indices) - 1, output_size)).astype(int)
    )
    return tracking_indices, output_positions
