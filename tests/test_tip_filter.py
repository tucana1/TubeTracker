"""Unit tests for the forecast-guided tip filter (H149 recovery-pass)."""

import numpy as np
import pandas as pd

from prototypes.timesfm_tip_forecast.guided_tip_filter import apply_tip_filter


def _frame(tips, verdicts):
    # tips: [(sample_index, x, y)]; verdicts: {sample_index: verdict}
    m = pd.DataFrame(
        {
            "pollen_id": [7] * len(tips),
            "accepted": [1] * len(tips),
            "sample_index": [t[0] for t in tips],
            "source_frame": [t[0] * 10 for t in tips],
            "tip_x_px": [t[1] for t in tips],
            "tip_y_px": [t[2] for t in tips],
        }
    )
    r = pd.DataFrame(
        {"sample_index": list(verdicts), "verdict": list(verdicts.values())}
    )
    return m, r


def test_fault_departure_freezes():
    m, r = _frame([(0, 0.0, 0.0), (1, 1.0, 0.0), (2, 50.0, 0.0)],
                  {2: "fault-suspect"})
    g = apply_tip_filter(m, r, 7)
    assert bool(g[g.sample_index == 2].held.iloc[0])
    assert (g[g.sample_index == 2].guided_x.iloc[0], ) == (1.0,)


def test_flicker_back_passes():
    # Fault tip returns within RECOVERY_PX of a recent keep: pass, no hold.
    m, r = _frame([(0, 0.0, 0.0), (1, 1.0, 0.0), (2, 50.0, 0.0), (3, 1.5, 0.5)],
                  {2: "fault-suspect", 3: "fault-suspect"})
    g = apply_tip_filter(m, r, 7)
    assert bool(g[g.sample_index == 2].held.iloc[0])
    assert not bool(g[g.sample_index == 3].held.iloc[0])
    assert g[g.sample_index == 3].guided_x.iloc[0] == 1.5


def test_burst_and_keep_pass_untouched():
    m, r = _frame([(0, 0.0, 0.0), (1, 30.0, 0.0)], {1: "burst"})
    g = apply_tip_filter(m, r, 7)
    assert not bool(g.held.any())


def test_redetect_ok_and_unsupported():
    from prototypes.timesfm_tip_forecast.guided_redetect import redetect_sample

    g, held, tag = redetect_sample((0.0, 0.0), "keep", [(3.0, 4.0, 0.9)], [], None)
    assert (tag, held) == ("OK", False)
    g, held, tag = redetect_sample((0.0, 0.0), "keep", [(50.0, 0.0, 0.9)], [], (0.0, 0.0))
    assert (tag, held) == ("UNSUPPORTED", True)


def test_redetect_hold_and_recover():
    from prototypes.timesfm_tip_forecast.guided_redetect import redetect_sample

    g, held, tag = redetect_sample(
        (50.0, 0.0), "fault-suspect", [], [(0.0, 0.0)], (0.0, 0.0))
    assert (tag, held, g) == ("HOLD", True, (0.0, 0.0))
    g, held, tag = redetect_sample(
        (1.0, 1.0), "fault-suspect", [], [(0.0, 0.0)], (0.0, 0.0))
    assert (tag, held) == ("RECOVER", False)


def test_redetect_burst_passes():
    from prototypes.timesfm_tip_forecast.guided_redetect import redetect_sample

    _, held, tag = redetect_sample((30.0, 0.0), "burst", [], [], None)
    assert (tag, held) == ("BURST", False)


def test_grain_mask_drops_disk_edge_not_bar_walls():
    # H162: dark disk (grain); path through the disk interior must read
    # zero (masked), path on the outside bar keeps bar walls.
    from prototypes.timesfm_tip_forecast.switch_cut import (
        grain_masked_wall_energy,
        transverse_wall_energy,
    )

    g = np.full((100, 100), 100.0)
    yy, xx = np.mgrid[0:100, 0:100]
    g[np.hypot(xx - 30.0, yy - 50.0) < 15.0] = 20.0
    g[48:53, 44:80] = 130.0
    inside = np.column_stack([np.arange(20.0, 28.0), np.full(8, 50.0)])
    mi = grain_masked_wall_energy(g, inside, np.array([30.0, 50.0]), 17.0)
    assert bool(np.all(mi == 0.0))  # fully inside: nothing kept
    outside = np.column_stack([np.arange(50.0, 60.0), np.full(10, 50.0)])
    mo = grain_masked_wall_energy(g, outside, np.array([30.0, 50.0]), 17.0)
    assert bool((mo > 0).any())  # bar walls survive outside the disk
