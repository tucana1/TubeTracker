"""Unit tests for corner-negative loading (H181)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np

from train_cnn_points import load_corner_negatives, load_faint_keys

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv" / "bin" / "python"
GATE = REPO / "prototypes" / "timesfm_tip_forecast" / "promote_gate.py"


def test_corner_loader_reads_points(tmp_path):
    f = tmp_path / "corners.csv"
    f.write_text(
        "image_name,source_frame,x,y,kind\n"
        "dense_000001.png,100,10.5,20.5,elbow\n"
        "dense_000001.png,100,30.0,40.0,elbow\n"
        "dense_000002.png,200,50.0,60.0,elbow\n"
        "bad,row\n"
    )
    out = load_corner_negatives(f)
    assert out == {
        "dense_000001.png": [(10.5, 20.5), (30.0, 40.0)],
        "dense_000002.png": [(50.0, 60.0)],
    }


def test_corner_loader_missing_file_is_off(tmp_path):
    assert load_corner_negatives(tmp_path / "nope.csv") == {}


def _gate_csv(path, rows):
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_promote_gate_passes_baseline_itself(tmp_path):
    import subprocess

    base = _gate_csv(tmp_path / "base.csv", [
        {"tag": "P58-clean", "dist_nearest": 4.9, "heat_at_truth": 0.1},
        {"tag": "P3-faint", "dist_nearest": 4.3, "heat_at_truth": 0.2},
        {"tag": "P13-clean", "dist_nearest": 4.3, "heat_at_truth": 0.1},
        {"tag": "P22-clean", "dist_nearest": 17.6, "heat_at_truth": 0.0},
        {"tag": "P11-faint", "dist_nearest": 10.8, "heat_at_truth": 0.0},
        {"tag": "P8-apex", "dist_nearest": 0.5, "heat_at_truth": 0.8},
        {"tag": "P22-ghost", "dist_nearest": 21.9, "heat_at_truth": 0.0},
        {"tag": "P55-background", "dist_nearest": 32.7, "heat_at_truth": 0.0},
        {"tag": "P22-pre47", "dist_nearest": 218.0, "heat_at_truth": 0.0},
        {"tag": "P22-pre58", "dist_nearest": 187.0, "heat_at_truth": 0.0},
    ])
    r = subprocess.run(
        [str(PY), str(GATE),
         "--eval-csv", base, "--baseline-csv", base],
        capture_output=True, text=True,
    )
    assert (r.returncode, "PROMOTION: PASS") == (0, r.stdout.strip().splitlines()[-1])


def test_promote_gate_fails_confirm_flip(tmp_path):
    import subprocess

    base = _gate_csv(tmp_path / "base.csv", [
        {"tag": "P58-clean", "dist_nearest": 4.9, "heat_at_truth": 0.1},
        {"tag": "P3-faint", "dist_nearest": 4.3, "heat_at_truth": 0.2},
        {"tag": "P13-clean", "dist_nearest": 4.3, "heat_at_truth": 0.1},
        {"tag": "P22-clean", "dist_nearest": 17.6, "heat_at_truth": 0.0},
        {"tag": "P11-faint", "dist_nearest": 10.8, "heat_at_truth": 0.0},
        {"tag": "P8-apex", "dist_nearest": 0.5, "heat_at_truth": 0.8},
        {"tag": "P22-ghost", "dist_nearest": 21.9, "heat_at_truth": 0.0},
        {"tag": "P55-background", "dist_nearest": 32.7, "heat_at_truth": 0.0},
    ])
    cand = _gate_csv(tmp_path / "cand.csv", [
        {"tag": "P58-clean", "dist_nearest": 5.8, "heat_at_truth": 0.1},
        {"tag": "P3-faint", "dist_nearest": 4.3, "heat_at_truth": 0.2},
        {"tag": "P13-clean", "dist_nearest": 5.7, "heat_at_truth": 0.1},
        {"tag": "P22-clean", "dist_nearest": 18.6, "heat_at_truth": 0.0},
        {"tag": "P11-faint", "dist_nearest": 37.8, "heat_at_truth": 0.0},
        {"tag": "P8-apex", "dist_nearest": 1.5, "heat_at_truth": 0.7},
        {"tag": "P22-ghost", "dist_nearest": 21.9, "heat_at_truth": 0.0},
        {"tag": "P55-background", "dist_nearest": 88.0, "heat_at_truth": 0.0},
    ])
    r = subprocess.run(
        [str(PY), str(GATE),
         "--eval-csv", cand, "--baseline-csv", base],
        capture_output=True, text=True,
    )
    assert r.returncode == 1 and "PROMOTION: FAIL" in r.stdout
    assert "P11-faint" in r.stdout


def test_sigma_overrides_widen_faint_label():
    from collections import namedtuple
    from tubetracker.cnn_prototype import point_heatmaps

    L = namedtuple("L", ["label_type", "x", "y"])
    base = point_heatmaps(64, 64, [L("tip", 32.0, 32.0)])
    wide = point_heatmaps(64, 64, [L("tip", 32.0, 32.0)],
                          sigma_overrides=[4.5])
    assert wide[1, 32, 32] > 0.99 and base[1, 32, 32] > 0.99
    assert wide[1, 32, 40] > base[1, 32, 40] + 0.05
    # None override == default
    same = point_heatmaps(64, 64, [L("tip", 32.0, 32.0)],
                          sigma_overrides=[None])
    assert np.allclose(same, base)


def test_load_faint_keys_threshold_and_missing(tmp_path):
    from scripts.train_cnn_points import load_faint_keys

    assert load_faint_keys(tmp_path / "nope.csv", 12.0) == set()
    p = tmp_path / "c.csv"
    p.write_text("image_name,x,y,contrast\nf.png,10.4,20.6,8.0\nf.png,30.1,40.9,25.0\nbad,row\n")
    assert load_faint_keys(p, 12.0) == {("f.png", 10, 21)}
