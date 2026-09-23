"""rev12 P0.3/P0.4: projection geometry + manifest comparator tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.targets import (  # noqa: E402
    polyline_point_projection, project_to_arclength)


def test_projection_onto_segment_midpoint():
    # a sparse polyline: the point sits 3 px from the MIDDLE of a long
    # segment; nearest-vertex distance would be much larger
    route = [[0.0, 0.0], [100.0, 0.0]]
    d, s, xy = polyline_point_projection(route, [50.0, 3.0])
    assert d == pytest.approx(3.0, abs=1e-6)
    assert s == pytest.approx(50.0, abs=1e-6)
    assert xy == pytest.approx((50.0, 0.0), abs=1e-6)


def test_projection_clamps_beyond_ends():
    route = [[0.0, 0.0], [10.0, 0.0]]
    d, s, xy = polyline_point_projection(route, [-5.0, 0.0])
    assert d == pytest.approx(5.0, abs=1e-6)
    assert s == pytest.approx(0.0, abs=1e-6)
    d2, s2, _ = polyline_point_projection(route, [20.0, 0.0])
    assert d2 == pytest.approx(10.0, abs=1e-6)
    assert s2 == pytest.approx(10.0, abs=1e-6)


def test_projection_repeated_zero_length_vertices():
    # duplicated vertices must not break the projection (zero-length
    # segments degenerate to their endpoint)
    route = [[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [10.0, 0.0]]
    d, s, xy = polyline_point_projection(route, [5.0, 2.0])
    assert d == pytest.approx(2.0, abs=1e-6)
    assert s == pytest.approx(5.0, abs=1e-6)


def test_projection_matches_arclength_wrapper():
    route = np.asarray([[0, 0], [10, 0], [10, 10]], float)
    q = [9.0, 5.0]
    d, s, xy = polyline_point_projection(route, q)
    assert project_to_arclength(route, q) == pytest.approx(s)
    assert d == pytest.approx(1.0, abs=1e-6)


def test_projection_sparse_polyline_gap_case():
    # the audit's motivating case: a route passing 5.17 px from the cap
    # but whose nearest stored VERTEX is 27.51 px away
    route = [[0.0, 0.0], [60.0, 0.0]]
    tip = [30.0, 5.17]
    d_vertex = float(np.hypot(
        *(np.asarray(tip)[None, :] - np.asarray(route)).T).min())
    d, s, xy = polyline_point_projection(route, tip)
    assert d == pytest.approx(5.17, abs=1e-6)
    assert d_vertex > 27.0


def _write_manifest(p: Path, **over) -> Path:
    base = {
        "config": {"variant": "temporal", "lr": 1e-4},
        "init_param_hash": "aaaa", "seed": 0,
        "updates": 200, "block": 50,
        "body_loss_config": {"objective": "dice_selectors",
                             "bg_region": "strata"},
    }
    base.update(over)
    p.write_text(json.dumps(base))
    return p


def test_comparator_accepts_declared_single_variable(tmp_path):
    from scripts.compare_run_manifests import compare
    a = _write_manifest(tmp_path / "a.json")
    b = _write_manifest(tmp_path / "b.json",
                        body_loss_config={"objective": "dice_selectors",
                                          "bg_region": "band"})
    res = compare(a, b, "body_loss_config.bg_region")
    assert res["verdict"] == "single-variable", res
    assert not res["undeclared_diffs"]


def test_comparator_rejects_undeclared_confound(tmp_path):
    from scripts.compare_run_manifests import compare
    a = _write_manifest(tmp_path / "a.json")
    b = _write_manifest(tmp_path / "b.json", updates=580,
                        body_loss_config={"objective": "dice_selectors",
                                          "bg_region": "band"})
    res = compare(a, b, "body_loss_config.bg_region")
    assert res["verdict"] == "NOT-single-variable"
    fields = {d["field"] for d in res["undeclared_diffs"]}
    assert "updates" in fields


def test_comparator_rejects_init_change(tmp_path):
    from scripts.compare_run_manifests import compare
    a = _write_manifest(tmp_path / "a.json")
    b = _write_manifest(tmp_path / "b.json", init_param_hash="bbbb")
    res = compare(a, b, "seed")
    assert res["verdict"] == "NOT-single-variable"
