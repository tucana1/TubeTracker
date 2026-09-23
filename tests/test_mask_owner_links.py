"""rev8: a mask that carries its OWN named owner is linked, not reused.

The paired-crop round names each grain's owner in the task itself
("ld|rev8p-42000-g0"). That is an explicit identity — the mask must be
consumable. The quarantine was built for legacy masks whose owner
could only ever be the "|task:<uuid>" fallback, and those must STILL
quarantine (attaching them by proximity was the original sin).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.targets import (  # noqa: E402
    encode_mask_raster, samples_from_snapshot)


def _row(mask_uuid, owner_uuid, owner_key, task_uuid, raster):
    return {"mask_uuid": mask_uuid, "mask_revision": 1,
            "task_uuid": task_uuid, "project": "p", "movie": "ld",
            "movie_path": "/m.mp4", "source_frame": 100,
            "painted_xy": [], "brush_px": 9.0, "complete": True,
            "mask_raster": raster, "source_obs_uuid": "",
            "review_region": [[0.0, 0.0], [64.0, 64.0]],
            "owner_uuid": owner_uuid, "owner_key": owner_key,
            "tube_uuid": owner_uuid, "target_xy": [10.0, 10.0],
            "target_r": 14.0, "annotator": "t"}


def test_named_owner_links_and_task_fallback_stays_quarantined(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "observations.json").write_text("[]")
    (snap / "regions.json").write_text("[]")
    # masks resolve their movie from the manifest (a snapshot can hold
    # masks with no observation sample at all)
    (snap / "snapshot_manifest.json").write_text(json.dumps(
        {"movies": {"ld": {"path": "/m.mp4"}}}))
    paint = np.zeros((64, 64), bool)
    paint[8:14, 8:14] = True
    ras = encode_mask_raster(paint)
    rows = [
        # rev8 paired grain: identity named by its own task
        _row("mask-000", "rev8p-100-g0", "ld|rev8p-100-g0",
             "rev8p-000", ras),
        # legacy mask: owner is only the task fallback -> unknown owner
        _row("mask-001", "", "ld|task:old-000", "old-000", ras),
    ]
    (snap / "body_masks.json").write_text(json.dumps(rows))
    got = [s for s in samples_from_snapshot(str(snap))
           if s.kind == "body_mask"]
    by = {s.mask_uuid: s for s in got}
    assert set(by) == {"mask-000", "mask-001"}, by
    linked = by["mask-000"]
    assert linked.quarantine_reason == "", linked.quarantine_reason
    assert linked.link_source == "owner-self"
    assert linked.owner_link_source == "task-derived"
    assert linked.owner_key == "ld|rev8p-100-g0"
    assert linked.target_xy == (10.0, 10.0)
    assert linked.complete is True and linked.review_region
    legacy = by["mask-001"]
    assert legacy.quarantine_reason == "unlinked-mask"
    assert legacy.link_source == ""
