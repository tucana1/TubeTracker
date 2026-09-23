import numpy as np
import pytest

from prototypes.v30_video_apex.cap_evidence import CapLabelIndex
from scripts.train_native_caps import scoped_negative_spot, sample_cap_batch


def test_hard_peak_at_polygon_edge_never_labels_unreviewed_pixels():
    index = CapLabelIndex([], [{'kind': 'verified_negative', 'confirmed': True,
        'class_scope': 'cap', 'movie': 'm', 'source_frame': 10,
        '_region_uuid': 'r', '_region_revision': 1, '_project': 'p',
        'polygon_xy': [[10, 10], [14, 10], [14, 14], [10, 14]]}])
    targets = index.spatial('m', 10, [8, 8], (20, 20))
    mask, audit = scoped_negative_spot(targets['negative'], [8, 8], [14, 12], 8.)
    assert mask[4, 6]  # the measured peak is licensed
    assert not mask[4, 7]  # immediately outside the reviewed right edge
    assert mask[3:6, 3:6].all()  # reviewed polygon interior stays supervised
    assert mask.sum() == targets['negative'].sum()
    assert audit['unlicensed_spot_pixels_excluded'] > 0
    assert not (mask & ~targets['negative']).any()


def test_peak_with_no_reviewed_scope_keeps_entire_crop_unknown():
    mask, audit = scoped_negative_spot(np.zeros((20, 20), bool), [0, 0], [10, 10], 8.)
    assert not mask.any() and audit['licensed_spot_pixels'] == 0
    with pytest.raises(ValueError):
        scoped_negative_spot(mask, [0, 0], [10, 10], -1.)


def test_hard_spots_reach_every_batch_without_displacing_all_ordinary_negatives():
    rng = np.random.default_rng(41)
    positive, negative, focused = [0,1], [2,3], [4,5,6]
    visited = set()
    for _ in range(100):
        batch = sample_cap_batch(rng, positive, negative, focused)
        assert len(batch) == 4
        assert all(i in positive for i in batch[:2])
        assert batch[2] in negative and batch[3] in focused
        visited.add(batch[3])
    assert visited == set(focused)


def test_without_focus_cases_sampling_preserves_legacy_rng_sequence():
    actual_rng, expected_rng = np.random.default_rng(41), np.random.default_rng(41)
    for _ in range(25):
        actual = sample_cap_batch(actual_rng, [0,1,2], [3,4,5])
        expected = list(expected_rng.choice([0,1,2],2)) + list(expected_rng.choice([3,4,5],2))
        assert actual == expected
