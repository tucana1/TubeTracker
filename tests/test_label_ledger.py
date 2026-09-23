"""rev9 WP-A.6: the consumption table, and quarantined content is zero.

"Do not infer consumed supervision from sample counts." A label that is
quarantined must contribute nothing to any head, and the run must be
able to show which unique labels fed which head, with magnitudes.
"""
from __future__ import annotations

import pytest

from prototypes.v30_video_apex.label_ledger import (
    CONSUMER_MAP, build_consumption_table)
from prototypes.v30_video_apex.targets import SampleDef


def _sample(key, kind, quarantine=""):
    return SampleDef(entry_id=key, movie="ld", movie_path="x.mp4",
                     source_frame=1, kind=kind, sample_key=key,
                     quarantine_reason=quarantine)


def test_table_lists_unique_labels_per_head_with_magnitudes():
    samples = [_sample("ld|a", "body_mask"), _sample("ld|b", "tip_only")]
    usage = {"body_valid": {"ld|a"}, "apex_pos": {"ld|b"},
             "route": {"ld|a|rotated"}}
    counts = {"body_valid": {"ld|a": {"paint_px": 820, "valid_px": 4992}},
              "route": {"ld|a|rotated": {"routes": 1}}}
    t = build_consumption_table(samples, usage, counts=counts)
    assert t["unique_total"] == 3
    assert t["heads"]["body_valid"]["n_unique"] == 1
    assert t["heads"]["body_valid"]["counts"]["ld|a"]["paint_px"] == 820
    assert t["heads"]["route"]["counts"]["ld|a|rotated"]["routes"] == 1
    assert t["heads"]["body_valid"]["kinds"] == {"body_mask": 1}
    assert t["quarantined_consumed"] == []
    assert "body_mask" in t["consumer_map"]


def test_quarantined_label_contributing_raises():
    samples = [_sample("ld|good", "body_mask"),
               _sample("ld|bad", "comparison", "duel-conflict-quarantined")]
    usage = {"route": {"ld|good", "ld|bad"}}      # the leak
    with pytest.raises(ValueError) as e:
        build_consumption_table(samples, usage)
    assert "quarantined" in str(e.value)


def test_quarantined_label_present_but_unused_is_fine():
    samples = [_sample("ld|good", "body_mask"),
               _sample("ld|bad", "comparison", "duel-conflict-quarantined")]
    t = build_consumption_table(samples, {"body_valid": {"ld|good"}})
    assert t["quarantined_labels"] == 1
    assert t["quarantined_consumed"] == []


def test_off_consumer_map_is_rejected_not_warned():
    """A comparison feeding the body head is a design error, not a warning.

    rev10 WP-A: this used to RECORD the pair and let the run continue
    ("records an off-map band consumer without failing" — the review).
    It now raises, so an invalid consumer cannot reach a long run."""
    samples = [_sample("ld|c", "comparison")]
    with pytest.raises(ValueError) as e:
        build_consumption_table(samples, {"body_valid": {"ld|c"}})
    assert "comparison->body_valid" in str(e.value)
    # audit tooling may still inspect the table, explicitly
    t = build_consumption_table(samples, {"body_valid": {"ld|c"}},
                                strict=False)
    assert t["heads"]["body_valid"]["off_consumer_map"] == [
        "comparison->body_valid"]
    assert set(CONSUMER_MAP["comparison"]) == {"route"}


def test_the_new_body_channels_are_on_map_for_body_masks():
    """body_band / body_bg_reviewed / body_overlap are legitimate
    consumers of a body_mask, so adding them did not silently create
    off-map errors (rev10 WP-A plumbed all three)."""
    samples = [_sample("ld|m", "body_mask")]
    for head in ("body_valid", "body_confusable", "body_band",
                 "body_bg_reviewed", "body_overlap"):
        t = build_consumption_table(samples, {head: {"ld|m"}})
        assert head in t["heads"]
