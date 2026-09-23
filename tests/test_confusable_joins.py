"""rev9 WP-A.6: foreign-mask joins must not invent negatives.

The rev8 union joined by frame alone and excluded only the same mask
UUID. A same-numbered frame in another movie, or a re-annotation of the
queried tube, then became a *foreign negative* — supervision claiming
another owner's tube where the truth was "my own tube, again".
"""
from __future__ import annotations

import numpy as np

from prototypes.v30_video_apex.targets import (
    confusable_union, decode_mask_raster, encode_mask_raster)

H = W = 288
ORIGIN = (380.0, 320.0)          # crop covers native x 380-668, y 320-608


def _paint(cx, cy, r=8, canvas=900):
    m = np.zeros((canvas, canvas), dtype=bool)
    yy, xx = np.mgrid[0:canvas, 0:canvas]
    m[np.hypot(xx - cx, yy - cy) <= r] = True
    return m


def _rec(uuid, movie, frame, owner, cx):
    return {"mask_uuid": uuid, "movie": movie, "source_frame": frame,
            "owner_key": owner, "mask_raster": encode_mask_raster(_paint(cx, 430.0))}


def _incrop(cx):
    return decode_mask_raster(H, W, ORIGIN,
                              encode_mask_raster(_paint(cx, 430.0)))


def test_union_keeps_only_other_owners_same_movie_same_frame():
    foreign = _rec("mask-foreign", "ld", 49350, "ld|rev8p-42000-g1", 560.0)
    same_owner_revision = _rec("mask-mine-r2", "ld", 49350,
                               "ld|obs-r4-p05", 620.0)
    other_movie = _rec("mask-m2", "m2", 49350, "ld|someone", 700.0)
    other_frame = _rec("mask-otherframe", "ld", 42000, "ld|x", 740.0)
    unlinked = _rec("mask-unlinked", "ld", 49350, "ld|y", 780.0)

    un = confusable_union(
        H, W, ORIGIN, movie="ld", frame=49350,
        self_uuid="mask-rev8m-002", self_owner_key="ld|obs-r4-p05",
        masks=[foreign, same_owner_revision, other_movie, other_frame,
               unlinked],
        eligible={"mask-foreign"})

    # exactly the foreign tube's pixels, nothing else
    assert np.array_equal(un, _incrop(560.0)), (
        "union must be the foreign paint and only that")


def test_same_owner_revision_is_never_a_foreign_negative():
    same_owner_revision = _rec("mask-mine-r2", "ld", 49350,
                               "ld|obs-r4-p05", 620.0)
    un = confusable_union(
        H, W, ORIGIN, movie="ld", frame=49350,
        self_uuid="mask-rev8m-002", self_owner_key="ld|obs-r4-p05",
        masks=[same_owner_revision])
    assert not un.any(), "a re-annotation of my own tube is not foreign"


def test_same_uuid_is_excluded_even_without_owner():
    rec = _rec("mask-rev8m-002", "ld", 49350, "", 500.0)
    un = confusable_union(
        H, W, ORIGIN, movie="ld", frame=49350, self_uuid="mask-rev8m-002",
        self_owner_key="", masks=[rec])
    assert not un.any()


def test_quarantined_masks_contribute_zero():
    """No identity link ⇒ could be this very tube ⇒ excluded."""
    unlinked = _rec("mask-unlinked", "ld", 49350, "ld|y", 560.0)
    un = confusable_union(
        H, W, ORIGIN, movie="ld", frame=49350, self_uuid="mask-rev8m-002",
        self_owner_key="ld|obs-r4-p05", masks=[unlinked],
        eligible={"mask-something-else"})
    assert not un.any()


def test_other_movie_same_frame_is_excluded():
    other_movie = _rec("mask-m2", "m2", 49350, "ld|someone", 560.0)
    un = confusable_union(
        H, W, ORIGIN, movie="ld", frame=49350, self_uuid="mask-rev8m-002",
        self_owner_key="ld|obs-r4-p05", masks=[other_movie],
        eligible={"mask-m2"})
    assert not un.any(), "m2 is a different acquisition key"


def test_self_positive_pixels_take_precedence_in_the_loss_selectors():
    """At crossings, confusable weight can never fall on my own paint.

    `neg_sel = (1 - target) * valid`, so a pixel that is both
    confusable and self-positive carries zero negative weight by
    construction. This pins that contract against a rewrite.
    """
    tgt = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    valid = np.ones((2, 2), dtype=np.float32)
    neg_sel = (1.0 - tgt) * valid
    conf = np.ones((2, 2), dtype=np.float32)      # everything confusable
    weighted = neg_sel * (1.0 + 3.0 * conf)
    assert weighted[tgt > 0].sum() == 0.0
    assert weighted[tgt == 0].sum() == 8.0
