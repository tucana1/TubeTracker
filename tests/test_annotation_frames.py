"""Exact-frame reader: identity verification on the real lowdens movie."""

import numpy as np

from tubetracker.annotation_frames import FrameReader

MOVIE = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"


def test_frame_identity_exact_on_real_movie():
    r = FrameReader(MOVIE)
    try:
        assert len(r) > 50000
        assert r.native_size == (1280, 1024)
        for fid in (0, 15000, 52500):
            out = r.read(fid)
            assert out.frame.shape[:2] == (1024, 1280)
            assert out.verified_id == fid and out.exact
        a = r.read(1000).frame
        b = r.read(1000).frame
        assert np.array_equal(a, b)
    finally:
        r.close()
