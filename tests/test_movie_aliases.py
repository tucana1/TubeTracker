"""rev9 WP-A.6: aliased movies are ONE acquisition.

The live `m1`/`m2` keys point at identical bytes. A split that treats
them as independent could put the same frames on both sides (or score
the same acquisition twice). samples_from_snapshot collapses them.
"""
from __future__ import annotations

from prototypes.v30_video_apex.targets import samples_from_snapshot

SNAP = "runs/prototypes/v30/snapshots/snap23"


def test_alias_collapsed_in_real_snapshot():
    samples = samples_from_snapshot(SNAP)
    assert samples
    movies = {s.movie for s in samples}
    assert "m2" not in movies, (
        f"m2 must be collapsed into its identical-bytes alias; got {movies}")
    # every sample still resolves to a real path
    assert all(s.movie_path for s in samples)


def test_alias_collapse_is_recorded_not_silent():
    """The collapse prints (stderr/stdout of the loader) — a silent
    identity rewrite would be its own defect."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        samples_from_snapshot(SNAP)
    text = buf.getvalue()
    assert "movie aliases collapsed" in text
    assert "'m2': 'm1'" in text
