"""The per-bin decoder's tip-growth continuity option (``reach_grain(continuity="path")``, off by default), on small
synthetic inputs: a clean tube reads as without it; a foreign tube joining mid-path, or lying across the tip's way
before the tip got there, is not read as the grain's; a region that reconnects is still read in full; thin holes of a
wide tube are filled, the eye of a coil is not."""
import numpy as np

from prototypes.learned_evidence import reach as R


def _movie(T=60, own_max=45.0, foreign=None, t_foreign=0, gap=None, jump_at=None, gx=80.0, gy=100.0, gr=10.0,
           f_half=60.0, wave=None):
    """A dark grain with a thin tube growing to its right (1.5 px per bin from bin 10, up to ``own_max`` px) and its
    probability map (x16). ``foreign``: x offset from the rim of a vertical foreign tube (120 px long, centred on the
    tube's line; ``f_half`` px either side) in the probability map from bin ``t_foreign`` on; ``gap=(a, z, s)``: in
    bins a..z-1 the tube's evidence is missing 2 px at s px from the rim; ``jump_at``: the tube is first seen at bin
    ``jump_at``, already grown; ``wave=(amp, wavelength)``: the tube meanders about its line."""
    from sparsetrack.render import Renderer
    rng = np.random.default_rng(0)
    H, W = 200, 200
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    img = np.full((T, H, W), 130.0, np.float32)
    prob = np.zeros((T, H, W), np.float32)
    rg = np.hypot(xx - gx, yy - gy)
    for t in range(T):
        L = float(np.clip((t - 10) * 1.5, 0.0, own_max))
        if jump_at is not None and t < jump_at:
            L = 0.0
        img[t][rg < gr] = 90.0
        img[t][(rg >= gr - 2) & (rg < gr)] = 60.0
        tube = np.zeros((H, W), bool)
        if L > 0:
            yc = gy + (wave[0] * np.sin(2 * np.pi * (xx - gx - gr) / wave[1]) if wave else 0.0)
            tube = (xx >= gx + gr) & (xx <= gx + gr + L) & (np.abs(yy - yc) <= 1.5)
            if gap is not None and gap[0] <= t < gap[1]:
                tube &= ~((xx >= gx + gr + gap[2]) & (xx <= gx + gr + gap[2] + 2))
        if foreign is not None and t >= t_foreign:
            tube |= (np.abs(xx - (gx + gr + foreign)) <= 1.5) & (np.abs(yy - gy) <= f_half)
        img[t] -= np.where(tube, 30.0, 0.0).astype(np.float32)
        prob[t] = np.where(tube, 16.0, 0.0)
        img[t] += rng.normal(0, 2.0, (H, W)).astype(np.float32)
    meta = {"shifts": np.zeros((T, 2)).tolist(), "n_bins": T, "frames_per_bin": 1, "ref_start": 0}
    return Renderer(prob, meta), Renderer(img, meta), meta, {"id": "g001", "x": gx, "y": gy, "r": gr}


def _own(t, own_max=45.0):
    return float(np.clip((t - 10) * 1.5, 0.0, own_max))


def _read(movie, **kw):
    RP, RI, meta, g = movie
    off = R.reach_grain(RP, RI, meta, g, [], half=80, vmax=4.0, **kw)
    on = R.reach_grain(RP, RI, meta, g, [], half=80, vmax=4.0, continuity="path", **kw)
    return off, on


def test_a_clean_tube_reads_as_without_continuity():
    off, on = _read(_movie())
    assert on["raw_reach_px"] == off["raw_reach_px"] and on["length"] == off["length"]
    assert on["continuity"] == {"reread_bins": [], "held_bins": [], "switched_at": []} and "continuity" not in off


def test_a_tube_first_seen_long_is_read_as_without_continuity():
    off, on = _read(_movie(jump_at=35))  # first seen 37 px long: never locked on
    assert on["raw_reach_px"] == off["raw_reach_px"]


def test_a_foreign_tube_joining_mid_path_is_not_read():
    off, on = _read(_movie(foreign=12.0, t_foreign=40))  # crosses the tube 12 px from the rim from bin 40
    raw_off, raw_on = np.array(off["raw_reach_px"]), np.array(on["raw_reach_px"])
    own = np.array([_own(t) for t in range(60)])
    assert raw_off[45:].min() > own[45:].max() + 20  # the farthest point runs along the foreign tube
    assert np.all(np.abs(raw_on[40:] - own[40:]) <= 3.0)  # continuity keeps reading the grain's tube
    assert abs(on["final_length_px"] - 45.0) <= 3.0 < off["final_length_px"] - 45.0
    assert on["continuity"]["reread_bins"] and not on["continuity"]["held_bins"]


def test_a_tube_lying_across_the_tips_way_before_it_got_there_is_not_read():
    off, on = _read(_movie(foreign=24.0, t_foreign=0))  # there from the start; the tip reaches it at bin 26
    raw_off, raw_on = np.array(off["raw_reach_px"]), np.array(on["raw_reach_px"])
    own = np.array([_own(t) for t in range(60)])
    assert raw_off[30:].min() > own[30:].max() + 20
    assert np.all(np.abs(raw_on[30:] - own[30:]) <= 4.0)  # the tip grows on over it, read as the grain's


def test_a_region_that_reconnects_is_read_in_full():
    off, on = _read(_movie(gap=(25, 35, 8.0)))  # bins 25-34: the tube's evidence breaks 8 px from the rim
    raw_off, raw_on = np.array(off["raw_reach_px"]), np.array(on["raw_reach_px"])
    assert raw_off[30] < 12 and raw_on[30] < 12  # the break reads short either way
    assert np.allclose(raw_on[35:], raw_off[35:])  # back in full at once: a catch-up, not a jump


def test_fill_small_holes_fills_a_hollow_band_not_a_coil():
    yy, xx = np.mgrid[0:80, 0:80]
    band = (np.abs(yy - 40) <= 5) & (xx >= 10) & (xx <= 70)
    hollow = band & ~((np.abs(yy - 40) <= 1) & (xx >= 20) & (xx <= 60))  # a double-walled wide tube
    assert np.array_equal(R.fill_small_holes(hollow), band)
    d = np.hypot(xx - 40, yy - 40)
    coil = (d >= 12) & (d <= 16)
    assert np.array_equal(R.fill_small_holes(coil), coil)


def test_continuity_is_off_by_default_and_checks_its_inputs():
    import pytest
    RP, RI, meta, g = _movie(T=30)
    default = R.reach_grain(RP, RI, meta, g, [], half=80, vmax=4.0)
    assert default == R.reach_grain(RP, RI, meta, g, [], half=80, vmax=4.0, continuity="off")
    with pytest.raises(ValueError):
        R.reach_grain(RP, RI, meta, g, [], half=80, vmax=4.0, continuity="path", length="smooth")
    with pytest.raises(ValueError):
        R.reach_grain(RP, RI, meta, g, [], half=80, vmax=4.0, continuity="yes")


def test_a_foreign_tube_the_grains_tube_grows_through_is_never_taken_for_it():
    # from bin 35 (the tube 37 px long) a static tube crosses it 12 px from the rim, a path of 52 px along its arm,
    # while the grain's tube grows on past it: that path is steady but does not grow, so it never replaces the tube's
    RP, RI, meta, g = mv = _movie(T=70, own_max=60.0, foreign=12.0, t_foreign=35, f_half=40.0)
    off, on = _read(mv)
    fit = np.array(on["length"]["px"])
    assert on["continuity"]["switched_at"] == []
    assert all(abs(fit[t] - _own(t, 60.0)) <= 3.0 for t in range(35, 70)), fit[35:]
    assert max(np.array(off["length"]["px"])[35:46] - [_own(t, 60.0) for t in range(35, 46)]) > 8  # off reads it


def test_a_meandering_tube_reads_as_without_continuity():
    # a tube that bends back and forth (amplitude 4 px, wavelength 40 px) turns ~45 deg per 20 px at each inflection
    off, on = _read(_movie(T=70, own_max=70.0, wave=(4.0, 40.0)))
    assert np.max(np.abs(np.array(on["length"]["px"]) - np.array(off["length"]["px"]))) <= 1.0


def test_a_branch_leaving_a_stopped_tube_before_its_apex_is_not_read():
    # the grain's tube stops at 70 px; from bin 60 a tube crosses it 55 px out (25 px arms): 80 px along the arm
    RP, RI, meta, g = mv = _movie(T=75, own_max=70.0, foreign=55.0, t_foreign=60, f_half=25.0)
    off, on = _read(mv)
    assert max(on["length"]["px"][60:]) <= 72.0 < max(off["length"]["px"][60:])


def test_review_pictures_draw_what_was_measured_not_the_foreign_tube_left_out():
    RP, RI, meta, g = _movie(foreign=12.0, t_foreign=40)
    keep = (30, 45, 55)
    on = R.reach_grain(RP, RI, meta, g, [], half=80, vmax=4.0, continuity="path", keep=keep)
    off = R.reach_grain(RP, RI, meta, g, [], half=80, vmax=4.0, keep=keep)
    assert set(keep[1:]) <= set(on["continuity"]["reread_bins"])
    for i in keep[1:]:
        _, region, axis = on["_views"][i]
        ys, _ = np.nonzero(region)
        assert ys.max() - ys.min() + 1 <= 8  # the grain's own tube (3 px wide), not the 120 px foreign one
        assert abs(int(axis.sum()) - on["raw_reach_px"][i]) <= 6  # the drawn path is the one measured
        _, region_off, _ = off["_views"][i]
        ys, _ = np.nonzero(region_off)
        assert ys.max() - ys.min() + 1 >= 100  # without continuity the foreign tube is read, and drawn
    assert np.array_equal(on["_views"][30][1], off["_views"][30][1])  # before it joins: the same picture


def test_a_held_bin_draws_nothing():
    region = np.zeros((20, 20), bool)
    region[9:12, 2:18] = True
    assert R.drawn_path(region, None) == (None, None)
    drawn, axis = R.drawn_path(region, np.array([[2.0, 10.0], [17.0, 10.0]]))
    assert axis.sum() == 16 and (drawn == region).all()
