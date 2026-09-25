"""The per-bin decoder's real-footage options, each on small synthetic inputs: the per-bin frame-edge mask, hold while
the grain is gone, the grown-length floor and the ghost-disc test (the defaults), and the ones tried and left off
(re-acquiring tracker, burst runs, abrupt changes, pass-by ownership)."""
import numpy as np
import pytest

from prototypes.learned_evidence import reach as R


# ----------------------------------------------------------------------------- fits
def test_monotone_fit_matches_frozen_l1():
    rng = np.random.default_rng(1)
    for _ in range(50):
        raw = np.maximum(0, np.cumsum(rng.uniform(0, 3, 70)) + rng.normal(0, 5, 70)) * (rng.random(70) > 0.2)
        assert np.array_equal(R.monotone_fit(raw, 4.0), R.monotone_l1(raw, 4.0))


def test_monotone_fit_skips_nan_readings():
    raw = np.array([0, 5, 10, 15, np.nan, np.nan, 0.0, 25, 30])
    fit = R.monotone_fit(raw, 6.0)
    assert np.all(np.diff(fit) >= 0) and fit[-1] == pytest.approx(30.0, abs=0.5)


def test_grown_floor_follows_a_steady_climb_not_a_jump():
    climb = np.concatenate([np.zeros(5), np.arange(1, 41) * 3.0, np.zeros(60)])  # grows to 120, then reads 0
    fl = R.grown_floor(climb, vmax=16.0)
    assert fl[-1] == pytest.approx(0.9 * 114.0)  # 3rd longest grown reading, less 10%
    jump = np.concatenate([np.zeros(30), np.full(10, 150.0), np.zeros(60)])  # a foreign tube joins for 10 bins
    assert R.grown_floor(jump, vmax=16.0).max() == 0.0


def test_grown_fit_keeps_the_length_after_a_run_of_failed_readings():
    raw = np.concatenate([np.zeros(5), np.arange(1, 41) * 3.0, np.zeros(60)])
    l1, _ = R.fit_lengths(raw, 16.0, burst=False, fit="l1")
    grown, _ = R.fit_lengths(raw, 16.0, burst=False, fit="grown")
    assert l1[-1] < 30  # the frozen fit is dragged down by the 60 failed readings
    assert grown[-1] >= 0.9 * 114.0 - 0.5
    assert np.all(np.abs(grown[10:40] - raw[10:40]) <= 0.1 * raw[10:40] + 3.0)  # the climb itself is followed


def test_burst_cut_run_and_skip():
    raw = np.concatenate([np.arange(0, 30, 3.0), [2.0], np.full(4, 40.0), np.full(50, 1.0)])
    assert R.burst_cut(raw) == 10  # frozen rule: one low reading starts the burst
    assert R.burst_cut(raw, run=5) == 15  # a run of 5 low readings is needed
    skip = np.zeros(raw.size, bool)
    skip[15:18] = True
    assert R.burst_cut(raw, run=5, skip=skip) == 18


def test_burst_cut_needs_readings_not_bins_after_it():
    # one low reading just before the grain leaves (NaN = not a reading) is not a burst
    raw = np.r_[np.arange(0, 30, 3.0), [1.0], np.full(20, np.nan)]
    assert R.burst_cut(raw) is None
    assert R.burst_cut(np.r_[np.arange(0, 30, 3.0), [1.0], np.full(20, 1.0)]) == 10


# ----------------------------------------------------------------------------- frame edge
def test_frame_outside_is_per_bin():
    # grain 10 px from the left edge; the crop is 40 px wide
    out0 = R.frame_outside(100, 100, 30.0, 50.0, 20, np.array([0.0, 0.0]))
    out1 = R.frame_outside(100, 100, 30.0, 50.0, 20, np.array([-15.0, 0.0]))  # drifted 15 px left
    assert not out0.any()
    assert out1[:, :6].all() and not out1[:, 7:].any()


# ----------------------------------------------------------------------------- grain presence
def test_gone_bins_runs_and_short_returns():
    c = np.array([10.0] * 20 + [2] * 6 + [10] * 3 + [2] * 5 + [10] * 10 + [2] * 7 + [10] * 2)
    g = R.gone_bins(c)
    assert g[20:34].all() and not g[34:44].any() and g[44:].all() and not g[:20].any()
    assert not R.gone_bins(np.r_[np.full(20, 10.0), [2, 2, 2, 10, 10]]).any()  # 3 low bins: not gone
    assert not R.gone_bins(np.r_[np.full(10, 1.0), np.full(20, 0.0)]).any()  # no early contrast: never gone


def _disc_movie(T=30, jump_at=None, jump=(0, 0), leave_at=None, size=160):
    yy, xx = np.mgrid[0:size, 0:size]
    frames = []
    pos = np.array([size / 2 - 0.5, size / 2 - 0.5])
    for t in range(T):
        p = pos + (np.array(jump, float) if jump_at is not None and t >= jump_at else 0.0)
        im = np.full((size, size), 200.0)
        if leave_at is None or t < leave_at:
            d = np.hypot(xx - p[0], yy - p[1])
            im -= 80.0 * (d < 12) + 60.0 * ((d >= 12) & (d < 14))
            im += 30.0 * ((d < 5))  # an off-centre-free bright core
        frames.append(im + np.random.default_rng(t).normal(0, 1.0, im.shape))
    return np.stack(frames).astype(np.float32)


def test_reacquire_follows_a_jump_beyond_the_local_window():
    img = _disc_movie(jump_at=12, jump=(40, 0))
    c = img.shape[1] / 2 - 0.5
    yy, xx = np.mgrid[0:img.shape[1], 0:img.shape[2]]
    rg = np.hypot(xx - c, yy - c)
    ls = np.zeros((len(img), 2))  # a tracker that stayed behind
    con = R.disc_contrast(img, ls, c, 12.0, rg)
    gone = R.gone_bins(con)
    assert gone[12:].all()
    ls2, gone2 = R.reacquire(img, ls, con, gone, c, 12.0, rg)
    assert not gone2.any()
    assert np.allclose(ls2[12:], [40, 0], atol=1.0)


def test_reacquire_leaves_a_grain_that_left_gone():
    img = _disc_movie(leave_at=12)
    c = img.shape[1] / 2 - 0.5
    yy, xx = np.mgrid[0:img.shape[1], 0:img.shape[2]]
    rg = np.hypot(xx - c, yy - c)
    ls = np.zeros((len(img), 2))
    con = R.disc_contrast(img, ls, c, 12.0, rg)
    gone = R.gone_bins(con)
    _, gone2 = R.reacquire(img, ls, con, gone, c, 12.0, rg)
    assert gone2[12:].all()


# ----------------------------------------------------------------------------- abrupt changes
def test_abrupt_bins_finds_jumps_not_busy_stretches():
    d = np.full(60, 1.0)
    d[17] = 3.0  # a jump
    d[38:46] = [1.5, 2.0, 2.6, 2.6, 2.6, 2.6, 2.6, 2.6]  # a busy stretch building up
    assert R.abrupt_bins(d) == [17]
    assert R.abrupt_bins(np.zeros(12)) == []  # a static movie: no change, no jump


# ----------------------------------------------------------------------------- pass-by rivals
def test_passby_tangential_vs_radial():
    comp = np.zeros((80, 80), bool)
    comp[40:43, 5:75] = True  # a horizontal tube
    yy, xx = np.mgrid[0:80, 0:80]
    below = np.hypot(xx - 40, yy - 55)  # a grain centred below the tube: the tube runs along its rim
    ring = (below >= 11) & (below <= 15)
    assert R.passby(comp, ring, 40, 55)
    comp2 = np.zeros((80, 80), bool)
    comp2[5:44, 39:42] = True  # a vertical tube ending at that grain's rim: attached, radial
    assert not R.passby(comp2, ring, 40, 55)


# ----------------------------------------------------------------------------- ghost discs
def test_disc_sharpness_soft_disc_scores_low():
    import cv2
    yy, xx = np.mgrid[0:120, 0:120]
    d = np.hypot(xx - 60, yy - 60)
    sharp = 200.0 - 80.0 * (d < 14)
    soft = cv2.GaussianBlur(sharp.astype(np.float32), (0, 0), 5.0)
    s_sharp, f_sharp = R._disc_sharpness(sharp.astype(np.float32), 60, 60, 14)
    s_soft, f_soft = R._disc_sharpness(soft, 60, 60, 14)
    assert s_soft < 0.3 * s_sharp and f_soft < 0.65 * f_sharp


def test_passby_two_contacts_one_attached():
    comp = np.zeros((80, 80), bool)
    comp[40:43, 5:75] = True  # passes along the top of the grain's rim...
    comp[54:57, 0:10] = True  # ...and a second tube reaches the rim radially from the left
    yy, xx = np.mgrid[0:80, 0:80]
    d = np.hypot(xx - 21, yy - 55)
    ring = (d >= 11) & (d <= 15)
    assert not R.passby(comp, ring, 21, 55)
    comp[54:57, 0:10] = False
    assert R.passby(comp, ring, 21, 55)


# ----------------------------------------------------------------------------- end to end: the frame edge
def _edge_movie(T=40, jump_at=30, jump=-40.0):
    """A grain 60 px from the left edge of a 160 x 160 frame with a tube growing left from its rim (1.5 px per bin
    from bin 5); from ``jump_at`` the stage moves ``jump`` px in x (the image shows it, the registration undoes it),
    so pixels within 40 px of the left edge have no source in the frame from then on."""
    from sparsetrack.render import Renderer
    H = W = 160
    gx, gy, gr = 60.0, 80.0, 10.0
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    shifts = np.zeros((T, 2))
    shifts[jump_at:, 0] = jump
    img = np.full((T, H, W), 130.0, np.float32)
    prob = np.zeros((T, H, W), np.float32)
    rng = np.random.default_rng(0)
    for t in range(T):
        L = float(np.clip((t - 5) * 1.5, 0, 45))
        ex, ey = xx + shifts[t, 0] * 0, yy  # reference coordinates of the probability cache
        rg = np.hypot(ex - gx, ey - gy)
        tube = (ex <= gx - gr) & (ex >= gx - gr - L) & (np.abs(ey - gy) <= 1.5)
        prob[t] = np.where(tube, 16.0, 0.0)
        # the image in frame coordinates: content at reference x appears at x + shift
        fx = xx - shifts[t, 0]
        frg = np.hypot(fx - gx, yy - gy)
        img[t][frg < gr] = 90.0
        img[t][(frg >= gr - 2) & (frg < gr)] = 60.0
        ftube = (fx <= gx - gr) & (fx >= gx - gr - L) & (np.abs(yy - gy) <= 1.5)
        img[t][ftube] -= 30.0
        img[t][fx < 0] = img[t][:, :1].mean()  # no scene outside the frame
        img[t] += rng.normal(0, 1.5, (H, W)).astype(np.float32)
        del rg
    meta_i = {"shifts": shifts.tolist(), "n_bins": T, "frames_per_bin": 1, "ref_start": 0}
    meta_p = {"shifts": np.zeros((T, 2)).tolist(), "n_bins": T, "frames_per_bin": 1, "ref_start": 0}
    return Renderer(prob, meta_p), Renderer(img, meta_i), meta_p, {"id": "g001", "x": gx, "y": gy, "r": gr}


def test_per_bin_edge_mask_keeps_the_tube_while_it_is_in_the_frame():
    RP, RI, meta, g = _edge_movie()
    frozen = R.reach_grain(RP, RI, meta, g, [], half=60, vmax=4.0, edge_mask="movie")
    per_bin = R.reach_grain(RP, RI, meta, g, [], half=60, vmax=4.0, edge_mask="bin")
    truth = np.clip((np.arange(40) - 5) * 1.5, 0, 45)
    raw_f, raw_b = np.array(frozen["raw_reach_px"]), np.array(per_bin["raw_reach_px"])
    assert raw_f[:30].max() <= 12.0  # the whole movie's mask cuts the tube ~10 px from the rim in every bin
    assert np.all(np.abs(raw_b[12:30] - truth[12:30]) <= 3.0)  # per bin: the tube is read while it is visible
    assert raw_b[30:].max() <= 12.0  # ...and cut at the edge after the stage moved


# ----------------------------------------------------------------------------- end to end: growth, leaving, frozen
def _grow_movie(T=50, ref_start=0, fpb=1, shifts=None, away=None, burst_at=None, gx=80.0, gy=80.0, gr=10.0):
    """A dark grain with a thin dark tube growing to its right (1.5 px per bin from bin 10, up to 36 px) and the
    tube's probability map (x16). ``away=(a, z)``: the grain's disc is not in the image in bins a..z-1 (z=None: to
    the end); ``burst_at``: the tube is gone from then on."""
    from sparsetrack.render import Renderer
    rng = np.random.default_rng(0)
    H = W = 160
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    shifts = np.zeros((T, 2)) if shifts is None else shifts
    img = np.full((T, H, W), 130.0, np.float32)
    prob = np.zeros((T, H, W), np.float32)
    for t in range(T):
        L = 0.0 if burst_at is not None and t >= burst_at else float(np.clip((t - 10) * 1.5, 0.0, 36.0))
        fx, fy = xx - shifts[t, 0], yy - shifts[t, 1]
        rg = np.hypot(fx - gx, fy - gy)
        here = away is None or not (away[0] <= t < (away[1] if away[1] is not None else T))
        if here:
            img[t][rg < gr] = 90.0
            img[t][(rg >= gr - 2) & (rg < gr)] = 60.0
        if L > 0:
            along = (fx >= gx + gr) & (fx <= gx + gr + L)
            d = np.abs(fy - gy)
            img[t] -= np.where(along, 35.0 * np.exp(-d ** 2 / 2.0), 0.0).astype(np.float32)
            prob[t] = np.where(along & (d <= 1.5), 16.0, 0.0)
        img[t] += rng.normal(0, 2.0, (H, W)).astype(np.float32)
    meta = {"shifts": np.asarray(shifts).tolist(), "n_bins": T, "frames_per_bin": fpb, "ref_start": ref_start}
    return Renderer(prob, meta), Renderer(img, meta), meta, {"id": "g001", "x": gx, "y": gy, "r": gr}


FROZEN = dict(edge_mask="movie", gone="flag", fit="l1")


def test_the_frozen_options_reproduce_the_decoder_as_it_was():
    """Outputs of reach_grain as of commit db6c752^ (the decoder the development numbers were measured with)."""
    T = 50
    sh = np.c_[np.linspace(0, 6, T), np.linspace(0, -4, T)]
    RP, RI, meta, g = _grow_movie(ref_start=5, fpb=3, shifts=sh, burst_at=38)
    res = R.reach_grain(RP, RI, meta, g, [{"id": "g002", "x": 80.0, "y": 118.0, "r": 10.0}], half=60, burst=True,
                        **FROZEN)
    assert (res["status"], res["onset_frame"], res["flags"], res["burst_frame"]) == \
        ("emerged_within", 34, ["burst_after:112"], 115)
    raw = [0.0] * 6 + [2.51, 2.93, 3.93, 5.93, 6.93, 8.93, 10.93, 11.51, 13.93, 14.93, 16.34, 17.93, 18.93, 20.93,
                       22.93, 23.93, 25.93, 26.51, 27.93, 29.93, 31.34, 32.93, 34.93] + [35.93] * 4 + [0.0] * 12
    fit = [0.0] * 6 + [2.5, 3.0, 4.0, 6.0, 7.0, 9.0, 11.0, 11.5, 14.0, 15.0, 16.5, 18.0, 19.0, 21.0, 23.0, 24.0, 26.0,
                       26.5, 28.0, 30.0, 31.5, 33.0, 35.0] + [36.0] * 16
    assert np.allclose(res["raw_reach_px"], raw, atol=0.05) and np.allclose(res["length"]["px"], fit, atol=0.05)
    RP, RI, meta, g = _grow_movie(away=(35, None))
    res = R.reach_grain(RP, RI, meta, g, [], half=60, **FROZEN)
    assert (res["status"], res["onset_frame"], res["flags"]) == ("emerged_within", 12, ["no_grain_after:35"])
    assert "gone_bins" not in res
    fit = [0.0] * 11 + [1.5, 2.5, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0, 13.0, 14.0, 16.0, 17.0, 19.0] + [20.0] * 21 \
        + [22.5] + [23.0] * 4
    assert np.allclose(res["length"]["px"], fit, atol=0.05)


def test_a_spell_away_is_flagged_with_its_frames_and_a_grain_that_left_for_good_as_left():
    RP, RI, meta, g = _grow_movie(away=(30, 38))
    res = R.reach_grain(RP, RI, meta, g, [], half=60)
    assert "gone:30-37" in res["flags"] and not any(f.startswith("no_grain_after") for f in res["flags"])
    assert res["gone_bins"] == list(range(30, 38))
    assert abs(res["final_length_px"] - 36.0) <= 3.0  # read again once it is back
    RP, RI, meta, g = _grow_movie(away=(35, None))
    assert "no_grain_after:35" in R.reach_grain(RP, RI, meta, g, [], half=60)["flags"]


def test_an_onset_length_the_tube_never_reaches_is_no_onset():
    # calibration may adopt onset_px above min_tube_px; a tube stopping between the two has not germinated by it
    RP, RI, meta, g = _grow_movie()
    res = R.reach_grain(RP, RI, meta, g, [], half=60, onset_px=40.0, min_tube_px=8.0)
    assert res["status"] == "no_emergence_by_end" and res["onset_frame"] is None


# ----------------------------------------------------------------------------- end to end: ghost discs
def _field(mover=None, jump=(0.0, 0.0), at=0, glide=False, ghost=6, T=30):
    """Seven discs (r 12) on a 400 x 400 field; disc ``ghost`` is out of focus; disc ``mover`` moves by ``jump``
    px from bin ``at`` (at once, or gliding over 8 bins)."""
    import cv2
    from sparsetrack.render import Renderer
    H = W = 400
    pos = [(80.0, 80.0), (200.0, 80.0), (320.0, 80.0), (80.0, 250.0), (200.0, 250.0), (320.0, 250.0), (200.0, 350.0)]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    rng = np.random.default_rng(0)
    frames = []
    for t in range(T):
        im = np.full((H, W), 180.0)
        for k, (x, y) in enumerate(pos):
            if k == mover and t >= at:
                f = min(1.0, (t - at + 1) / 8.0) if glide else 1.0
                x, y = x + f * jump[0], y + f * jump[1]
            d = np.hypot(xx - x, yy - y)
            disc = np.where(d < 12, -60.0, 0.0) + np.where((d >= 10) & (d < 13), -40.0, 0.0)
            if k == ghost:
                disc = cv2.GaussianBlur(disc.astype(np.float32), (0, 0), 5.0) * 0.5
            im = im + disc
        frames.append(im + rng.normal(0, 2.0, (H, W)))
    meta = {"shifts": np.zeros((T, 2)).tolist(), "n_bins": T, "frames_per_bin": 1, "ref_start": 0}
    census = [{"id": f"g{k + 1:03d}", "x": x, "y": y, "r": 12.0} for k, (x, y) in enumerate(pos)]
    return Renderer(np.stack(frames).astype(np.float32), meta), census


def test_ghost_test_keeps_a_sharp_grain_that_moves_early():
    R_img, census = _field()
    assert set(R.census_ghosts(R_img, census)) == {"g007"}
    for kw in (dict(jump=(15.0, 0.0), at=3), dict(jump=(25.0, 0.0), at=5), dict(jump=(20.0, 0.0), at=2, glide=True)):
        R_img, census = _field(mover=0, **kw)
        assert set(R.census_ghosts(R_img, census)) == {"g007"}, kw  # g001 is judged where it was counted


def test_ghost_test_judges_excluded_discs_too_against_the_included_ones():
    R_img, census = _field()
    census[6]["excluded"], census[6]["exclude_reason"] = True, "out_of_focus"  # an annotator's call
    assert set(R.census_ghosts(R_img, census)) == {"g007"}
    census[0]["excluded"], census[0]["exclude_reason"] = True, "clump"  # a sharp excluded disc is still a disc
    assert set(R.census_ghosts(R_img, census)) == {"g007"}


def test_analyze_drops_an_excluded_ghost_from_the_neighbours(tmp_path, monkeypatch):
    import json

    from sparsetrack import stack
    R_img, census = _field()
    census[6]["excluded"], census[6]["exclude_reason"] = True, "out_of_focus"  # g007, the ghost, excluded by hand
    T = R_img.n_bins
    for name, bins in (("img", np.asarray(R_img.bins)), ("prob", np.zeros_like(np.asarray(R_img.bins)))):
        (tmp_path / name).mkdir()
        np.save(tmp_path / name / "bins.npy", bins.astype(np.float16))
        (tmp_path / name / "meta.json").write_text(json.dumps({
            "schema": stack.SCHEMA, "frames_per_bin": 1, "n_bins": T, "shifts": [[0.0, 0.0]] * T,
            "movie": {"name": "t.mp4", "size_bytes": 1, "n_frames": T, "width": 400, "height": 400}}))
        (tmp_path / name / "grains.json").write_text(json.dumps({"grains": census}))
    seen = {}
    monkeypatch.setattr(R, "reach_grain", lambda RP, RI, meta, g, others, **kw: seen.setdefault(
        g["id"], [o["id"] for o in others]) and {"id": g["id"]})
    pred = R.analyze(tmp_path / "prob", tmp_path / "img", log=lambda *a: None)
    assert set(pred["ghosts"]) == {"g007"}
    assert "g007" not in seen and all("g007" not in others for others in seen.values())
    seen.clear()
    R.analyze(tmp_path / "prob", tmp_path / "img", log=lambda *a: None, ghosts=False)
    assert seen and all("g007" in others for others in seen.values())  # without the test it blocks as before
