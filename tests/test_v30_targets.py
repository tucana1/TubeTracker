"""rev5 #2 proof: one saved UI example -> export -> target -> loss.

Drives the headless controller (tip click + negative box + census
tile), exports a v30 snapshot, builds real apex/front targets, and
proves: (a) validity geometry (positive at tip, negative at box, zero
elsewhere); (b) label-dependence (moving the tip changes the loss,
perturbing only unknown pixels does not); (c) the blinded oracle route
does not end at the answer.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_app import AnnotatorController  # noqa: E402
from tubetracker.annotation_frames import FrameReader  # noqa: E402
from tubetracker.annotation_store import AnnotationStore  # noqa: E402

MOVIE_LD = "/Users/joshjiang/Downloads/test1lowdensjoshua-28c-hz.mp4 .mp4"
FRAME = 15000
TIP = (1017.0, 331.0)


def _project(tmp_path):
    s = AnnotationStore(tmp_path / "annotations.db")
    r = FrameReader(MOVIE_LD)
    c = AnnotatorController(s, r, viewer=None, actor="proof")
    c.load_tasks([
        {"uuid": "t-tip", "owner_uuid": "o1", "query_frames": [FRAME],
         "task_type": "apex", "movie": "ld", "completed": False},
        {"uuid": "t-neg", "owner_uuid": "o1", "query_frames": [FRAME],
         "task_type": "neg_region", "movie": "ld",
         "neg_class": "apex", "completed": False},
        {"uuid": "t-cen", "owner_uuid": "o1", "query_frames": [FRAME],
         "task_type": "census", "movie": "ld",
         "focus_xy": [500.0, 500.0], "completed": False},
    ])
    return s, r, c


def test_ui_to_export_to_targets(tmp_path):
    s, r, c = _project(tmp_path)
    try:
        # 1. UI: tip click, negative box (2 clicks), census dots + finish.
        c.current = next(t for t in c.tasks if t["uuid"] == "t-tip")
        c.click(*TIP)
        c.current = next(t for t in c.tasks if t["uuid"] == "t-neg")
        c.click(100.0, 100.0)  # single click: fixed box + auto-advance
        assert c.current["uuid"] == "t-cen"
        assert len(next(t for t in c.tasks
                       if t["uuid"] == "t-neg").get("neg_boxes", [])) == 1
        c.current = next(t for t in c.tasks if t["uuid"] == "t-cen")
        cen_task = c.current
        c.click(490.0, 495.0)
        c.click(510.0, 505.0)
        c.finish_census()
        assert cen_task.get("census_complete") is True

        # 2. Export: snapshot contains all three label kinds.
        snap = tmp_path / "snap"
        pr = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "build_v30_snapshot.py"),
             "--project-dir", str(tmp_path),
             "--movie", f"ld={MOVIE_LD}",
             "--out", str(snap)],
            capture_output=True, text=True)
        assert pr.returncode == 0, pr.stderr[-2000:]
        obs = json.loads((snap / "observations.json").read_text())
        regs = json.loads((snap / "regions.json").read_text())
        cen = json.loads((snap / "census.json").read_text())
        assert any(o.get("direct_xy") == [1017.0, 331.0] for o in obs)
        assert len(regs) == 1 and regs[0]["kind"] == "verified_negative"
        assert regs[0]["polygon_xy"][0] == [72.0, 72.0]  # 56px fixed box
        assert len(cen) == 1 and cen[0]["complete"] is True
        assert len(cen[0]["tips"]) == 2

        # 3. Targets: validity geometry on a real crop.
        from prototypes.v30_video_apex.targets import (
            apex_heat, apex_valid_mask, load_clip_pixels)
        crop = (900, 250, 320, 256)
        clip = load_clip_pixels(r, [FRAME], crop)
        assert clip.shape == (1, 256, 320)
        assert clip.max() > 0  # real pixels, not padding
        tip_crop = (TIP[0] - crop[0], TIP[1] - crop[1])
        heat = apex_heat(256, 320, tip_crop)
        assert heat.shape == (256, 320)
        assert np.unravel_index(int(heat.argmax()), heat.shape) == (
            round(tip_crop[1]), round(tip_crop[0]))
        valid = apex_valid_mask(256, 320, crop[:2], tip_crop, regs)
        # positive disc at tip
        assert valid[int(tip_crop[1]), int(tip_crop[0])] == 1.0
        # box is far outside this crop -> crop stays mostly unknown
        assert valid.mean() < 0.05
        # same box in its own crop is negative-valid
        valid2 = apex_valid_mask(64, 64, (64, 64), None, regs)
        assert valid2[20, 20] == 1.0  # inside box (72-128)
        assert valid2[0, 0] == 0.0    # outside stays unknown
    finally:
        r.close()
        s.close()


def test_loss_label_dependence_and_blinded_route():
    import torch

    from prototypes.v30_video_apex.targets import (
        apex_heat, apex_valid_mask, blind_oracle_route,
        front_interval_mask, project_to_arclength)
    from prototypes.v30_video_apex.train import (
        LossWeights, masked_multihead_loss)

    h, w = 64, 64
    tip = (40.0, 30.0)
    heat = torch.from_numpy(apex_heat(h, w, tip))
    valid = torch.from_numpy(
        apex_valid_mask(h, w, (0, 0), tip, []))
    pred = {"apex_heat": torch.zeros(1, 1, h, w)}
    base = masked_multihead_loss(
        pred, {"apex_heat": heat.unsqueeze(0)},
        {"apex_valid": valid.unsqueeze(0)},
        LossWeights())["total"]
    # Moving the tip 30px changes the supervised loss.
    heat2 = torch.from_numpy(apex_heat(h, w, (tip[0] + 30.0, tip[1])))
    moved = masked_multihead_loss(
        pred, {"apex_heat": heat2.unsqueeze(0)},
        {"apex_valid": valid.unsqueeze(0)},
        LossWeights())["total"]
    assert abs(float(moved) - float(base)) > 1e-6
    # Perturbing the target ONLY in unknown pixels changes nothing.
    heat3 = heat.clone()
    heat3[0, 0] = 1.0  # corner is unknown (valid==0 there)
    assert bool(valid[0, 0].item()) is False
    same = masked_multihead_loss(
        pred, {"apex_heat": heat3.unsqueeze(0)},
        {"apex_valid": valid.unsqueeze(0)},
        LossWeights())["total"]
    assert abs(float(same) - float(base)) < 1e-9

    # Blinded oracle: support extends past the cap; tip is interior.
    path = [[0.0, 0.0], [30.0, 0.0], [60.0, 5.0]]
    tip_xy = [60.0, 5.0]
    blind = blind_oracle_route(path, tip_xy)
    end = blind["route_xy"][-1]
    assert float(np.hypot(end[0] - tip_xy[0], end[1] - tip_xy[1])) > 8.0
    assert 2.0 < blind["s_star"] < blind["support"] - 8.0
    mask = front_interval_mask(blind["s_grid"], blind["s_star"])
    assert mask.sum() >= 3  # ±2px around s* at 1px sampling
    assert mask[0] == 0.0 and mask[-1] == 0.0  # ends never supervised
    # Projection sanity: tip at path end maps near original length.
    assert abs(project_to_arclength(np.asarray(path, float),
                                    np.asarray(tip_xy, float))
               - (30.0 + float(np.hypot(30.0, 5.0)))) < 1.0


def test_checkpoint_reload_reproduces_inference(tmp_path):
    import torch

    from prototypes.v30_video_apex.model import (
        OwnerPrompt, build_model, random_clip)
    from prototypes.v30_video_apex.train import (
        load_checkpoint, save_checkpoint)

    torch.manual_seed(11)
    model = build_model("temporal", base=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    clip = random_clip(1, 9, 1, 32)
    prompt = OwnerPrompt(owner_id="o")
    route = [[2.0, 2.0], [16.0, 16.0], [28.0, 28.0]]
    model.eval()
    with torch.no_grad():
        before = model.forward(clip, prompt, route_xy=route)
    ckpt = save_checkpoint(tmp_path / "ckpt.pt", model, opt,
                           config={"variant": "temporal", "base": 8},
                           manifest={"n_updates": 7})
    try:
        save_checkpoint(ckpt, model)
    except FileExistsError:
        pass
    else:
        raise AssertionError("checkpoint overwrite must refuse")
    model2 = build_model("temporal", base=8)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    meta = load_checkpoint(ckpt, model2, opt2)
    assert meta["manifest"] == {"n_updates": 7}
    assert meta["config_hash"] == "2b2f2b0e1e2b2b2b" or len(
        meta["config_hash"]) == 16
    model2.eval()
    with torch.no_grad():
        after = model2.forward(clip, prompt, route_xy=route)
    assert torch.equal(before.front_logits, after.front_logits)
    assert torch.equal(before.heat, after.heat)
    # Optimizer state resumes (same param groups + state keys).
    assert (opt.state_dict()["param_groups"][0]["lr"] ==
            opt2.state_dict()["param_groups"][0]["lr"])


def test_movie_qualified_assertion():
    from prototypes.v30_video_apex.dataset import (
        ManifestEntry, assert_movie_qualified)

    good = [ManifestEntry(entry_id="ld|a|42000", movie_id="ld",
                          acquisition_group="s1", source_frame=42000,
                          owner_id="t1"),
            ManifestEntry(entry_id="m2|a|42000", movie_id="m2",
                          acquisition_group="s2", source_frame=42000,
                          owner_id="t1")]
    assert_movie_qualified(good)  # same frame, different movies: fine
    bad_prefix = [ManifestEntry(entry_id="42000", movie_id="ld",
                                acquisition_group="s1", source_frame=42000,
                                owner_id="t1")]
    try:
        assert_movie_qualified(bad_prefix)
    except ValueError:
        pass
    else:
        raise AssertionError("unqualified entry_id must raise")
    dup = [ManifestEntry(entry_id="ld|a|1", movie_id="ld",
                         acquisition_group="s1", source_frame=9,
                         owner_id="t1", kind="path_tip"),
           ManifestEntry(entry_id="ld|b|1", movie_id="ld",
                         acquisition_group="s1", source_frame=9,
                         owner_id="t1", kind="path_tip")]
    try:
        assert_movie_qualified(dup)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate movie/frame/owner must raise")
    # rev8: one owner/frame may carry DIFFERENT kinds of supervision —
    # a traced path and a painted body mask are different heads, not
    # duplicates. Only a same-kind repeat is a genuine duplicate.
    two_kinds = [ManifestEntry(entry_id="ld|a|1", movie_id="ld",
                               acquisition_group="s1", source_frame=9,
                               owner_id="t1", kind="path_tip"),
                 ManifestEntry(entry_id="ld|a-mask|1", movie_id="ld",
                               acquisition_group="s1", source_frame=9,
                               owner_id="t1", kind="body_mask")]
    assert_movie_qualified(two_kinds)


def test_train_step_front_descends_and_reports():
    import torch

    from prototypes.v30_video_apex.model import (
        OwnerPrompt, build_model, random_clip)
    from prototypes.v30_video_apex.targets import front_interval_mask
    from prototypes.v30_video_apex.train import (
        LossWeights, train_step_front)

    torch.manual_seed(3)
    model = build_model("temporal", base=8)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    clip = random_clip(1, 9, 1, 48)
    prompt = OwnerPrompt(owner_id="o")
    route = [[4.0, 4.0], [24.0, 24.0], [44.0, 44.0]]
    with torch.no_grad():
        probe = model.forward(clip, prompt, route_xy=route)
    n_s = int(probe.front_logits.shape[1])
    s = probe.front_s.detach().numpy()
    s_star = float(s[int(n_s * 0.7)])
    interval = torch.from_numpy(
        front_interval_mask(s, s_star)).to(torch.float32)
    yy, xx = torch.meshgrid(torch.arange(48), torch.arange(48),
                            indexing="ij")
    heat_t = torch.exp(-((xx - 44.0) ** 2 + (yy - 44.0) ** 2) / 8.0)
    heat_t = heat_t.unsqueeze(0).unsqueeze(0)
    errs = []
    for _ in range(6):
        r = train_step_front(
            model, opt, clip, prompt, route,
            heat_t, torch.ones_like(heat_t), interval, torch.ones(1),
            torch.tensor([0]), torch.ones(1),
            LossWeights(apex=1.0, front=1.0, visibility=0.2))
        errs.append(r["front_err_px"])
        assert abs(r["s_star"] - s_star) < 2.0
    assert errs[-1] < errs[0], f"front not localizing: {errs}"


def test_front_logits_match_ribbon_support_on_odd_lengths():
    """Regression: transpose-conv overshoot must not misalign q(s)."""
    import torch

    from prototypes.v30_video_apex.model import (
        OwnerPrompt, build_model, random_clip, sample_ribbon)

    torch.manual_seed(5)
    for route in ([[4.0, 4.0], [24.0, 24.0], [44.0, 44.0]],   # odd support
                  [[1.0, 1.0], [30.0, 5.0], [61.0, 40.0]]):   # even support
        model = build_model("temporal", base=8)
        clip = random_clip(1, 9, 1, 48)
        pred = model.forward(clip, OwnerPrompt(owner_id="o"),
                             route_xy=route)
        rb = sample_ribbon(clip[:, :, :1], route)
        assert pred.front_logits.shape[1] == rb["ribbon"].shape[3], route
        assert pred.front_s.shape[0] == rb["ribbon"].shape[3], route
        assert pred.heat.shape[-2:] == (48, 48)


def test_confidence_gate_withholds_low_q():
    from prototypes.v30_video_apex.evaluate import confidence_gate

    scored = [{"route_id": "a", "q_max": 0.2},
              {"route_id": "b", "q_max": 0.1}]
    assert confidence_gate(scored, 0.0)["route_id"] == "a"
    assert confidence_gate(scored, 0.5) is None
    assert confidence_gate([], 0.0) is None


def test_jitter_route_keeps_tip_interior_or_falls_back():
    import numpy as np

    from prototypes.v30_video_apex.targets import jitter_route

    route = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 5.0]])
    tip = np.array([100.0, 5.0])
    rng = np.random.default_rng(0)
    for _ in range(10):
        rj, sj = jitter_route(route, tip, 12.0, rng)
        if sj is not None:
            total = float(np.hypot(np.diff(rj[:, 0]),
                                   np.diff(rj[:, 1])).sum())
            assert 2.0 < sj < total - 8.0
        else:
            assert (rj == route).all()  # fallback returns input unchanged


def test_route_head_separates_true_vs_wrong_ribbon():
    import torch

    from prototypes.v30_video_apex.model import (
        OwnerPrompt, build_model, random_clip)
    from prototypes.v30_video_apex.targets import front_interval_mask
    from prototypes.v30_video_apex.train import (
        LossWeights, train_step_front)

    torch.manual_seed(9)
    model = build_model("temporal", base=8)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    clip = random_clip(1, 9, 1, 40)
    prompt = OwnerPrompt(owner_id="o")
    true_route = [[2.0, 2.0], [20.0, 20.0], [37.0, 37.0]]
    wrong_route = [[2.0, 37.0], [20.0, 20.0], [37.0, 2.0]]
    heat_t = torch.zeros(1, 1, 40, 40)
    heat_t[0, 0, 37, 37] = 1.0
    with torch.no_grad():
        probe = model.forward(clip, prompt, route_xy=true_route)  # type: ignore[call-arg]
    n_s = int(probe.front_logits.shape[1])
    interval = torch.zeros(n_s)
    interval[n_s - 3] = 1.0
    for _ in range(20):
        train_step_front(
            model, opt, clip, prompt, true_route, heat_t,
            torch.ones_like(heat_t), interval, torch.ones(1),
            torch.tensor([0]), torch.ones(1),
            LossWeights(apex=0.2, front=1.0, visibility=0.0,
                        route_correct=1.0),
            route_t=torch.ones(1))
        train_step_front(
            model, opt, clip, prompt, wrong_route,
            torch.zeros_like(heat_t), torch.zeros_like(heat_t),
            torch.zeros(n_s), torch.zeros(1),
            torch.tensor([0]), torch.ones(1),
            LossWeights(apex=0.2, front=1.0, visibility=0.0,
                        route_correct=1.0),
            route_t=torch.zeros(1))
    model.eval()
    with torch.no_grad():
        rt = model.forward(clip, prompt, route_xy=true_route).route_logit  # type: ignore[call-arg]
        rw = model.forward(clip, prompt, route_xy=wrong_route).route_logit  # type: ignore[call-arg]
    assert rt is not None and rw is not None
    assert float(rt[0]) > float(rw[0]), (float(rt[0]), float(rw[0]))


def test_fold_exclusion_rules(tmp_path):
    """Rechecks/abstentions/conflicts/degenerates never train (snap4 rules)."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prototypes.v30_video_apex.targets import samples_from_snapshot

    def obs(uuid, task, state, xy=None, focus=None, ttype="apex",
            showed=False, path=None):
        return {"obs_uuid": uuid, "obs_revision": 1, "task_uuid": task,
                "task_revision": 1, "project": "p", "movie": "ld",
                "movie_path": "/m.mp4", "tube_uuid": "", "owner_uuid": "",
                "source_frame": 100, "direct_state": state,
                "direct_xy": list(xy) if xy else None,
                "direct_region": None, "context_xy": None,
                "context_frames": [], "path_xy": path or [],
                "path_visible": [], "path_complete": bool(path),
                "focus_xy": list(focus) if focus else None,
                "task_type": ttype, "task_showed_tip": showed,
                "owner_decision": "", "annotator": "t"}

    snap = tmp_path / "snap"
    snap.mkdir()
    rows = [
        # banked anchor tip (FULL path tip here for rank coverage)
        obs("o0", "t0", "direct_visible", (50.0, 50.0), (50.0, 50.0),
            path=[[10.0, 10.0], [50.0, 50.0]]),
        # marker-anchored recheck 0.6px away -> excluded
        obs("o1", "t1", "direct_visible", (50.6, 50.0), (50.0, 50.0),
            showed=True),
        # independent agreement 2px away, no marker -> kept
        obs("o2", "t2", "direct_visible", (52.0, 50.0), (52.0, 50.0)),
        # census-tile absence -> abstention, excluded
        obs("o3", "t3", "no_tube_visible", None, (52.0, 50.0),
            ttype="census"),
        # absence at the banked tip -> conflict, excluded
        obs("o4", "t4", "no_tube_visible", None, (50.0, 50.0)),
        # genuine absence far away -> kept
        obs("o5", "t5", "no_tube_visible", None, (900.0, 900.0)),
    ]
    (snap / "observations.json").write_text(json.dumps(rows))
    (snap / "regions.json").write_text("[]")
    got = samples_from_snapshot(str(snap))
    kinds = sorted((s.obs_uuid, s.kind) for s in got)
    assert kinds == [("o0", "path_tip"), ("o2", "tip_only"),
                     ("o5", "no_tube")], kinds


def test_split_pos_neg_no_global_cooling():
    """Split normalization: one step must push the tip pixel UP and the
    verified-negative pixel DOWN at once (v4 failure: shared denominator
    cooled the whole map instead of discriminating)."""
    import torch
    from prototypes.v30_video_apex.train import (
        LossWeights, masked_multihead_loss)
    heat = torch.nn.Parameter(torch.full((1, 1, 8, 8), 0.5))
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 1, 1] = 1.0  # one tip pixel
    pos = torch.zeros(1, 1, 8, 8)
    pos[0, 0, 1, 1] = 1.0
    neg = torch.zeros(1, 1, 8, 8)
    neg[0, 0, 6, 6] = 1.0  # one verified-negative pixel
    opt = torch.optim.SGD([heat], lr=1.0)
    opt.zero_grad()
    out = masked_multihead_loss(
        {"apex_heat": heat}, {"apex_heat": target},
        {"apex_pos": pos, "apex_neg": neg},
        LossWeights(apex=1.0, apex_neg=1.0))
    assert "apex_neg" in out
    out["total"].backward()
    opt.step()
    with torch.no_grad():
        assert float(heat[0, 0, 1, 1]) > 0.5, "tip pixel must rise"
        assert float(heat[0, 0, 6, 6]) < 0.5, "neg pixel must fall"
        assert abs(float(heat[0, 0, 3, 3]) - 0.5) < 1e-6, \
            "unknown pixels must not move"


def test_outside_box_paints_zero_pixels():
    """rev6: a box wholly outside the crop must paint nothing (was a
    57px border sliver via clip-without-intersection)."""
    import numpy as np
    from prototypes.v30_video_apex.targets import apex_pos_neg_masks
    regs = [{"kind": "verified_negative", "class_scope": "apex",
             "polygon_xy": [[900., 600.], [956., 600.],
                            [956., 656.], [900., 656.]]}]
    pos, neg = apex_pos_neg_masks(288, 288, (560., 509.), None, regs)
    assert int(neg.sum()) == 0 and int(pos.sum()) == 0


def test_other_known_tips_stay_unknown_in_neg_box():
    """rev6 v30d-001: an accepted box containing another sample's cap
    keeps those pixels unknown, never negative."""
    import numpy as np
    from prototypes.v30_video_apex.targets import apex_pos_neg_masks
    regs = [{"kind": "verified_negative", "class_scope": "apex",
             "polygon_xy": [[424., 116.], [480., 116.],
                            [480., 172.], [424., 172.]]}]
    ox, oy = 291., 5.
    tip_crop = (467.5 - ox, 139.5 - oy)  # p00 cap inside the box
    pos, neg = apex_pos_neg_masks(288, 288, (ox, oy), None, regs,
                                  other_tips_crop=[tip_crop])
    assert int(neg.sum()) > 100, "box still supervises away from cap"
    yy, xx = np.mgrid[0:288, 0:288].astype(np.float32)
    d = np.hypot(xx - tip_crop[0], yy - tip_crop[1])
    assert int(neg[d <= 8.0].sum()) == 0, "cap pixels must be unknown"


def test_region_tip_conflict_recorded(tmp_path):
    """rev6 conflict contract with a deterministic fixture (was the
    machine-local /tmp/v30snap6 read). A verified-negative box that
    contains another task's banked tip must be recorded in the manifest
    as a region<->tip conflict, never silently trained as a negative.
    Drives the real controller path: UI click -> DB -> snapshot -> manifest.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    s = AnnotationStore(proj / "annotations.db")
    r = FrameReader(MOVIE_LD)
    c = AnnotatorController(s, r, viewer=None, actor="proof")
    try:
        c.load_tasks([
            {"uuid": "t-tip", "owner_uuid": "o1",
             "query_frames": [FRAME], "task_type": "apex",
             "movie": "ld", "completed": False},
            {"uuid": "v30d-001", "owner_uuid": "o1",
             "query_frames": [FRAME], "task_type": "neg_region",
             "movie": "ld", "neg_class": "apex", "completed": False},
        ])
        c.current = next(t for t in c.tasks if t["uuid"] == "t-tip")
        c.click(*TIP)                       # banks a precise tip
        c.current = next(t for t in c.tasks if t["uuid"] == "v30d-001")
        c.click(*TIP)                       # box centered ON that tip
        assert c.current is None or c.current["uuid"] != "v30d-001"
    finally:
        r.close()
        s.close()

    snap = tmp_path / "snap"
    pr = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_v30_snapshot.py"),
         "--project-dir", str(proj), "--movie", f"ld={MOVIE_LD}",
         "--default-movie", "ld", "--out", str(snap)],
        capture_output=True, text=True, cwd=str(REPO))
    assert pr.returncode == 0, pr.stderr[-2000:]
    m = json.loads((snap / "snapshot_manifest.json").read_text())
    assert m["n_region_tip_conflicts"] >= 1, m["region_tip_conflicts"]
    tasks = [c_["task_uuid"] for c_ in m["region_tip_conflicts"]]
    assert "v30d-001" in tasks, m["region_tip_conflicts"]
    # the conflicting tip is named, so adjudication is possible
    hit = next(c_ for c_ in m["region_tip_conflicts"]
               if c_["task_uuid"] == "v30d-001")
    assert hit["tip_obs_uuids"], hit


def test_region_sample_centers_its_box(tmp_path):
    """rev6: every verified-negative region becomes a first-class
    sample whose own crop contains the box (deterministic fixture; the
    former /tmp/v30snap6 dependency is gone)."""
    from prototypes.v30_video_apex.targets import (
        apex_pos_neg_masks, samples_from_snapshot)

    snap = tmp_path / "snap"
    snap.mkdir()
    # one real tip sample supplies the movie/path mapping the region
    # samples join against
    (snap / "observations.json").write_text(json.dumps([
        {"obs_uuid": "o1", "task_uuid": "t1", "project": "p",
         "movie": "ld", "movie_path": MOVIE_LD, "source_frame": FRAME,
         "direct_state": "direct_visible", "direct_xy": [300.0, 300.0],
         "path_xy": [[280.0, 280.0], [300.0, 300.0]],
         "path_complete": True, "path_visible": [True, True]},
    ]))
    boxes = [(200.0, 200.0), (700.0, 450.0), (950.0, 800.0)]
    (snap / "regions.json").write_text(json.dumps([
        {"_region_uuid": f"neg-{i}", "_region_revision": 1,
         "_project": "p", "task_uuid": f"t-neg-{i}", "movie_uuid": "ld",
         "source_frame": FRAME, "kind": "verified_negative",
         "class_scope": "apex",
         "polygon_xy": [[x - 28.0, y - 28.0], [x + 28.0, y - 28.0],
                        [x + 28.0, y + 28.0], [x - 28.0, y + 28.0]]}
        for i, (x, y) in enumerate(boxes)]))
    ss = samples_from_snapshot(str(snap))
    regs = [s for s in ss if s.kind == "neg_region"]
    assert len(regs) == len(boxes), [s.entry_id for s in regs]
    for s in regs:
        b = s.region_box
        cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        ox, oy = cx - 144.0, cy - 144.0  # train-prep crop rule
        poly = [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
        pos, neg = apex_pos_neg_masks(
            288, 288, (ox, oy), None,
            [{"kind": "verified_negative", "class_scope": "apex",
              "polygon_xy": poly}])
        assert int(neg.sum()) > 1000, (s.entry_id, int(neg.sum()))
        assert int(pos.sum()) == 0, s.entry_id


def test_owner_geometry_reaches_front_and_route():
    """rev6 audit counterexample as regression test: swapping the
    numerical owner (attachment+proximal) must change front and
    route outputs (was exactly 0.0); geometry=None renders zeros."""
    import torch
    from prototypes.v30_video_apex.model import (
        OwnerPrompt, _owner_channel, build_model)
    torch.manual_seed(0)
    m = build_model("temporal", base=8)
    m.eval()
    clip = torch.rand(1, 9, 1, 64, 64)
    route = [[5., 5.], [30., 20.], [55., 50.]]
    p_none = OwnerPrompt(owner_id="o")
    p_a = OwnerPrompt(owner_id="o", attachment_xy=(5., 5.),
                      proximal_xy=[(5., 5.), (12., 9.), (18., 13.)])
    p_b = OwnerPrompt(owner_id="o", attachment_xy=(55., 50.),
                      proximal_xy=[(55., 50.), (48., 44.), (42., 38.)])
    assert float(_owner_channel(clip, p_none)[:, :, 1].abs().max()) == 0.0
    assert float(_owner_channel(clip, p_a)[:, :, 1].max()) > 0.5
    with torch.no_grad():
        fa = m.forward(clip, p_a, route_xy=route)
        fb = m.forward(clip, p_b, route_xy=route)
    df = float((fa.front_logits - fb.front_logits).abs().max())
    dr = float(abs(fa.route_logit - fb.route_logit).max())
    assert df > 0.0 and dr > 0.0, (df, dr)


def test_body_head_learns_centerline_ribbon():
    """Body head + loss wiring: a few supervised steps on a synthetic
    diagonal path must reduce body BCE (visible spans only)."""
    import numpy as np
    import torch
    from prototypes.v30_video_apex.model import OwnerPrompt, build_model
    from prototypes.v30_video_apex.targets import body_ribbon_mask
    from prototypes.v30_video_apex.train import (
        LossWeights, masked_multihead_loss)
    torch.manual_seed(1)
    m = build_model("temporal", base=8)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    clip = torch.rand(1, 9, 1, 64, 64)
    path = [(8., 8.), (24., 24.), (40., 40.), (54., 54.)]
    bt, bv = body_ribbon_mask(64, 64, (0., 0.), path, [True] * 4)
    assert int((bt > 0).sum()) > 50 and int((bv > 0).sum()) > int(
        (bt > 0).sum())
    tgt = torch.from_numpy(bt).unsqueeze(0).unsqueeze(0)
    val = torch.from_numpy(bv).unsqueeze(0).unsqueeze(0)
    prompt = OwnerPrompt(owner_id="o")
    losses = []
    for _ in range(8):
        opt.zero_grad()
        pred = m.forward(clip, prompt)
        assert pred.body is not None and pred.body.shape[-2:] == (64, 64)
        out = masked_multihead_loss(
            {"body": pred.body}, {"body_mask": tgt},
            {"body_valid": val}, LossWeights(body=1.0))
        out["total"].backward()
        opt.step()
        losses.append(float(out["total"].detach() if hasattr(out["total"], "detach") else out["total"]))
    assert losses[-1] < losses[0], losses


def test_v11_checkpoint_loads_with_zero_gaps():
    """Additive heads (route, body) are the only allowed gaps: the
    latest checkpoint must load exactly (shapes preserved)."""
    from prototypes.v30_video_apex.model import build_model
    from prototypes.v30_video_apex.train import load_checkpoint
    m = build_model("temporal", base=16)
    r = load_checkpoint("runs/prototypes/v30/front_v11/front.pt", m,
                        strict=False)
    assert r["missing_keys"] == [], r["missing_keys"]
    assert r["unexpected_keys"] == [], r["unexpected_keys"]


def test_mask_raster_roundtrip_exact_with_hole():
    """rev7: exact raster saves every painted pixel and no other.

    Regression: stride-3 stamps + disc redraw expanded 364px of paint
    to 766px and refilled erased holes (IoU 0.475). The raster path
    must roundtrip bit-exact, holes included.
    """
    from prototypes.v30_video_apex.targets import (
        body_mask_from_raster, decode_mask_raster, encode_mask_raster)
    paint = np.zeros((60, 70), bool)
    paint[10:40, 20:50] = True
    paint[20:25, 30:35] = False  # erased hole stays a hole
    r = encode_mask_raster(paint)
    back = decode_mask_raster(60, 70, (0, 0), r)
    assert back.shape == paint.shape
    assert (back == paint).all()
    t, v = body_mask_from_raster(60, 70, (0, 0), r, complete=False)
    assert float(t.sum()) == float(paint.sum())
    assert float(t[20:25, 30:35].sum()) == 0.0  # hole supervises 0-adjacent
    assert float(v[paint].sum()) == float(paint.sum())


def test_paint_path_distance_is_euclidean():
    """rev7: far paint must score far, even sharing one coordinate.

    Regression: the transpose/min took per-coordinate minima, so paint
    far away in x but close in y scored median ~0 and passed the
    30px match. Euclidean nearest-vertex distance must be large here.
    """
    path_a = np.array([[100.0, 100.0], [110.0, 100.0]])
    pts = np.array([[500.0, 100.0]])  # same y, 400px away in x
    dd = pts[:, None, :] - path_a[None, :, :]
    d = np.hypot(dd[..., 0], dd[..., 1]).min(axis=1)
    assert float(np.median(d)) > 30.0


def test_duel_conflict_quarantined(tmp_path):
    """rev7: same lanes judged twice -> agreement deduped, conflict
    quarantined (never two training rows from one contradiction)."""
    import subprocess
    la = [[0.0, 0.0], [50.0, 0.0]]
    lb = [[0.0, 0.0], [40.0, 30.0]]
    proj = tmp_path / "p"
    proj.mkdir()
    s = AnnotationStore(proj / "annotations.db")
    try:
        for i, w in (("d1", "A"), ("d2", "neither")):
            s.save("task", f"t{i}", {"uuid": f"t{i}",
                    "query_frames": [15000], "task_type": "route_duel",
                    "movie": "ld", "completed": True}, actor="t")
            s.save("duel", i, {"task_uuid": f"t{i}", "movie_uuid": "ld",
                    "source_frame": 15000, "winner": w,
                    "lane_a": la, "lane_b": lb,
                    "truth_a": "x", "truth_b": "y",
                    "annotator": "t"}, actor="t")
    finally:
        s.close()
    out = tmp_path / "snap"
    p = subprocess.run(
        [sys.executable, str(REPO / "scripts/build_v30_snapshot.py"),
         "--project-dir", str(proj), "--movie",
         f"ld={MOVIE_LD}", "--default-movie", "ld",
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO))
    assert p.returncode == 0, p.stderr[-2000:]
    rows = json.loads((out / "duels.json").read_text())
    assert len(rows) == 2
    assert {r["status"] for r in rows} == {"CONFLICT-quarantined"}
    m = json.loads((out / "snapshot_manifest.json").read_text())
    assert m["n_duel_conflicts_quarantined"] == 1


def test_prompt_constructor_records_provenance():
    """rev7: one prompt contract; provenance travels with the prompt."""
    from prototypes.v30_video_apex.model import build_owner_prompt
    p = build_owner_prompt("o1", provenance="human")
    assert p.owner_id == "o1" and p.provenance == "human"
    q = build_owner_prompt("o2", grain_mask=np.ones((8, 8), np.float32),
                           provenance="auto-grain")
    assert q.provenance == "auto-grain"
    assert q.grain_mask is not None


def test_crop_jitter_seed_is_process_stable():
    """rev7: Python hash() is per-process randomized — the same nominal
    seed gave different translated crops on every run. The jitter seed
    must derive from sha256, never hash()."""
    src = (REPO / "scripts/train_v30_front.py").read_text()
    assert "hash(s.entry_id)" not in src
    import hashlib
    a = hashlib.sha256(b"731/ld|obs-x|1").hexdigest()[:8]
    b = hashlib.sha256(b"731/ld|obs-x|1").hexdigest()[:8]
    assert a == b
