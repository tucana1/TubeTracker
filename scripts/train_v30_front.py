"""Train the v30 native ribbon front scorer on banked human labels (rev5 #3).

Supervision (unknown-by-default, see targets.py):
  path_tip samples: apex heat + front interval on a BLINDED oracle route
                    + visibility. The route never ends at the answer.
  tip_only samples: apex heat + visibility (no front target).
  no_tube samples:  visibility only (never a tip target).

Split is by EVENT (tube_ref): no (movie, frame) appears on both sides.
Clips are preloaded once (exact frames, recorded hashes); init/shuffle
are seeded and every update is counted in the manifest. Checkpoints are
faithful torch states (save_checkpoint), immutable.

Dev evaluation is oracle-route localization (heat-peak err + front err)
— predicted-route selection is scored separately in run_v30_movie.py.
"""

from __future__ import annotations

import os

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# rev11: GRAIN_R_NOMINAL is no longer used for training prompts — the
# `auto-grain` arm is the DETECTOR's output (radius included), the same
# construction the runtime uses.
GRAIN_R_NOMINAL = 16.0


def owner_query_valid(sample, resolved_kind: str = "") -> bool:
    """rev12 P0.2: is this an owner-specific loss with a VALID query?

    Owner-specific losses (visibility, body, owner-route) require both
    a resolved owner-conditioned prompt and an explicit identity:
    a real owner UUID, or a grain/tube-scoped query. The string
    "unassigned" is not an identity, and a focus-scoped legacy record
    is not an owner query — its owner-specific gradient is zero.
    """
    if str(resolved_kind or "").strip().lower() == "none":
        return False
    u = str(getattr(sample, "owner_uuid", "") or "").strip().lower()
    if u not in ("", "unassigned"):
        return True
    return str(getattr(sample, "query_kind", "") or "") in ("grain",
                                                            "tube")


def sample_prompt(s, clip_np, ox: int, oy: int, w: int, h: int,
                  prompt_kind, owner_drop: bool):
    """The TRAINING owner prompt — one construction site (rev11 item 4).

    Requested kind -> resolved kind -> typed geometry, through
    `inference.build_typed_prompt`:

    * `auto-grain` renders routes.auto_grain's output on the query frame
      around the root anchor (path start, else focus) — exactly the
      runtime's construction. It is NEVER a disc drawn at the annotation
      focus (the audit measured those centers 45–68 px apart under one
      label).
    * `human` renders the human attachment + proximal path (no disc).
    * only an explicit `none` renders zeros.

    Returns (prompt, provenance_dict, resolved_kind).
    """
    from prototypes.v30_video_apex.model import resolve_prompt_kind
    from prototypes.v30_video_apex.inference import build_typed_prompt
    from prototypes.v30_video_apex import routes as _R
    from prototypes.v30_video_apex.dataset import QUERY_OFFSETS
    import numpy as np

    pk = "none" if owner_drop else str(prompt_kind)
    _anchor = None
    if s.path_xy and len(s.path_xy) >= 2:
        _anchor = (float(s.path_xy[0][0]), float(s.path_xy[0][1]))
    elif s.focus_xy and len(s.focus_xy) == 2:
        _anchor = (float(s.focus_xy[0]), float(s.focus_xy[1]))
    _has_att = bool(s.attachment_xy and len(s.attachment_xy) == 2
                    and s.proximal_xy and len(s.proximal_xy) >= 2)
    _has_grain = _anchor is not None
    if not owner_drop:
        pk = resolve_prompt_kind(pk, _has_att, _has_grain)
    _det_xy = _det_r = None
    if pk == "auto-grain" and _anchor is not None:
        _qoff = list(QUERY_OFFSETS).index(0)
        _qframe = np.asarray(clip_np)[_qoff]
        _gc, _gr, _gsrc = _R.auto_grain(
            _qframe, (float(_anchor[0]) - ox, float(_anchor[1]) - oy))
        # the detector works in crop coords; the typed constructor takes
        # native coordinates and subtracts the origin itself
        _det_xy = (float(_gc[0]) + ox, float(_gc[1]) + oy)
        _det_r = float(_gr)
    prompt, _pprov = build_typed_prompt(
        s.tube_ref or s.obs_uuid, crop_wh=(w, h), crop_origin=(ox, oy),
        detected_grain_xy=(_det_xy if pk == "auto-grain" else None),
        detected_radius_px=_det_r,
        attachment_xy=(s.attachment_xy if pk == "human" else None),
        proximal_xy=(s.proximal_xy if pk == "human" else None))
    return prompt, _pprov, pk


def sha_of_obj(o) -> str:
    return hashlib.sha256(
        json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dev-refs", default="",
                    help="comma-separated tube_ref substrings held out")
    ap.add_argument("--dev-frames", default="",
                    help="comma-separated movie:frame held out "
                    "(frame-level guarantee, e.g. ld:47250)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=288)
    ap.add_argument("--front-halfwidth", type=float, default=2.0)
    ap.add_argument("--init-checkpoint", default="",
                    help="resume/fine-tune from a faithful checkpoint")
    ap.add_argument("--fresh-optimizer", action="store_true",
                    help="weights-only init: do not restore optimizer "
                    "state (required when the architecture grew; the "
                    "run manifest records optimizer_fresh=true)")
    ap.add_argument("--zero-owner-channel", action="store_true",
                    help="zero the encoder's owner-channel input weights "
                    "after init (the source run never sent signal down "
                    "that channel, so zeroing reproduces its behavior "
                    "exactly and lets geometry fade in by learning)")
    ap.add_argument("--path-jitter", type=int, default=0,
                    help="extra translated copies per path_tip sample "
                    "(rev6: 4-6 paths cannot drive the body head alone; "
                    "copies shift ox/oy by +-48px with a seeded RNG and "
                    "pass the same route-in-crop check, so every head's "
                    "targets stay consistent)")
    ap.add_argument("--owner-dropout", type=float, default=0.0,
                    help="fraction of training sights with owner geometry "
                    "blanked (rev6: without it the body head mirrors the "
                    "rendered prompt — proximal IoU 0.42, distal 0.00 — "
                    "instead of seeing the tube)")
    ap.add_argument("--support-jitter", type=float, nargs=2,
                    default=[24.0, 24.0], metavar=("MIN_PX", "MAX_PX"),
                    help="per-sight blinded-route post-tip extension range "
                    "(rev6: fixed 24px lets the front memorize "
                    "support-minus-24; jitter forces image reading)")
    ap.add_argument("--tail-mix", type=float, default=0.0,
                    help="rev11: probability that a sight uses the EXACT "
                    "24px tail even when --support-jitter is a range "
                    "(two-component training distribution: jitter for "
                    "robustness + exact-24 samples to keep the "
                    "diagnostic construction precise). 0.0 = pure jitter, "
                    "the historic behavior")
    ap.add_argument("--route-jitter", type=float, default=0.0,
                    help="online route perturbation ±deg about the root "
                    "(rev5: train on perturbed routes)")
    ap.add_argument("--wrong-routes", type=int, default=1,
                    help="wrong-ribbon items per path sample per epoch "
                    "(route-head negatives; 0 = off)")
    ap.add_argument("--wrong-angle", type=float, nargs=2,
                    default=[60.0, 120.0], metavar=("MIN_DEG", "MAX_DEG"),
                    help="wrong-ribbon rotation range about the root")
    ap.add_argument("--only-refs", default="",
                    help="rev8 plumbing test: keep ONLY train samples whose "
                    "entry contains one of these substrings (comma-"
                    "separated). Used to check whether the heads can fit a "
                    "single real example end-to-end at all.")
    ap.add_argument("--hard-routes", default="",
                    help="JSON from mine_hard_routes.py: real mistaken "
                    "proposals per path sample, trained as route-head "
                    "negatives (rev6: compare actual candidates, not "
                    "just rotations)")
    ap.add_argument("--hard-per-path", type=int, default=2,
                    help="mined hard negatives per path sample per epoch")
    ap.add_argument("--weak-projects", default="",
                    help="comma-separated project substrings whose tips "
                    "train weak (own apex weight + wider sigma)")
    ap.add_argument("--weak-apex-weight", type=float, default=0.3)
    ap.add_argument("--weak-sigma", type=float, default=6.0)
    ap.add_argument("--head-weights", default="",
                    help="comma key=value overrides for LossWeights "
                    "(apex, apex_neg, body, visibility, route, front, "
                    "aux_mask, route_correct, grain). Used to isolate "
                    "one head's learning from multi-task interference: "
                    "the body head fits a single tube perfectly when it "
                    "trains alone (IoU 1.000) but stalls in the joint "
                    "mixture.")
    ap.add_argument("--prompt-mix", default="auto-grain:0.4,human:0.4,none:0.2",
                    help="rev8: prompt kinds sampled per training query, "
                    "as kind:weight (auto-grain matches deployment's "
                    "detection disc; human = attachment+fused root "
                    "geometry; none = zeros)")
    ap.add_argument("--head-lr", type=float, default=None,
                    help="learning rate for the output heads (default: the "
                         "same as the trunk); heads need a much larger step")
    ap.add_argument("--allow-legacy-semantics", action="store_true",
                    help="replay a checkpoint that declares no "
                         "activation/schema")
    ap.add_argument("--body-dice-region", default="extent",
                    choices=["extent", "band"],
                    help="Dice evaluated over the whole reviewed extent, or "
                         "only paint+13.5px band (learn3's scale)")
    ap.add_argument("--body-bg-region", default="strata",
                    choices=["reviewed", "band", "strata"],
                    help="loss negatives: whole reviewed background, only "
                         "the paint-adjacent band (hard negatives), or "
                         "(default) balanced strata — licensed reviewed "
                         "negatives inside AND outside the band, each "
                         "normalized on its own mass (rev12 P0.1)")
    ap.add_argument("--body-dice-balanced", action="store_true",
                    help="class-balanced Dice term (equal total weight per "
                         "class) — needed once reviewed regions are large")
    ap.add_argument("--body-objective", default="",
                    choices=["", "split", "dice_pixel",
                             "dice_selectors"],
                    help="split = per-class mass normalisation (has a "
                         "degenerate flat valley: the uniform direction gets "
                         "zero gradient, so the head oscillates between "
                         "all-foreground and all-background forever); "
                         "dice_pixel = Dice + a small pixel-normalised BCE; "
                         "dice_selectors = rev9 WP-B: Dice + separately "
                         "normalised BCE from self / reviewed-background / "
                         "foreign-exclusive selectors (explicit and "
                         "disjoint; unknown contributes zero)")
    ap.add_argument("--body-pixel-bce", type=float, default=None,
                    help="weight of the pixel-normalised BCE term in the "
                         "dice_pixel objective")
    ap.add_argument("--body-self-bce", type=float, default=None,
                    help="dice_selectors: weight of the self-foreground "
                         "BCE term (own denominator)")
    ap.add_argument("--body-bg-bce", type=float, default=None,
                    help="dice_selectors: weight of the reviewed ordinary "
                         "background BCE term")
    ap.add_argument("--body-foreign-bce", type=float, default=None,
                    help="dice_selectors: weight of the foreign-exclusive "
                         "BCE term (balance-capped at the paint mass)")
    ap.add_argument("--body-dice", type=float, default=None,
                    help="weight of the soft-Dice term in the body loss "
                         "(0 = plain BCE, the setting every run before "
                         "H310 used); Dice cannot be reduced by a "
                         "spatially constant field, which plain BCE can")
    ap.add_argument("--confusable-mode", default="",
                    choices=["", "fixed", "balance"],
                    help="fixed = CONFUSABLE_W per pixel; balance = scale "
                         "the term so confusables carry at most the "
                         "paint's total weight (a fixed 10x collapsed the "
                         "body head on the full dataset)")
    ap.add_argument("--confusable-weight", type=float, default=None,
                    help="override the confusable-negative weight "
                    "(0 disables it; default is the declared 10x)")
    ap.add_argument("--mask-jitter", type=int, default=0,
                    help="rev8: seeded translated copies per body-mask "
                    "sample, so the queried grain is NOT always at the "
                    "crop centre. The swap test holds ONE crop fixed "
                    "while the query moves, so a model that only ever "
                    "saw its query centred would read the framing "
                    "instead of the query.")
    ap.add_argument("--multiscale", action="store_true",
                    help="rev8 step 3: crop-scale encoder (4 levels, "
                    "dilated bottleneck, owner injection per decoder "
                    "level, 3x3x3 temporal). Old checkpoints are NOT "
                    "shape-compatible with it.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save ep{N}.pt every K epochs (0 = final only)")
    a = ap.parse_args()

    out = Path(a.out_dir)
    if out.exists():
        print(f"refusing to overwrite {out}")
        return 1
    out.mkdir(parents=True)

    import numpy as np
    import torch

    from prototypes.v30_video_apex.dataset import (
        ClipSampler, GroupedManifest, ManifestEntry, QUERY_OFFSETS,
        assert_movie_qualified)
    from prototypes.v30_video_apex.batch_builder import (
        body_mask_tensors, finalize_body_supervision, linked_mask_target,
        own_mask_target)
    from prototypes.v30_video_apex.route_pref import pairwise_route_step
    from prototypes.v30_video_apex.targets import (
        confusable_union, paint_overlap_union)
    from prototypes.v30_video_apex.model import (
        OwnerPrompt, build_model)
    from prototypes.v30_video_apex.targets import (
        VISIBILITY_INDEX, apex_heat, apex_pos_neg_masks,
        blind_oracle_route, body_ribbon_mask,
        front_interval_mask, jitter_route, load_clip_pixels,
        resample_polyline, samples_from_snapshot)
    from prototypes.v30_video_apex.train import (
        LossWeights, config_hash, load_checkpoint, save_checkpoint,
        train_step_front, train_step)
    from tubetracker.annotation_frames import FrameReader

    dev_refs = [s for s in a.dev_refs.split(",") if s]
    weak_projects = [s for s in a.weak_projects.split(",") if s]
    dev_frames = set()
    for spec in [s for s in a.dev_frames.split(",") if s]:
        mk, _, fr = spec.partition(":")
        dev_frames.add((mk, int(fr)))
    samples = samples_from_snapshot(a.snapshot)
    regions = json.loads(
        (Path(a.snapshot) / "regions.json").read_text())
    hard_routes = {}
    if a.hard_routes:
        hard_routes = json.loads(Path(a.hard_routes).read_text())
        print(f"hard routes: {sum(len(v) for v in hard_routes.values())} "
              f"proposals across {len(hard_routes)} events")
    body_masks = []
    bmp = Path(a.snapshot) / "body_masks.json"
    if bmp.exists():
        body_masks = json.loads(bmp.read_text())
        print(f"body masks: {len(body_masks)} human masks")
    # rev9 WP-A.6: which masks may serve as FOREIGN negatives — only
    # those with a resolvable identity. A quarantined/unlinked mask
    # could be this very tube, so it contributes zero.
    confusable_eligible = {
        str(_s.mask_uuid) for _s in samples
        if _s.kind == "body_mask" and _s.mask_uuid
        and not _s.quarantine_reason}
    # rev9 WP-A.6: the legacy duel join (movie+frame, winner=1.0 as an
    # ABSOLUTE target) is retired. Comparisons are consumed as their own
    # samples by the pairwise route step; the count stays for the record.
    duel_rows = []
    _legacy_duels: list = []
    dp = Path(a.snapshot) / "duels.json"
    if dp.exists():
        _legacy_duels = json.loads(dp.read_text())
        print(f"duels: {len(_legacy_duels)} human preferences "
              f"(consumed as comparison samples, not by the legacy join)")
    # Frame -> all banked precise tips at (movie, frame), any sample.
    # A verified-negative box overlapping ANY known cap keeps those
    # pixels unknown (rev6 v30d-001 conflict), not just the current
    # sample's own tip.
    frame_tips: dict[tuple, list] = {}
    for s in samples:
        if s.tip_xy and len(s.tip_xy) == 2:
            frame_tips.setdefault(
                (s.movie, int(s.source_frame)),
                []).append((float(s.tip_xy[0]), float(s.tip_xy[1])))
    snap_manifest = json.loads(
        (Path(a.snapshot) / "snapshot_manifest.json").read_text())

    # rev8: ACTUAL unique-label consumption per head. Recorded at the
    # exact sites that generate a loss, so the manifest proves which
    # annotations were really used — presence in a snapshot is not
    # consumption, and this is what makes loss accountable.
    usage: dict[str, set] = {}
    usage_kind: dict[str, str] = {}
    # rev9 WP-A.6: per-label magnitudes (pixels / routes) so the
    # consumption table can say HOW MUCH a label fed a head, not only
    # that it did.
    metric_counts: dict[str, dict] = {}

    def note(head: str, key: str, kind: str = "", **cnt) -> None:
        usage.setdefault(head, set()).add(str(key))
        if kind and str(key) not in usage_kind:
            usage_kind[str(key)] = kind
        if cnt:
            row = metric_counts.setdefault(head, {}).setdefault(str(key), {})
            for k, v in cnt.items():
                if isinstance(v, (int, float)):
                    row[k] = round(float(v), 3)

    # Event split: any overlapping (movie, frame) across sides is a leak.
    train_samples, dev_samples = [], []
    for s in samples:
        if not s.movie_path or not Path(s.movie_path).exists():
            print(f"skip {s.entry_id}: movie missing")
            continue
        if any(d in s.tube_ref or d in s.obs_uuid for d in dev_refs):
            dev_samples.append(s)
        elif (s.movie, s.source_frame) in dev_frames:
            dev_samples.append(s)
        else:
            train_samples.append(s)
    if a.only_refs:
        # plumbing test: restrict the TRAIN side to named samples; the
        # dev side stays whatever dev-refs/dev-frames selected (the
        # run needs both splits, and the probe is the point).
        _keep = [w.strip() for w in a.only_refs.split(",") if w.strip()]
        _before = len(train_samples)
        train_samples = [s for s in train_samples
                         if any(w in s.entry_id or w in (s.owner_key or "")
                                or w in (s.obs_uuid or "") for w in _keep)]
        print(f"only-refs {_keep}: train {len(train_samples)}/{_before}; "
              f"dev {len(dev_samples)}")
    train_frames = {(s.movie, s.source_frame) for s in train_samples}
    dev_frames = {(s.movie, s.source_frame) for s in dev_samples}
    leak = train_frames & dev_frames
    if leak:
        print(f"REFUSE: {len(leak)} (movie,frame) in both splits: {leak}")
        return 1
    if not train_samples or not dev_samples:
        print(f"need both sides nonempty "
              f"(train={len(train_samples)} dev={len(dev_samples)})")
        return 1

    manifest_entries = [
        ManifestEntry(entry_id=s.entry_id, movie_id=s.movie,
                      acquisition_group=s.movie, source_frame=s.source_frame,
                      # rev11: an owner-selection task can carry SEVERAL
                      # absence grains (one ball each) — each is its own
                      # physical case. Keying them by the task ref would
                      # collide six ways; the per-grain region uuid is
                      # the record's identity.
                      owner_id=(s.obs_uuid if s.kind == "no_tube"
                                else (s.tube_ref or s.obs_uuid)),
                      kind=s.kind)
        for s in train_samples + dev_samples]
    assert_movie_qualified(manifest_entries)

    readers: dict[str, FrameReader] = {}

    def reader_for(path: str) -> FrameReader:
        if path not in readers:
            readers[path] = FrameReader(path)
        return readers[path]

    # Frozen sample records: clip frames, crop, route hash, update plan.
    CS = a.crop
    frozen = []
    prepared = []  # (sample, side, clip_np, crop, blind_or_none)
    for s in train_samples + dev_samples:
        # neg_region samples are built inside samples_from_snapshot
        # (they are in neither split list): a region on a dev frame
        # joins dev (probe-only), else train. Without this the leak
        # guard would refuse the run.
        if s.kind == "neg_region":
            side = "dev" if (s.movie, int(s.source_frame)) in dev_frames \
                else "train"
        else:
            side = "dev" if s in dev_samples else "train"
        frames = tuple(s.source_frame + o for o in QUERY_OFFSETS)
        missing = tuple(1 if f < 0 else 0 for f in frames)
        frames = tuple(max(0, f) for f in frames)
        # Owner-centered crop: route centroid, else tip, else task focus.
        # neg_region samples center on their own box (rev6: a region
        # must not depend on sharing a frame with a tip crop).
        if s.kind == "neg_region":
            b = s.region_box
            cx, cy = (float(b[0]) + float(b[2])) / 2.0, \
                (float(b[1]) + float(b[3])) / 2.0
        elif s.kind == "path_tip":
            pts = np.asarray(s.path_xy, float)
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        elif s.tip_xy:
            cx, cy = float(s.tip_xy[0]), float(s.tip_xy[1])
        elif s.focus_xy:
            cx, cy = float(s.focus_xy[0]), float(s.focus_xy[1])
        else:
            print(f"skip {s.entry_id}: no crop anchor")
            continue
        rr = reader_for(s.movie_path)
        H, W = rr.native_size[1], rr.native_size[0]
        ox = int(min(max(cx - CS / 2, 0), max(0, W - CS)))
        oy = int(min(max(cy - CS / 2, 0), max(0, H - CS)))
        # rev6: translated copies of path samples (seeded). Each copy
        # is an independent crop through the same consistent-target
        # machinery below — never a duplicated gradient.
        # rev7: process-stable seed (Python hash() is randomized per
        # process — the same nominal seed gave different crops).
        shifts = [(0, 0)]
        if side == "train" and s.kind == "body_mask" and a.mask_jitter > 0:
            import hashlib as _hl2
            _seed2 = int(_hl2.sha256(
                f"{a.seed}/mask/{s.entry_id}".encode()).hexdigest()[:8], 16)
            mrng = np.random.default_rng(_seed2)
            for _ in range(a.mask_jitter):
                shifts.append((int(mrng.integers(-48, 49)),
                               int(mrng.integers(-48, 49))))
        if side == "train" and s.kind == "path_tip" and a.path_jitter > 0:
            import hashlib as _hl
            _seed = int(_hl.sha256(
                f"{a.seed}/{s.entry_id}".encode()).hexdigest()[:8], 16)
            jrng = np.random.default_rng(_seed)
            for _ in range(a.path_jitter):
                shifts.append((int(jrng.integers(-48, 49)),
                               int(jrng.integers(-48, 49))))
        for dx, dy in shifts:
            oxj = int(min(max(ox + dx, 0), max(0, W - CS)))
            oyh = int(min(max(oy + dy, 0), max(0, H - CS)))
            crop = (oxj, oyh, min(CS, W), min(CS, H))
            if s.kind == "body_mask" and (dx, dy) != (0, 0):
                # a translated mask sight is only useful if the paint is
                # still inside it (an all-zero target wastes the step)
                _r = s.mask_raster or {}
                try:
                    _bx, _by = int(_r["x0"]), int(_r["y0"])
                    _bw, _bh = int(_r["w"]), int(_r["h"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (oxj <= _bx and _bx + _bw <= oxj + CS
                        and oyh <= _by and _by + _bh <= oyh + CS):
                    continue
            blind = None
            if s.kind == "path_tip":
                blind = blind_oracle_route(s.path_xy, s.tip_xy)
                # Route (with blinded extension) must fit the crop.
                r = np.asarray(blind["route_xy"])
                if not (r[:, 0].min() >= oxj and r[:, 0].max() <= oxj + CS and
                        r[:, 1].min() >= oyh and r[:, 1].max() <= oyh + CS):
                    continue
            clip_np = load_clip_pixels(rr, frames, crop, missing)
            frozen.append({
                "entry_id": s.entry_id, "side": side, "kind": s.kind,
                "movie": s.movie, "source_frame": s.source_frame,
                "frames": list(frames), "missing": list(missing),
                "crop_xywh": list(crop), "obs_uuid": s.obs_uuid,
                "obs_revision": s.obs_revision,
                "route_hash": sha_of_obj(blind["route_xy"].tolist())
                if blind else "",
                "s_star": blind["s_star"] if blind else None,
                "clip_sha": hashlib.sha256(
                    clip_np.tobytes()).hexdigest()[:16],
            })
            prepared.append((s, side, clip_np, crop, blind))
    train_prep = [p for p in prepared if p[1] == "train"]
    dev_prep = [p for p in prepared if p[1] == "dev"]
    if not train_prep or not dev_prep:
        print("empty side after crop checks")
        return 1
    n_path_train = sum(1 for s, _, _, _, b in train_prep if b is not None)
    # batch-1 steps: one per sample plus one wrong-ribbon item per path.
    # rev7: this is NOMINAL — hard-route and duel steps are counted
    # actually (history rows carry updates_this_epoch; the manifest
    # carries the true total). Never present nominal as actual.
    updates_per_epoch = len(train_prep) + n_path_train * a.wrong_routes
    total_updates = updates_per_epoch * a.epochs

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = build_model("temporal", base=a.base,
                        multiscale=bool(a.multiscale))
    model.train()
    # rev8 H312: heads need a much larger step than the trunk. Measured
    # in the fast probe: with a frozen trunk the 1x1 body head grows its
    # logit spread 0.65 in 1500 steps at lr 0.5 but only 0.046 at 0.05,
    # and fitting ONE tube needs ~10-20k updates -- a run at the trunk
    # lr (0.003) is ~150x too slow to move the head out of a constant.
    # One definition of the head/trunk split, used by BOTH the optimiser
    # groups and the per-step gradient logging below.
    _head_prefixes = ("body", "heat", "vis_head", "route_head",
                      "front_head", "front_conv", "mode_head", "nobs_head")

    def _group_grad_norms() -> dict:
        """Head/trunk gradient magnitudes from the step that just ran.

        Called immediately after train_step, while grads are still on
        the parameters (they are cleared at the next zero_grad), so this
        reports the gradient the loss actually produced rather than a
        re-derived approximation.
        """
        _h2 = _t2 = 0.0
        for _n, _p in model.named_parameters():
            if _p.grad is None:
                continue
            _g = float(_p.grad.detach().pow(2).sum())
            if _n.split(".")[0] in _head_prefixes:
                _h2 += _g
            else:
                _t2 += _g
        return {"head": _h2 ** 0.5, "trunk": _t2 ** 0.5}

    if a.head_lr is not None and float(a.head_lr) != float(a.lr):
        _hp, _tp = [], []
        for _n, _p in model.named_parameters():
            (_hp if _n.split(".")[0] in _head_prefixes
             else _tp).append(_p)
        opt = torch.optim.AdamW(
            [{"params": _hp, "lr": float(a.head_lr)},
             {"params": _tp, "lr": float(a.lr)}])
        print(f"optimiser: {len(_hp)} head tensors at lr {a.head_lr}, "
              f"{len(_tp)} trunk tensors at lr {a.lr}", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(a.seed)
    resumed_from = None
    optimizer_fresh = False
    if a.init_checkpoint:
        # strict=False across additive head changes; gaps are reported,
        # never silent (only route_head.* may be missing).
        # rev9 WP-A.4: if the checkpoint DECLARES its architecture, it
        # must match the flags we just built the model with — refuse
        # instead of discovering a mismatch mid-run.
        from prototypes.v30_video_apex.model_factory import (
            CheckpointContractError, checkpoint_config,
            optimizer_groups_summary, read_checkpoint_payload)
        try:
            _cfg = checkpoint_config(
                read_checkpoint_payload(a.init_checkpoint))
            _want = {"variant": "temporal", "base": int(a.base),
                     "multiscale": bool(a.multiscale)}
            _mismatch = {k: {"cli": v, "checkpoint": _cfg[k]}
                         for k, v in _want.items()
                         if k in _cfg and _cfg[k] != v}
            if _mismatch:
                print(f"REFUSE: --init-checkpoint declares a different "
                      f"architecture than the CLI: {_mismatch}")
                return 1
            print(f"init-checkpoint declares "
                  f"{ {k: _cfg[k] for k in _want if k in _cfg} }"
                  f" | groups: {optimizer_groups_summary(_cfg)}")
        except CheckpointContractError as e:
            print(f"note: legacy checkpoint ({e}); architecture cannot be "
                  f"verified before loading")
        optimizer_fresh = bool(a.fresh_optimizer)
        resumed_from = load_checkpoint(
            a.init_checkpoint, model,
            None if a.fresh_optimizer else opt, strict=False)
        if a.fresh_optimizer:
            print("weights-only init: fresh optimizer "
                  f"(lr={a.lr})")
        if a.zero_owner_channel:
            # New-information channel starts closed: with zero input
            # weights, rendered geometry contributes exactly 0 and the
            # loaded heads behave as in the source run.
            with torch.no_grad():
                w = model.unet.enc[0].weight
                w[:, 1:2] = 0.0
            print("zeroed encoder owner-channel input weights")
        gaps = (resumed_from.get("missing_keys", []) +
                resumed_from.get("unexpected_keys", []))
        print(f"resumed from {a.init_checkpoint} "
              f"(hash={resumed_from.get('config_hash', '')} gaps={gaps})")
        if not a.fresh_optimizer:
            # rev7: restore the saved epoch RNG — continuing with a
            # re-seeded stream silently reshuffles every epoch order.
            rs = (resumed_from.get("rng_state", {}) or {}).get(
                "epoch_rng")
            if rs is not None:
                try:
                    rng.bit_generator.state = rs
                    print("restored epoch RNG from checkpoint")
                except (ValueError, KeyError, TypeError) as e:
                    print(f"REFUSE: saved RNG state unrestorable ({e})")
                    return 1
            else:
                print("note: checkpoint holds no epoch RNG; "
                      "epoch order re-seeds from --seed")
        bad = [k for k in gaps
               if not (k.startswith("route_head.") or
                       k.startswith("body."))]
        if bad:
            print(f"REFUSE: unexpected checkpoint gaps: {bad}")
            return 1
    if a.body_objective:
        from prototypes.v30_video_apex.train import (
            set_body_bg_region, set_body_dice_balanced, set_body_dice_region,
            set_body_objective)
        set_body_objective(a.body_objective)
        set_body_dice_balanced(bool(a.body_dice_balanced))
        set_body_bg_region(a.body_bg_region)
        set_body_dice_region(a.body_dice_region)
        print(f"body negatives -> {a.body_bg_region}", flush=True)
        print(f"body objective -> {a.body_objective}", flush=True)
    if a.body_pixel_bce is not None:
        from prototypes.v30_video_apex.train import set_body_pixel_bce_weight
        set_body_pixel_bce_weight(a.body_pixel_bce)
        print(f"body pixel-bce weight -> {a.body_pixel_bce}", flush=True)
    if a.body_self_bce is not None:
        from prototypes.v30_video_apex.train import set_body_self_bce_weight
        set_body_self_bce_weight(a.body_self_bce)
        print(f"body self-bce weight -> {a.body_self_bce}", flush=True)
    if a.body_bg_bce is not None:
        from prototypes.v30_video_apex.train import set_body_bg_bce_weight
        set_body_bg_bce_weight(a.body_bg_bce)
        print(f"body bg-bce weight -> {a.body_bg_bce}", flush=True)
    if a.body_foreign_bce is not None:
        from prototypes.v30_video_apex.train import (
            set_body_foreign_bce_weight)
        set_body_foreign_bce_weight(a.body_foreign_bce)
        print(f"body foreign-bce weight -> {a.body_foreign_bce}",
              flush=True)
    if a.body_dice is not None:
        from prototypes.v30_video_apex.train import set_body_dice_weight
        set_body_dice_weight(a.body_dice)
        print(f"body dice weight -> {a.body_dice}", flush=True)
    elif str(a.body_objective) in ("dice_selectors", "dice_selectors2"):
        # rev10 WP-A: Dice is the DEFINING term of these modes, but the
        # global default is 0.0 and the selector branch now honours the
        # weight (it used to ignore it entirely). Leaving the default
        # would silently drop the mode's defining term, so state it
        # explicitly and log it — "keep any weights explicit, resolved
        # and logged".
        from prototypes.v30_video_apex.train import set_body_dice_weight
        set_body_dice_weight(1.0)
        print("body dice weight -> 1.0 (selector-mode default, explicit)",
              flush=True)
    if a.confusable_mode:
        from prototypes.v30_video_apex.train import set_confusable_mode
        set_confusable_mode(a.confusable_mode)
        print(f"confusable mode -> {a.confusable_mode}", flush=True)
    if a.confusable_weight is not None:
        from prototypes.v30_video_apex.train import (
            set_confusable_mode, set_confusable_weight)
        set_confusable_weight(a.confusable_weight)
        print(f"confusable weight -> {a.confusable_weight}")
    weights = LossWeights(apex=1.0, front=1.0, visibility=0.3)
    if a.head_weights:
        import dataclasses as _dc
        _kw = {}
        for _part in str(a.head_weights).split(","):
            _part = _part.strip()
            if not _part:
                continue
            _k, _, _v = _part.partition("=")
            _k = _k.strip()
            if not hasattr(weights, _k):
                raise SystemExit(f"unknown head weight {_k!r}")
            _kw[_k] = float(_v)
        weights = _dc.replace(weights, **_kw)
        print(f"head weights: {_kw}")
    # rev6: the EFFECTIVE lr is whatever the optimizer holds after a
    # resume (front_pert declared 0.001 but ran 0.003). Record it and
    # use it as the run's declared setting — never the CLI default.
    effective_lr = float(opt.param_groups[0]["lr"])
    if abs(effective_lr - a.lr) > 1e-12:
        print(f"note: effective optimizer lr={effective_lr} "
              f"differs from --lr {a.lr}; recording the effective value")
    # rev9 WP-A.4: record EVERY effective knob the audit found missing
    # (model, loss, prompt, sampling) plus NAMED optimizer groups, so a
    # consumer can rebuild the exact run from the checkpoint alone.
    def _num_or_none(x):
        """Optional numeric flags (None means 'the declared default')."""
        return float(x) if x is not None else None

    _group_lrs = {}
    for _i, _g in enumerate(opt.param_groups):
        _name = "head" if (len(opt.param_groups) > 1 and _i == 0) else (
            "trunk" if len(opt.param_groups) > 1 else "all")
        _group_lrs[_name] = float(_g["lr"])
    run_config = {"variant": "temporal", "base": a.base,
                  "lr": effective_lr, "epochs": a.epochs, "seed": a.seed,
                  "crop": CS, "front_halfwidth": a.front_halfwidth,
                  "route_jitter": a.route_jitter,
                  "path_jitter": a.path_jitter,
                  "owner_dropout": a.owner_dropout,
                  "support_jitter_px": list(a.support_jitter),
                  "tail_mix": float(a.tail_mix),
                  "wrong_routes": a.wrong_routes,
                  "hard_routes": a.hard_routes,
                  "hard_per_path": a.hard_per_path,
                  "n_legacy_duels": int(len(_legacy_duels)),
                  "weak_projects": sorted(weak_projects),
                  # rev9 WP-A.4 (audit: these ten were omitted)
                  "multiscale": bool(a.multiscale),
                  "mask_jitter": float(a.mask_jitter or 0.0),
                  "body_objective": str(a.body_objective),
                  # H345: the two knobs I added in this stretch were not
                  # recorded (checkpoint read back None/None) — the same
                  # class of omission WP-A.4 fixed for ten others.
                  "body_dice_balanced": bool(a.body_dice_balanced),
                  "body_bg_region": str(a.body_bg_region),
                  "trunk_lr": _num_or_none(a.lr),
                  "body_dice_region": str(a.body_dice_region),
                  "body_pixel_bce": _num_or_none(a.body_pixel_bce),
                  "body_self_bce": _num_or_none(a.body_self_bce),
                  "body_bg_bce": _num_or_none(a.body_bg_bce),
                  "body_foreign_bce": _num_or_none(a.body_foreign_bce),
                  "body_dice": _num_or_none(a.body_dice),
                  "head_lr": _num_or_none(a.head_lr),
                  "prompt_mix": str(a.prompt_mix),
                  "only_refs": str(a.only_refs),
                  "confusable_weight": _num_or_none(a.confusable_weight),
                  "confusable_mode": str(a.confusable_mode),
                  "head_weights": str(a.head_weights or ""),
                  "optimizer_group_lrs": _group_lrs,
                  "weak_sigma": float(a.weak_sigma),
                  "weak_apex_weight": float(a.weak_apex_weight),
                  # dataset / split identity (what was actually trained)
                  "snapshot_content_sha": config_hash(
                      dict(snap_manifest.get("content_sha256", {}))),
                  "train_refs": sorted({
                      (s.obs_uuid or s.entry_id) for s in train_samples}),
                  "n_train_samples": len(train_samples),
                  "dev_refs": str(a.dev_refs),
                  "dev_frames": str(a.dev_frames)}

    def to_clip(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(2)  # B,T,C,H,W

    # rev8 diagnostics: how often a body-mask sight actually carries
    # confusable negatives (should be every sight in a multi-tube crop)
    conf_stats = {"body_sights": 0, "with_confusable": 0,
                  "confusable_px": 0, "valid_extended": 0}

    def sample_tensors(s, clip_np, crop, blind, owner_drop=False,
                       prompt_kind="human"):
        import dataclasses

        clip = to_clip(clip_np)
        ox, oy = crop[0], crop[1]
        # Other known caps at this frame (crop coords), excluding the
        # sample's own tip: disputed box pixels stay unknown.
        others = [(x - ox, y - oy)
                  for (x, y) in frame_tips.get(
                      (s.movie, int(s.source_frame)), [])
                  if not (s.tip_xy and abs(x - s.tip_xy[0]) < 1e-6
                          and abs(y - s.tip_xy[1]) < 1e-6)]
        h, w = clip_np.shape[1], clip_np.shape[2]
        weak = any(p in (s.provenance or "") for p in weak_projects)
        sigma = a.weak_sigma if (weak and s.tip_xy) else 4.0
        step_weights = dataclasses.replace(
            weights, apex=(a.weak_apex_weight if weak else 1.0))
        tip_crop = None
        if s.tip_xy:
            tip_crop = (s.tip_xy[0] - ox, s.tip_xy[1] - oy)
            heat_t = torch.from_numpy(
                apex_heat(h, w, tip_crop,
                          sigma_px=sigma)).unsqueeze(0).unsqueeze(0)
        else:
            heat_t = torch.zeros(1, 1, h, w)
        regs = [r for r in regions
                if r.get("movie_uuid", "") in (s.movie,)
                and int(r.get("source_frame", -2)) == s.source_frame]
        pos_np, neg_np = apex_pos_neg_masks(h, w, (ox, oy), tip_crop,
                                            regs, other_tips_crop=others)
        # rev6/rev7/rev11: the ONE typed construction site for training
        # prompts (module-level `sample_prompt`; the runtime and the
        # evaluator route through the same constructor).
        prompt, _pprov, pk = sample_prompt(
            s, clip_np, ox, oy, w, h, prompt_kind, owner_drop)
        # rev9 WP-A.6 / rev12 P0.2: visibility is supervised ONLY where
        # the label is about a known owner's state AND the resolved
        # owner query is VALID — a resolved `none` prompt zeroes it (it
        # used to keep gradient: 30 no-query absence updates in the
        # saved schedule), and "unassigned" is not an identity. A
        # grain- or tube-scoped query counts as an explicit identity;
        # a focus-scoped legacy record does not.
        vis_valid = (torch.zeros(1) if (
            s.kind in ("neg_region", "body_mask", "census_tile",
                       "comparison")
            or not owner_query_valid(s, pk))
            else torch.ones(1))
        masks = {
            "apex_pos": torch.from_numpy(pos_np
                                         ).unsqueeze(0).unsqueeze(0),
            "apex_neg": torch.from_numpy(neg_np
                                         ).unsqueeze(0).unsqueeze(0),
            "vis_valid": vis_valid,
        }
        vis = torch.tensor(
            [VISIBILITY_INDEX.get(s.direct_state, 2)])
        # rev8: other LABELLED tubes in this frame, in crop coords.
        # These are the confusable negatives the body loss needs: with
        # plain background averaging a wrong-instance activation is ~1%
        # of the negative term, so predicting every tube in the crop is
        # nearly free (~0.003 vs ~0.000) and training collapses to a
        # query-invariant answer.
        confusable_t = None
        _overlap_np = None
        if s.kind == "body_mask" and body_masks:
            # rev9 WP-A.6 joins: canonical movie + same frame + NOT the
            # same physical owner (other annotations/revisions of this
            # very tube are not foreign), and only masks with a
            # resolvable identity (quarantined ones contribute zero).
            _un = confusable_union(
                h, w, (ox, oy), movie=s.movie, frame=s.source_frame,
                self_uuid=s.mask_uuid, self_owner_key=s.owner_key,
                masks=body_masks, eligible=confusable_eligible)
            # rev10 WP-A: pixels painted by >=2 distinct owners are
            # ambiguous at a crossing, not foreign negatives.
            _overlap_np = paint_overlap_union(
                h, w, (ox, oy), movie=s.movie, frame=s.source_frame,
                masks=body_masks, eligible=confusable_eligible)
            conf_stats["body_sights"] += 1
            if _un.any():
                conf_stats["with_confusable"] += 1
                conf_stats["confusable_px"] += int(_un.sum())
                confusable_t = torch.from_numpy(
                    _un.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        # rev6: visible-body target for path samples (visible spans
        # only); everything else gets zeros with zero validity.
        # A HUMAN mask replaces the centerline ribbon where it matches
        # this tube (median paint-to-path distance < 30px): the hand,
        # not the old geometry, is the evidence. Match recorded below.
        body_t = torch.zeros(1, 1, h, w)
        body_v = torch.zeros(1, 1, h, w)
        body_source = "none"
        if s.kind == "path_tip" and len(s.path_xy) >= 2:
            pcrop = [(q[0] - ox, q[1] - oy) for q in s.path_xy]
            used_mask = None
            mask_link = "none"
            if body_masks and s.path_xy:
                path_a = np.asarray(s.path_xy, float)
                best, best_m = 1e18, None
                best_link = "none"
                for bm in body_masks:
                    if str(bm.get("movie", "")) != s.movie or \
                            int(bm.get("source_frame", -2)) != \
                            int(s.source_frame):
                        continue
                    # rev8: a mask explicitly linked to ANOTHER
                    # observation is that observation's evidence — the
                    # proximity fallback must never hand it over.
                    bm_obs = str(bm.get("source_obs_uuid", "") or "")
                    bm_owner = str(bm.get("owner_key", "") or "")
                    if bm_obs and bm_obs != str(s.obs_uuid):
                        continue
                    if not bm_obs and bm_owner and s.owner_key and \
                            bm_owner != s.owner_key:
                        continue
                    linked = bool(bm_obs) and bm_obs == str(s.obs_uuid)
                    pts = np.asarray(bm.get("painted_xy", []), float)
                    if pts.shape[0] == 0:
                        # raster-only mask: anchor distance on the
                        # raster bbox centre instead of skipping it
                        ras = bm.get("mask_raster") or {}
                        if not ras:
                            continue
                        bc = np.asarray([[float(ras.get("x0", 0)) +
                                          float(ras.get("w", 0)) / 2.0,
                                          float(ras.get("y0", 0)) +
                                          float(ras.get("h", 0)) / 2.0]])
                        d = np.hypot(
                            (bc[:, None, :] - path_a[None, :, :])[..., 0],
                            (bc[:, None, :] - path_a[None, :, :])[..., 1]
                        ).min(axis=1)
                        med = float(d[0])
                    else:
                        # Euclidean nearest-path-vertex per paint point
                        # (rev7: the old transpose/min took per-
                        # COORDINATE minima, letting far paint score
                        # median ~0).
                        _dd = pts[:, None, :] - path_a[None, :, :]
                        d = np.hypot(_dd[..., 0], _dd[..., 1]).min(axis=1)
                        med = float(np.median(d))
                    if linked:
                        best, best_m, best_link = med, bm, "explicit"
                        break
                    if med < best:
                        best, best_m, best_link = med, bm, "proximity"
                # rev8: proximity alone may only attach a mask that has
                # no competing owner claim; the link is recorded.
                if best_m is not None and (best_link == "explicit"
                                           or best < 30.0):
                    used_mask = best_m
                    mask_link = best_link
            _bt = None  # rev10: set only on the body-mask branch below;
            # the band/bg channels are read from it unconditionally, so a
            # ribbon-path sample must find None here, not raise.
            if used_mask is not None:
                # rev9 WP-A.3: one shared builder for the trainer and the
                # dev probe (the source id keeps its historical spelling).
                _bt = linked_mask_target(h, w, (ox, oy), used_mask,
                                         link=mask_link)
                # rev12 P0.1: ALWAYS finalize — the explicit selectors
                # are the authoritative loss domains, and a sample that
                # skipped this step would silently train the legacy
                # (unlicensed) path instead.
                _bt = finalize_body_supervision(
                    _bt,
                    confusable=(confusable_t.numpy()[0, 0]
                                if confusable_t is not None else None),
                    overlap=_overlap_np)
                bt, bv = _bt.target, _bt.valid
                body_source = _bt.source
                note("body_valid", body_source, "body_mask",
                     paint_px=float((bt > 0).sum()),
                     valid_px=float((bv > 0).sum()))
            else:
                bt, bv = body_ribbon_mask(h, w, (ox, oy), pcrop,
                                          list(s.path_visible or []))
                body_source = "ribbon"
            body_t = torch.from_numpy(bt).unsqueeze(0).unsqueeze(0)
            body_v = torch.from_numpy(bv).unsqueeze(0).unsqueeze(0)
            if _bt is not None:
                # rev11: same shared channel builder as the body-mask
                # branch and the fit helper.
                _ch = body_mask_tensors(
                    _bt,
                    confusable=(confusable_t.numpy()[0, 0]
                                if confusable_t is not None else None),
                    overlap=_overlap_np)
                for _ck, _cv in _ch.items():
                    masks[_ck] = torch.from_numpy(
                        np.asarray(_cv, dtype=np.float32)
                    ).unsqueeze(0).unsqueeze(0)
        elif s.kind == "body_mask" and not s.quarantine_reason:
            # rev8: a mask is its OWN sample — the body head learns on
            # the mask's own crop, with its reviewed extent.
            # rev9 WP-A.3: built by the shared builder, which the dev
            # probe also calls, so the two cannot drift apart.
            _bt = own_mask_target(s, h, w, (ox, oy))
            body_source = _bt.source
            # rev9/rev11/rev12: ONE shared step (batch_builder
            # .finalize_body_supervision), now UNCONDITIONAL so every
            # sample trains the licensed selectors. The rev11 audit: the
            # trainer extended validity over verified foreign paint while
            # the fit helper did not — the two consumers fed DIFFERENT
            # supervision to the same objective (g1: 623 of 683 known
            # foreign pixels discarded by the helper). Verified
            # foreign-EXCLUSIVE paint becomes supervised negative
            # material; ambiguous multi-owner overlap stays
            # UNSUPERVISED; self-positive paint stays positive.
            _bt = finalize_body_supervision(
                _bt,
                confusable=(confusable_t.numpy()[0, 0]
                            if confusable_t is not None else None),
                overlap=_overlap_np)
            if confusable_t is not None:
                conf_stats["valid_extended"] += 1
            # rev11: the channel tensors come from the SAME builder the
            # fit helper uses — same sample, same crop, same pixels.
            _ch = body_mask_tensors(
                _bt,
                confusable=(confusable_t.numpy()[0, 0]
                            if confusable_t is not None else None),
                overlap=_overlap_np)
            for _ck, _cv in _ch.items():
                masks[_ck] = torch.from_numpy(
                    np.asarray(_cv, dtype=np.float32)
                ).unsqueeze(0).unsqueeze(0)
            bt, bv = _bt.target, _bt.valid
            body_t, body_v = _bt.tensors()
            note("body_valid", body_source, "body_mask",
                 paint_px=float((bt > 0).sum()),
                 valid_px=float((bv > 0).sum()),
                 reviewed_bg_px=float(_bt.bg_reviewed.sum()))
            if os.environ.get("TUBETRACKER_DEBUG_BODY") and \
                    s.kind == "body_mask":
                print(f"DEBUG mask body target: {s.entry_id} "
                      f"frame={s.source_frame} crop={crop} "
                      f"src={body_source} "
                      f"paint_px={int((bt > 0).sum())} "
                      f"valid_px={int((bv > 0).sum())} "
                      f"complete={s.complete}", flush=True)
        elif s.kind == "census_tile":
            # rev8: census tiles supervise discovery — declared tips
            # are positives, checked background (complete tiles only)
            # is negative, known caps stay unknown.
            from prototypes.v30_video_apex.targets import \
                census_pos_neg_masks
            from prototypes.v30_video_apex.targets import \
                _region_in_crop as _ric
            tb = s.region_box
            x0n, y0n = float(tb[0]), float(tb[1])
            others_c = [(q[0] - ox, q[1] - oy) for q in others] if others \
                else []
            pos_np, neg_np = census_pos_neg_masks(
                h, w, (ox, oy), s.census_tips, s.census_grains,
                (x0n, y0n, float(tb[2]), float(tb[3])),
                bool(s.census_complete), other_tips_crop=others_c,
                class_scopes=s.census_class_scopes)
            masks["apex_pos"] = torch.from_numpy(
                pos_np).unsqueeze(0).unsqueeze(0)
            masks["apex_neg"] = torch.from_numpy(
                neg_np).unsqueeze(0).unsqueeze(0)
            masks["vis_valid"] = torch.zeros(1)
            bt = np.zeros((h, w), np.float32)
            bv = np.zeros((h, w), np.float32)  # no body claim here
        masks["body_valid"] = body_v
        if confusable_t is not None:
            masks["body_confusable"] = confusable_t
        return clip, prompt, heat_t, masks, vis, step_weights, body_t, \
            body_source

    actual_use: dict[str, dict] = {}
    actual_mass: dict[str, dict] = {}
    actual_zeroed: dict[str, dict] = {}
    _HEAD_W = (("vis_valid", "visibility"), ("route_valid",
                                             "route_correct"),
               ("front_valid", "front"), ("front_present_valid",
                                          "front_present"),
               ("body_valid", "body"))

    def _consumed_from(masks: dict, w) -> dict:
        """Consumed() semantics for call sites without a loss boundary
        (dev probe): final masks + effective weights, nothing overwrites
        them afterwards."""
        _out2: dict = {}
        for _mk, _wk in _HEAD_W:
            _m = (masks or {}).get(_mk)
            if _m is None:
                continue
            try:
                _mass = float(torch.as_tensor(_m).detach().sum())
            except (TypeError, ValueError, RuntimeError):
                continue
            _wt = float(getattr(w, _wk, 0.0))
            _out2[_mk] = {"mass": _mass, "weight": _wt,
                          "consumed": bool(_mass > 0 and _wt > 0)}
        return _out2

    def _account_consumed(cons, s, side: str = "train") -> None:
        """rev13 W1: contribution accounting at the LOSS BOUNDARY.
        The train side passes r['consumed'] — the FINAL masks the loss
        actually saw (the audit showed pre-step counters were bypassed
        by mask overwrites). Keys keep head | owner | grain | annotation
        and invalid heads are counted with their reason."""
        _base = "|".join([
            str(getattr(s, "owner_uuid", "") or getattr(s, "owner_key", "")),
            str(getattr(s, "grain_id", "") or ""),
            str(getattr(s, "obs_uuid", "") or getattr(s, "mask_uuid", "")
                or "")])
        for _mk, info in (cons or {}).items():
            _ac = actual_use.setdefault(_mk, {"train": {}, "dev": {}})
            _am = actual_mass.setdefault(_mk, {"train": 0.0, "dev": 0.0})
            if info["consumed"]:
                _ac[side][_base] = _ac[side].get(_base, 0) + 1
                _am[side] += float(info["mass"])
            else:
                _why = "zero-mass" if info["mass"] <= 0 else "zero-weight"
                _rz = actual_zeroed.setdefault(
                    _mk, {"train": {}, "dev": {}})
                _rz[side][_why] = _rz[side].get(_why, 0) + 1
    body_schedule = {"opportunities": 0, "zeroed_no_query": 0,
                     "nonzero": 0}
    history = []
    n_updates = 0
    best_dev_front: float | None = None
    best_body: tuple[float, int] | None = None
    best_front: tuple[float, int] | None = None

    def _param_hash() -> str:
        import hashlib
        h = hashlib.sha256()
        for _k, _v in sorted(model.state_dict().items()):
            h.update(_k.encode())
            h.update(_v.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    def _save_best(kind: str, value: float, domain: str, epoch: int) -> None:
        """rev10 WP-A: the manifest selected on a FRONT metric even when
        the front head was untrained. Each head now keeps its own best,
        labelled with the metric's domain and the parameter hash."""
        try:
            save_checkpoint(
                out / f"best_{kind}_ep{epoch}.pt", model, opt,
                config=dict(run_config, epoch=epoch, best_of=kind),
                manifest={"epoch": epoch, "best_of": kind,
                          "metric_domain": domain, "metric_value": value,
                          "param_hash": _param_hash()},
                rng_state={"epoch_rng": rng.bit_generator.state})
        except FileExistsError:
            pass
        (out / f"best_{kind}.json").write_text(json.dumps(
            {"kind": kind, "epoch": epoch, "metric_domain": domain,
             "metric_value": value, "param_hash": _param_hash(),
             "checkpoint": f"best_{kind}_ep{epoch}.pt"}, indent=1))
    best_epoch = -1
    t0 = time.time()
    # rev8 step 3: parse the declared prompt mix once, and count what
    # was actually drawn (reported in label_usage.json).
    prompt_mix = []
    for _part in str(a.prompt_mix).split(","):
        _part = _part.strip()
        if not _part:
            continue
        _k, _, _w = _part.partition(":")
        _k = _k.strip()
        if _k not in ("auto-grain", "human", "none"):
            raise SystemExit(f"bad --prompt-mix kind: {_k!r}")
        prompt_mix.append((_k, float(_w) if _w else 1.0))
    if not prompt_mix:
        prompt_mix = [("human", 1.0)]
    prompt_mix = [(k, w / sum(w for _, w in prompt_mix))
                  for k, w in prompt_mix]
    prompt_counts: dict[str, int] = {k: 0 for k, _ in prompt_mix}

    for ep in range(a.epochs):
        order = rng.permutation(len(train_prep))
        ep_loss, ep_front = 0.0, []
        ep_body = 0.0
        ep_updates0 = n_updates
        ep_grads = []
        ep_bparts: dict[str, float] = {}
        from collections import Counter as _Counter
        ep_bsrc = _Counter()
        for j in order:
            idx = int(j)
            s, _, clip_np, crop, blind = train_prep[idx]
            if s.kind == "comparison":
                # rev9 WP-A.6: comparisons are consumed ONLY by the
                # pairwise route pass below — never by the generic
                # branch, which would quietly give them apex/body/vis
                # supervision the label never carried (the consumption
                # table caught exactly that: comparison -> apex_neg).
                continue
            drop = False
            if a.owner_dropout > 0 and s.kind == "path_tip":
                drng = np.random.default_rng([a.seed + 31337, ep, idx])
                drop = bool(drng.random() < a.owner_dropout)
            # rev8 step 3: prompt kind drawn per query from the
            # declared mix (seeded), so every head sees the prompt
            # distribution deployment will actually provide.
            prng = np.random.default_rng([a.seed + 6060, ep, idx])
            _u = float(prng.random())
            _acc = 0.0
            pk = prompt_mix[-1][0]
            for _k, _wt in prompt_mix:
                _acc += _wt
                if _u < _acc:
                    pk = _k
                    break
            prompt_counts[pk] += 1
            clip, prompt, heat_t, masks, vis, step_w, body_t, _bsrc = \
                sample_tensors(
                s, clip_np, crop, blind, owner_drop=drop, prompt_kind=pk)
            _resolved_pk = str(getattr(prompt, "provenance", pk))
            if body_t is not None:
                body_schedule["opportunities"] += 1
            if (not owner_query_valid(s, _resolved_pk)) \
                    and body_t is not None:
                # rev10 WP-A / rev11 / rev12 P0.2: a no-owner-prompt
                # update — or an owner-specific loss whose resolved
                # query is NOT valid (focus-scoped legacy record,
                # "unassigned" owner) — must carry ZERO owner-specific
                # loss. The condition is the RESOLVED prompt
                # (construction may remap a requested kind; the
                # requested type alone is insufficient) plus the
                # identity check. Counted as its own category.
                body_t = torch.zeros_like(body_t)
                masks["body_valid"] = torch.zeros_like(
                    masks.get("body_valid", body_t))
                for _k2 in ("body_band", "body_bg_reviewed",
                            "body_confusable", "body_overlap",
                            "body_sel_self", "body_sel_bg",
                            "body_sel_foreign", "body_sel_unknown"):
                    masks.pop(_k2, None)
                _zero_reason = ("no-owner-prompt"
                                if _resolved_pk == "none"
                                else "invalid-owner-query")
                _bsrc = f"{_zero_reason}:{_bsrc}"
                conf_stats["no_prompt_body_zeroed"] = (
                    conf_stats.get("no_prompt_body_zeroed", 0) + 1)
                if _zero_reason == "invalid-owner-query":
                    conf_stats["invalid_owner_body_zeroed"] = (
                        conf_stats.get("invalid_owner_body_zeroed", 0)
                        + 1)
                body_schedule["zeroed_no_query"] += 1
            # rev12 P0.2: visibility gating is counted by REASON, so the
            # no-query / invalid-owner zeroing is auditable per run.
            if float(masks.get("vis_valid",
                               torch.ones(1)).sum()) == 0:
                _vwhy = ("no-owner-prompt" if _resolved_pk == "none"
                         else "invalid-owner-query"
                         if not owner_query_valid(s, _resolved_pk)
                         else "kind-excluded")
                conf_stats["vis_valid_zeroed_" + _vwhy] = (
                    conf_stats.get("vis_valid_zeroed_" + _vwhy, 0) + 1)
            # rev13 W1: owner-route and owner-specific front targets are
            # gated on the SAME resolved-query validity as visibility and
            # body — an invalid/no query carries no owner-specific
            # claim, and the finalized masks travel as one set.
            _oq = owner_query_valid(s, _resolved_pk)
            masks["front_valid"] = (torch.ones(1) if _oq
                                    else torch.zeros(1))
            masks["route_valid"] = (torch.ones(1) if _oq
                                    else torch.zeros(1))
            ep_bsrc[str(_bsrc).split(":")[0]] += 1
            if blind is not None:
                # rev6: per-sight support variation (seeded). A fixed
                # 24px extension lets the front memorize support-24.
                if a.support_jitter[0] != a.support_jitter[1]:
                    srng = np.random.default_rng([a.seed + 7171, ep, idx])
                    if a.tail_mix > 0.0 and srng.random() < a.tail_mix:
                        # rev11 two-component distribution: exact-24
                        # samples keep the diagnostic construction
                        # precise while the jitter range forces image
                        # reading. One declared flag; default 0.0.
                        post = 24.0
                    else:
                        post = float(srng.uniform(a.support_jitter[0],
                                                  a.support_jitter[1]))
                    try:
                        bj = blind_oracle_route(s.path_xy, s.tip_xy,
                                                post_px=post)
                    except ValueError:
                        bj = None
                    if bj is not None:
                        rj = np.asarray(bj["route_xy"])
                        if rj[:, 0].min() >= crop[0] and \
                                rj[:, 0].max() <= crop[0] + crop[2] and \
                                rj[:, 1].min() >= crop[1] and \
                                rj[:, 1].max() <= crop[1] + crop[3]:
                            blind = bj
                route_nat = np.asarray(blind["route_xy"])
                s_star = float(blind["s_star"])
                if a.route_jitter > 0 and s.tip_xy:
                    jrng = np.random.default_rng(
                        [a.seed, ep, idx])
                    rj, sj = jitter_route(route_nat, s.tip_xy,
                                          a.route_jitter, jrng)
                    if sj is not None:
                        route_nat, s_star = rj, float(sj)
                route_crop = (route_nat -
                              np.array([crop[0], crop[1]])).tolist()
                with torch.no_grad():
                    probe = model.forward(
                        clip, prompt, route_xy=route_crop)  # type: ignore[call-arg]
                interval = torch.from_numpy(front_interval_mask(
                    probe.front_s.detach().numpy(), s_star,
                    a.front_halfwidth)).to(torch.float32)
                r = train_step_front(
                    model, opt, clip, prompt, route_crop,
                    heat_t, masks, interval, torch.ones(1),
                    vis, torch.ones(1), step_w,
                    route_t=torch.ones(1), body_t=body_t)
                # rev13 W1: count what the loss consumed, at the loss
                # boundary (final masks, effective weights, reasons).
                _account_consumed(r.get("consumed"), s)
                if (r.get("consumed") or {}).get(
                        "body_valid", {}).get("consumed"):
                    body_schedule["nonzero"] += 1
                ep_front.append(r["front_err_px"])
                if a.wrong_routes > 0 and s.tip_xy:
                    # Wrong-ribbon negative: large rotation about the root.
                    # Verified wrong (tip far from support) so the route
                    # head learns true-vs-wrong ribbons. No apex/front
                    # claim travels on a wrong route (validity zero).
                    wrng = np.random.default_rng(
                        [a.seed + 999, ep, idx])
                    for _ in range(a.wrong_routes):
                        ang = float(wrng.uniform(*a.wrong_angle)) * \
                            wrng.choice([-1.0, 1.0])
                        c, sn = np.cos(ang * np.pi / 180.0), np.sin(
                            ang * np.pi / 180.0)
                        rot = np.array([[c, -sn], [sn, c]])
                        rnat = np.asarray(blind["route_xy"])
                        rw = (rnat - rnat[0]) @ rot.T + rnat[0]
                        tip = np.asarray(s.tip_xy, float)
                        seg = rw[1:] - rw[:-1]
                        sl = np.hypot(seg[:, 0], seg[:, 1]).clip(
                            min=1e-9)
                        w = (((tip - rw[:-1]) * seg).sum(axis=1) /
                             (sl ** 2))
                        proj = rw[:-1] + np.clip(w, 0, 1)[:, None] * seg
                        dmin = float(np.hypot(
                            *(proj - tip).T).min())
                        if dmin < 15.0:
                            rw = (rnat - rnat[0]) @ (-np.eye(2)) + \
                                rnat[0]  # 180 deg fallback
                        rw_crop = (rw -
                                   np.array([crop[0], crop[1]])).tolist()
                        zero_heat = torch.zeros_like(heat_t)
                        from prototypes.v30_video_apex.batch_builder import negative_route_masks
                        from prototypes.v30_video_apex.route_supervision import wrong_route_certificate
                        route_certificate = wrong_route_certificate(
                            s, rw, current_s=s_star)
                        zero_masks = negative_route_masks(
                            masks, query_valid=owner_query_valid(s, _resolved_pk),
                            certified=route_certificate["certified_wrong_route"])
                        with torch.no_grad():
                            wprobe = model.forward(
                                clip, prompt,
                                route_xy=rw_crop)  # type: ignore[call-arg]
                            n_s = int(
                                wprobe.front_logits.shape[1])
                        dummy = torch.zeros(n_s)
                        # rev12 P0.2: the owner-route negative carries
                        # gradient only for a valid owner query.
                        _rv = (torch.ones(1) if owner_query_valid(
                            s, _resolved_pk) else torch.zeros(1))
                        rw2 = train_step_front(
                            model, opt, clip, prompt, rw_crop,
                            zero_heat, zero_masks, dummy,
                            torch.zeros(1), vis, torch.ones(1), step_w,
                            route_t=torch.zeros(1), route_valid=_rv)
                        ep_loss += float(rw2["total"])
                        _account_consumed(rw2.get("consumed"), s)
                        n_updates += 1
                        note("route", s.sample_key + "|rotated",
                             "route-rotation", routes=1)
                if blind is not None and hard_routes and s.tip_xy:
                    # Mined real mistakes for THIS owner (seeded subset
                    # per epoch): tip-near but body-poor proposals are
                    # route-head negatives with zero apex/front/vis
                    # validity — the comparison rev6 asks for, against
                    # actual alternatives rather than rotations.
                    hrng = np.random.default_rng(
                        [a.seed + 4242, ep, idx])
                    cands = [d for d in hard_routes.get(s.entry_id, [])
                             if d.get("hard")]
                    pick = [cands[i] for i in hrng.permutation(len(cands))
                            [:a.hard_per_path]] if cands else []
                    if pick:
                        zero_heat_h = torch.zeros_like(heat_t)
                        from prototypes.v30_video_apex.batch_builder import negative_route_masks
                    for hrow in pick:
                        from prototypes.v30_video_apex.route_supervision import wrong_route_certificate
                        # Revalidate against the loaded revision rather than
                        # trusting a stale mined JSON's boolean certificate.
                        route_certificate_h = wrong_route_certificate(
                            s, hrow["polyline_native"])
                        zero_masks_h = negative_route_masks(
                            masks, query_valid=owner_query_valid(s, _resolved_pk),
                            certified=route_certificate_h["certified_wrong_route"])
                        hpoly = (np.asarray(
                            hrow["polyline_native"], float) -
                            np.array([crop[0], crop[1]])).tolist()
                        try:
                            _rh, _ = resample_polyline(hpoly)
                        except ValueError:
                            continue
                        _rha = np.asarray(_rh)
                        if _rha[:, 0].min() < 0 or \
                                _rha[:, 0].max() > crop[2] or \
                                _rha[:, 1].min() < 0 or \
                                _rha[:, 1].max() > crop[3]:
                            continue  # outside this crop: no claim
                        with torch.no_grad():
                            hprobe = model.forward(
                                clip, prompt,
                                route_xy=hpoly)  # type: ignore[call-arg]
                            n_s = int(
                                hprobe.front_logits.shape[1])
                        dummy = torch.zeros(n_s)
                        rh = train_step_front(
                            model, opt, clip, prompt, hpoly,
                            zero_heat_h, zero_masks_h, dummy,
                            torch.zeros(1), vis, torch.ones(1), step_w,
                            route_t=torch.zeros(1))
                        ep_loss += float(rh["total"])
                        _account_consumed(rh.get("consumed"), s)
                        n_updates += 1
                        note("route", s.sample_key + "|hard:"
                             + str(hrow.get("route_id", "")), "hard-route")
            else:
                r = train_step(model, opt, clip, prompt,
                               {"apex_heat": heat_t,
                                "visibility": vis,
                                "body_mask": body_t},
                               masks, step_w)
            # rev10 WP-A: head/trunk gradient magnitudes from the step
            # that just ran (grads are still on the parameters until the
            # next zero_grad).
            ep_grads.append(_group_grad_norms())
            ep_loss += float(r["total"])
            ep_body += float(r.get("body") or 0.0)
            # rev9 WP-B: keep the body terms of THIS step so the epoch
            # row can show what the selectors objective actually did
            # (a changed objective changes the loss SCALE — comparing
            # raw numbers across objectives is meaningless without them).
            for _k in ("body_dice", "body_self_bce", "body_bg_bce",
                       "body_foreign_bce"):
                if _k in r:
                    try:
                        ep_bparts[_k] = ep_bparts.get(_k, 0.0) + float(r[_k])
                    except (TypeError, ValueError):
                        pass
            n_updates += 1
            # rev8: attribute the loss to the heads that actually had
            # supervised pixels in this sample.
            for _hk, _hv in masks.items():
                try:
                    if float(torch.as_tensor(_hv).sum()) > 0:
                        note(_hk, s.sample_key, s.kind)
                except (TypeError, RuntimeError):
                    continue
        # rev9 WP-A.6: comparisons are consumed HERE and only here, as
        # pairwise route judgments. The legacy path branch joined duels
        # by (movie, frame) and trained the preferred lane with an
        # absolute `route_t = 1.0` — claiming a correctness the
        # annotator never gave. A preference only orders two lanes;
        # "neither" rejects both.
        n_cmp = 0
        for s2, _x2, clip_np2, crop2, blind2 in train_prep:
            if s2.kind != "comparison" or s2.quarantine_reason:
                continue
            _clip2, _prompt2, _h2, _m2, _v2, _sw2, _bt2, _bs2 = \
                sample_tensors(s2, clip_np2, crop2, blind2)
            _o2 = np.array([crop2[0], crop2[1]], dtype=float)
            _la2 = (np.asarray(s2.lanes_a, float) - _o2).tolist()
            _lb2 = (np.asarray(s2.lanes_b, float) - _o2).tolist()
            try:
                _ci = pairwise_route_step(model, opt, _clip2, _prompt2,
                                          _la2, _lb2, s2.preference)
            except (RuntimeError, ValueError) as e:
                print(f"comparison {s2.entry_id}: skipped ({e})")
                continue
            note("route", s2.sample_key, "comparison", routes=2,
                 margin_loss=_ci.get("loss"))
            n_updates += 1
            n_cmp += 1
        if n_cmp:
            print(f"comparison steps this epoch: {n_cmp}", flush=True)
        # Dev probe (oracle routes, no updates).
        model.eval()
        dev_rows = []
        with torch.no_grad():
            for s, _, clip_np, crop, blind in dev_prep:
                # rev8 step 3: the dev probe reads the DEPLOYMENT
                # prompt distribution (auto-grain disc from detection),
                # so the numbers predict runtime, not a hint-favoured
                # setup. Falls back to "none" where no focus exists.
                _dpk = "auto-grain" if (s.focus_xy and len(s.focus_xy) == 2) \
                    else "none"
                clip, prompt, heat_t, valid, vis, _sw, _bt, _bs = \
                    sample_tensors(
                    s, clip_np, crop, blind, prompt_kind=_dpk)
                # rev13 W1: dev contributions use the same consumed()
                # semantics. The dev probe performs no updates and
                # nothing overwrites its masks, so its masks are final
                # by construction; keys stay identical to the train side.
                _account_consumed(
                    _consumed_from(valid, _sw), s, side="dev")
                if s.kind == "neg_region":
                    # Region probe: raw heat max inside the box vs crop
                    # max (suppression readout, not a tip error).
                    pred = model.forward(clip, prompt)
                    hp = pred.heat[0, 0].detach().numpy()
                    b = s.region_box
                    xs = slice(max(0, int(min(b[0], b[2]) - crop[0])),
                               min(hp.shape[1],
                                   int(max(b[0], b[2]) - crop[0])))
                    ys = slice(max(0, int(min(b[1], b[3]) - crop[1])),
                               min(hp.shape[0],
                                   int(max(b[1], b[3]) - crop[1])))
                    inmax = float(hp[ys, xs].max()) \
                        if hp[ys, xs].size else float("nan")
                    # The heat head is an unrestricted regression score
                    # and can be <= 0 while untrained; dividing by a
                    # near-zero max produced ratios like -2.4e8, which
                    # look like a finding and are arithmetic noise. A
                    # non-positive max means the ratio is undefined.
                    _hmax = float(hp.max())
                    dev_rows.append({
                        "entry": s.entry_id, "kind": s.kind,
                        "box_raw_max": inmax,
                        "box_to_max_ratio": (inmax / _hmax
                                             if _hmax > 1e-6 else None),
                        "crop_max": _hmax})
                    continue
                if blind is not None:
                    route_crop = (np.asarray(blind["route_xy"]) -
                                  np.array([crop[0], crop[1]])).tolist()
                    pred = model.forward(  # type: ignore[call-arg]
                        clip, prompt, route_xy=route_crop)
                    q = torch.softmax(pred.front_logits, -1)[0]
                    s_grid = pred.front_s.detach().numpy()
                    s_hat = float(s_grid[int(q.argmax())])
                    dev_rows.append({
                        "entry": s.entry_id, "kind": s.kind,
                        "front_err_px": abs(s_hat - blind["s_star"]),
                        "s_hat": s_hat, "s_star": blind["s_star"]})
                    hp = pred.heat[0, 0].detach().numpy()
                    py, px = np.unravel_index(int(hp.argmax()), hp.shape)
                    dev_rows[-1]["heat_err_px"] = float(np.hypot(
                        px - (s.tip_xy[0] - crop[0]),
                        py - (s.tip_xy[1] - crop[1])))
                    # Argmax distance is meaningless in multi-apex crops
                    # (the argmax lands on a foreign apex at equal height).
                    # Honest readout: heat_at_truth (recall at the owned
                    # tip) + truth_rank (1 = truth is the crop max).
                    # rev7 correction: heat_margin = max - truth, so a
                    # LARGER margin means the truth sits FARTHER below
                    # the max — margin-up is worse discrimination, not
                    # better (H284/H290 read it backwards). A foreign
                    # real cap can legitimately be the max, so margin
                    # alone never proves owned discrimination either.
                    j = int(round(s.tip_xy[0] - crop[0]))
                    i = int(round(s.tip_xy[1] - crop[1]))
                    if 0 <= i < hp.shape[0] and 0 <= j < hp.shape[1]:
                        dev_rows[-1]["heat_at_truth"] = float(hp[i, j])
                        dev_rows[-1]["heat_margin"] = float(
                            hp.max() - hp[i, j])
                        dev_rows[-1]["truth_rank"] = int(
                            1 + (hp > hp[i, j]).sum())
                    # rev6: held-out DISTAL body IoU (proximal excluded:
                    # mirroring the rendered prompt scores proximal for
                    # free). Owner geometry ON here, as in deployment.
                    pcrop = [(q[0] - crop[0], q[1] - crop[1])
                             for q in s.path_xy]
                    n = len(pcrop)
                    k = max(2, n // 3)
                    _tp, _vp = body_ribbon_mask(
                        hp.shape[0], hp.shape[1], crop[:2], pcrop,
                        list(s.path_visible or []))
                    _pp, _ = body_ribbon_mask(
                        hp.shape[0], hp.shape[1], crop[:2], pcrop[:k + 1],
                        [True] * k)
                    distal = _tp * (1 - np.clip(_pp, 0, 1))
                    bp = (torch.sigmoid(
                        pred.body[0, 0]).detach().numpy() > 0.5
                    ).astype(float)
                    _inter = float(((bp == 1) & (distal == 1)).sum())
                    _union = float(((bp == 1) | (distal == 1)).sum())
                    dev_rows[-1]["distal_body_iou"] = (
                        _inter / _union if _union > 0 else 0.0)
                elif s.kind in ("census_tile", "body_mask", "comparison"):
                    # rev8: these sample types have no single tip target;
                    # report a discovery readout instead of a tip error.
                    pred = model.forward(clip, prompt)
                    hp = pred.heat[0, 0].detach().numpy()
                    row_d = {"entry": s.entry_id, "kind": s.kind,
                             "heat_max": float(hp.max())}
                    if s.kind == "census_tile" and s.census_tips:
                        vals = []
                        for q in s.census_tips:
                            jj = int(round(float(q[0]) - crop[0]))
                            ii = int(round(float(q[1]) - crop[1]))
                            if 0 <= ii < hp.shape[0] and \
                                    0 <= jj < hp.shape[1]:
                                vals.append(float(hp[ii, jj]))
                        row_d["census_tip_heat"] = (
                            float(np.mean(vals)) if vals else None)
                        row_d["census_n_out_of_crop"] = (
                            len(s.census_tips) - len(vals))
                    if s.kind == "body_mask":
                        _bl = pred.body[0, 0].detach().numpy()
                        bp = (torch.sigmoid(
                            pred.body[0, 0]).detach().numpy() > 0.5
                        ).astype(float)
                        # rev9 WP-A.3: the SAME builder the training path
                        # uses (this was a nested reimplementation).
                        _dt = own_mask_target(s, hp.shape[0], hp.shape[1],
                                              crop[:2])
                        tgt, val = _dt.target, _dt.valid
                        row_d.update(_dt.readout())
                        from prototypes.v30_video_apex.targets import (
                            mask_iou_split)
                        # rev8: distal = farther than 40px from the
                        # queried grain; the near end is the easy part.
                        _anc = None
                        if s.target_xy and len(s.target_xy) == 2:
                            _anc = (float(s.target_xy[0]) - crop[0],
                                    float(s.target_xy[1]) - crop[1])
                        # split_px="auto": the median painted
                        # distance, so a 30px tube still has a distal
                        # half (an absolute 40px split made every
                        # short-tube distal number vacuous).
                        row_d.update(mask_iou_split(
                            bp, tgt, val, anchor_xy=_anc,
                            split_px="auto"))
                        # rev8 H309: the honest metrics. `mask_iou` is
                        # computed inside the valid region, which is
                        # mostly a band around the paint -- so a
                        # SPATIALLY CONSTANT field above 0.5 scores
                        # ~1.0 there while localising nothing (measured:
                        # front_rev8_plumb scored 1.000 by that metric
                        # and 0.009 crop-wide). Report the logit spread
                        # and the crop-wide IoU so a flat field can
                        # never be mistaken for a prediction again.
                        _t = (tgt > 0)
                        _pi = float((bp > 0) [:, :].sum())
                        _inter = float(((bp > 0) & _t).sum())
                        _union = float(((bp > 0) | _t).sum())
                        row_d["body_logit_std"] = float(_bl.std())
                        row_d["body_logit_mean"] = float(_bl.mean())
                        row_d["crop_iou"] = (_inter / _union
                                             if _union else 0.0)
                        row_d["pred_frac_crop"] = _pi / float(bp.size)
                        row_d["flat_field"] = bool(float(_bl.std()) < 1e-3)
                    dev_rows.append(row_d)
                else:
                    pred = model.forward(clip, prompt)
                    hp = pred.heat[0, 0].detach().numpy()
                    py, px = np.unravel_index(int(hp.argmax()), hp.shape)
                    j = int(round(s.tip_xy[0] - crop[0]))
                    i = int(round(s.tip_xy[1] - crop[1]))
                    at = (float(hp[i, j]) if 0 <= i < hp.shape[0]
                          and 0 <= j < hp.shape[1] else 0.0)
                    dev_rows.append({
                        "entry": s.entry_id, "kind": s.kind,
                        "heat_err_px": float(np.hypot(
                            px - (s.tip_xy[0] - crop[0]),
                            py - (s.tip_xy[1] - crop[1]))),
                        "heat_at_truth": at,
                        "heat_margin": float(hp.max() - at)})
        model.train()
        _nupd = max(1, n_updates - ep_updates0)
        row = {"epoch": ep,
               "train_loss": ep_loss / max(1, len(order)),
               **{f"mean_{k}": v / _nupd
                  for k, v in sorted(ep_bparts.items())},
               "mean_body_loss": (ep_body / max(1, n_updates - ep_updates0)
                                  if n_updates > ep_updates0 else None),
               "updates_this_epoch": n_updates - ep_updates0,
               "body_sources": dict(ep_bsrc),
               "mean_front_err_px": float(np.mean(ep_front))
               if ep_front else None,
               "dev": dev_rows}
        history.append(row)
        # write the trajectory EVERY epoch (atomic replace). Runs here
        # get killed or slept through; a history.json that only lands
        # at the end makes an interrupted arm unreadable, and the
        # equal-updates comparison is exactly the case that needs both
        # arms readable.
        _tmp = out / "history.json.tmp"
        _tmp.write_text(json.dumps(history, indent=2))
        _tmp.replace(out / "history.json")
        dev_fronts = [d["front_err_px"] for d in dev_rows
                      if d.get("front_err_px") is not None]
        if dev_fronts:
            mean_dev = float(np.mean(dev_fronts))
            row["mean_dev_front_px"] = mean_dev
            if best_dev_front is None or mean_dev < best_dev_front:
                best_dev_front, best_epoch = mean_dev, ep
        _body_vals = [r.get("mask_iou") for r in dev_rows
                      if r.get("kind") == "body_mask"
                      and r.get("mask_iou") is not None]
        _front_vals = [r.get("front_err_px") for r in dev_rows
                       if r.get("front_err_px") is not None]
        if _body_vals:
            _bm = float(np.mean(_body_vals))
            if best_body is None or _bm > best_body[0]:
                best_body = (_bm, ep)
                _save_best("body", _bm,
                           "dev body-mask reviewed-region mask_iou "
                           "(mean over dev body rows)", ep)
        if _front_vals:
            _fm = float(np.mean(_front_vals))
            if best_front is None or _fm < best_front[0]:
                best_front = (_fm, ep)
                _save_best("front", _fm,
                           "mean dev front_err_px (lower is better; "
                           "untrained heads score the null default)", ep)
        if a.save_every > 0 and (ep + 1) % a.save_every == 0:
            save_checkpoint(
                out / f"ep{ep}.pt", model, opt,
                config=dict(run_config, epoch=ep),
                manifest={"epoch": ep,
                          "mean_dev_front_px": row.get(
                              "mean_dev_front_px")},
                rng_state={"epoch_rng": rng.bit_generator.state})
        _flat = [r["entry"] for r in dev_rows
                 if r.get("flat_field") and r.get("kind") == "body_mask"]
        if _flat:
            # rev8 H309: a flat body field makes every IoU reading
            # meaningless (it scores ~1.0 inside the valid band). Say so
            # in the run's own log instead of leaving a healthy-looking
            # number for someone to interpret later.
            print(f"WARNING ep {ep}: FLAT body field on {len(_flat)} dev "
                  f"mask(s) -- IoU readings this epoch are threshold "
                  f"artifacts: {_flat[:3]}", flush=True)
        _comp = {k: round(float(v), 4) for k, v in sorted(row.items())
                 if k not in ("train_loss", "mean_front_err_px")
                 and isinstance(v, (int, float))}
        print(f"ep {ep}: loss={row['train_loss']:.4f} parts={_comp} "
              f"front={row['mean_front_err_px']} dev={dev_rows}",
              flush=True)

    # rev8/rev9: the consumption manifest — which unique labels actually
    # generated each head's loss, verified against epochs run. rev9
    # WP-A.6: built by `label_ledger.build_consumption_table`, which also
    # REFUSES a table in which a quarantined label contributed anything.
    try:
        from prototypes.v30_video_apex.label_ledger import (
            build_consumption_table)
        _bykind: dict[str, set] = {}
        for h_, keys in usage.items():
            for k_ in keys:
                base = k_.split("|rotated")[0].split("|hard:")[0]
                base = base.split("|lane:")[0]
                kind = usage_kind.get(k_) or "unknown"
                _bykind.setdefault(kind, set()).add(k_)
        table = build_consumption_table(
            samples, usage, counts=metric_counts, kind_of=usage_kind)
        usage_out = {
            "snapshot": str(a.snapshot),
            "snapshot_content_sha256": snap_manifest.get(
                "content_sha256", {}),
            "epochs": a.epochs,
            "heads": table["heads"],
            "unique_total": table["unique_total"],
            "quarantined_labels": table["quarantined_labels"],
            "quarantined_consumed": table["quarantined_consumed"],
            "consumer_map": table["consumer_map"],
            "by_kind": {k: len(v) for k, v in sorted(_bykind.items())},
            "prompt_mix": {k: w for k, w in prompt_mix},
            "prompt_counts": dict(prompt_counts),
            # what the loss actually consumed, after selection, after the
            # resolved-prompt no-query zeroing, gated by the effective
            # weight; phase-separated, with pixel mass (rev11).
            "actual_contributions": {
                h: {"train_labels": len(v["train"]),
                    "dev_labels": len(v["dev"]),
                    "train_updates": int(sum(v["train"].values())),
                    "dev_updates": int(sum(v["dev"].values())),
                    "train_mass_px": actual_mass.get(h, {}).get(
                        "train", 0.0),
                    "dev_mass_px": actual_mass.get(h, {}).get("dev", 0.0),
                    "keys": sorted(set(v["train"])
                                   | set(v["dev"]))[:2000]}
                for h, v in sorted(actual_use.items())},
            # rev13 W1: invalid-head counts at the LOSS BOUNDARY with the
            # reason (zero-mass vs zero-weight). A head can appear here
            # even when it never appears in actual_contributions: that is
            # exactly the bypass the audit demonstrated.
            "zeroed_at_loss": {
                h: {"train": dict(z["train"]), "dev": dict(z["dev"])}
                for h, z in sorted(actual_zeroed.items())},
            "body_prompt_schedule": {
                **body_schedule,
                "note": "counted AFTER the resolved-prompt no-query "
                        "zeroing (rev11): opportunities = updates whose "
                        "sample carried a body target; zeroed_no_query "
                        "= those zeroed by a resolved 'none' prompt; "
                        "nonzero = updates that carried a non-zero body "
                        "target into the weighted loss.",
            },
        }
        (out / "label_usage.json").write_text(json.dumps(
            usage_out, indent=1, default=str))
        print(f"label usage -> {out / 'label_usage.json'}: "
              f"{usage_out['unique_total']} unique keys, "
              f"heads={ {h: len(v) for h, v in usage.items()} }",
              flush=True)
    except TypeError as e:
        # Type-level failures (a counter shape we cannot serialize) stay
        # non-fatal, but a CONTRACT failure must stop the run: the review
        # forbids swallowing invalid consumers in bookkeeping.
        print(f"label usage manifest skipped (type): {e}")

    print(f"confusable negatives: {conf_stats['with_confusable']} of "
          f"{conf_stats['body_sights']} body-mask sights "
          f"({conf_stats['confusable_px']} px total, validity extended on "
          f"{conf_stats['valid_extended']})", flush=True)
    # rev12 P0.2: the owner-query gating receipt (zeroed gradients by
    # reason) — the no-query visibility count must be explicit.
    _gate = {k: v for k, v in conf_stats.items()
             if k.startswith("vis_valid_zeroed_")
             or k.endswith("_body_zeroed")}
    print(f"owner-query gating: {_gate}", flush=True)
    (out / "gating_counters.json").write_text(json.dumps(
        _gate, indent=1) + "\n")
    ckpt = save_checkpoint(
        out / "front.pt", model, opt,
        config=run_config,
        manifest={"snapshot": str(a.snapshot),
                  "snapshot_manifest": snap_manifest,
                  "n_train": len(train_prep), "n_dev": len(dev_prep),
                  "updates_per_epoch": updates_per_epoch,
                  "total_updates": n_updates,
                  "elapsed_s": time.time() - t0},
        rng_state={"epoch_rng": rng.bit_generator.state})
    # Reload check: fresh model reproduces dev rows exactly.
    model2 = build_model("temporal", base=a.base,
                         multiscale=bool(a.multiscale))
    load_checkpoint(ckpt, model2)
    model2.eval()
    model.eval()
    with torch.no_grad():
        s, _, clip_np, crop, blind = dev_prep[0]
        clip, prompt, _, _, _, _, _, _ = sample_tensors(s, clip_np, crop, blind)
        kw = {}
        if blind is not None:
            kw["route_xy"] = (np.asarray(blind["route_xy"]) -
                              np.array([crop[0], crop[1]])).tolist()
        h1 = model.forward(clip, prompt, **kw).heat
        h2 = model2.forward(clip, prompt, **kw).heat
        assert torch.equal(h1, h2), "reload must reproduce inference"
    (out / "samples.json").write_text(json.dumps(frozen, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    from prototypes.v30_video_apex.train import (
        body_loss_config as _body_loss_config)
    (out / "run_manifest.json").write_text(json.dumps({
        "snapshot": str(a.snapshot),
        "snapshot_manifest": snap_manifest,
        "dev_refs": dev_refs,
        "dev_frames": sorted([m + ":" + str(f) for m, f in dev_frames]),
        "updates_per_epoch": updates_per_epoch,
        "total_updates": n_updates,
        "resumed_from": resumed_from.get("config_hash", "")
        if resumed_from else "",
        "optimizer_fresh": optimizer_fresh,
        "zero_owner_channel": bool(a.zero_owner_channel),
        "route_jitter_deg": a.route_jitter,
        "wrong_routes_per_path": a.wrong_routes,
        "wrong_angle_deg": list(a.wrong_angle),
        "weak_projects": weak_projects,
        "weak_apex_weight": a.weak_apex_weight,
        "weak_sigma": a.weak_sigma,
        "best_epoch": best_epoch,
        "best_mean_dev_front_px": best_dev_front,
        # rev12 P0.1: the complete effective body-loss configuration —
        # comparisons are checked mechanically, never from prose.
        "body_loss_config": _body_loss_config(),
        "config": {"variant": "temporal", "base": a.base, "lr": a.lr,
                   "epochs": a.epochs, "seed": a.seed, "crop": CS},
        # rev7: the checkpoint hash covers the FULL run config, not
        # just architecture — resuming a differently-configured run
        # must refuse, not silently continue.
        "config_hash": config_hash(run_config),
    }, indent=2, default=str))
    for rr in readers.values():
        rr.close()
    print(f"done: {n_updates} updates, ckpt={ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
