"""Bounded native owned-body fit using preserved exact masks and scopes."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prototypes.v30_video_apex.native_body import (
    NativeBodyNet, TILE, body_input, body_loss, load_body_checkpoint, save_body_checkpoint)
from prototypes.v30_video_apex.native_caps import extract_tile, file_hash
from prototypes.v30_video_apex.span_supervision import reviewed_span_definitions

PAD = 16
STORED = TILE + 2*PAD
# Reviewed tube half-width (physics): a scoped centreline span is painted
# as a 2*half-width band of positive pixels; the rest of the crop is unknown.
SPAN_HALF_PX = 4.0


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def query_in_crop(xy, image_origin, offset, turns=0, flipped=False, size=TILE):
    """Transform any native query with the IMAGE crop's transform.

    A partner grain's own crop origin must never enter this calculation:
    both prompts are evaluated on exactly the same transformed pixels.
    """
    if xy is None:
        return None
    q = np.asarray(xy, dtype=float) - np.asarray(image_origin) - np.asarray(offset)
    for _ in range(turns % 4):
        q = np.array([q[1], size - 1 - q[0]])
    if flipped:
        q[0] = size - 1 - q[0]
    return q


def presence_pool_mask(case, offset, size, turns=0, flipped=False):
    """Bool emergence-ring mask (1r..2.5r) in crop coordinates.

    Uses only the case's licensed query geometry (grain centre + radius),
    never any label: it tells the head WHERE to look, not what is there.
    The grain point undergoes the same crop transform as the input pixels.
    """
    g = query_in_crop(np.asarray(case["grain"], float),
                      np.asarray(case["origin"], float),
                      offset, turns=turns, flipped=flipped, size=size)
    r = float(case.get("radius", 13))
    yy, xx = np.mgrid[:size, :size]
    d = np.hypot(xx - g[0], yy - g[1])
    return (d >= r) & (d <= 2.5 * r)


def body_owner_pairs(doc, arrays):
    """All same-frame distinct-owner pairs, using exact licensed paint.

    A positive centreline band does not license foreign negatives. Duplicate
    masks of one physical grain must not compete, even when older exports
    repeat a movie prefix or give the same grain different observation IDs.
    """
    from collections import defaultdict

    def owner_key(case):
        owner = str(case.get('owner') or '')
        prefix = case['movie'] + '|'
        while owner.startswith(prefix):
            owner = owner[len(prefix):]
        return case['movie'], owner

    def shift(mask, origin_from, origin_to):
        h, w = mask.shape
        dy, dx = (np.asarray(origin_from) - np.asarray(origin_to))[::-1].astype(int)
        out = np.zeros_like(mask, dtype=bool)
        y0, y1 = max(0, dy), min(h, dy + h)
        x0, x1 = max(0, dx), min(w, dx + w)
        if y0 < y1 and x0 < x1:
            out[y0:y1, x0:x1] = mask[y0-dy:y1-dy, x0-dx:x1-dx]
        return out

    groups, partners, eligible, report = defaultdict(list), defaultdict(list), {}, []
    for i, case in enumerate(doc['cases']):
        if case['kind'] == 'owned_body' and case.get('owner'):
            groups[(case['movie'], case['frame'])].append(i)
    for (movie, frame), members in sorted(groups.items()):
        for a, i in enumerate(members):
            for j in members[a+1:]:
                ci, cj = doc['cases'][i], doc['cases'][j]
                # Coincident reviewed grain centres are unresolved aliases,
                # not evidence of two distinct physical grains.
                separation = np.linalg.norm(np.asarray(ci['grain']) - cj['grain'])
                if owner_key(ci) == owner_key(cj) or separation <= 2.:
                    continue
                pi, pj = (np.asarray(arrays['positive'][k], bool) for k in (i, j))
                pj_i = shift(pj, cj['origin'], ci['origin'])
                pi_j = shift(pi, ci['origin'], cj['origin'])
                ei, ej = pi & ~pj_i, pj & ~pi_j
                if not ei.any() or not ej.any():
                    continue
                partners[i].append(j); partners[j].append(i)
                eligible[i, j], eligible[j, i] = ei, ej
                report.append({'a': ci['id'], 'b': cj['id'], 'movie': movie,
                    'frame': int(frame), 'eligible_pixels_a': int(ei.sum()),
                    'eligible_pixels_b': int(ej.sum()),
                    'overlap_pixels_excluded': int((pi & pj_i).sum()),
                    'grain_separation_px': float(separation),
                    'rule': 'exact owned pixels only; same-owner, overlap and weak spans excluded'})
    return dict(partners), eligible, report


def local_owner_pairs(doc, pairs, max_distance):
    """Keep nearby queries that can both be present in the same native crop."""
    local = {}
    for i, partners in pairs.items():
        a = doc['cases'][i]
        for j in partners:
            b = doc['cases'][j]
            q = np.asarray([a['grain'], b['grain']]) - a['origin']
            lower = np.maximum(0, np.ceil(q.max(0) - (TILE-1))).astype(int)
            upper = np.minimum(2*PAD, np.floor(q.min(0))).astype(int)
            if np.linalg.norm(q[0]-q[1]) <= max_distance and (lower <= upper).all():
                local.setdefault(i, []).append(j)
    return local


def align_native_array(array, source_origin, destination_origin):
    """Translate stored pixels/selectors without resampling native geometry."""
    h, w = array.shape
    displacement = np.asarray(source_origin) - np.asarray(destination_origin)
    if displacement.shape != (2,) or not np.isfinite(displacement).all() or np.any(displacement != np.floor(displacement)):
        raise ValueError('stored image origins must differ by whole native pixels')
    dx, dy = displacement.astype(int)
    x0, x1, y0, y1 = max(0, dx), min(w, w+dx), max(0, dy), min(h, h+dy)
    result, covered = np.zeros_like(array), np.zeros_like(array, dtype=bool)
    if x0 < x1 and y0 < y1:
        result[y0:y1, x0:x1] = array[y0-dy:y1-dy, x0-dx:x1-dx]
        covered[y0:y1, x0:x1] = True
    return result, covered


def same_crop_owner_samples(doc, arrays, mode, max_distance=80.):
    """Explicit per-query samples, including complete licensed partner targets.

    Both arms use the same images/queries and sample ordering. The control
    extra query has only foreign paint; the symmetric arm adds the queried
    owner's own positives and reviewed background, aligned to the image.
    """
    if mode not in ('negative_only', 'symmetric_positive'):
        raise ValueError('unknown same-crop supervision mode')
    names = ('positive', 'ordinary', 'foreign')
    samples = [{'image_case': i, 'query_case': i, 'kind': 'primary',
                'targets': {k: arrays[k][i] for k in names}} for i in range(len(doc['cases']))]
    pairs, eligible, _ = body_owner_pairs(doc, arrays)
    pairs = local_owner_pairs(doc, pairs, max_distance)
    audits = []
    for i, others in sorted(pairs.items()):
        ci = doc['cases'][i]
        for j in others:
            cj = doc['cases'][j]
            pixels, covered = align_native_array(arrays['pixels'][j], cj['origin'], ci['origin'])
            if not np.array_equal(pixels[covered], arrays['pixels'][i][covered]):
                raise ValueError('paired masks do not describe identical native image pixels')
            pos, bg, foreign = [align_native_array(arrays[k][j], cj['origin'], ci['origin'])[0]
                                for k in names]
            overlap = pos & arrays['positive'][i]
            pos &= ~overlap
            foreign = (foreign | eligible[i, j]) & ~overlap
            bg &= ~overlap & ~foreign
            symmetric = dict(zip(names, (pos, bg, foreign)))
            if (pos & (bg | foreign) | bg & foreign).any():
                raise ValueError('aligned partner supervision domains overlap')
            zeros = np.zeros_like(pos)
            targets = (symmetric if mode == 'symmetric_positive' else
                       dict(zip(names, (zeros, zeros, eligible[i, j]))))
            samples.append({'image_case': i, 'query_case': j, 'kind': 'same_crop_partner',
                            'targets': targets})
            crop = lambda value: value[PAD:PAD+TILE, PAD:PAD+TILE]
            audits.append({'image_case': ci['id'], 'query_case': cj['id'],
                'origin': (np.asarray(ci['origin'])+PAD).tolist(), 'grain_query': cj['grain'],
                'identical_source_pixels': int(crop(covered).sum()),
                'positive_pixels': int(crop(pos).sum()), 'ordinary_pixels': int(crop(bg).sum()),
                'foreign_pixels': int(crop(foreign).sum()),
                'negative_only_pixels': int(crop(eligible[i, j]).sum()),
                'overlap_pixels_excluded': int(crop(overlap).sum())})
    if not audits:
        raise ValueError('same-crop supervision needs a local pair of reviewed distinct owners')
    return samples, audits


def transfer_body_parameters(model, initial):
    """Strict transfer, allowing only inert additions of root/query channels."""
    import torch
    target, old = model.state_dict(), initial.state_dict()
    extra = set(old) - set(target)
    if extra:
        raise ValueError(f'cannot discard learned body parameters: {sorted(extra)}')
    transferred, padded, added = {}, [], []
    for key, value in target.items():
        if key not in old:
            if key.startswith('query_') and not getattr(initial, 'query_adapters', False):
                transferred[key] = value
                added.append(key)
                continue
            if key.startswith('presence_fc.') and not getattr(initial, 'presence_head', False):
                # rev15 task 3: a new presence head starts at zero (sigmoid 0.5,
                # maximally uncertain) so enabling it never perturbs the dense path.
                transferred[key] = torch.zeros_like(value)
                added.append(key)
                continue
            raise ValueError(f'init is missing tensor: {key}')
        source = old[key]
        if source.shape == value.shape:
            transferred[key] = source
        elif (key == 'e1.0.weight' and model.root_channel and not initial.root_channel
              and value.shape[0] == source.shape[0] and value.shape[1] == source.shape[1]+1
              and value.shape[2:] == source.shape[2:]):
            tensor = torch.zeros_like(value)
            tensor[:, :source.shape[1]] = source
            transferred[key] = tensor
            padded.append({'tensor': key, 'from': list(source.shape), 'to': list(value.shape)})
        else:
            raise ValueError(f'cannot reconcile init tensor {key}: {tuple(source.shape)} -> {tuple(value.shape)}')
    model.load_state_dict(transferred, strict=True)
    return {'zero_padded': padded, 'zero_initialized_query_tensors': added}


def owner_query_loss(correct, wrong, eligible, *, margin, margin_weight, foreign_weight):
    """A relative advantage alone does not make the wrong owner negative."""
    import torch
    mask = eligible.bool()
    zero = (correct.sum() + wrong.sum()) * 0.
    if not bool(mask.any()):
        return zero, zero
    relative = (torch.relu(margin - (correct[mask] - wrong[mask])).mean() * margin_weight
                if margin > 0 else zero)
    foreign = torch.nn.functional.softplus(wrong[mask]).mean() * foreign_weight
    return relative, foreign


def filter_training_records(samples, observations, regions, masks):
    """A held-out frame cannot train through a different owner's body query."""
    from tubetracker.review_semantics import TRAINING_REVIEW_ROLES, training_review_role
    records = observations + regions + masks
    excluded = [{'movie': r.get('movie_uuid') or r.get('movie'),
        'frame': int(r.get('source_frame', -1)), 'role': training_review_role(r),
        'source_id': r.get('obs_uuid') or r.get('_region_uuid') or r.get('mask_uuid'),
        'revision': r.get('obs_revision', r.get('_region_revision', r.get('mask_revision', 1))),
        'project': r.get('project') or r.get('_project', '')}
        for r in records if training_review_role(r) not in TRAINING_REVIEW_ROLES]
    blocked = {(r['movie'], r['frame']) for r in excluded}
    keep = lambda r: (r.get('movie_uuid') or r.get('movie'), int(r.get('source_frame', -1))) not in blocked
    return ([s for s in samples if (s.movie, s.source_frame) not in blocked],
            [r for r in observations if keep(r)], [r for r in regions if keep(r)], excluded)


def reviewed_presence_definitions(observations, owners_by_id, manifest_movies):
    """Per-grain owned-presence observations near the pollen (rev15 task 3).

    Accepts human review_tip answers with direct_state in
    ('no_tube_visible', 'direct_visible', 'not_directly_visible') that name an
    owner — INCLUDING validation-role records, which are carried as held-out
    presence checks, never pixel training. Each entry carries explicit query
    geometry (owner grain centre from the corrected registry) and review
    provenance (observation id/revision/project/role). A scalar absence
    verdict NEVER becomes a dense background mask: presence cases enter the
    panel with zero pixel loss and are measured, not trained.
    """
    definitions = []
    for o in observations:
        state = o.get('direct_state')
        if state not in ('no_tube_visible', 'direct_visible', 'not_directly_visible'):
            continue
        owner_id = o.get('owner_uuid')
        if not owner_id:
            continue
        movie = o.get('movie')
        if movie not in (manifest_movies or {}):
            continue
        owner = (owners_by_id or {}).get(owner_id)
        grain = list(map(float, owner['grain_native'])) if owner and owner.get('grain_native') else None
        if not grain:
            continue
        frame = int(o.get('source_frame', -1))
        if frame < 0:
            continue
        if state == 'direct_visible' and not (o.get('direct_xy') or o.get('path_xy')):
            presence = 'uncertain'
        elif state == 'direct_visible':
            presence = 'visible'
        elif state == 'no_tube_visible':
            presence = 'absent'
        else:
            presence = 'uncertain'
        definitions.append((movie, frame, grain,
            {'_presence': presence,
             '_presence_uuid': o.get('obs_uuid'),
             '_presence_revision': o.get('obs_revision', 1),
             '_presence_project': o.get('project'),
             '_presence_role': o.get('annotation_role') or 'development',
             '_presence_review_origin': o.get('review_origin'),
             '_presence_tip_source': o.get('tip_source'),
             '_presence_focus_xy': o.get('focus_xy'),
             '_presence_owner': owner_id,
             '_presence_state': state}))
    return definitions


def event_bracket_presence_definitions(events, owners_by_id, extra_grains, manifest_movies):
    """Scalar presence cases from human germination-event brackets (rev16).

    A human `emerged_within` verdict with at least one endpoint supplies one
    scalar case per endpoint: absent at last_absent_frame, visible at
    first_visible_frame. Endpoints are approximate (the reviewer marks close
    frames, not certified exact onset) — recorded as training role with event
    lineage. Query geometry comes from the owners registry, else the explicit
    event-grain registry (e.g. survey-verified G0); events without resolvable
    geometry are skipped. `unobservable`/censored verdicts and non-human
    drafts never enter: unknown stays unknown, never a negative. Downstream
    these cases carry zero pixel loss, like review_tip presence cases.
    """
    definitions = []
    for e in (events or []):
        d = e.get('data', e) if isinstance(e, dict) else {}
        if d.get('verdict') != 'emerged_within':
            continue
        if d.get('review_origin') != 'human':
            continue
        movie = d.get('movie_uuid')
        if movie not in (manifest_movies or {}):
            continue
        owner_id = d.get('owner_uuid')
        grain, radius, grain_source = None, 13.0, 'owners-registry'
        owner = (owners_by_id or {}).get(owner_id)
        if owner and owner.get('grain_native'):
            grain = list(map(float, owner['grain_native']))
            radius = float(owner.get('grain_radius_px') or 13.0)
        elif (extra_grains or {}).get(owner_id):
            g = extra_grains[owner_id]
            grain = list(map(float, g['grain_native']))
            radius = float(g.get('grain_radius_px') or 13.0)
            grain_source = str(g.get('source', 'event-grain-registry'))
        if not grain:
            continue
        la, fv = d.get('last_absent_frame'), d.get('first_visible_frame')
        if la is None and fv is None:
            continue
        ws, we = int(d.get('window_start', 0)), int(d.get('window_end', 0))
        mismatch = ((la is not None and not (ws <= int(la) <= we))
                    or (fv is not None and not (ws <= int(fv) <= we)))
        for frame, presence, state, suffix in (
                (la, 'absent', 'event_bracket_absent', 'absent'),
                (fv, 'visible', 'event_bracket_visible', 'visible')):
            if frame is None:
                continue
            definitions.append((movie, int(frame), list(grain),
                {'_presence': presence,
                 '_presence_uuid': '%s:%s' % (d.get('uuid', 'event'), suffix),
                 '_presence_revision': int(d.get('revision', 1)),
                 '_presence_project': d.get('project') or e.get('project'),
                 '_presence_role': 'training',
                 '_presence_review_origin': 'human',
                 '_presence_tip_source': None,
                 '_presence_focus_xy': None,
                 '_presence_owner': owner_id,
                 '_presence_state': state,
                 '_presence_radius': radius,
                 '_presence_grain_source': grain_source,
                 '_presence_window_mismatch': mismatch,
                 '_presence_event_bracket': [la, fv]}))
    return definitions


def prepare(args):
    import cv2
    from tubetracker.annotation_frames import FrameReader
    from prototypes.v30_video_apex.targets import samples_from_snapshot, confusable_union, paint_overlap_union
    from prototypes.v30_video_apex.batch_builder import own_mask_target, finalize_body_supervision
    from prototypes.v30_video_apex.cap_evidence import inside_polygon
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    snap = Path(args.snapshot).resolve()
    manifest = json.loads((snap/"snapshot_manifest.json").read_text())
    samples = [s for s in samples_from_snapshot(snap) if s.kind == "body_mask"
               and s.mask_raster and s.review_region and not s.quarantine_reason]
    regions = json.loads((snap/"regions.json").read_text())
    # rev14 W3: a reviewed PARTIAL centreline is scoped positive evidence.
    # The span enters as a tube-half-width positive band; every other pixel
    # is unknown (no loss) - never background, never a negative. The
    # implementation visibility field is deliberately not consulted.
    census = (json.loads((snap/"census.json").read_text())
              if (snap/"census.json").exists() else [])
    observations = (json.loads((snap/'observations.json').read_text())
                    if (snap/'observations.json').exists() else [])
    masks = (json.loads((snap/'body_masks.json').read_text())
             if (snap/'body_masks.json').exists() else [])
    samples, observations, regions, excluded_roles = filter_training_records(
        samples, observations, regions, masks)
    pool = [{"mask_uuid": s.mask_uuid, "movie": s.movie, "source_frame": s.source_frame,
             "owner_key": s.owner_key, "mask_raster": s.mask_raster} for s in samples]
    eligible = {s.mask_uuid for s in samples}
    definitions = [(s.movie, s.source_frame, s.target_xy, s) for s in samples]
    definitions += [(r.get("movie_uuid") or r["movie"], r["source_frame"], r["grain_xy"], r)
                    for r in regions if r.get("kind") == "owned_absence"
                    and r.get("class_scope") == "owned_tube"
                    and r.get("source") == "owner-task-no-tube"]
    spans, excluded_spans = reviewed_span_definitions(observations, census, manifest['movies'])
    definitions += spans
    # rev15 task 3: per-grain owned-presence observations (review_tip
    # absence/visible answers). Built from the PRE-FILTER observation list so
    # validation-role answers are carried as held-out presence checks;
    # workflow and withdrawn records are still excluded. Query geometry comes
    # from the corrected owners registry; each case records its review
    # provenance. Presence cases NEVER carry pixel supervision (see the
    # sampling branch below): a scalar verdict is measured, not painted.
    from tubetracker.review_semantics import is_workflow_record, is_withdrawn
    _reg_path = Path(args.owners_registry) if getattr(args, 'owners_registry', '') else None
    _owners_by_id = {}
    if _reg_path and _reg_path.exists():
        _reg = json.loads(_reg_path.read_text())
        for _o in (_reg if isinstance(_reg, list) else _reg.get('owners', [])):
            _owners_by_id[_o['id']] = _o
    _presence_source_obs = [o for o in
        (json.loads((snap/'observations.json').read_text()) if (snap/'observations.json').exists() else [])
        if not is_workflow_record(o) and not is_withdrawn(o)]
    presence_definitions = reviewed_presence_definitions(
        _presence_source_obs, _owners_by_id, manifest['movies'])
    # rev16: human germination-event brackets as scalar presence cases. G0's
    # emerged_within [8000,8050] supplies one absent + one visible case from
    # the same grain (endpoints approximate, not certified exact onset).
    # Unobservable/censored verdicts never enter; geometry must resolve via
    # the owners registry or the explicit event-grain registry (survey), else
    # the event is skipped — no invented queries.
    _event_reg_path = Path(args.event_grain_registry) if getattr(args, 'event_grain_registry', '') else None
    _extra_grains = {}
    if _event_reg_path and _event_reg_path.exists():
        _extra_grains = json.loads(_event_reg_path.read_text())
    _events = (json.loads((snap/'germination_events.json').read_text())
               if (snap/'germination_events.json').exists() else [])
    presence_definitions += event_bracket_presence_definitions(
        _events, _owners_by_id, _extra_grains, manifest['movies'])
    # WO1-pose: resolve frame-specific runtime grain poses for presence
    # queries. The registry centre is a late-frame reference; the app
    # follows moving poses (cf70 differs by ~11px early). Each case records
    # which geometry it used; unusable registration stays flagged, never a
    # confident negative.
    _pose_index = {}
    for _pp in (getattr(args, 'grain_poses', None) or []):
        try:
            _plan = json.loads(Path(_pp).read_text())
        except Exception:
            continue
        for _p in (_plan.get('poses', []) if isinstance(_plan, dict) else []):
            _pose_index[(_p.get('owner_id'), int(_p.get('source_frame', -1)))] = _p
    _posed = []
    for _movie, _frame, _grain, _source in presence_definitions:
        if isinstance(_source, dict) and '_presence' in _source:
            _oid = _source.get('_presence_owner')
            _pose = (_pose_index.get((_oid, int(_frame)))
                     or _pose_index.get(('ld|' + str(_oid), int(_frame))))
            _usable = bool(_pose and _pose.get('grain_native')
                           and _pose.get('geometry_complete', True)
                           and _pose.get('pose_status') != 'unusable')
            if _usable:
                _ev = _pose.get('evidence', {}) or {}
                _source = dict(_source, **{
                    '_presence_grain': [float(v) for v in _pose['grain_native']],
                    '_presence_radius': float(_pose.get('grain_radius_px', 13)),
                    '_presence_pose_source': 'runtime_pose',
                    '_presence_pose_reference_frame': (_pose.get('evidence', {}) or {}).get('reference_anchor_frame'),
                    '_presence_pose_status': _pose.get('pose_status'),
                    '_presence_pose_correlation': _ev.get('patch_correlation'),
                    '_presence_pose_flow_points': _ev.get('flow_points'),
                    '_presence_pose_translation_px': [
                        round(float(_pose['grain_native'][0]) - float(_grain[0]), 2),
                        round(float(_pose['grain_native'][1]) - float(_grain[1]), 2)]})
                _grain = [float(v) for v in _pose['grain_native']]
            else:
                _source = dict(_source, **{
                    '_presence_pose_source': 'registry_fallback',
                    '_presence_pose_status': (_pose or {}).get('pose_status', 'no_pose')})
        _posed.append((_movie, _frame, _grain, _source))
    definitions += _posed
    arrays = {k: [] for k in ("pixels", "positive", "ordinary", "foreign")}
    cases, readers, frames = [], {}, {}
    try:
        for movie, frame, grain, source in definitions:
            if len(grain) != 2:
                raise ValueError("every body sample requires an explicit reviewed grain query")
            origin = np.floor(np.asarray(grain)-STORED/2).astype(int)
            if movie not in readers:
                readers[movie] = FrameReader(manifest["movies"][movie]["path"])
            if (movie, frame) not in frames:
                r = readers[movie].read(frame)
                if not r.exact:
                    raise ValueError("inexact native frame")
                frames[(movie, frame)] = cv2.cvtColor(r.frame, cv2.COLOR_BGR2GRAY)
            gray = frames[(movie, frame)]
            pixels = extract_tile([gray], origin, STORED)[0]
            if isinstance(source, dict) and "_presence" in source:
                # rev15 task 3: scalar owned-presence verdict. ZERO pixel
                # loss by construction — no dense mask is fabricated from
                # the verdict. The crop (grain + emergence neighbourhood)
                # is stored so eval can measure the presence response.
                pos = np.zeros((STORED, STORED), bool)
                negative = np.zeros((STORED, STORED), bool)
                foreign = np.zeros((STORED, STORED), bool)
                case = {"id": source["_presence_uuid"],
                        "revision": source["_presence_revision"],
                        "kind": "owned_presence",
                        "radius": float(source.get("_presence_radius", 13)),
                        "source": source["_presence_project"] or "snapshot",
                        "owner": source["_presence_owner"],
                        "presence": source["_presence"],
                        "presence_role": source["_presence_role"],
                        "presence_review_origin": source["_presence_review_origin"],
                        "presence_tip_source": source["_presence_tip_source"],
                        "presence_focus_xy": source["_presence_focus_xy"],
                        "presence_state": source["_presence_state"],
                        "presence_grain_source": source.get("_presence_grain_source", "owners-registry"),
                        "presence_window_mismatch": bool(source.get("_presence_window_mismatch", False)),
                        "query_pose_source": source.get("_presence_pose_source", "registry_fallback"),
                        "query_pose_status": source.get("_presence_pose_status"),
                        "query_pose_reference_frame": source.get("_presence_pose_reference_frame"),
                        "query_pose_correlation": source.get("_presence_pose_correlation"),
                        "query_pose_flow_points": source.get("_presence_pose_flow_points"),
                        "query_pose_translation_px": source.get("_presence_pose_translation_px"),
                        "supervision": "scalar owned-presence verdict; zero pixel loss (measured, not trained)"}
            elif isinstance(source, dict) and "_span" in source:
                # scoped positive band along the reviewed centreline
                band = np.zeros((STORED, STORED), np.uint8)
                segments = np.asarray(source["_span"], float)
                for a, b in segments:
                    cv2.line(band, tuple(np.round(a-origin).astype(int)),
                             tuple(np.round(b-origin).astype(int)), 1,
                             int(round(2*SPAN_HALF_PX)))
                pos = band.astype(bool)
                negative = np.zeros((STORED, STORED), bool)
                foreign = np.zeros((STORED, STORED), bool)
                case = {"id": source["_span_uuid"],
                        "revision": source["_span_revision"],
                        "kind": "owned_body_scoped_span", "radius": 13,
                        "source": source["_span_project"] or "snapshot",
                        "owner": source["_span_owner"],
                        "span_segments": int(len(source["_span"])),
                        "span_role": source.get("_span_role", "development"),
                        "supervision": "derived positive centreline band over visible segments; outside is unknown (no loss)",
                        "lineage": source['_span_lineage'], "derived_band_half_width_px": SPAN_HALF_PX}
            elif isinstance(source, dict):
                y, x = np.mgrid[:STORED, :STORED]
                xy = np.stack((x.ravel()+origin[0], y.ravel()+origin[1]), 1)
                negative = inside_polygon(xy, source["polygon_xy"]).reshape(STORED, STORED)
                pos, foreign = np.zeros_like(negative), np.zeros_like(negative)
                case = {"id": source["_region_uuid"], "revision": source["_region_revision"],
                        "kind": "owned_absence", "radius": source.get("grain_r", 14),
                        "source": source["_project"], "owner": source.get("grain_id", "")}
            else:
                s = source
                bt = own_mask_target(s, STORED, STORED, tuple(origin))
                fo = confusable_union(STORED, STORED, tuple(origin), movie=movie, frame=frame,
                    self_uuid=s.mask_uuid, self_owner_key=s.owner_key, masks=pool, eligible=eligible)
                overlap = paint_overlap_union(STORED, STORED, tuple(origin),
                    movie=movie, frame=frame, masks=pool, eligible=eligible)
                bt = finalize_body_supervision(bt, confusable=fo, overlap=overlap)
                # The legacy validity array includes a geometric paint band.
                # Only finalized selectors license native-model supervision.
                pos = np.asarray(bt.sel_self, bool)
                negative = np.asarray(bt.sel_bg, bool)
                foreign = np.asarray(bt.sel_foreign, bool)
                case = {"id": s.mask_uuid, "revision": s.obs_revision, "kind": "owned_body",
                        "radius": s.target_r or 13, "source": s.provenance, "owner": s.owner_key,
                        "extent_scope_policy": bt.extent_scope_policy,
                        "unlicensed_band_pixels_excluded": int(((bt.valid > 0) & ~bt.licensed()).sum())}
            y, x = np.mgrid[:STORED, :STORED]
            real = (x+origin[0]>=0)&(x+origin[0]<gray.shape[1])&(y+origin[1]>=0)&(y+origin[1]<gray.shape[0])
            pos, negative, foreign = pos&real, negative&real, foreign&real
            case.update(movie=movie, frame=int(frame), grain=list(grain), origin=origin.tolist(),
                        image_size=[gray.shape[1], gray.shape[0]],
                        positive_pixels=int(pos.sum()), negative_pixels=int(negative.sum()),
                        foreign_pixels=int(foreign.sum()))
            for k, value in zip(arrays, (pixels, pos, negative, foreign)):
                arrays[k].append(value)
            cases.append(case)
    finally:
        for reader in readers.values():
            reader.close()
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    np.savez_compressed(out/"panel.npz", **arrays)
    doc = {"schema": "tubetracker.native_body_panel.v1", "snapshot": str(snap),
           "snapshot_sha256": file_hash(snap/"snapshot_manifest.json"),
           "dataset_sha256": file_hash(out/"panel.npz"), "cases": cases,
           "scope": "development fit on preserved native masks; unknown pixels have no loss",
           "excluded_spans": excluded_spans,
           "excluded_role_frames": excluded_roles,
           "supervision_contract": "finalized_self_reviewed_background_foreign_v1",
           "legacy_rectangle_policy": "intersection_of_recorded_and_swapped_canvas_dimensions",
           "grain_poses_plans": {str(_pp): file_hash(_pp) for _pp in (getattr(args, 'grain_poses', None) or [])},
           "event_grain_registry": ({str(_event_reg_path): file_hash(_event_reg_path)} if _event_reg_path and _event_reg_path.exists() else {}),
           "preparation_source": {f:file_hash(ROOT/f) for f in (
                'scripts/train_native_body.py', 'prototypes/v30_video_apex/span_supervision.py', 'prototypes/v30_video_apex/targets.py',
               'prototypes/v30_video_apex/batch_builder.py', 'tubetracker/review_region.py')},
           "query": "reviewed grain centre/radius; no full path or distal tip supplied"}
    dump(out/"panel.json", doc)
    render(cases, arrays, out/"input-target-sheet.png")
    print(json.dumps({"cases": len(cases), "body_masks": len(samples),
                      "foreign_pixels": sum(c["foreign_pixels"] for c in cases)}), flush=True)


def render(cases, arrays, path, predictions=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots((len(cases)+3)//4, 4, figsize=(12, 3*((len(cases)+3)//4)), squeeze=False)
    for i, case in enumerate(cases):
        ax = axes.flat[i]
        ax.imshow(arrays["pixels"][i], cmap="gray", vmin=0, vmax=255)
        overlay = np.zeros((*arrays["pixels"][i].shape, 4))
        overlay[arrays["ordinary"][i]] = [1,.5,0,.10]
        overlay[arrays["foreign"][i]] = [1,0,0,.35]
        overlay[arrays["positive"][i]] = [0,1,0,.4]
        ax.imshow(overlay)
        q = np.asarray(case["grain"])-case["origin"]
        ax.plot(*q, "o", color="magenta", markersize=4)
        if predictions is not None:
            ax.contour(np.pad(predictions[i], PAD), levels=[.5], colors=["cyan"], linewidths=.8)
        ax.set_title(f"{case['id']} / {case['frame']}", fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def evaluate(model, arrays, doc, render_path=None, roots=None):
    import torch
    predictions, rows = [], []
    for i, case in enumerate(doc["cases"]):
        gray = arrays["pixels"][i, PAD:PAD+TILE, PAD:PAD+TILE]
        origin = np.asarray(case["origin"])+PAD
        x = body_input(gray, origin, case["grain"], case["radius"], model.image_gain,
                       with_root=getattr(model, "root_channel", False),
                       root_xy=(roots or {}).get(i))
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(x)[None].to(next(model.parameters()).device)))[0,0].cpu().numpy()
        predictions.append(p)
        pos, ordinary, foreign = [arrays[k][i,PAD:PAD+TILE,PAD:PAD+TILE] for k in ("positive","ordinary","foreign")]
        valid = pos | ordinary | foreign
        pred = p >= .5
        iou = float((pred & pos).sum()/max(1, ((pred | pos)&valid).sum())) if pos.any() else None
        row = {"id": case["id"], "kind": case["kind"], "iou": iou,
            "recall": float((pred&pos).sum()/pos.sum()) if pos.any() else None,
            "ordinary_positive_rate": float(pred[ordinary].mean()) if ordinary.any() else None,
            "foreign_positive_rate": float(pred[foreign].mean()) if foreign.any() else None,
            "query_field_std": float(p.std())}
        if case.get("kind") == "owned_presence":
            # rev15 task 3: presence response in the emergence neighbourhood
            # (ring 1r..2.5r around the queried grain), measured — never trained.
            import numpy as _np
            _r = float(case.get("radius", 13))
            _qy, _qx = _np.mgrid[:p.shape[0], :p.shape[1]]
            _g = (_np.asarray(case["grain"], float)
                  - (_np.asarray(case["origin"], float) + PAD))
            _d = _np.hypot(_qx - _g[0], _qy - _g[1])
            _ring = (_d >= 1.0 * _r) & (_d <= 2.5 * _r)
            row["presence"] = case.get("presence")
            row["presence_role"] = case.get("presence_role")
            row["presence_response_max"] = float(p[_ring].max()) if _ring.any() else None
            row["presence_response_mean"] = float(p[_ring].mean()) if _ring.any() else None
            if getattr(model, "presence_head", False) and case.get("presence") in ("visible", "absent"):
                _xt = torch.from_numpy(x)[None].to(next(model.parameters()).device)
                _pm = torch.as_tensor(
                    presence_pool_mask(case, (PAD, PAD), x.shape[-1]),
                    dtype=torch.bool, device=_xt.device)
                row["presence_head_logit"] = float(
                    model.presence_logit(_xt, _pm)[0].detach().cpu())
        rows.append(row)
    body = [r for r in rows if r["kind"]=="owned_body"]
    absent = [r for r in rows if r["kind"]=="owned_absence"]
    presence = [r for r in rows if r["kind"]=="owned_presence"]
    _p_abs = [r for r in presence if r.get("presence") == "absent"]
    _p_vis = [r for r in presence if r.get("presence") == "visible"]
    result = {"scope": doc["scope"], "rows": rows,
        "mean_body_iou": float(np.mean([r["iou"] for r in body])) if body else None,
        "min_body_iou": min((r["iou"] for r in body), default=None),
        "min_body_recall": min((r["recall"] for r in body), default=None),
        "max_foreign_positive_rate": max((r["foreign_positive_rate"] for r in body if r["foreign_positive_rate"] is not None), default=None),
        "max_absence_region_positive_rate": max((r["ordinary_positive_rate"] for r in absent), default=None),
        "presence_cases": len(presence),
        "presence_absent_cases": len(_p_abs),
        "presence_visible_cases": len(_p_vis),
        "presence_false_presence_rate": (
            sum(1 for r in _p_abs if (r.get("presence_response_max") or 0) >= .5) / len(_p_abs)
            if _p_abs else None),
        "presence_visible_response_rate": (
            sum(1 for r in _p_vis if (r.get("presence_response_max") or 0) >= .5) / len(_p_vis)
            if _p_vis else None)}
    _p_head = [r for r in presence if "presence_head_logit" in r]
    if _p_head:
        _h_abs = [r for r in _p_head if r.get("presence") == "absent"]
        _h_vis = [r for r in _p_head if r.get("presence") == "visible"]
        result["presence_head_cases"] = len(_p_head)
        result["presence_head_accuracy"] = (
            sum(1 for r in _p_head
                if (r["presence_head_logit"] >= 0) == (r.get("presence") == "visible"))
            / len(_p_head))
        result["presence_head_false_presence_rate"] = (
            sum(1 for r in _h_abs if r["presence_head_logit"] >= 0) / len(_h_abs)
            if _h_abs else None)
        # Grouped by annotation role: held-out validation/test never trained.
        _TRAIN = ("training", "development")
        for _grp in ("train", "heldout"):
            _g = [r for r in _p_head
                  if ((r.get("presence_role") or "training") in _TRAIN) == (_grp == "train")]
            _ga = [r for r in _g if r.get("presence") == "absent"]
            result[f"presence_head_cases_{_grp}"] = len(_g)
            result[f"presence_head_accuracy_{_grp}"] = (
                sum(1 for r in _g
                    if (r["presence_head_logit"] >= 0) == (r.get("presence") == "visible"))
                / len(_g) if _g else None)
            result[f"presence_head_false_presence_rate_{_grp}"] = (
                sum(1 for r in _ga if r["presence_head_logit"] >= 0) / len(_ga)
                if _ga else None)
    result["fit_gate_passed"] = (result["min_body_iou"]>=.9 and result["min_body_recall"]>=.95
        and result["max_foreign_positive_rate"] is not None and result["max_foreign_positive_rate"]<=.05
        and result["max_absence_region_positive_rate"] is not None and result["max_absence_region_positive_rate"]<=.05)
    if render_path:
        render(doc["cases"], arrays, render_path, predictions)
    return result


def fit(args):
    import torch
    from prototypes.v30_video_apex.model_factory import parameter_hash
    panel, out = Path(args.panel), Path(args.out)
    doc = json.loads((panel/'panel.json').read_text())
    legacy_panel = doc.get('supervision_contract') != 'finalized_self_reviewed_background_foreign_v1'
    if legacy_panel and not getattr(args, 'allow_legacy_panel_control', False):
        raise ValueError('Body panel predates the reviewed-geometry/selector fix. Rebuild it with prepare; '
                         '--allow-legacy-panel-control is only for a recorded historical comparison.')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    pair_rng = np.random.default_rng(args.seed + 1)
    focus = bool(getattr(args, 'owner_focus', False))
    focus_distance = float(getattr(args, 'owner_focus_distance', 80.))
    foreign_weight = float(getattr(args, 'owner_foreign_weight', 0.))
    contrast = args.owner_margin > 0 or foreign_weight > 0
    owner_supervision = getattr(args, 'owner_supervision', 'legacy')
    canonical = bool(getattr(args, 'canonical_views', False))
    canonical_replay = bool(getattr(args, 'canonical_replay', False))
    batch_size = int(getattr(args, 'batch_size', 4))
    hard_negative_ratio = float(getattr(args, 'hard_negative_ratio', 3.))
    if owner_supervision != 'legacy' and (contrast or focus):
        raise ValueError('explicit same-crop samples cannot be combined with legacy margin/focus sampling')
    if canonical_replay and (canonical or owner_supervision == 'legacy'):
        raise ValueError('canonical replay requires augmented explicit same-crop samples')
    if getattr(args, 'presence_head', False) and owner_supervision != 'legacy':
        raise ValueError('presence-head sampling requires legacy minibatch rotation')
    if batch_size < 1 or not np.isfinite(hard_negative_ratio) or hard_negative_ratio <= 0:
        raise ValueError('batch size and hard-negative ratio must be positive')
    if min(args.owner_margin, args.owner_weight, foreign_weight) < 0 or focus_distance <= 0:
        raise ValueError('Owner loss weights must be nonnegative; focus distance must be positive')
    out.mkdir(parents=True, exist_ok=False)
    if file_hash(panel/"panel.npz") != doc["dataset_sha256"]:
        raise ValueError("body panel changed after preparation")
    arrays = dict(np.load(panel/"panel.npz"))
    initial = None
    if args.init:
        initial, _ = load_body_checkpoint(args.init)
    if args.image_gain is None:
        # A fit must never silently use a different input gain than the
        # checkpoint it continues (rev14: the CLI default 1.0 vs the
        # pinned 8.0 collapsed every evaluation). Inherit the
        # checkpoint gain unless the caller explicitly overrides it;
        # the effective value is recorded in the manifest either way.
        args.image_gain = float(getattr(initial, "image_gain", 1.0)) if initial else 1.0
    query_adapters = getattr(args, 'query_adapters', None)
    if query_adapters is None:
        query_adapters = bool(getattr(initial, 'query_adapters', False))
    root_channel = bool(getattr(args, 'root_channel', False) or getattr(initial, 'root_channel', False))
    presence_head = bool(getattr(args, 'presence_head', False) or getattr(initial, 'presence_head', False))
    presence_weight = float(getattr(args, 'presence_weight', 1.0))
    if presence_weight < 0 or not np.isfinite(presence_weight):
        raise ValueError('presence weight must be finite/nonnegative')
    presence_lr = float(getattr(args, 'presence_lr', 1e-3))
    freeze_trunk = bool(getattr(args, 'freeze_trunk', False)) and presence_head
    if not np.isfinite(presence_lr) or presence_lr <= 0:
        raise ValueError('presence lr must be positive and finite')
    model = NativeBodyNet(base=initial.base if initial else 12,
                          image_gain=args.image_gain, normalization=args.normalization,
                          root_channel=root_channel, query_adapters=query_adapters,
                          presence_head=presence_head)
    normalization_transfer = None
    root_init_note = None
    if initial is not None:
        if initial.normalization != model.normalization:
            if not args.allow_normalization_transfer:
                raise ValueError('changing normalization at initialization requires --allow-normalization-transfer')
            normalization_transfer = {'from': initial.normalization, 'to': model.normalization,
                                      'method': 'same initial tensors, new normalization operator; retraining required'}
        root_init_note = transfer_body_parameters(model, initial)
    elif args.allow_normalization_transfer:
        raise ValueError('--allow-normalization-transfer requires --init')
    # rev14 P5 fallback: genuinely known roots only.
    #   registry attachments marked verified, plus reviewed FULL exits.
    #   Everything else is an explicit unknown (zero channel).
    case_roots, root_sources = {}, {}
    if root_channel:
        case_roots, root_sources = resolve_case_roots(doc, doc.get('snapshot'), args.owners_registry)
    # rev14 P5: paired owner queries for the same image with already
    # licensed, DISJOINT owned pixels. Off unless --owner-margin > 0.
    # Overlapping or ambiguous pixels never become foreign negatives.
    owner_pairs, pair_eligible, pair_report = {}, {}, []
    if contrast or focus:
        owner_pairs, pair_eligible, pair_report = body_owner_pairs(doc, arrays)
    focused_pairs = []
    if focus:
        owner_pairs = local_owner_pairs(doc, owner_pairs, focus_distance)
        focused_pairs = sorted((i, j) for i, js in owner_pairs.items() for j in js
                               if i < j and i in owner_pairs.get(j, []))
        if not focused_pairs:
            raise ValueError('Owner focus requires a reviewed pair whose queries fit the same crop')
        allowed = {(doc['cases'][i]['id'], doc['cases'][j]['id']) for i, j in focused_pairs}
        pair_report = [p for p in pair_report if (p['a'], p['b']) in allowed]
    shared_samples, shared_audit = [], []
    if owner_supervision != 'legacy':
        shared_samples, shared_audit = same_crop_owner_samples(doc, arrays, owner_supervision, focus_distance)
    inputs = {"panel": str(panel.resolve()), "panel_sha256": file_hash(panel/"panel.json"), "dataset_sha256": doc["dataset_sha256"],
              "initial_parameter_hash": parameter_hash(model), "model": model.config(),
              "init_checkpoint_sha256": file_hash(args.init) if args.init else None,
              "normalization_transfer": normalization_transfer,
              'supervision_contract': doc.get('supervision_contract', 'legacy_unverified_domains'),
              'legacy_panel_control': legacy_panel,
              "root_channel": ({"enabled": True, "roots": root_sources,
                  "rule": "verified attachment or reviewed FULL exit; unknown root is a zero channel"}
                  if root_channel else None),
              "presence_head": ({"enabled": True, "weight": presence_weight,
                  "lr": presence_lr, "freeze_trunk": freeze_trunk,
                  "pooling": "emergence ring 1r..2.5r (query geometry only)",
                  "rule": "scalar owned-presence verdicts (visible=1/absent=0, uncertain excluded) train the head only; pixel loss on presence cases stays zero"}
                  if presence_head else None),
              "parameter_transfer": root_init_note,
              "root_init": root_init_note if root_channel else None,
              "same_crop_supervision": ({'mode': owner_supervision, 'sample_count': len(shared_samples),
                  'pairs': shared_audit, 'schedule': 'identical deterministic minibatch rotation over primary then partner samples'}
                  if shared_samples else None),
              "owner_discrimination": ({"margin": args.owner_margin,
                   "weight": args.owner_weight, "pairs": pair_report,
                   "foreign_weight": foreign_weight,
                   "rule": "relative margin plus absolute wrong-owner BCE on licensed, disjoint pixels"}
                   if contrast else None),
              "training": {"seed": args.seed, "updates": args.updates, "lr": args.lr,
                           "hard_negative_weight": args.hard_negative_weight,
                           "hard_negative_ratio": hard_negative_ratio,
                           "optimizer": getattr(args, 'optimizer', 'adamw'),
                           "device": args.device,
                           "forward_batching": "batched native queries",
                           "batch_size": batch_size if shared_samples else 'legacy sampling',
                           "canonical_replay": {"enabled": canonical_replay,
                                "views_per_sample": 2 if canonical_replay else 1,
                                "rule": "equal loss weight for canonical and augmented views of each sampled query"},
                           "stop_on_fit_gate": args.stop_on_fit_gate,
                           "augmentation": "canonical views only" if canonical else "native crop translations, quarter turns and flips",
                            "loss": "per-query balanced positive/ordinary/foreign BCE plus masked Dice",
                            "owner_focus": {"enabled": focus, "max_distance_px": focus_distance,
                                "pairs": pair_report if focus else [],
                                "schedule": "both members of each local pair in rotation plus one corpus/absence replay"}
                                if focus else None},
              "source": {f:file_hash(ROOT/f) for f in [
                  "scripts/train_native_body.py", "prototypes/v30_video_apex/span_supervision.py", "prototypes/v30_video_apex/native_body.py",
                  "prototypes/v30_video_apex/native_caps.py",
                  "prototypes/v30_video_apex/batch_builder.py", "prototypes/v30_video_apex/targets.py",
                  'tubetracker/review_region.py']}}
    save_body_checkpoint(out/"init.pt", model, inputs)
    def require_unchanged_inputs():
        changed = [f for f, digest in inputs['source'].items() if file_hash(ROOT/f) != digest]
        if (file_hash(panel/'panel.json') != inputs['panel_sha256']
                or file_hash(panel/'panel.npz') != inputs['dataset_sha256']):
            changed.append(str(panel))
        if changed:
            raise RuntimeError(f'body training inputs changed during the run: {changed}')
    model.to(args.device).train()
    if presence_head and not freeze_trunk:
        # Trunk and head learn at their own rates: the dense pixel path
        # must not be dragged by the scalar presence objective.
        _trunk_params = [p for n, p in model.named_parameters()
                         if not n.startswith('presence_fc.')]
        _head_params = [p for n, p in model.named_parameters()
                        if n.startswith('presence_fc.')]
        _opt_params = [{"params": _trunk_params, "lr": args.lr},
                       {"params": _head_params, "lr": presence_lr}]
    elif freeze_trunk:
        _opt_params = [{"params": [p for n, p in model.named_parameters()
                                   if n.startswith('presence_fc.')],
                        "lr": presence_lr}]
    else:
        _opt_params = model.parameters()
    optimizer = (torch.optim.Adam(_opt_params, lr=args.lr)
                 if getattr(args, 'optimizer', 'adamw') == 'adam'
                 else torch.optim.AdamW(_opt_params, lr=args.lr, weight_decay=1e-4))
    body = [i for i,c in enumerate(doc["cases"])
            if c["kind"] in ("owned_body", "owned_body_scoped_span")]
    absence = [i for i,c in enumerate(doc["cases"]) if c["kind"]=="owned_absence"]
    clump = [i for i,c in enumerate(doc["cases"]) if c["id"].startswith("mask-rev8p")]
    # rev15 task 3: scalar presence supervision. Uncertain verdicts are
    # measured only, never trained. Validation/test roles are held out
    # from scalar training too (audit: 7 validation records were admitted)
    # — they stay in eval, grouped by role.
    _TRAIN_ROLES = ("training", "development")
    presence_train = [i for i,c in enumerate(doc["cases"])
                      if c["kind"]=="owned_presence" and c.get("presence") in ("visible", "absent")
                      and c.get("presence_role", "training") in _TRAIN_ROLES]
    presence_heldout = [i for i,c in enumerate(doc["cases"])
                        if c["kind"]=="owned_presence" and c.get("presence") in ("visible", "absent")
                        and c.get("presence_role", "training") not in _TRAIN_ROLES]
    presence = presence_train
    presence_labels = {i: 1.0 if doc["cases"][i].get("presence")=="visible" else 0.0
                       for i in presence_train}
    if presence_head and not presence_train:
        raise ValueError('presence head requires supervised presence cases (visible/absent)')
    # Exposure audit: every training pick is logged by record id/role;
    # validation/test targets must never be sampled for training.
    exposures = {}
    def _log_exposure(case_index, kind):
        case = doc["cases"][case_index]
        key = (case.get("id"), kind)
        rec = exposures.setdefault(key, {"id": case.get("id"), "kind": kind,
            "role": case.get("presence_role", case.get("annotation_role", "training")),
            "revision": case.get("revision", case.get("obs_revision")),
            "source": case.get("source"), "picks": 0})
        rec["picks"] += 1
    history, started, best_rank, best = [], time.monotonic(), None, None
    pair_usage = {}
    for step in range(1,args.updates+1):
        focus_partner = {}
        presence_pick = []
        if presence_head and presence:
            # One presence sample per step, alternating visible/absent when
            # both exist so the tiny absent set is not drowned.
            if step % 2 == 0:
                _vis = [i for i in presence if presence_labels[i] == 1.0]
                _abs = [i for i in presence if presence_labels[i] == 0.0]
                pool = _vis if (step // 2) % 2 == 0 and _vis else (_abs or presence)
                if not pool:
                    pool = presence
                presence_pick = [int(rng.choice(pool))]
            else:
                presence_pick = [int(rng.choice(presence))]
        if shared_samples:
            indices = [(step*batch_size+i) % len(shared_samples) for i in range(batch_size)]
        elif focus:
            i, j = focused_pairs[(step-1) % len(focused_pairs)]
            replay = absence if step % 3 == 0 and absence else body
            indices = [i, j, int(rng.choice(replay))]
            focus_partner = {i: j, j: i}
        else:
            indices = clump if step%4==0 else [int(rng.choice(body)),int(rng.choice(body if step%2 else absence))]
        indices = list(indices) + [i for i in presence_pick if i not in indices]
        presence_targets = {i: presence_labels[i] for i in presence_pick}
        n_base = len(indices) - len([i for i in presence_pick if i in indices])
        for pos in indices[:n_base]:
            _log_exposure(shared_samples[pos]["image_case"] if shared_samples else pos,
                          "pixel")
        for i in presence_pick:
            _log_exposure(i, "scalar")
        xs, ys, partners = [], [], []
        presence_meta = []
        for sample_index in indices:
            record = shared_samples[sample_index] if shared_samples else None
            i = record['image_case'] if record else sample_index
            query_index = record['query_case'] if record else i
            case = doc["cases"][i]
            query_case = doc['cases'][query_index]
            ox, oy = (PAD, PAD) if canonical else rng.integers(0,2*PAD+1,2)
            partner = focus_partner.get(i)
            if partner is None and i in owner_pairs:
                partner = int(pair_rng.choice(owner_pairs[i]))
            if focus and partner is not None:
                q = np.asarray([case['grain'], doc['cases'][partner]['grain']]) - case['origin']
                lower = np.maximum(0, np.ceil(q.max(0) - (TILE-1))).astype(int)
                upper = np.minimum(2*PAD, np.floor(q.min(0))).astype(int)
                ox, oy = rng.integers(lower, upper+1)
            sample = {k:v[i,oy:oy+TILE,ox:ox+TILE].copy() for k,v in arrays.items()}
            if record:
                sample.update({k:v[oy:oy+TILE,ox:ox+TILE].copy() for k,v in record['targets'].items()})
            if partner is not None:
                sample["eligible"] = pair_eligible[i, partner][oy:oy+TILE, ox:ox+TILE].copy()
            grain = np.asarray(query_case["grain"])-np.asarray(case["origin"])-[ox,oy]
            _root = case_roots.get(query_index)
            root = (None if _root is None
                    else np.asarray(_root)-np.asarray(case["origin"])-[ox,oy])
            turns = 0 if canonical else int(rng.integers(4))
            for _ in range(turns):
                sample = {k:np.rot90(v).copy() for k,v in sample.items()}
                grain = np.array([grain[1],TILE-1-grain[0]])
                if root is not None:
                    root = np.array([root[1],TILE-1-root[0]])
            flipped = False if canonical else rng.random()<.5
            if flipped:
                sample = {k:v[:,::-1].copy() for k,v in sample.items()}
                grain[0] = TILE-1-grain[0]
                if root is not None:
                    root[0] = TILE-1-root[0]
            xs.append(body_input(sample["pixels"], [0,0], grain, query_case["radius"], model.image_gain,
                                 with_root=getattr(model, "root_channel", False), root_xy=root))
            ys.append([sample[k] for k in ("positive","ordinary","foreign")])
            if sample_index in presence_targets:
                # Ring-pool mask via the same crop transform as the input
                # pixels: WHERE to look, never what is there.
                presence_meta.append({
                    "batch": len(xs) - 1,
                    "mask": presence_pool_mask(
                        {"grain": query_case["grain"], "origin": case["origin"],
                         "radius": query_case.get("radius", 13)},
                        (ox, oy), TILE, turns, flipped)})
            if canonical_replay:
                # Retain the original reviewed view while learning geometric
                # transforms. Reuse only its already licensed selectors.
                gray = arrays['pixels'][i, PAD:PAD+TILE, PAD:PAD+TILE]
                xs.append(body_input(gray, np.asarray(case['origin'])+PAD,
                    query_case['grain'], query_case['radius'], model.image_gain,
                    with_root=model.root_channel, root_xy=case_roots.get(query_index)))
                ys.append([record['targets'][k][PAD:PAD+TILE, PAD:PAD+TILE]
                           for k in ('positive','ordinary','foreign')])
            # rev14 P5: the partner's query on the SAME transformed crop.
            if partner is not None and contrast:
                pcase = doc["cases"][partner]
                pgrain = query_in_crop(pcase['grain'], case['origin'], [ox, oy], turns, flipped)
                proot = query_in_crop(case_roots.get(partner), case['origin'], [ox, oy], turns, flipped)
                partners.append((body_input(sample["pixels"], [0,0], pgrain,
                                            pcase["radius"], model.image_gain,
                                            with_root=getattr(model, "root_channel", False),
                                            root_xy=proot),
                                 sample.get("eligible", sample["positive"]).astype(bool),
                                  len(xs)-1))
                key = case['id'] + ' -> ' + pcase['id']
                usage = pair_usage.setdefault(key, {'queries': 0, 'eligible_pixels': 0,
                                                    'both_queries_in_crop': 0})
                usage['queries'] += 1
                usage['eligible_pixels'] += int(sample['eligible'].sum())
                usage['both_queries_in_crop'] += int(
                    (grain >= 0).all() and (grain < TILE).all()
                    and (pgrain >= 0).all() and (pgrain < TILE).all())
        optimizer.zero_grad(set_to_none=True)
        logits = model(torch.from_numpy(np.stack(xs)).to(args.device))
        losses, terms = [], []
        for i,y in enumerate(ys):
            loss, term = body_loss(logits[i:i+1], *[torch.from_numpy(v)[None,None].to(args.device) for v in y],
                                   hard_negative_weight=args.hard_negative_weight,
                                   hard_negative_ratio=hard_negative_ratio)
            losses.append(loss); terms.append(term)
        loss = torch.stack(losses).mean()
        presence_term = None
        if presence_head and presence_targets:
            # rev15 task 3: scalar presence BCE on the head only. Pixel
            # masks on presence cases are all-zero, so body_loss on them
            # contributes nothing; only this term trains the head. Each
            # sample is pooled over its own transformed emergence ring.
            xb = torch.from_numpy(np.stack(xs)).to(args.device)
            pt, pl = [], []
            for meta in presence_meta:
                pm = torch.as_tensor(meta["mask"], dtype=torch.bool,
                                     device=args.device)
                pt.append(model.presence_logit(xb[meta["batch"]:meta["batch"] + 1], pm))
                pl.append(presence_targets[indices[meta["batch"]]])
            if pt:
                presence_term = torch.nn.functional.binary_cross_entropy_with_logits(
                    torch.stack(pt).squeeze(1), torch.tensor(pl, device=args.device))
                loss = loss + presence_weight * presence_term
        owner_term = owner_foreign_term = None
        if contrast and partners:
            # Masked margin on correct-owner vs foreign-owner logits at the
            # correct owner's licensed pixels (never at overlap or unknown).
            xp = torch.from_numpy(np.stack([p[0] for p in partners])).to(args.device)
            logits_p = model(xp)[:, 0]
            margins, foreign_terms = [], []
            for k, (_, pos, batch_i) in enumerate(partners):
                pos_t = torch.from_numpy(pos).to(args.device)
                relative, foreign = owner_query_loss(logits[batch_i][0], logits_p[k], pos_t,
                    margin=args.owner_margin, margin_weight=args.owner_weight, foreign_weight=foreign_weight)
                margins.append(relative); foreign_terms.append(foreign)
            owner_term, owner_foreign_term = torch.stack(margins).mean(), torch.stack(foreign_terms).mean()
            loss = loss + owner_term + owner_foreign_term
        if not torch.isfinite(loss):
            raise ValueError("nonfinite body loss")
        loss.backward()
        gradient = float(model.body.weight.grad.norm())
        optimizer.step()
        if step==1 or step%args.block==0 or step==args.updates:
            require_unchanged_inputs()
            model.eval()
            result = evaluate(model, arrays, doc, roots=case_roots)
            model.train()
            ckpt = out/f"step-{step:04d}.pt"
            save_body_checkpoint(ckpt, model, {"inputs":inputs,"update":step})
            dump(out/f"step-{step:04d}.eval.json",result)
            rank = (result["fit_gate_passed"], result["min_body_iou"], result["mean_body_iou"],
                    -result["max_absence_region_positive_rate"])
            if best_rank is None or rank>best_rank:
                best_rank,best=rank,str(ckpt)
            row={"update":step,"loss":float(loss.detach()),"terms":terms,"body_gradient":gradient,
                  "presence_loss": float(presence_term.detach()) if presence_term is not None else None,
                  "owner_margin_loss": float(owner_term.detach()) if owner_term is not None else None,
                  "owner_foreign_loss": float(owner_foreign_term.detach()) if owner_foreign_term is not None else None,
                 **{k:v for k,v in result.items() if k not in ("rows","scope")},"elapsed_s":time.monotonic()-started}
            history.append(row)
            print(json.dumps(row),flush=True)
            dump(out/"run.json",{"manifest_schema":"tubetracker.run.v1","inputs":inputs,
                                "results":{"history":history,"best_checkpoint":best,
                                            "actual_updates":step, "owner_pair_usage": pair_usage}})
            if args.stop_on_fit_gate and result["fit_gate_passed"]:
                break
    model,_=load_body_checkpoint(best,args.device)
    result=evaluate(model,arrays,doc,render_path=out/"predictions.png",roots=case_roots)
    result["checkpoint"]=best
    # Exposure audit: exact per-record training picks; validation/test
    # targets must have zero training exposure of any kind.
    exposure_rows = sorted(exposures.values(), key=lambda r: (-r["picks"], r["id"] or ""))
    leaked = [r for r in exposure_rows if r["role"] in ("validation", "test")]
    if leaked:
        raise ValueError("validation/test training exposure: "
                         + ", ".join(f"{r['id']}/{r['kind']}x{r['picks']}" for r in leaked))
    result["training_exposures"] = exposure_rows
    result["training_exposure_leak_count"] = 0
    result["presence_heldout_cases"] = len(presence_heldout)
    dump(out/"exposures.json", {"exposures": exposure_rows,
        "presence_train_cases": len(presence_train),
        "presence_heldout_cases": len(presence_heldout)})
    dump(out/"evaluation.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}),flush=True)


def resolve_case_roots(doc, snapshot, registry_path=None):
    """Genuinely known roots only: verified registry attachments + reviewed
    FULL exits. Reserved validation routes never contribute (their geometry
    stays held out), and every unmatched case is an explicit unknown (zero
    channel) -- no geometry is ever invented for an unlicensed root."""
    import numpy as _np
    RESERVED = {"obs-rev14-route-001", "obs-rev14-route-002", "obs-rev14-route-003"}
    registry = []
    reg_path = Path(registry_path) if registry_path else None
    if reg_path and reg_path.exists():
        owners = json.loads(reg_path.read_text())
        for o in (owners if isinstance(owners, list) else owners.get('owners', [])):
            if o.get('attachment_verified') is False:
                continue
            if o.get('root_review_required') and not o.get('attachment_verified'):
                continue
            if o.get('attachment_native') and o.get('grain_native'):
                registry.append({'id': o['id'], 'movie': o.get('movie'),
                                 'grain': list(map(float, o['grain_native'])),
                                 'attachment': list(map(float, o['attachment_native'])),
                                 'basis': 'verified attachment (' + str(o.get('source_task') or 'registry') + ')'})
    exits = {}
    snap = Path(snapshot) if snapshot else None
    obs_path = (snap/'observations.json') if snap else None
    if obs_path and obs_path.exists():
        for o in json.loads(obs_path.read_text()):
            pts = o.get('path_xy') or []
            if (o.get('path_complete') and len(pts) >= 2
                    and o.get('obs_uuid') not in RESERVED
                    and o.get('review_origin') != 'workflow_test'):
                exits.setdefault(o.get('owner_uuid'), list(map(float, pts[0])))
    case_roots, root_sources = {}, {}
    for i, case in enumerate(doc['cases']):
        owner = case.get('owner') or ''
        root, why = None, None
        for entry in registry:
            if entry['movie'] and case.get('movie') and entry['movie'] != case['movie']:
                continue
            if entry['grain'] and case.get('grain') and \
                    _np.hypot(entry['grain'][0]-case['grain'][0], entry['grain'][1]-case['grain'][1]) <= 2.0:
                root, why = entry['attachment'], entry['basis']
                break
        if root is None and owner in exits:
            root, why = exits[owner], 'reviewed FULL exit'
        case_roots[i] = root
        root_sources[case['id']] = why if root is not None else 'unknown root (zero channel)'
    return case_roots, root_sources


def run_eval(args):
    """Frozen evaluation: checkpoint load, eval mode, zero optimizer updates.

    Records the exact hashes of the checkpoint, panel and dataset so a
    reported number can never be detached from what produced it.
    """
    import torch
    from prototypes.v30_video_apex.model_factory import parameter_hash
    if not args.checkpoint:
        raise ValueError("eval mode requires --checkpoint")
    panel, out = Path(args.panel), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    doc = json.loads((panel/'panel.json').read_text())
    if file_hash(panel/"panel.npz") != doc["dataset_sha256"]:
        raise ValueError("body panel changed after preparation")
    arrays = dict(np.load(panel/"panel.npz"))
    model, _ = load_body_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)
    roots, root_sources = None, None
    if getattr(model, "root_channel", False):
        roots, root_sources = resolve_case_roots(doc, doc.get("snapshot"))
    result = evaluate(model, arrays, doc, render_path=out/"predictions.png", roots=roots)
    result["root_sources"] = root_sources
    result.update({"schema": "tubetracker.native_body_eval.v1",
                   "checkpoint": str(Path(args.checkpoint).resolve()),
                   "checkpoint_sha256": file_hash(args.checkpoint),
                   "panel": str(panel.resolve()),
                   "panel_sha256": file_hash(panel/"panel.json"),
                   "dataset_sha256": doc["dataset_sha256"],
                   "parameter_hash": parameter_hash(model),
                   "device": args.device, "updates": 0,
                   "note": "frozen evaluation: checkpoint load, eval mode, zero optimizer updates"})
    dump(out/"evaluation.json", result)
    print(json.dumps({k: result[k] for k in ("mean_body_iou", "min_body_iou",
        "min_body_recall", "fit_gate_passed", "checkpoint_sha256")}, indent=1))
    return 0


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","fit","eval"))
    p.add_argument("--snapshot",default="runs/prototypes/v30/snap30_rev14")
    p.add_argument("--panel",default="runs/prototypes/v30/rev14_review_geometry/native-panel")
    p.add_argument("--out",required=True)
    p.add_argument("--updates",type=int,default=800)
    p.add_argument("--block",type=int,default=100)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--seed",type=int,default=53)
    p.add_argument("--device",default="cpu",choices=("cpu","mps","cuda"))
    p.add_argument("--init")
    p.add_argument("--checkpoint", help="eval mode: frozen checkpoint to evaluate")
    p.add_argument("--image-gain", type=float, default=None,
                   help="input contrast gain; default: inherit from --init (recorded)")
    p.add_argument("--normalization", choices=("spatial", "pixel"), default="spatial")
    p.add_argument("--allow-normalization-transfer", action="store_true",
                   help="Explicitly reuse initial tensors across normalization operators for a recorded training experiment")
    p.add_argument('--allow-legacy-panel-control', action='store_true',
                   help='Use a historical panel only as an explicitly recorded comparison')
    p.add_argument("--hard-negative-weight", type=float, default=0.)
    p.add_argument("--hard-negative-ratio", type=float, default=3.,
                   help="Reviewed background top-k size per positive pixel; minimum 64, default preserves prior loss")
    p.add_argument("--query-adapters", action=argparse.BooleanOptionalAction, default=None,
                   help="Add inert decoder query adapters, or inherit them from --init")
    p.add_argument("--owner-supervision", choices=("legacy", "negative_only", "symmetric_positive"), default="legacy",
                   help="Explicit same-crop query samples; symmetric mode includes each owner's licensed positive targets")
    p.add_argument("--canonical-views", action="store_true", help="Disable image translations/rotations/flips for a bounded fit diagnostic")
    p.add_argument("--canonical-replay", action="store_true", help="Pair each augmented query with its canonical reviewed view to prevent forgetting")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size for explicit same-crop samples")
    p.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    p.add_argument("--owner-margin", type=float, default=0.,
                   help="rev14 P5: margin for paired owner-discrimination queries (0 disables)")
    p.add_argument("--owner-weight", type=float, default=1.)
    p.add_argument("--owner-foreign-weight", type=float, default=0.,
                   help="Absolute wrong-owner BCE on disjoint reviewed owned pixels")
    p.add_argument("--owner-focus", action="store_true",
                   help="Balanced local-pair batches plus corpus and absence replay; also valid for a no-contrast control")
    p.add_argument("--owner-focus-distance", type=float, default=80.)
    p.add_argument("--root-channel", action="store_true",
                   help="rev14 P5 fallback: prompt the root/exit as an extra input channel")
    p.add_argument("--owners-registry", default="runs/prototypes/v30/rev15_execution_20260920T061445Z/ownership-correction-20260920T174154Z/owners-corrected.json",
                   help="verified attachment registry for genuinely known roots (rev15 corrected; rev11 registry obsolete for 0002/0003 roots)")
    p.add_argument("--grain-poses", action="append", default=[],
                   help="runtime grain-motion poses.json plan for frame-specific presence queries (repeatable); missing frames fall back to the registry centre with provenance")
    p.add_argument("--event-grain-registry", default="",
                   help="JSON mapping owner_uuid -> {grain_native, grain_radius_px, source} for germination-event bracket queries missing from the owners registry (e.g. survey-verified G0); events without resolved geometry are skipped")
    p.add_argument("--presence-head", action="store_true",
                   help="rev15 task 3: train a scalar owned-presence head on review_tip verdicts (pixel loss stays zero)")
    p.add_argument("--presence-weight", type=float, default=1.0,
                   help="BCE weight for the presence head")
    p.add_argument("--presence-lr", type=float, default=1e-3,
                   help="learning rate for the presence head parameters")
    p.add_argument("--freeze-trunk", action="store_true",
                   help="train only the presence head; the dense pixel trunk stays exactly at init")
    p.add_argument("--stop-on-fit-gate", action="store_true")
    a=p.parse_args()
    if a.mode=="prepare":
        return prepare(a)
    if a.mode=="eval":
        return run_eval(a)
    return fit(a)


if __name__=="__main__":
    main()
