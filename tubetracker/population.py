"""Grain-specific evidence and population reports shared by analysis and census."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from prototypes.v30_video_apex.census import (
    CensusScope, CensusTile, GrainRecord, build_census, emergence_intervals,
    germination_report, merge_grain_records)
from .grain_registry import canonical_id
from .review_semantics import is_withdrawn, is_workflow_record, is_workflow_measurement, precise_tip


def snapshot_data(directory):
    return {name: json.loads((Path(directory) / (name + ".json")).read_text())
            if (Path(directory) / (name + ".json")).exists() else []
            for name in ("observations", "regions", "body_masks", "tubes", "census", "rulings")}


def census_tiles(records):
    tiles = []
    for r in records:
        box = r.get("tile_xywh") or r.get("box_xywh") or r.get("box")
        if not box or len(box) != 4 or min(box[2:]) <= 0:
            continue
        tiles.append(CensusTile(
            str(r.get("task_uuid") or r.get("tile_id")), str(r.get("movie", "")),
            int(r.get("source_frame", r.get("frame", -1))), tuple(box),
            tuple(r.get("class_scopes") or ()),
            bool(r.get("complete") and not r.get("tile_derived", False)),
            int(r.get("task_revision", r.get("revision", 0))),
            r.get("border_policy", ""), r.get("clump_policy", "")))
    return tiles


def census_review_records(entities, snapshot=""):
    records = snapshot_data(snapshot)["census"] if snapshot else []
    initial = len(records)
    for entity in entities:
        task = entity["data"]
        if entity["kind"] != "task" or task.get("task_type") != "census":
            continue
        region = task.get("review_region") or []
        if len(region) != 4:
            continue
        x0, y0 = np.min(region, axis=0); x1, y1 = np.max(region, axis=0)
        if set(map(tuple, region)) != {(x0, y0), (x1, y0), (x1, y1), (x0, y1)}:
            continue
        records.append({"task_uuid": entity["uuid"], "task_revision": entity["revision"],
            "movie": task.get("movie"), "source_frame": task["query_frames"][0],
            "tile_xywh": [float(x0), float(y0), float(x1-x0), float(y1-y0)],
            "tile_derived": False, "instances": copy.deepcopy(task.get("census_instances", [])),
            "class_scopes": task.get("census_class_scopes", []),
            "border_policy": task.get("census_border_policy", ""),
            "clump_policy": task.get("census_clump_policy", ""),
            "membership_resolved": bool(task.get("census_membership_resolved")),
            "complete": bool(task.get("census_complete")),
            "review_status": 'withdrawn' if is_withdrawn(task) else 'active',
            "review_origin": "workflow_test" if is_workflow_record(task) else "human"})
    live_ids = {(e["data"].get("movie"), e["uuid"]) for e in entities
                if e["kind"] == "task" and e["data"].get("task_type") == "census"}
    # A current unfinished revision supersedes an earlier exhaustive snapshot revision.
    return [r for i, r in enumerate(records) if not is_withdrawn(r)
            and (i >= initial or (r.get("movie"), r.get("task_uuid")) not in live_ids)]


def apply_reviewed_inventory(owners, census, *, movie, frames=None,
                             roi_xyxy=None, selected_owner_ids=None):
    """Resolve identity anchors separately from the requested measurements.

    ``frames`` is retained for callers of the original API, but never filters
    identity evidence. An offline run can use a reviewed census anchor outside
    its measurement schedule. Explicit selection addresses physical IDs; the
    default field selection uses reference grain centres. Neither mutates the
    persistent registry or licenses a complete population denominator.
    """
    inventory = {o["id"]: copy.deepcopy(o) for o in owners}
    applied, removed, excluded = [], [], []
    eligible = []
    for r in census:
        if (r.get("movie") != movie or is_withdrawn(r)
                or not r.get("complete") or not r.get("membership_resolved")
                or "grains" not in r.get("class_scopes", []) or r.get("tile_derived")):
            continue
        if (r.get("border_policy") != "centre-inside"
                or r.get("clump_policy") != "individual-physical-grains"):
            continue
        instances = r.get("instances", [])
        if any(not item.get("confirmed") for item in instances):
            continue
        eligible.append(r)
    # A copied analysis owner must not keep a withdrawn census anchor alive.
    # Newer active revisions are applied below and invalidate old pose files.
    active_ids = {r['task_uuid'] for r in eligible}
    for owner in owners:
        source = owner.get('census_source') or {}
        if source.get('task') and source['task'] not in active_ids:
            raise ValueError('grain census anchor is unavailable, withdrawn or incomplete; '
                             'refresh the reviewed inventory before regenerating motion')
    for r in sorted(eligible, key=lambda v: (v['source_frame'], v['task_uuid'], v['task_revision'])):
        instances = r.get('instances', [])
        x, y, w, h = r["tile_xywh"]
        identifiers = [g["grain_id"] for g in instances]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("census contains a duplicate physical grain ID")
        for item in instances:
            px, py = item["xy"]
            if not (x <= px < x+w and y <= py < y+h):
                raise ValueError("census grain centre lies outside its reviewed rectangle")
        for gid, owner in list(inventory.items()):
            px, py = owner["grain_native"]
            # A missing grain in a different frame may have moved out of view.
            # Only replace an unverified proposal from this census frame.
            same_frame = owner.get('identified_at_frame', owner.get('first_seen_frame')) == r['source_frame']
            if (same_frame and not owner.get('identity_verified') and
                    x <= px < x+w and y <= py < y+h and gid not in identifiers):
                removed.append({"grain_id": gid, "source": r["task_uuid"], "revision": r["task_revision"]})
                del inventory[gid]
        for item in instances:
            gid = item["grain_id"]
            owner = inventory.get(gid, {"id": gid, "movie": movie, "grain_radius_px": 13})
            if np.linalg.norm(np.asarray(owner.get("grain_native", item["xy"]))-item["xy"]) > 1:
                owner["attachment_verified"] = False
            owner.update(grain_native=item["xy"], identity_verified=not is_workflow_record(r),
                         identified_at_frame=r["source_frame"], census_source={
                             "task": r["task_uuid"], "revision": r["task_revision"],
                             "review_origin": r.get("review_origin", "human")})
            inventory[gid] = owner
        applied.append({"task": r["task_uuid"], "revision": r["task_revision"],
                        "review_origin": r.get("review_origin", "human")})
    known_ids = sorted(inventory)
    if selected_owner_ids is not None:
        unknown = set(selected_owner_ids) - set(inventory)
        if unknown:
            raise ValueError('selected physical grain IDs are not in the current inventory: '
                             + ', '.join(sorted(unknown)))
        wanted = set(selected_owner_ids)
        basis = 'explicit_physical_owner_ids'
    elif roi_xyxy is not None:
        x0, y0, x1, y1 = roi_xyxy
        wanted = {gid for gid, owner in inventory.items()
                  if x0 <= owner['grain_native'][0] < x1 and y0 <= owner['grain_native'][1] < y1}
        basis = 'reference_centre_inside_analysis_roi'
    else:
        wanted = set(inventory)
        basis = 'all_known_physical_owners'
    for gid in sorted(set(inventory) - wanted):
        excluded.append({'grain_id': gid, 'reason': 'outside analysis owner selection'})
    return [inventory[gid] for gid in sorted(wanted)], {
        "applied": applied, "removed": removed, 'excluded_from_analysis': excluded,
        'known_inventory_ids': known_ids, 'selection_basis': basis,
        'selected_owner_ids': sorted(wanted)}


def reviewed_mask_pixels(mask, image_size=None):
    """Decode exact positive paint and its native origin, excluding Unknown."""
    from prototypes.v30_video_apex.targets import decode_mask_raster
    raster = mask.get('mask_raster')
    if not raster:
        return np.zeros((0, 0), dtype=bool), (0, 0)
    h, w = int(raster['h']), int(raster['w'])
    origin = int(raster['x0']), int(raster['y0'])
    pixels = decode_mask_raster(h, w, origin, raster)
    unknown = mask.get('mask_unknown_raster')
    if unknown:
        pixels &= ~decode_mask_raster(h, w, origin, unknown)
    if image_size is not None:
        y, x = np.mgrid[:h, :w]
        pixels &= ((x+origin[0] >= 0) & (x+origin[0] < image_size[0])
                   & (y+origin[1] >= 0) & (y+origin[1] < image_size[1]))
    return pixels, origin


def reviewed_mask_positive_pixels(mask, image_size=None):
    """An explicitly Unknown pixel cannot license presence."""
    return int(reviewed_mask_pixels(mask, image_size)[0].sum())


def resolve_owned_masks(entities, request, owners, *, source='', tombstones=(), image_size=None,
                        all_movie_frames=False):
    """Resolve current exact masks and provenance without promoting model output.

    Snapshot masks were partitioned at export. Live revisions and parent-task
    exclusions still shadow those historical copies under logical project IDs.
    Offline model assistance can consume other frames from the same movie;
    current-frame presence callers retain their requested-frame restriction.
    """
    from .analysis_contracts import project_identity, stable_hash
    from .review_semantics import partition_review_entities, mask_review_provenance
    records = []
    if request.snapshot:
        for m in snapshot_data(request.snapshot)['body_masks']:
            record = dict(m, source_project=project_identity(m.get('project') or request.snapshot),
                          live_review=False)
            record.update(mask_review_provenance(record))
            records.append(record)
    tasks = {e['uuid']: e['data'] for e in entities if e['kind'] == 'task'}
    grouped = {}
    for entity in entities:
        grouped.setdefault(entity['kind'], []).append(entity)
    _, excluded = partition_review_entities(grouped)
    exclusion = {e['uuid']: e['exclusion'] for e in excluded}
    project = project_identity(source)
    for entity in entities:
        if entity['kind'] != 'mask':
            continue
        m = dict(entity['data'])
        task = tasks.get(m.get('task_uuid'), {})
        m.update(mask_uuid=entity['uuid'], mask_revision=entity['revision'],
                 source_project=project, live_review=True,
                 movie=m.get('movie_uuid') or m.get('movie') or task.get('movie'),
                 owner_uuid=m.get('owner_uuid') or task.get('owner_uuid'),
                 _exclusion=exclusion.get(entity['uuid']))
        m.update(mask_review_provenance(m, task))
        if m.get('task_uuid') and m['task_uuid'] not in tasks:
            m['_exclusion'] = 'parent task missing'
        records.append(m)
    tomb = {(t.get('source_project') or project_identity(t.get('project') or t.get('source') or source),
             t['uuid']) for t in tombstones}
    latest = {}
    for i, record in enumerate(records):
        key = record['source_project'], record.get('mask_uuid')
        rank = (int(record.get('mask_revision', 1)), bool(record.get('live_review')), i)
        if key not in latest or rank > latest[key][0]:
            latest[key] = rank, record
    allowed = {o['id'] for o in owners if o.get('identity_verified', bool(o.get('source_task')))}
    facts, audit = [], []
    for _, mask in latest.values():
        if (mask.get('movie') != request.movie_id
                or (not all_movie_frames and int(mask.get('source_frame', -1)) not in request.frames)):
            continue
        uid, project = mask.get('mask_uuid'), mask['source_project']
        owner = request.owner_aliases.get(mask.get('owner_uuid'), mask.get('owner_uuid'))
        reason = mask.get('_exclusion')
        if any((project, ref) in tomb for ref in (uid, mask.get('task_uuid'), mask.get('source_obs_uuid')) if ref):
            reason = 'mask or parent deleted (tombstone)'
        elif is_withdrawn(mask):
            reason = 'review withdrawn'
        elif is_workflow_record(mask):
            reason = 'workflow verification'
        elif mask.get('review_origin') != 'human':
            reason = 'mask lacks human review provenance'
        elif owner not in allowed:
            reason = 'no verified physical owner in analysis'
        # A live task withdrawal must also suppress a snapshot mask that has
        # not itself been re-saved or is absent from the current entity list.
        if project == project_identity(source) and mask.get('task_uuid') in exclusion:
            reason = exclusion[mask['task_uuid']]
        count = 0
        if not reason:
            try:
                count = reviewed_mask_positive_pixels(mask, image_size)
            except (ValueError, KeyError, TypeError) as error:
                reason = f'invalid exact mask: {error}'
            if not reason and not count:
                reason = 'no exact visible positive pixels'
        record = {'owner_id': owner, 'frame': int(mask['source_frame']),
                  'source_id': uid, 'source_project': project,
                  'revision': int(mask.get('mask_revision', 1)), 'task_uuid': mask.get('task_uuid'),
                  'kind': 'present', 'basis': 'reviewed_owned_body_pixels',
                  'review_origin': mask.get('review_origin'),
                  'review_origin_basis': mask.get('review_origin_basis'),
                  'positive_pixels': count,
                  'mask_sha256': stable_hash({k: mask.get(k) for k in ('mask_raster', 'mask_unknown_raster')})}
        if reason:
            audit.append(dict(record, ignored_reason=reason))
        else:
            facts.append({'evidence': record, 'mask': mask})
    return sorted(facts, key=lambda x: tuple(x['evidence'][k] for k in
                  ('owner_id', 'frame', 'source_project', 'source_id'))), audit


def resolve_body_presence(entities, request, owners, *, source='', tombstones=(), image_size=None):
    """Exact masks license presence only, never a tip, root or complete path."""
    masks, audit = resolve_owned_masks(entities, request, owners, source=source,
        tombstones=tombstones, image_size=image_size)
    return [m['evidence'] for m in masks], audit


def analysis_population(owners, rows, constraints, census, *, movie, scope=None, acquisition=None,
                        model_rows=None, body_presence=(), selected_owner_ids=None,
                        germination_events=()):
    grains = []
    for owner in owners:
        frame = int(owner.get("identified_at_frame", -1))
        grain = GrainRecord(owner["id"], movie, tuple(owner["grain_native"]),
            provenance={"identity_verified": owner.get("identity_verified", False),
                        "census": owner.get("census_source"), "evidence": []},
            observed_frames=[frame] if frame >= 0 else [],
            clump_id=owner.get("clump_id", ""),
            membership_resolved=bool(owner.get("identity_verified")))
        for (gid, frame), review in constraints.items():
            if gid != grain.grain_id or is_workflow_record(review):
                continue
            if review["state"] == "no_tube_visible":
                grain.no_tube_observed_at.append(frame)
            elif review.get("tip_xy") or review.get("path_xy") or review.get("tip_region_xy"):
                grain.present_observed_at.append(frame)
            grain.provenance["evidence"].append(dict(copy.deepcopy(review), frame=frame,
                kind="absent" if review["state"] == "no_tube_visible" else "present",
                revision=review.get("revision")))
        for evidence in body_presence:
            if evidence['owner_id'] == grain.grain_id:
                grain.present_observed_at.append(evidence['frame'])
                grain.provenance['evidence'].append(copy.deepcopy(evidence))
        for r in census:
            if r.get("movie") != movie or is_workflow_record(r) or not r.get("complete"):
                continue
            item = next((g for g in r.get("instances", []) if g["grain_id"] == grain.grain_id and g.get("confirmed")), None)
            if item:
                grain.observed_frames.append(r["source_frame"])
                grain.membership_resolved = bool(r.get("membership_resolved"))
                if item.get("classification"):
                    grain.classification = dict(item["classification"], source=r["task_uuid"], revision=r["task_revision"])
        grain.observed_frames = sorted(set(grain.observed_frames))
        grains.append(grain)
    eligible_tiles = [r for r in census if not is_workflow_record(r)]
    report = build_population_report(grains, census_tiles(eligible_tiles), movie=movie,
                                     scope=scope, acquisition=acquisition)
    if selected_owner_ids is not None:
        report['census']['selected_owner_ids'] = sorted(selected_owner_ids)
        report['census']['census_completeness_certified'] = False
        report['census']['note'] = 'Explicitly selected grains; this subset does not certify a whole-field census.'
        report['germination']['denominator_complete'] = False
        report['germination']['denominator_scope'] = 'selected_physical_owners'
        report['germination']['population_fraction'] = None
    model_observations = [r for r in (model_rows if model_rows is not None else rows)
                          if not is_workflow_measurement(r)]
    report["model_observations"] = {state: sum(r["state"] == state for r in model_observations)
                                   for state in sorted({r["state"] for r in model_observations})}
    report["model_note"] = "Model states are provisional observations, never biological classification or exhaustive census truth."
    report['analysis_frame_interval'] = [min(r['source_frame'] for r in rows), max(r['source_frame'] for r in rows)] if rows else None
    report['grain_summaries'] = grain_summaries(report, rows, model_rows or [], acquisition or {},
                                              germination_events=germination_events)
    return report


def germination_event_records(entities):
    """Human germination-episode verdicts (WO2), never model inference."""
    out = []
    for entity in entities or []:
        if entity.get("kind") != "germination_event":
            continue
        data = dict(entity.get("data") or {})
        if data.get("review_status") == "withdrawn" or entity.get("exclusion"):
            continue
        out.append(data)
    return out


def model_event_bracket(model):
    """Model-inferred emergence bracket from state transitions (WO4).

    Event history persists through later disappearance: absent frames
    after first visibility are flicker/occlusion, never event erasure.
    Returns None when the grain is never model-absent (left-censored:
    emergence at or before schedule start cannot be timed).
    """
    absent = [r["source_frame"] for r in model if r.get("state") == "model_absent"]
    seen = [r["source_frame"] for r in model
            if r.get("state") in ("present", "identity_uncertain")]
    if not absent:
        return None
    first_seen = min(seen) if seen else None
    prior_absent = [f for f in absent if first_seen is None or f < first_seen]
    if not prior_absent:
        return None
    later_absent = [f for f in absent if first_seen is not None and f > first_seen]
    scores = [(r["source_frame"], r.get("event_score")) for r in model
              if r.get("event_score") is not None]
    sustained = None
    if seen:
        tail_absent = max(absent)
        after = [f for f in seen if f > tail_absent]
        sustained = min(after) if after else None
    return {"last_absent_frame": max(prior_absent),
            "first_visible_frame": first_seen,
            "flicker_absent_after_first_visible": len(later_absent),
            "sustained_visible_from_frame": sustained,
            "n_absent_frames": len(absent),
            "event_score_range": [min(v for _, v in scores), max(v for _, v in scores)] if scores else None,
            "derivation": ("last model_absent before first present/uncertain; "
                           "later absences are visibility flicker, not event erasure"),
            "source": "model-inferred"}


def grain_summaries(report, measured_rows, model_rows, acquisition, germination_events=()):
    """One useful row per grain, with reviewed bounds separate from model hints."""
    human_event = {}
    for event in germination_events or []:
        if event.get("owner_uuid") and event.get("verdict") in (
                "emerged_at_start", "emerged_within",
                "no_emergence_by_end", "unobservable"):
            human_event[event["owner_uuid"]] = event
    from .acquisition import AcquisitionClock
    clock = AcquisitionClock.from_metadata(acquisition)
    timing = {r['grain_id']: r for r in report['germination']['emergence_records']}
    def route_evidence(row):
        value = row.get('route_evidence')
        return value if isinstance(value, dict) else {}

    def geometry(row):
        value = route_evidence(row).get('geometry')
        return value if isinstance(value, dict) else {}

    def length_withheld_reasons(row):
        if is_workflow_measurement(row):
            return ['workflow_verification']
        if row.get('path_complete') and row.get('length_px') is not None:
            return []
        reasons = route_evidence(row).get('withheld_reasons')
        if isinstance(reasons, list) and reasons:
            return list(reasons)
        if not row:
            return ['no_observation']
        constraint = row.get('constraint') or {}
        if not isinstance(constraint, dict):
            constraint = {}
        if row.get('as_inference_constraint'):
            if constraint.get('path_xy') and not constraint.get('path_complete'):
                return ['reviewed_path_partial']
            if constraint.get('tip_xy') or constraint.get('tip_region_xy'):
                return ['tip_review_does_not_establish_complete_path']
        if row.get('state') == 'present':
            return ['complete_current_path_not_established']
        return [row.get('state') or 'unknown_observation']

    summaries = []
    for grain in report['grains']:
        gid = grain['grain_id']
        observations = sorted((r for r in measured_rows if r['owner_id'] == gid), key=lambda r: r['source_frame'])
        complete = [r for r in observations if r.get('state') == 'present'
                    and r.get('path_complete') and r.get('length_px') is not None
                    and not is_workflow_measurement(r)]
        model = sorted((r for r in model_rows if r['owner_id'] == gid), key=lambda r: r['source_frame'])
        supported, streak, first_streak = [], [], None
        for row in model:
            good = (row.get('state') == 'present' and geometry(row).get('geometrically_supported')
                    and not is_workflow_measurement(row)
                    and not row.get('as_inference_constraint') and not row.get('path_ambiguous')
                    and not any(row.get(k) for k in ('candidate_pool_truncated', 'assignment_search_truncated',
                                                     'temporal_search_truncated')))
            if good:
                supported.append(row); streak.append(row)
                if len(streak) >= 3 and first_streak is None:
                    first_streak = streak[0]['source_frame']
            else:
                streak = []
        last = complete[-1] if complete else {}
        estimate = supported[-1] if supported else {}
        latest = observations[-1] if observations else {}
        first = supported[0]['source_frame'] if supported else None
        classification = grain.get('classification') or {}
        row = {'grain_id': gid, 'movie': grain['movie'],
               'grain_x': grain['position_native'][0], 'grain_y': grain['position_native'][1],
               'membership_resolved': grain['membership_resolved'],
               'biological_class': classification.get('class'), 'classification_rule': classification.get('rule'),
               'classification_source': classification.get('source'),
               'classification_revision': classification.get('revision'),
               **timing[gid],
                'complete_measurement_count': len(complete), 'last_complete_length_frame': last.get('source_frame'),
                'last_complete_length_px': last.get('length_px'), 'last_complete_length_um': last.get('length_um'),
                'last_complete_length_provenance': last.get('provenance'),
                'last_complete_length_certificate': copy.deepcopy(last.get('path_certificate')),
                'latest_observation_frame': latest.get('source_frame'),
                'latest_observation_state': latest.get('state'),
                'latest_length_withheld_reasons': length_withheld_reasons(latest),
                'model_first_supported_presence_frame': first, 'model_first_supported_presence_time_s': clock.at(first),
                'model_first_sustained_presence_frame': first_streak,
                'model_first_sustained_presence_time_s': clock.at(first_streak),
                'model_presence_support_rule': '3 consecutive analyzed samples with supported current geometry; not verified onset',
                'model_last_supported_path_frame': estimate.get('source_frame'),
                'model_last_supported_path_length_px': geometry(estimate).get('proposal_length_px'),
                'model_timing_note': 'Presence proposals do not license an earlier absence or a biological classification.',
                'model_event_bracket': model_event_bracket(model),
                'model_latest_event_score': next(
                    (r.get('event_score') for r in reversed(model)
                     if r.get('event_score') is not None), None),
                'human_event_verdict': (human_event.get(gid) or {}).get('verdict'),
                'human_event_bracket': (
                    [human_event[gid].get('last_absent_frame'),
                     human_event[gid].get('first_visible_frame')]
                    if gid in human_event else None),
                'human_event_source': (
                    human_event[gid].get('task_uuid') if gid in human_event else None)}
        summaries.append(row)
    return summaries


def build_population_report(grains, tiles, *, movie, scope=None, acquisition=None):
    grains = merge_grain_records(grains)
    scope = CensusScope(**scope) if isinstance(scope, dict) else scope
    census = build_census(grains, tiles, scope=scope, scope_movie=movie)
    allowed = set(census["grain_ids"])
    selected = [copy.deepcopy(g) for g in grains if g.movie == movie and g.grain_id in allowed]
    for g in selected:
        reviewed = g.classification.get("reviewed_frames")
        if scope and reviewed and not set(scope.frames).issubset(reviewed):
            g.provenance["classification_out_of_scope"] = g.classification
            g.classification = {}
    intervals = emergence_intervals(selected)
    metadata = acquisition or {}
    germination = germination_report(intervals,
        acquisition_cadence_s=metadata.get("seconds_per_source_frame"),
        cadence_source=metadata.get("cadence_source", ""),
        source_times_s=metadata.get("source_times_s"))
    germination["denominator_complete"] = census["census_completeness_certified"]
    germination["population_fraction"] = (
        germination["numerator_germinated"] / len(selected)
        if selected and census["census_completeness_certified"]
        and not germination["n_biologically_unclassified"] else None)
    return {"schema": "tubetracker.population.v1", "census": census,
            "germination": germination, "grains": [asdict(g) for g in selected],
            "intervals": [dict(asdict(iv), bounds=iv.bounds()) for iv in intervals]}


def banked_grains(registry, snapshot, owners=(), rulings=()):
    """Join reviewed evidence by explicit aliases; never infer a grain from a tube centroid."""
    rows, unresolved, stubs = {}, [], []
    for gid, r in registry.get("grains", {}).items():
        xy = r.get("grain_native")
        if xy is None or len(xy) != 2:
            unresolved.append({"grain_id": gid, "reason": "physical grain location not reviewed"})
            continue
        frames = [int(r["first_seen_frame"])] if int(r.get("first_seen_frame", -1)) >= 0 else []
        rows[gid] = GrainRecord(gid, r["movie"], tuple(xy),
            provenance={"registry": copy.deepcopy(r)}, observed_frames=frames)

    def resolve(movie, *keys):
        matches = set()
        for key in keys:
            gid = registry.get("aliases", {}).get(key, key)
            gid = canonical_id(registry, gid)
            if gid in rows and rows[gid].movie == movie:
                matches.add(gid)
        if len(matches) > 1:
            raise ValueError("conflicting grain identities in banked evidence")
        return rows[next(iter(matches))] if matches else None

    def record(grain, frame, kind, source, revision=None):
        if frame < 0:
            return
        grain.observed_frames = sorted(set(grain.observed_frames + [frame]))
        if kind in ("present", "absent"):
            attr = "present_observed_at" if kind == "present" else "no_tube_observed_at"
            setattr(grain, attr, sorted(set(getattr(grain, attr) + [frame])))
        evidence = {"frame": frame, "kind": kind, "source": source, "revision": revision}
        if evidence not in grain.provenance.setdefault("evidence", []):
            grain.provenance["evidence"].append(evidence)

    for owner in owners:
        movie = owner.get("movie", "")
        grain = resolve(movie, "owner|" + owner["id"],
            f"task|rev11own|{owner.get('source_task')}|{owner.get('source_label')}")
        if grain:
            record(grain, int(owner.get("identified_at_frame", -1)),
                   "absent" if owner.get("no_tube") else "present", owner["id"])
    for task in snapshot.get("tubes", []):
        if is_withdrawn(task):
            continue
        project = Path(task.get("project", "")).name
        frame = int(task.get("source_frame", (task.get("query_frames") or [-1])[0]))
        for label, data in (task.get("grains") or {}).items():
            grain = resolve(task.get("movie", ""), f"task|{project}|{task.get('task_uuid')}|{label}")
            if grain and data.get("xy"):
                kind = "absent" if data.get("no_tube") else "present" if data.get("root_xy") else "grain"
                record(grain, frame, kind, task.get("task_uuid"), task.get("task_revision"))
    for obs in snapshot.get("observations", []):
        if is_workflow_record(obs) or is_withdrawn(obs):
            continue
        movie = obs.get("movie", "")
        ids = [obs.get(k, "") for k in ("owner_uuid", "tube_uuid", "obs_uuid")]
        grain = resolve(movie, *[key for uid in ids if uid for key in (uid, "owner|"+uid, f"{movie}|track-{uid}")])
        if grain and (precise_tip(obs) is not None or obs.get("path_xy")):
            record(grain, int(obs["source_frame"]), "present", obs.get("obs_uuid"), obs.get("obs_revision"))
    for region in snapshot.get("regions", []):
        if region.get("kind") != "owned_absence" or is_workflow_record(region) or is_withdrawn(region):
            continue
        grain = resolve(region.get("movie_uuid", ""), "region|" + region.get("_region_uuid", ""))
        if grain:
            record(grain, int(region["source_frame"]), "absent", region.get("_region_uuid"), region.get("_region_revision"))
    for mask in snapshot.get("body_masks", []):
        raster = mask.get("mask_raster")
        if not raster or is_workflow_record(mask) or is_withdrawn(mask):
            continue
        positive_pixels = reviewed_mask_positive_pixels(mask)
        movie = mask.get("movie", "")
        ids = [mask.get(k, "") for k in ("owner_uuid", "tube_uuid", "source_obs_uuid")]
        grain = resolve(movie, *[key for uid in ids if uid for key in (uid, "owner|"+uid, f"{movie}|track-{uid}")])
        item = {"mask": mask["mask_uuid"], "frame": int(mask["source_frame"]),
                "decoded_painted_pixels": positive_pixels, "grain_id": grain.grain_id if grain else None,
                "revision": mask.get("mask_revision")}
        stubs.append(item)
        if positive_pixels and grain:
            record(grain, item["frame"], "present", item["mask"], item["revision"])
        elif positive_pixels:
            unresolved.append(dict(item, reason="paint is preserved; physical-grain identity is not linked"))
    for entity in rulings or snapshot.get("rulings", []):
        ruling = entity.get("data", entity)
        if is_withdrawn(ruling):
            continue
        movie = ruling.get("movie", "ld")
        owner = str(ruling.get("grain_owner", ""))
        grain = resolve(movie, owner, "owner|"+owner, f"{movie}|track-{owner}")
        cls = ruling.get("class")
        if not cls and ruling.get("ruling", "").strip().lower() == "started germinated and never grew":
            cls = "germinated"
        if not grain or cls not in ("germinated", "nongerminating", "unresolved"):
            unresolved.append({"ruling": entity.get("uuid"), "reason": "unmapped grain or untyped biological ruling"})
            continue
        grain.classification = {"class": cls,
            "rule": ruling.get("ruling") or ruling.get("rule"),
            "source": entity.get("source") or entity.get("uuid"),
            "revision": entity.get("revision", ruling.get("revision")),
            "reviewed_frames": ruling.get("reviewed_frames", []),
            "details": copy.deepcopy(ruling)}
        grain.validate()
    return list(rows.values()), {"unresolved_evidence": unresolved, "mask_decoding": stubs}
