"""Build a v30 training snapshot from annotation project DBs (rev5 work order #2).

Reads task/observation/region/crossing entities (with SQLite revisions)
and writes a versioned snapshot dir the v30 loader consumes. The live
project DB is never read by training.

Snapshot layout:
  snapshot_manifest.json  version, schema, movies+hashes, counts
  observations.json       tip/visibility/path judgments w/ revision lineage
  regions.json            SupervisionRegion entities (verified negatives etc.)
  census.json             census review state, confirmed dots and proposal status
  tubes.json              per-task tube ids, grains/roots, crossing links
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prototypes.v30_video_apex.targets import (  # noqa: E402
    review_region_usable, licensed_review_region)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _project_identity(source) -> str:
    """Stable logical identity: declared project-identity.json, else path."""
    import json as _json
    p = Path(source)
    identity = p / "project-identity.json"
    if identity.exists():
        try:
            declared = _json.loads(identity.read_text()).get("project_id")
        except (ValueError, OSError):
            declared = None
        if declared:
            return f"project-id:{declared}"
    return str(p.resolve())


def load_entities(db_path: Path) -> dict[str, list[dict]]:
    import sqlite3

    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = c.execute(
            "select uuid, kind, data, revision from entities").fetchall()
    finally:
        c.close()
    out: dict[str, list[dict]] = {}
    for uuid, kind, data, rev in rows:
        out.setdefault(kind, []).append(
            {"uuid": uuid, "revision": int(rev),
             "data": json.loads(data)})
    return out


def _extent_hits_paint(mask: dict) -> bool:
    """Does a mask's reviewed extent actually cover its own paint?

    rev8 (H306): an extent that misses the paint cannot apply -- the
    loader rasterizes it into the crop, finds nothing, and falls back
    to band-only validity. That is a silent reduction in supervision
    for a mask flagged `complete`, so the manifest counts it.

    rev9 (WP-A.1): the predicate is now shared with the target builder
    (`targets.review_region_usable`) so the manifest's usable count and
    the trainer's extent quarantine agree by construction.
    """
    ext, _policy = licensed_review_region(mask.get('review_region'), mask.get('review_region_provenance'))
    ras = mask.get("mask_raster") or {}
    pts = mask.get("painted_xy") or []
    try:
        if ras.get("w") and ras.get("h"):
            bx = (float(ras["x0"]), float(ras["x0"]) + float(ras["w"]))
            by = (float(ras["y0"]), float(ras["y0"]) + float(ras["h"]))
        elif pts:
            xs = [float(q[0]) for q in pts]
            ys = [float(q[1]) for q in pts]
            bx, by = (min(xs), max(xs)), (min(ys), max(ys))
        else:
            return False
    except (TypeError, ValueError, IndexError):
        return False
    ok, _reason = review_region_usable(ext, (bx[0], by[0], bx[1], by[1]))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", action="append", default=[],
                    help="annotation project dir (repeatable)")
    ap.add_argument("--movie", action="append", default=[],
                    help="movie mapping as key=path (repeatable)")
    ap.add_argument("--default-movie", default="",
                    help="movie key for tasks with no movie field "
                    "(recorded, never silent)")
    ap.add_argument("--out", required=True, help="snapshot dir (must not exist)")
    a = ap.parse_args()
    # rev8: a project passed twice would load its entities twice and
    # double every row from it (measured: snap20 came out with the two
    # rev8masks2 masks duplicated because the rebuild passed a project
    # its predecessor already listed). Refuse loudly -- a snapshot is
    # provenance and must not silently double-count supervision.
    _seen_proj: dict[str, int] = {}
    for _pd in a.project_dir:
        _key = str(Path(_pd).resolve())
        _seen_proj[_key] = _seen_proj.get(_key, 0) + 1
    _dupes = [k for k, v in _seen_proj.items() if v > 1]
    if _dupes:
        print("REFUSE: project dir(s) passed more than once: "
              + ", ".join(sorted(_dupes)))
        return 2

    out = Path(a.out)
    if out.exists() or out.with_name(out.name + '.building').exists():
        print(f"refusing to overwrite {out}")
        return 1
    # rev14 P1: build atomically -- nothing appears at the final path
    # until the manifest is complete, so a rebuild can never leave a
    # partially written snapshot in place.
    _final_out = out
    out = out.with_name(out.name + '.building')
    movies: dict[str, str] = {}
    for spec in a.movie:
        key, _, path = spec.partition("=")
        if not key or not path:
            print(f"bad --movie {spec!r}, want key=path")
            return 1
        movies[key] = path
    if not a.project_dir:
        print("need at least one --project-dir")
        return 1

    from tubetracker.annotation_schema import SCHEMA_VERSION
    from tubetracker.review_semantics import census_grain_points, partition_review_entities

    tasks: dict[tuple, dict] = {}  # (project, uuid) — uuids collide
    observations: list[dict] = []   # across projects (v2 reused r3-* ids)
    regions: list[dict] = []
    masks: list[dict] = []
    duels: list[dict] = []
    crossings: list[dict] = []
    workflow_audit: list[dict] = []
    withdrawal_audit: list[dict] = []
    rulings: list[dict] = []
    grain_identities: list[dict] = []
    germination_events: list[dict] = []
    for proj in a.project_dir:
        db = Path(proj) / "annotations.db"
        if not db.exists():
            print(f"missing db {db}")
            return 1
        ents, excluded = partition_review_entities(load_entities(db))
        from tubetracker.grain_identity import validate_identity_observation
        for entity in ents.get('grain_identity', []):
            data = entity['data']
            validate_identity_observation(data)
            if data.get('review_origin') == 'human':
                grain_identities.append({'uuid': entity['uuid'], 'revision': entity['revision'],
                    'project': _project_identity(proj), 'data': data})
        workflow_audit.extend({"project": str(proj), **r} for r in excluded
                              if r['exclusion'] == 'workflow_verification')
        withdrawal_audit.extend({"project": str(proj), **r} for r in excluded
                                if r['exclusion'] == 'review_withdrawn')
        rulings.extend({"uuid": r["uuid"], "revision": r["revision"], "data": r["data"],
                        "source": str(db.resolve()) + "#" + r["uuid"]}
                       for r in ents.get("ruling", []))
        for t in ents.get("task", []):
            d = dict(t["data"])
            d["_uuid"] = t["uuid"]
            d["_revision"] = t["revision"]
            d["_project"] = str(proj)
            tasks[(str(proj), t["uuid"])] = d
        for o in ents.get("observation", []):
            d = dict(o["data"])
            d["_obs_uuid"] = o["uuid"]
            d["_obs_revision"] = o["revision"]
            d["_project"] = str(proj)
            observations.append(d)
        for r in ents.get("region", []):
            d = dict(r["data"])
            d["_region_uuid"] = r["uuid"]
            d["_region_revision"] = r["revision"]
            d["_project"] = str(proj)
            parent = tasks.get((str(proj), d.get('task_uuid')), {})
            if parent.get('annotation_role'):
                d['annotation_role'] = parent['annotation_role']
            regions.append(d)
        for m in ents.get("mask", []):
            d = dict(m["data"])
            d["_mask_uuid"] = m["uuid"]
            d["_mask_revision"] = m["revision"]
            d["_project"] = str(proj)
            masks.append(d)
        for u in ents.get("duel", []):
            d = dict(u["data"])
            d["_duel_uuid"] = u["uuid"]
            d["_duel_revision"] = u["revision"]
            d["_project"] = str(proj)
            duels.append(d)
        for x in ents.get("crossing", []):
            d = dict(x["data"])
            d["_xing_uuid"] = x["uuid"]
            d["_xing_revision"] = x["revision"]
            d["_project"] = str(proj)
            crossings.append(d)
        # WO2: germination-event answers are a dedicated record kind.
        # Only human-origin answers ingest; model drafts never supervise.
        from tubetracker.annotation_schema import GerminationEvent as _GerminationEvent
        for g in ents.get("germination_event", []):
            data = g["data"]
            try:
                errs = _GerminationEvent(**{k: data.get(k) for k in (
                    "uuid", "movie_uuid", "movie_content_hash", "owner_uuid",
                    "task_uuid", "window_start", "window_end", "verdict",
                    "last_absent_frame", "first_visible_frame",
                    "consulted_frames", "absence_region_xy", "grain_mask_uuid",
                    "tube_mask_uuid", "exit_xy", "apex_xy", "annotator",
                    "revision", "lineage")}).validate()
            except TypeError as exc:
                raise ValueError(f"bad germination_event record {g['uuid']}: {exc}")
            if errs:
                raise ValueError(f"bad germination_event record {g['uuid']}: {errs}")
            if data.get("review_origin") == "human":
                germination_events.append({"uuid": g["uuid"], "revision": g["revision"],
                    "project": _project_identity(proj), "data": data})

    # Join observations -> task movie; drop rows whose task is unknown.
    obs_rows: list[dict] = []
    orphans = 0
    n_defaulted = 0
    for o in observations:
        t = tasks.get((o["_project"], str(o.get("task_uuid", ""))))
        if t is None:
            orphans += 1
            continue
        movie_key = str(t.get("movie", "") or "")
        defaulted = False
        if not movie_key and a.default_movie:
            movie_key = a.default_movie
            defaulted = True
            n_defaulted += 1
        obs_rows.append({
            "obs_uuid": o["_obs_uuid"],
            "obs_revision": o["_obs_revision"],
            "task_uuid": str(o.get("task_uuid", "")),
            "task_revision": t["_revision"],
            "project": o["_project"],
            "movie": movie_key,
            "movie_path": movies.get(movie_key, ""),
            "movie_defaulted": defaulted,
            "tube_uuid": str(o.get("tube_uuid", "")),
            "owner_uuid": str(o.get("owner_uuid", "")),
            "source_frame": int(o.get("source_frame", -1)),
            "direct_state": str(o.get("direct_state", "")),
            "direct_xy": o.get("direct_xy"),
            "direct_region": o.get("direct_region"),
            "context_xy": o.get("context_xy"),
            "context_frames": o.get("context_frames", []),
            "path_xy": o.get("path_xy", []),
            "path_visible": o.get("path_visible", []),
            "path_complete": bool(o.get("path_complete", False)),
            "tip_source": o.get("tip_source"),
            **({"tip_review_basis": o["tip_review_basis"]} if o.get("tip_review_basis") else {}),
            **({"label_corrections": o["label_corrections"]} if o.get("label_corrections") else {}),
            **({"root_review_required": True} if o.get("root_review_required") else {}),
            "review_origin": o.get("review_origin") or t.get("review_origin", "human"),
            "distinct_cap_evidence": o.get("distinct_cap_evidence") or t.get("distinct_cap_evidence"),
            "reassign_from_owner": t.get("reassign_from_owner"),
            "focus_xy": t.get("focus_xy"),
            "task_type": str(t.get("task_type", "")),
            **({"annotation_role": t["annotation_role"]} if t.get("annotation_role") else {}),
            "task_showed_tip": t.get("known_tip_xy") is not None,
            "owner_decision": str(o.get("owner_decision", "")),
            "annotator": str(o.get("annotator", "")),
        })

    # Census tiles: tasks carrying census items (complete or not; flag it).
    census_rows: list[dict] = []
    for (_proj_key, uuid), t in tasks.items():
        tips = t.get("census_tips", []) or []
        grains = t.get("census_grains", []) or []
        if not tips and not grains and not t.get("census_complete"):
            continue
        focus = t.get("focus_xy") or [0.0, 0.0]
        region = t.get("review_region")
        if region:
            xs = [q[0] for q in region]
            ys = [q[1] for q in region]
            tile = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            derived = set(map(tuple, region)) != {
                (min(xs), min(ys)), (max(xs), min(ys)),
                (max(xs), max(ys)), (min(xs), max(ys))}
        else:
            tile = [float(focus[0]) - 160.0, float(focus[1]) - 160.0,
                    320.0, 320.0]
            derived = True
        row = {
            "task_uuid": uuid,
            "task_revision": t["_revision"],
            "project": t["_project"],
            "movie": str(t.get("movie", "")),
            "source_frame": (t.get("query_frames") or [-1])[0],
            "tile_xywh": tile,
            "tile_derived": derived,
            "tips": tips,
            "grains": grains,
            "complete": bool(t.get("census_complete", False)),
            "instances": t.get("census_instances", []),
            "instance_review_required": bool(t.get("analysis_census") or 'census_instances' in t),
            "class_scopes": t.get("census_class_scopes", []),
            "border_policy": t.get("census_border_policy", ""),
            "clump_policy": t.get("census_clump_policy", ""),
            "membership_resolved": bool(t.get("census_membership_resolved", False)),
        }
        row['grains'] = census_grain_points(row)
        census_rows.append(row)

    # Tubes: per-task stable ids, grains/roots, crossing continuation.
    tube_rows: list[dict] = []
    for (_proj_key2, uuid), t in tasks.items():
        if not (t.get("tube_ids") or t.get("grains") or
                t.get("draft_crossing")):
            continue
        tube_rows.append({
            "task_uuid": uuid,
            "task_revision": t["_revision"],
            "project": t["_project"],
            "movie": str(t.get("movie", "")),
            "tube_ids": t.get("tube_ids", {}),
            "source_frame": int((t.get("query_frames") or [-1])[0]),
            "grains": t.get("grains", {}),
            "draft_crossing": str(t.get("draft_crossing", "")),
        })

    movie_hashes: dict[str, dict] = {}
    for key, path in movies.items():
        p = Path(path)
        movie_hashes[key] = {
            "path": path,
            "bytes": p.stat().st_size if p.exists() else -1,
            "sha256": sha256_file(p) if p.exists() else "",
        }

    out.mkdir(parents=True)
    # Drop degenerate regions (zero-area slivers from double-clicks):
    # a 1px "negative" at the tip supervises nothing and risks confusion.
    kept_regions, n_degenerate = [], 0
    for r in regions:
        poly = r.get("polygon_xy", []) or []
        try:
            xs = [float(q[0]) for q in poly]
            ys = [float(q[1]) for q in poly]
        except (TypeError, IndexError):
            n_degenerate += 1
            continue
        if min(max(xs) - min(xs), max(ys) - min(ys)) < 4.0:
            n_degenerate += 1
            continue
        kept_regions.append(r)
    regions = kept_regions
    # Region<->tip conflicts (rev6): a verified-negative box containing
    # ANY banked precise tip at the same (movie, frame) is preserved
    # but flagged; the loader keeps those pixels unknown. Both human
    # records stand; the conflict is adjudicated by scope, never by
    # silently reinterpreting the box as a dense negative.
    region_tip_conflicts: list[dict] = []
    for r in regions:
        rmovie = str(r.get("movie_uuid", "") or "")
        rframe = int(r.get("source_frame", -2))
        if not rmovie or rframe < 0:
            rtask = tasks.get((r["_project"], str(r.get("task_uuid", ""))))
            if rtask is None:
                continue
            if not rmovie:
                rmovie = str(rtask.get("movie", "") or "")
                if not rmovie and a.default_movie:
                    rmovie = a.default_movie
            if rframe < 0:
                rframe = int((rtask.get("query_frames") or [-1])[0])
        poly = r.get("polygon_xy", []) or []
        try:
            rxs = [float(q[0]) for q in poly]
            rys = [float(q[1]) for q in poly]
        except (TypeError, IndexError):
            continue
        hits = []
        for o in obs_rows:
            if o["movie"] != rmovie or o["source_frame"] != rframe:
                continue
            tip = o.get("direct_xy")
            if not tip:
                continue
            try:
                tx, ty = float(tip[0]), float(tip[1])
            except (TypeError, IndexError):
                continue
            if min(rxs) <= tx <= max(rxs) and min(rys) <= ty <= max(rys):
                hits.append(o["obs_uuid"])
        if hits:
            region_tip_conflicts.append({
                "region_uuid": r["_region_uuid"],
                "task_uuid": str(r.get("task_uuid", "")),
                "project": r["_project"],
                "movie": rmovie,
                "source_frame": rframe,
                "tip_obs_uuids": hits,
            })
    (out / "observations.json").write_text(
        json.dumps(obs_rows, indent=2) + "\n")
    # rev8: augment each exported region with its owner identity (the
    # raw entity has none; the task it belongs to does), so the loader
    # never has to guess an owner from geometry.
    for _r in regions:
        _rt = tasks.get((_r.get("_project", ""),
                         str(_r.get("task_uuid", "")))) or {}
        _ru = str(_rt.get("owner_uuid", "") or "")
        _rm = str(_r.get("movie_uuid", "") or _rt.get("movie", "") or "")
        _r["owner_uuid"] = _ru
        _r["owner_key"] = (f"{_rm}|{_ru}" if _ru and _ru != "unassigned"
                           else f"{_rm}|task:{_r.get('task_uuid', '')}")
    (out / "regions.json").write_text(json.dumps(regions, indent=2) + "\n")
    (out / "census.json").write_text(json.dumps(census_rows, indent=2) + "\n")
    (out / "tubes.json").write_text(json.dumps(tube_rows, indent=2) + "\n")
    (out / "crossings.json").write_text(
        json.dumps(crossings, indent=2) + "\n")

    # Body masks: painted tube bodies (rev6). Empty paint is refused
    # (never supervision); movie falls back to the task join, then the
    # default movie, exactly like observations.
    mask_rows: list[dict] = []
    ball_neg_regions: list[dict] = []
    mask_provenance_audit: list[dict] = []
    from tubetracker.review_semantics import mask_review_provenance
    for m in masks:
        mt = tasks.get((m["_project"], str(m.get("task_uuid", "")))) or {}
        provenance = mask_review_provenance(m, mt)
        if provenance['review_origin'] != 'human':
            mask_provenance_audit.append({'mask_uuid': m['_mask_uuid'],
                'mask_revision': m['_mask_revision'], 'project': m['_project'],
                **provenance, 'ignored_reason': 'mask lacks human review provenance'})
            continue
        pts = m.get("painted_xy", []) or []
        if m.get("no_tube") or m.get("scope") == "ball" and not pts:
            # rev8: a ball-scoped "no tube here" verdict. It is a CAP
            # negative at the queried grain ONLY (H242) — never a body
            # mask and never a frame/region negative.
            mt = tasks.get((m["_project"], str(m.get("task_uuid", "")))) or {}
            mv = str(m.get("movie_uuid", "") or mt.get("movie", "")
                     or a.default_movie or "")
            tgt = m.get("target_xy") or mt.get("target_xy") \
                or mt.get("focus_xy") or []
            if len(tgt) == 2:
                r = float(m.get("target_r") or mt.get("target_r", 14.0))
                x, y = float(tgt[0]), float(tgt[1])
                own_u = str(m.get("owner_uuid", "")
                            or mt.get("owner_uuid", "") or "")
                _r = {
                    "_region_uuid": f"ballneg-{m['_mask_uuid']}",
                    "_region_revision": int(m.get("_mask_revision", 1)),
                    "_project": m["_project"],
                    "movie": mv, "movie_uuid": mv,
                    "source_frame": int(m.get("source_frame", -1)),
                    # rev9 WP-A.6: an OWNED absence is not a generic cap
                    # negative. `verified_negative`+cap scope suppresses
                    # the generic terminal head over this disc, so a real
                    # (unlabelled) cap of a neighbouring grain inside it
                    # would be supervised negative. This record instead
                    # trains OWNED presence/visibility for this owner at
                    # this grain, and nothing else.
                    "kind": "owned_absence",
                    "class_scope": "owned_tube",
                    "polygon_xy": [[x - r, y - r], [x + r, y - r],
                                   [x + r, y + r], [x - r, y + r]],
                    "source": "mask-no-tube",
                    "grain_xy": [x, y], "grain_r": r,
                    "review_region": m.get("review_region") or [],
                    "annotator": str(m.get("annotator", "")),
                    "owner_uuid": own_u,
                    "owner_key": (f"{mv}|{own_u}" if own_u
                                  and own_u != "unassigned"
                                  else f"{mv}|task:"
                                       f"{m.get('task_uuid', '')}"),
                }
                ball_neg_regions.append(_r)
                regions.append(_r)
            continue
        if not pts and not m.get("mask_raster"):
            continue  # neither geometry nor raster: nothing to supervise
        movie_key = str(m.get("movie_uuid", "") or "")
        mt = tasks.get((m["_project"], str(m.get("task_uuid", "")))) or {}
        if not movie_key:
            movie_key = str(mt.get("movie", "") or "")
            if not movie_key and a.default_movie:
                movie_key = a.default_movie
        # rev8: explicit identity + reviewed extent travel with the mask
        owner_uuid = str(m.get("owner_uuid", "")
                         or mt.get("owner_uuid", "") or "")
        owner_key = (f"{movie_key}|{owner_uuid}"
                     if owner_uuid and owner_uuid != "unassigned" else
                     f"{movie_key}|task:{m.get('task_uuid', '')}")
        mask_rows.append({
            **provenance,
            **({'annotation_role': mt.get('annotation_role') or m.get('annotation_role')}
               if mt.get('annotation_role') or m.get('annotation_role') else {}),
            "mask_uuid": m["_mask_uuid"],
            "mask_revision": m["_mask_revision"],
            "task_uuid": str(m.get("task_uuid", "")),
            "project": m["_project"],
            "movie": movie_key,
            "movie_path": movies.get(movie_key, ""),
            "source_frame": int(m.get("source_frame", -1)),
            "painted_xy": [[float(q[0]), float(q[1])] for q in pts],
            "brush_px": float(m.get("brush_px", 9.0)),
            "complete": bool(m.get("complete", False)),
            # rev7: exact raster (pixel-exact) + explicit source-tube
            # link; either may be absent on legacy records.
            "mask_raster": m.get("mask_raster"),
            **({"mask_unknown_raster": m["mask_unknown_raster"]} if m.get("mask_unknown_raster") else {}),
            "source_obs_uuid": str(m.get("source_obs_uuid", "") or
                                   mt.get("source_obs_uuid", "")),
            # rev8: the extent the painter actually reviewed; a
            # background claim needs it, a "complete" flag alone never
            # licensed background outside the view.
            "review_region": m.get("review_region")
            or mt.get("review_region") or [],
            **({'review_region_provenance': m['review_region_provenance']}
               if m.get('review_region_provenance') else {}),
            "owner_uuid": owner_uuid,
            "owner_key": owner_key,
            "tube_uuid": str(m.get("tube_uuid", "")
                             or mt.get("tube_uuid", "") or ""),
            # rev8: the queried object (grain) must travel with the
            # mask; without it an evaluation cannot rebuild the
            # deployment prompt for this sample.
            "target_xy": [float(q) for q in (
                m.get("target_xy") or mt.get("target_xy")
                or mt.get("focus_xy") or [])],
            "target_r": float(m.get("target_r")
                              or mt.get("target_r", 14.0)),
            "annotator": str(m.get("annotator", "")),
            **({"label_corrections": m["label_corrections"]} if m.get("label_corrections") else {}),
        })
    (out / "body_masks.json").write_text(
        json.dumps(mask_rows, indent=2) + "\n")
    (out / 'mask_provenance_audit.json').write_text(
        json.dumps(mask_provenance_audit, indent=2) + '\n')

    # rev11 item 7: owner-task "No tube" verdicts become OWNED ABSENCE
    # regions — a human-identified grain + explicit frame with its tube
    # absent. Same semantics as the mask-based ball-neg record: owned
    # presence/visibility for THIS owner at THIS grain only, never a
    # generic cap negative, never a frame negative (H242).
    owner_abs_regions: list[dict] = []
    # rev12 P0.2: each region carries its durable physical grain ID
    # from the append-only grain registry (aliases: region uuid or
    # task|project|task|label — never row order).
    _grain_alias: dict[str, str] = {}
    _gid_path = REPO / "runs/prototypes/v30/grain_ids.json"
    if _gid_path.exists():
        _grain_alias = json.loads(_gid_path.read_text()).get(
            "aliases", {})
    for (_pk2, tuuid), t in tasks.items():
        if str(t.get("task_type", "")) != "owner":
            continue
        grains = t.get("grains") or {}
        frame = int((t.get("query_frames") or [-1])[0])
        mv = str(t.get("movie", "") or a.default_movie or "")
        own_u = str(t.get("owner_uuid", "") or "")
        _proj = Path(str(t.get("_project", ""))).name
        for lab, g in sorted(grains.items()):
            if not isinstance(g, dict) or not g.get("no_tube"):
                continue
            xy = g.get("xy") or []
            if len(xy) != 2:
                continue
            x, y = float(xy[0]), float(xy[1])
            r = float(t.get("target_r", 14.0) or 14.0)
            _ruuid = f"ownabs-{tuuid}-{lab}"
            _gid = (_grain_alias.get(f"region|{_ruuid}")
                    or _grain_alias.get(f"task|{_proj}|{tuuid}|{lab}")
                    or "")
            _r = {
                "_region_uuid": _ruuid,
                "_region_revision": int(t.get("_revision", 1)),
                "_project": t.get("_project", ""),
                "movie": mv, "movie_uuid": mv,
                "source_frame": frame,
                "kind": "owned_absence",
                "class_scope": "owned_tube",
                "polygon_xy": [[x - r, y - r], [x + r, y - r],
                               [x + r, y + r], [x - r, y + r]],
                "source": "owner-task-no-tube",
                "grain_xy": [x, y], "grain_r": r,
                "grain_id": _gid,
                "review_region": [],
                "annotator": str(t.get("annotator", "rev11own") or ""),
                "owner_uuid": own_u,
                "owner_key": (f"{mv}|{own_u}" if own_u
                              and own_u != "unassigned"
                              else f"{mv}|task:{tuuid}"),
                "task_uuid": tuuid,
            }
            owner_abs_regions.append(_r)
            regions.append(_r)
    if ball_neg_regions or owner_abs_regions:
        # regions.json was written before the mask pass; ball-scoped
        # no-tube verdicts are only known here, so rewrite it.
        for region in regions:
            parent = tasks.get((region.get('_project'), region.get('task_uuid')), {})
            if parent.get('annotation_role'):
                region['annotation_role'] = parent['annotation_role']
        (out / "regions.json").write_text(
            json.dumps(regions, indent=2) + "\n")
        if ball_neg_regions:
            print(f"ball-scoped no-tube negatives: +{len(ball_neg_regions)} "
                  "(cap scope, never a frame negative)")
        if owner_abs_regions:
            print(f"owner-task owned-absence regions: "
                  f"+{len(owner_abs_regions)} (owned visibility only)")

    # Route duels: human pairwise preferences on real alternatives.
    # winner A/B = the voted lane's polyline is preferred for that
    # owner; 'neither' = BOTH lanes are negatives. Lanes ship with the
    # verdict (no later join needed); movie falls back like masks.
    duel_rows: list[dict] = []
    # rev7: dedupe identical geometry pairs + quarantine
    # contradictions. The same lanes judged twice with different
    # winners is a conflict to resolve, not two training rows; the
    # trainer must never see both sides of a contradiction.
    import hashlib as _hl
    seen_pairs: dict[str, dict] = {}
    n_duel_duplicates = 0
    n_duel_conflicts = 0
    for u in duels:
        w = str(u.get("winner", ""))
        if w not in ("A", "B", "neither"):
            continue
        la = [[float(q[0]), float(q[1])]
              for q in u.get("lane_a", []) or []]
        lb = [[float(q[0]), float(q[1])]
              for q in u.get("lane_b", []) or []]
        if len(la) < 2 or len(lb) < 2:
            continue
        movie_key = str(u.get("movie_uuid", "") or "")
        if not movie_key:
            mt = tasks.get((u["_project"], str(u.get("task_uuid", ""))))
            if mt is not None:
                movie_key = str(mt.get("movie", "") or "")
            if not movie_key and a.default_movie:
                movie_key = a.default_movie
        pair_key = _hl.sha256(repr(sorted([
            (movie_key, int(u.get("source_frame", -1)),
             tuple((round(float(q[0]), 1), round(float(q[1]), 1))
                   for q in la)),
            (movie_key, int(u.get("source_frame", -1)),
             tuple((round(float(q[0]), 1), round(float(q[1]), 1))
                   for q in lb)),
        ])).encode()).hexdigest()[:16]
        # Canonicalize the verdict to WINNING GEOMETRY (lane labels
        # are presentation order — an A/B swap with the same judgment
        # is agreement, not conflict; 'neither' = both lanes wrong).
        # Normalized to plain lists: tuple/list mixups must never read
        # as disagreement.
        if w == "neither":
            win_geo = "both-wrong"
        else:
            win_geo = [list(q) for q in
                       (la if w == "A" else lb)]
            win_geo = [[round(float(v), 1) for v in q] for q in win_geo]
        status = "unique"
        if pair_key in seen_pairs:
            prev = seen_pairs[pair_key]
            if prev["win_geo"] == win_geo:
                status = "duplicate-agreement"
                n_duel_duplicates += 1
            else:
                status = "CONFLICT-quarantined"
                n_duel_conflicts += 1
                prev["status"] = "CONFLICT-quarantined"
        duel_rows.append({
            "duel_uuid": u["_duel_uuid"],
            "duel_revision": u["_duel_revision"],
            "task_uuid": str(u.get("task_uuid", "")),
            "project": u["_project"],
            "movie": movie_key,
            "movie_path": movies.get(movie_key, ""),
            "source_frame": int(u.get("source_frame", -1)),
            # rev8: explicit owner link at build time (task carries it);
            # the loader must never have to guess by frame alone.
            "owner_uuid": str(
                (tasks.get((u["_project"], str(u.get("task_uuid", ""))))
                 or {}).get("owner_uuid", "") or u.get("owner_uuid", "")),
            "owner_key": str(
                (tasks.get((u["_project"], str(u.get("task_uuid", ""))))
                 or {}).get("owner_key", "")
                or f"{movie_key}|task:{u.get('task_uuid', '')}"),
            "owner_link_source": "task" if (
                tasks.get((u["_project"], str(u.get("task_uuid", ""))))
                or {}).get("owner_uuid") else "task-derived",
            "winner": w,
            "lane_a": la, "lane_b": lb,
            "truth_a": str(u.get("truth_a", "")),
            "truth_b": str(u.get("truth_b", "")),
            "pair_key": pair_key,
            "win_geo": win_geo,
            "status": status,
            "annotator": str(u.get("annotator", "")),
        })
        seen_pairs.setdefault(pair_key, duel_rows[-1])
    (out / "duels.json").write_text(
        json.dumps(duel_rows, indent=2) + "\n")
    (out / "workflow_review_records.json").write_text(
        json.dumps(workflow_audit, indent=2) + "\n")
    (out / "withdrawn_review_records.json").write_text(
        json.dumps(withdrawal_audit, indent=2) + "\n")
    (out / "rulings.json").write_text(json.dumps(rulings, indent=2) + "\n")
    (out / 'grain_identities.json').write_text(json.dumps(grain_identities, indent=2) + '\n')
    (out / "germination_events.json").write_text(json.dumps(germination_events, indent=2) + "\n")
    # rev14 P1: export deletion tombstones (Undo included) so a later fold
    # cannot resurrect a deleted observation from an older snapshot copy.
    tomb_rows = []
    for pd in a.project_dir:
        db = Path(pd) / "annotations.db"
        if not db.exists():
            continue
        identity = _project_identity(pd)
        import sqlite3 as _sq
        con = _sq.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT uuid, kind, created_utc, actor FROM revisions"
                " WHERE rowid IN (SELECT MAX(rowid) FROM revisions GROUP BY uuid)"
                "   AND revision = -1").fetchall()
        finally:
            con.close()
        for uuid, kind, created, actor in rows:
            tomb_rows.append({"uuid": uuid, "kind": kind,
                              "deleted_utc": created, "actor": actor,
                              "project": identity})
    (out / "tombstones.json").write_text(
        json.dumps(tomb_rows, indent=2) + "\n")

    def _sha(name: str) -> str:
        import hashlib
        return hashlib.sha256((out / name).read_bytes()).hexdigest()

    manifest = {
        "snapshot_version": 2,
        "schema_version": SCHEMA_VERSION,
        "review_semantics_version": 1,
        "n_workflow_records_excluded": len(workflow_audit),
        "n_withdrawn_records_excluded": len(withdrawal_audit),
        "n_rulings": len(rulings),
        'n_grain_identities': len(grain_identities),
        "n_germination_events": len(germination_events),
        "projects": list(a.project_dir),
        "movies": movie_hashes,
        "n_tasks": len(tasks),
        "n_observations": len(obs_rows),
        "n_orphan_observations": orphans,
        "n_movie_defaulted": n_defaulted,
        # rev8: identity/link/extent accounting — what is actually
        # consumable, and what is stored but cannot supervise.
        "n_obs_without_owner": sum(
            1 for o in obs_rows
            if not str(o.get("owner_uuid", "") or "")
            or str(o.get("owner_uuid", "")) == "unassigned"),
        "n_obs_without_tube_id": sum(
            1 for o in obs_rows if not str(o.get("tube_uuid", "") or "")),
        "n_masks_unlinked": sum(
            1 for m in mask_rows if not m.get("source_obs_uuid")),
        "n_masks_raster_only": sum(
            1 for m in mask_rows if m.get("mask_raster")
            and not m.get("painted_xy")),
        "n_masks_with_review_region": sum(
            1 for m in mask_rows if m.get("review_region")),
        # rev8 honesty counter (H306): a reviewed extent that does not
        # intersect the mask's own paint is unusable — the loader then
        # falls back to band-only validity, so the mask supervises less
        # than its 'complete' flag claims. Extents saved before the
        # camera-centre fix all had this shape (y symmetric about 0),
        # and counting them as "with review_region" hid the loss.
        "n_masks_extent_usable": sum(
            1 for m in mask_rows if _extent_hits_paint(m)),
        "n_masks_extent_unusable": sum(
            1 for m in mask_rows
            if m.get("review_region") and not _extent_hits_paint(m)),
        "n_duels_owner_linked": sum(
            1 for d_ in duel_rows
            if str(d_.get("owner_uuid", "") or "")
            and str(d_.get("owner_uuid", "")) != "unassigned"),
        "n_census_with_extent": sum(
            1 for c_ in census_rows if not c_.get("tile_derived")),
        "n_regions": len(regions),
        "n_degenerate_regions_dropped": n_degenerate,
        "n_region_tip_conflicts": len(region_tip_conflicts),
        "region_tip_conflicts": region_tip_conflicts,
        "n_census_tiles": len(census_rows),
        "n_census_confirmed_grains": sum(len(c['grains']) for c in census_rows),
        "n_census_unconfirmed_instances": sum(
            not g.get('confirmed') for c in census_rows for g in c['instances']),
        "n_tube_tasks": len(tube_rows),
        "n_crossings": len(crossings),
        "n_body_masks": len(mask_rows),
        "n_masks_without_human_provenance": len(mask_provenance_audit),
        "n_duels": len(duel_rows),
        "n_duel_duplicates": n_duel_duplicates,
        "n_duel_conflicts_quarantined": n_duel_conflicts,
        # rev6: file-content hashes, not just counts — a manifest that
        # cannot detect swapped label files is not provenance.
        "content_sha256": {
            "observations.json": _sha("observations.json"),
            "regions.json": _sha("regions.json"),
            "census.json": _sha("census.json"),
            "tubes.json": _sha("tubes.json"),
            "crossings.json": _sha("crossings.json"),
            "body_masks.json": _sha("body_masks.json"),
            "mask_provenance_audit.json": _sha("mask_provenance_audit.json"),
            "duels.json": _sha("duels.json"),
            "workflow_review_records.json": _sha("workflow_review_records.json"),
            "withdrawn_review_records.json": _sha("withdrawn_review_records.json"),
            "rulings.json": _sha("rulings.json"),
            'grain_identities.json': _sha('grain_identities.json'),
            "germination_events.json": _sha("germination_events.json"),
            "tombstones.json": _sha("tombstones.json"),
        },
        "n_tombstones": len(tomb_rows),
    }
    (out / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    out.rename(_final_out)
    out = _final_out
    print(f"snapshot -> {out} "
          f"({len(obs_rows)} obs, {len(regions)} regions, "
          f"{len(census_rows)} census, {len(tube_rows)} tube tasks, "
          f"{orphans} orphans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
