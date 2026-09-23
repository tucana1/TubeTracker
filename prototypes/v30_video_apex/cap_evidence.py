"""Owner-independent local cap truth and one native candidate emitter.

Distance from an owner's tip never licenses a negative. Only explicit
reviewed non-cap polygons do; owner absence is a separate inference task.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from tubetracker.review_semantics import (precise_tip, is_workflow_record,
    TRAINING_REVIEW_ROLES, training_review_role)


def inside_polygon(points, polygon):
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    v = np.asarray(polygon, dtype=float)
    if v.ndim != 2 or len(v) < 3 or v.shape[1] != 2 or not np.isfinite(v).all():
        return np.zeros(len(p), dtype=bool)
    x, y = p[:, 0], p[:, 1]
    inside = np.zeros(len(p), dtype=bool)
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        crossing = (a[1] > y) != (b[1] > y)
        if abs(b[1] - a[1]) > 1e-12:
            inside ^= crossing & (x <= a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
        # Include reviewed boundary pixels explicitly.
        delta, edge = p - a, b - a
        length2 = float(edge @ edge)
        if length2:
            t = np.clip((delta @ edge) / length2, 0, 1)
            on = np.linalg.norm(p - (a + t[:, None] * edge), axis=1) < 1e-6
            inside |= on
    return inside


class CapLabelIndex:
    def __init__(self, observations, regions, *, source=""):
        self.source = str(source)
        self.points = {}
        self.negatives = {}
        self.owned_absences = []
        self.quarantined = []
        self.resolved_observations = {}
        self.observation_records = []
        self.resolved_regions = []
        self.training_scope = None
        self.excluded_workflow_records = 0
        # Later entries in a versioned snapshot are explicit follow-up
        # projects. Retain lineage while resolving a repeated owner/frame.
        grouped = {}
        for o in observations:
            if is_workflow_record(o):
                self.excluded_workflow_records += 1
                continue
            self.observation_records.append(o)
            owner = str(o.get("owner_uuid") or o.get("tube_uuid") or o.get("obs_uuid", ""))
            if owner == "unassigned":
                owner = str(o.get("obs_uuid", ""))
            key = (str(o.get("movie", "")), int(o.get("source_frame", -1)), owner)
            grouped.setdefault(key, []).append(o)
        for (movie, frame, owner), rows in grouped.items():
            # A dedicated later tip review overrides the earlier draft.
            o = max(enumerate(rows), key=lambda item: (
                item[1].get("task_type") == "review_tip", item[0]))[1]
            self.resolved_observations[(movie, frame, owner)] = o
            xy = precise_tip(o)
            if (o.get("direct_state") != "direct_visible" or not xy
                    or len(xy) != 2 or not np.isfinite(xy).all()):
                continue
            self.points.setdefault((movie, frame), []).append({
                "xy": list(map(float, xy)), "owner": owner,
                "obs_uuid": o["obs_uuid"], "revision": int(o.get("obs_revision", 1)),
                "project": o.get("project", ""), "task_uuid": o.get("task_uuid", ""),
                "lineage": [{"obs_uuid": r["obs_uuid"], "revision": int(r.get("obs_revision", 1)),
                             "project": r.get("project", "")} for r in rows]})
        for r in regions:
            if is_workflow_record(r):
                self.excluded_workflow_records += 1
                continue
            self.resolved_regions.append(r)
            movie = str(r.get("movie_uuid") or r.get("movie", ""))
            frame = int(r.get("source_frame", -1))
            if r.get("kind") == "owned_absence":
                self.owned_absences.append(r)
                continue
            if (r.get("kind") != "verified_negative" or not r.get("confirmed")
                    or r.get("class_scope") not in ("apex", "cap", "tip")):
                continue
            poly = r.get("polygon_xy", [])
            points = self.points.get((movie, frame), [])
            hit = inside_polygon([p["xy"] for p in points], poly)
            if hit.any():
                self.quarantined.append({"region": r.get("_region_uuid"),
                    "reason": "reviewed_noncap_conflicts_with_known_cap",
                    "observations": [p["obs_uuid"] for p, h in zip(points, hit) if h]})
                continue
            self.negatives.setdefault((movie, frame), []).append(r)

    @classmethod
    def from_snapshot(cls, directory):
        p = Path(directory)
        return cls(json.loads((p / "observations.json").read_text()),
                   json.loads((p / "regions.json").read_text()), source=p)

    def for_training(self, *, context_offsets=(0,), reserved_intervals=()):
        """Reserve held-out query/context frames before sampling or rasterizing.

        Resolve revisions first, then select training scope. Keeping evaluation
        truth in this index would leak it into another owner's crop on the same
        frame, even if its own sampling case had been removed.
        """
        offsets = tuple(int(d) for d in context_offsets)
        reserved = [(str(m), int(a), int(b)) for m, a, b in reserved_intervals]
        if not offsets or any(a > b for _, a, b in reserved):
            raise ValueError('training scope requires context offsets and ordered intervals')
        roles = TRAINING_REVIEW_ROLES
        # Independent FULL/hidden reviews reserve their frames even when a
        # dedicated point review wins the tip-only resolution on that frame.
        # Resolve copies of the same source revision before consulting roles.
        from tubetracker.analysis_contracts import project_identity
        latest = {}
        for i, row in enumerate(self.observation_records + self.resolved_regions):
            uid = row.get('obs_uuid') or row.get('_region_uuid') or f'anonymous-{i}'
            project = project_identity(row.get('project') or row.get('_project'))
            key = project, uid
            rank = int(row.get('obs_revision', row.get('_region_revision', 1))), i
            if key not in latest or rank > latest[key][0]:
                latest[key] = rank, row
        records = [row for _, row in latest.values()]
        held_out, role_records = set(), []
        for row in records:
            role = training_review_role(row)
            if role in roles:
                continue
            movie = str(row.get('movie_uuid') or row.get('movie', ''))
            frame = int(row.get('source_frame', -1))
            held_out.add((movie, frame))
            role_records.append({'movie': movie, 'frame': frame, 'role': role,
                'source_id': row.get('obs_uuid') or row.get('_region_uuid'),
                'revision': row.get('obs_revision', row.get('_region_revision', 1)),
                'project': row.get('project', row.get('_project', ''))})

        def reason(movie, frame):
            if any(m == movie and any(a <= frame+d <= b for d in offsets)
                   for m, a, b in reserved):
                return 'reserved_query_or_context_frame'
            if any((movie, frame+d) in held_out for d in offsets):
                return 'held_out_annotation_query_or_context_frame'
            return None

        view = copy.deepcopy(self)
        excluded = []
        for key, points in self.points.items():
            why = reason(*key)
            if why:
                excluded.extend({'movie': key[0], 'frame': key[1], 'obs_uuid': p['obs_uuid'],
                    'project': p['project'], 'revision': p['revision'], 'reason': why} for p in points)
                view.points.pop(key, None)
        for key, regions in self.negatives.items():
            why = reason(*key)
            if why:
                excluded.extend({'movie': key[0], 'frame': key[1],
                    'region_uuid': r.get('_region_uuid'), 'revision': r.get('_region_revision', 1),
                    'project': r.get('_project', ''), 'reason': why} for r in regions)
                view.negatives.pop(key, None)
        view.owned_absences = [r for r in view.owned_absences if not reason(
            str(r.get('movie_uuid') or r.get('movie', '')), int(r.get('source_frame', -1)))]
        view.training_scope = {'allowed_roles': sorted(roles),
            'legacy_missing_role': 'development', 'context_offsets': list(offsets),
            'reserved_intervals': reserved, 'role_reservations': role_records,
            'excluded': excluded,
            'policy': 'all labels on held-out query/context frames are unavailable for training'}
        return view

    def locations(self, movie, frame, xy, *, positive_radius=6.0,
                  negative_exclusion=10.0):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        caps = self.points.get((str(movie), int(frame)), [])
        distance = np.full(len(xy), np.inf)
        offset = np.zeros((len(xy), 2), np.float32)
        source = np.full(len(xy), -1, dtype=int)
        if caps:
            truth = np.asarray([p["xy"] for p in caps])
            d = np.linalg.norm(xy[:, None, :] - truth[None, :, :], axis=2)
            source = d.argmin(1)
            distance = d[np.arange(len(d)), source]
            offset = (truth[source] - xy).astype(np.float32)
        positive = distance <= positive_radius
        negative = np.zeros(len(xy), dtype=bool)
        for r in self.negatives.get((str(movie), int(frame)), []):
            negative |= inside_polygon(xy, r["polygon_xy"])
        negative &= (distance > negative_exclusion) & ~positive
        valid = np.isfinite(xy).all(1)
        positive &= valid
        negative &= valid
        return {"positive": positive, "negative": negative,
                "unknown": ~(positive | negative), "target": positive.astype(np.float32),
                "offset_xy": offset, "nearest_source": source,
                "distance_px": distance}

    def spatial(self, movie, frame, origin, shape, **kwargs):
        h, w = shape
        y, x = np.mgrid[:h, :w]
        xy = np.stack([x.ravel() + origin[0], y.ravel() + origin[1]], 1)
        t = self.locations(movie, frame, xy, **kwargs)
        return {k: v.reshape(h, w, 2) if k == "offset_xy" else v.reshape(h, w)
                for k, v in t.items()}

    def summary(self):
        return {"source": self.source, "cap_points": sum(map(len, self.points.values())),
                "excluded_workflow_records": self.excluded_workflow_records,
                "licensed_negative_regions": sum(map(len, self.negatives.values())),
                "owned_absences_separate": len(self.owned_absences),
                "quarantined": self.quarantined,
                "training_scope": self.training_scope,
                "label_rule": "all-owner direct caps positive; explicit reviewed noncap negative; otherwise unknown"}


def emit_cap_candidates(probabilities, locations_xy, valid, *, offsets_xy=None,
                        bounds_xyxy=None, threshold=0.5, nms_px=6.0,
                        max_candidates=256, movie="", source_frame=0,
                        log_variance=None):
    """Filter before ranking, refine in native XY, then share a global pool.

    Returns no sentinel, padded or nonfinite location. An evidence ID
    describes a cap hypothesis; it never contains an owner or route ID.
    """
    prob = np.asarray(probabilities, dtype=float).reshape(-1)
    loc = np.asarray(locations_xy, dtype=float).reshape(-1, 2)
    keep = np.asarray(valid, dtype=bool).reshape(-1).copy()
    if len(prob) != len(loc) or len(keep) != len(loc):
        raise ValueError("cap probabilities, locations and validity must align")
    refined = loc.copy()
    if offsets_xy is not None:
        refined += np.asarray(offsets_xy, dtype=float).reshape(-1, 2)
    keep &= np.isfinite(prob) & np.isfinite(loc).all(1) & np.isfinite(refined).all(1)
    keep &= (prob >= float(threshold)) & (prob >= 0) & (prob <= 1)
    if bounds_xyxy is not None:
        x0, y0, x1, y1 = bounds_xyxy
        for p in (loc, refined):
            keep &= (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
    candidates = np.flatnonzero(keep)
    order = candidates[np.lexsort((candidates, -prob[candidates]))]
    selected = []
    rows = []
    for i in order:
        if selected and np.linalg.norm(refined[selected] - refined[i], axis=1).min() < nms_px:
            continue
        point = refined[i].tolist()
        ident = hashlib.sha256(json.dumps([str(movie), int(source_frame),
                                         [round(v, 3) for v in point]]).encode()).hexdigest()[:20]
        row = {"cap_id": "cap-" + ident, "tip_xy": point,
               "probability": float(prob[i]), "sample_index": int(i),
               "location_xy": loc[i].tolist(), "source_frame": int(source_frame)}
        if log_variance is not None:
            lv = float(np.asarray(log_variance).reshape(-1)[i])
            row["uncertainty_px"] = float(np.exp(np.clip(lv, -10, 10) / 2)) if np.isfinite(lv) else None
        rows.append(row)
        selected.append(i)
        if len(rows) >= max_candidates:
            break
    return rows
