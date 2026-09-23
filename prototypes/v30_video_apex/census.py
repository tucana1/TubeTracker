"""rev12 P1.3: grain census, emergence intervals, germination report.

- The census keeps ALL physical grains, including nongerminating
  grains and unresolved clump memberships. Exhaustive-review flags
  attach to exact tiles/frames/revisions per class; a task marked
  completed does not license background truth.
- Emergence is stored as last-verified-absent / first-verified-present
  bounds with explicit censoring (movie-start presence, missing
  frames, occlusion, field exit). Unreviewed frames are NOT absence.
- The investigator's germination RULE is stored separately from
  appearance/visibility; it is never invented here (rule unset =>
  reported as unset).
- Growth rates require actual acquisition time: without cadence
  provenance, lengths are pixels and rates are NULL.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field

GRAIN_CLASSES = ("germinated", "nongerminating", "unresolved",
                 "censored-left", "censored-right")


@dataclass(frozen=True)
class CensusScope:
    movie: str
    roi_xywh: tuple
    frames: tuple
    class_scopes: tuple = ("grains",)
    border_policy: str = "centre-inside"
    clump_policy: str = "individual-physical-grains"

    def validate(self):
        _rectangle(self.roi_xywh)
        if not self.movie or not self.frames or not self.class_scopes:
            raise ValueError("census requires a movie, exact frames and reviewed classes")
        if any(int(f) != f or f < 0 for f in self.frames):
            raise ValueError("census frames must be nonnegative source-frame indices")
        if self.border_policy != "centre-inside" or self.clump_policy != "individual-physical-grains":
            raise ValueError("unsupported census border/clump protocol")


def _rectangle(box):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("rectangle needs four finite values")
    x, y, w, h = map(float, box)
    if min(w, h) <= 0:
        raise ValueError("rectangle width and height must be positive")
    return x, y, x+w, y+h


def _covered_area(boxes, roi):
    """Exact clipped rectangle-union area, including overlapping tiles."""
    x0, y0, x1, y1 = _rectangle(roi)
    clipped = []
    for box in boxes:
        a, b, c, d = _rectangle(box)
        a, b, c, d = max(a, x0), max(b, y0), min(c, x1), min(d, y1)
        if c > a and d > b:
            clipped.append((a, b, c, d))
    xs = sorted({v for b in clipped for v in (b[0], b[2])})
    area = 0.
    for left, right in zip(xs, xs[1:]):
        spans = sorted((b, d) for a, b, c, d in clipped if a < right and c > left)
        length, end = 0., -math.inf
        for start, stop in spans:
            length += max(0., stop-max(start, end))
            end = max(end, stop)
        area += (right-left)*length
    return area


@dataclass
class CensusTile:
    tile_id: str
    movie: str
    frame: int
    box_xywh: tuple
    class_scopes: tuple = ()       # classes this review licenses
    exhaustive: bool = False       # did the reviewer certify completeness
    revision: int = 0
    border_policy: str = ""
    clump_policy: str = ""

    def validate(self) -> None:
        _rectangle(self.box_xywh)
        if not self.tile_id or not self.movie or self.frame < 0:
            raise ValueError("CensusTile needs tile_id + box")


@dataclass
class GrainRecord:
    grain_id: str                  # durable movie-qualified id
    movie: str
    position_native: tuple
    provenance: dict = field(default_factory=dict)
    clump_id: str = ""             # non-empty => membership unresolved
    no_tube_observed_at: list = field(default_factory=list)  # frames
    present_observed_at: list = field(default_factory=list)
    observed_frames: list = field(default_factory=list)
    membership_resolved: bool = False
    classification: dict = field(default_factory=dict)

    def validate(self) -> None:
        if (not self.grain_id or not self.movie or len(self.position_native) != 2
                or not all(math.isfinite(float(v)) for v in self.position_native)):
            raise ValueError("GrainRecord needs grain_id + position")
        if self.classification:
            c = self.classification
            if c.get("class") not in ("germinated", "nongerminating", "unresolved"):
                raise ValueError("unknown biological classification")
            if not c.get("rule") or not c.get("source") or c.get("revision") is None:
                raise ValueError("biological classification requires its grain-specific rule/source/revision")


def merge_grain_records(grains):
    """Merge evidence only for an already canonical movie-qualified identity."""
    merged = {}
    for original in grains:
        original.validate()
        key = (original.movie, original.grain_id)
        if key not in merged:
            merged[key] = copy.deepcopy(original)
            continue
        g = merged[key]
        for name in ("no_tube_observed_at", "present_observed_at", "observed_frames"):
            setattr(g, name, sorted(set(getattr(g, name)) | set(getattr(original, name))))
        sources = g.provenance.setdefault("merged_sources", [])
        if original.provenance and original.provenance not in sources:
            sources.append(copy.deepcopy(original.provenance))
        evidence = g.provenance.get("evidence", []) + original.provenance.get("evidence", [])
        unique = {json.dumps(e, sort_keys=True): copy.deepcopy(e) for e in evidence}
        if unique:
            g.provenance["evidence"] = [unique[k] for k in sorted(unique)]
        if original.classification:
            if g.classification and g.classification != original.classification:
                g.provenance["classification_conflicts"] = [g.classification, original.classification]
                g.classification = {}
            elif not g.provenance.get("classification_conflicts"):
                g.classification = copy.deepcopy(original.classification)
        if not g.clump_id:
            g.clump_id = original.clump_id
        g.membership_resolved |= original.membership_resolved
    return [merged[k] for k in sorted(merged)]


@dataclass
class EmergenceInterval:
    grain_id: str
    last_verified_absent: dict | None   # {frame, source, revision}
    first_verified_present: dict | None
    censoring: list = field(default_factory=list)  # left|right|occluded|...
    unresolved_reason: str = ""
    # rev13 W5.3: germination CLASS is separate from emergence TIMING.
    # A grain classified germinated can be left-censored and still
    # belong in the germination numerator under the declared protocol.
    classified: str = ""           # germinated | "" (unresolved)
    # rev13 W5.4: contradictory bounds are quarantined, never emitted
    # as a confirmed interval (and never as a negative duration).
    contradictory: bool = False
    classification_rule: str = ""
    classification_source: str = ""
    classification_revision: int | None = None

    def bounds(self):
        if self.contradictory:
            return None  # a quarantined record has no confirmed interval
        a = (self.last_verified_absent or {}).get("frame")
        p = (self.first_verified_present or {}).get("frame")
        if a is None or p is None:
            return None
        if int(a) >= int(p):
            # never a negative-duration confirmed interval
            return None
        return (int(a), int(p))


def build_census(grains: list[GrainRecord], tiles: list[CensusTile], *,
                 scope: CensusScope | dict | None = None, scope_movie: str = "",
                 min_tile_px: float | None = None) -> dict:
    """Unique scoped inventory; certify only exact reviewed coverage.

    Legacy scope_movie callers get a filtered provisional inventory.
    A minimum tile size never substitutes for ROI/time/class coverage.
    """
    if isinstance(scope, dict):
        scope = CensusScope(**scope)
    if scope:
        scope.validate()
        if scope_movie and scope_movie != scope.movie:
            raise ValueError("conflicting census movies")
        scope_movie = scope.movie
    all_grains = merge_grain_records(grains)
    grains, excluded_grains = [], []
    for g in all_grains:
        reasons = []
        if scope_movie and g.movie != scope_movie:
            reasons.append("different movie")
        if scope:
            x0, y0, x1, y1 = _rectangle(scope.roi_xywh)
            x, y = g.position_native
            if not (x0 <= x < x1 and y0 <= y < y1):
                reasons.append("centre outside requested field")
            if g.observed_frames and not set(g.observed_frames).intersection(scope.frames):
                reasons.append("no observation at requested census frames")
        if reasons:
            excluded_grains.append({"grain_id": g.grain_id, "movie": g.movie, "reasons": reasons})
        else:
            grains.append(g)
    in_scope, out_of_scope = [], []
    for t in tiles:
        t.validate()
        reasons = []
        if scope is None:
            reasons.append("missing exact requested ROI/frame/class/protocol scope")
        elif t.movie != scope.movie:
            reasons.append("different movie")
        else:
            if t.frame not in scope.frames:
                reasons.append("different census frame")
            if not set(t.class_scopes).intersection(scope.class_scopes):
                reasons.append("different reviewed classes")
            if t.border_policy != scope.border_policy or t.clump_policy != scope.clump_policy:
                reasons.append("border/clump protocol missing or different")
            if _covered_area([t.box_xywh], scope.roi_xywh) == 0:
                reasons.append("tile outside requested field")
        if reasons:
            out_of_scope.append({"tile_id": t.tile_id, "movie": t.movie,
                "frame": t.frame, "exhaustive": bool(t.exhaustive), "reasons": reasons})
        else:
            in_scope.append(t)
    exhaustive_tiles = [t for t in in_scope if t.exhaustive]
    coverage = []
    if scope:
        total = float(scope.roi_xywh[2]*scope.roi_xywh[3])
        for frame in scope.frames:
            for cls in scope.class_scopes:
                matching = [t for t in exhaustive_tiles if t.frame == frame and cls in t.class_scopes]
                area = _covered_area([t.box_xywh for t in matching], scope.roi_xywh)
                coverage.append({"frame": frame, "class": cls, "covered_area_px2": area,
                    "requested_area_px2": total, "complete": math.isclose(area, total, rel_tol=1e-10),
                    "tiles": [{"id": t.tile_id, "revision": t.revision} for t in matching]})
    clumps = {}
    for g in grains:
        if g.clump_id and not g.membership_resolved:
            clumps.setdefault(g.clump_id, []).append(g.grain_id)
    present = sum(bool(g.present_observed_at) for g in grains)
    absent_only = sum(bool(g.no_tube_observed_at) and not g.present_observed_at for g in grains)
    return {
        "n_grains": len(grains), "scope_movie": scope_movie or None,
        "scope": asdict(scope) if scope else None,
        "grain_ids": [g.grain_id for g in grains], "grains_out_of_scope": excluded_grains,
        "n_grains_without_frame_evidence": sum(not g.observed_frames for g in grains),
        "n_exhaustive_tiles": len(exhaustive_tiles), "n_tiles_out_of_scope": len(out_of_scope),
        "tiles_out_of_scope": out_of_scope, "coverage": coverage,
        "census_completeness_certified": (bool(coverage) and all(c["complete"] for c in coverage)
            and not clumps and all(g.observed_frames for g in grains)),
        "clumps_unresolved": clumps,
        "classes": {"present_evidence": present, "no_visible_tube_evidence_only": absent_only,
                    "unresolved": len(grains)-present-absent_only},
        "note": "Provisional unique inventory unless exact ROI/frame/class coverage, "
                "grain frame evidence and physical clump membership are certified. "
                "Visibility evidence does not assign a biological class."}


def emergence_intervals(grains: list[GrainRecord]) -> list[EmergenceInterval]:
    """Bounds from banked observations; unreviewed frames stay unknown.

    - Biological class requires a grain-specific sourced ruling,
      independently of whether the timing interval is bounded or left-censored;
    - an absent bound must PRECEDE the first verified present; a
      contradictory pair is quarantined with both sources retained and
      can never become a confirmed interval.
    """
    out = []
    for g in merge_grain_records(grains):
        def bound(frame, kind, fallback):
            evidence = [e for e in g.provenance.get("evidence", [])
                        if e.get("frame") == frame and e.get("kind") == kind]
            revisions = {e["revision"] for e in evidence if e.get("revision") is not None}
            return {"frame": frame,
                    "source": "; ".join(sorted({str(e["source"]) for e in evidence if e.get("source")})) or fallback,
                    "revision": next(iter(revisions)) if len(revisions) == 1 else g.provenance.get("revision"),
                    "lineage": evidence}
        iv = EmergenceInterval(grain_id=g.grain_id,
                               last_verified_absent=None,
                               first_verified_present=None)
        if g.classification:
            c = g.classification
            iv.classified = c["class"]
            iv.classification_rule = c["rule"]
            iv.classification_source = c["source"]
            iv.classification_revision = c["revision"]
        if g.no_tube_observed_at:
            f = int(max(g.no_tube_observed_at))  # latest certified absent
            iv.last_verified_absent = bound(f, "absent", "certified owned-absence review")
        else:
            iv.censoring.append("left")  # no certified pre-absence
        if g.present_observed_at:
            f = int(min(g.present_observed_at))  # earliest verified present
            iv.first_verified_present = bound(f, "present", "human path/tip observation")
            # Visibility evidence does not invent a biological protocol.
        else:
            iv.censoring.append("right")  # no later verified presence
        if iv.last_verified_absent and iv.first_verified_present:
            _a = int(iv.last_verified_absent["frame"])
            _p = int(iv.first_verified_present["frame"])
            if _a >= _p:
                iv.contradictory = True
                iv.unresolved_reason = (
                    f"quarantined: certified absence at frame {_a} is "
                    f"not before the first verified present at frame "
                    f"{_p}; both bounds retained with their revisions, "
                    f"no confirmed interval emitted")
        if iv.bounds() is None and not iv.contradictory:
            iv.unresolved_reason = ("bounds incomplete: "
                                    + "+".join(iv.censoring))
        out.append(iv)
    return out


def germination_report(intervals: list[EmergenceInterval],
                       *, germination_rule: str = "",
                       acquisition_cadence_s: float | None = None,
                       cadence_source: str = "", source_times_s: dict | None = None) -> dict:
    """Numerator/denominator/unresolved/censored + timing bounds.

    The investigator's germination rule is stored separately and is
    NEVER invented here; timing is in source frames unless a
    provenance-carrying cadence is supplied (then seconds too).
    """
    from tubetracker.acquisition import AcquisitionClock
    clock = AcquisitionClock.from_metadata({'seconds_per_source_frame': acquisition_cadence_s,
        'cadence_source': cadence_source, 'source_times_s': source_times_s})
    bounded = [iv for iv in intervals if iv.bounds() is not None]
    germinated = [iv for iv in intervals if iv.classified == "germinated"]
    quarantined = [iv for iv in intervals if iv.contradictory]
    timing, emergence = [], []
    for iv in intervals:
        a = (iv.last_verified_absent or {}).get('frame')
        p = (iv.first_verified_present or {}).get('frame')
        state = ('contradictory' if iv.contradictory else 'bounded' if iv.bounds() else
                 'left_censored' if p is not None else 'right_censored' if a is not None else 'unknown')
        emergence.append({'grain_id': iv.grain_id, 'timing_status': state,
            'last_verified_absent_frame': a, 'first_verified_present_frame': p,
            'last_verified_absent_time_s': clock.at(a), 'first_verified_present_time_s': clock.at(p),
            'onset_after_frame': a if not iv.contradictory else None,
            'onset_by_frame': p if not iv.contradictory else None,
            'onset_after_time_s': clock.at(a) if not iv.contradictory else None,
            'onset_by_time_s': clock.at(p) if not iv.contradictory else None,
            'bound_inclusion': 'after last absence (exclusive), by first presence (inclusive)',
            'absent_evidence': iv.last_verified_absent, 'present_evidence': iv.first_verified_present,
            'timing_source': clock.source or None, 'reason': iv.unresolved_reason or None})
    for iv in bounded:
        _b = iv.bounds()
        assert _b is not None  # filtered above
        a, p = _b
        row = {"grain_id": iv.grain_id, "absent_frame": a,
               "present_frame": p, "interval_frames": p - a}
        ta, tp = clock.at(a), clock.at(p)
        row.update(absent_time_s=ta, present_time_s=tp,
                   interval_seconds=round(tp-ta, 6) if ta is not None and tp is not None else None,
                   timing_source=clock.source or None)
        timing.append(row)
    return {
        "germination_rule": germination_rule or None,
        "rule_note": "Global rule text is descriptive only; each counted grain requires its own sourced ruling.",
        "classification_records": [{"grain_id": iv.grain_id, "class": iv.classified or None,
            "rule": iv.classification_rule or None, "source": iv.classification_source or None,
            "revision": iv.classification_revision} for iv in intervals],
        "n_biologically_unclassified": sum(iv.classified not in ("germinated", "nongerminating") for iv in intervals),
        "n_with_present_evidence": sum(iv.first_verified_present is not None for iv in intervals),
        # rev13 W5.3: the germination numerator counts grains CLASSIFIED
        # germinated under the declared protocol — a germinated grain
        # with a left-censored interval still belongs here. Bounded
        # emergence intervals are a SEPARATE, smaller count.
        "numerator_germinated": len(germinated),
        "numerator_germinated_basis": (
            "explicit grain-specific sourced biological ruling; left-censored grains included"),
        "numerator_bounded_intervals": len(bounded),
        "numerator_confirmed": len(bounded),  # legacy alias (bounded)
        "denominator_grains": len(intervals),
        "n_quarantined_contradictory": len(quarantined),
        "quarantined": [{"grain_id": iv.grain_id,
                         "absent": iv.last_verified_absent,
                         "present": iv.first_verified_present,
                         "reason": iv.unresolved_reason}
                        for iv in quarantined],
        "unresolved": sum(1 for iv in intervals
                          if iv.bounds() is None
                          and not iv.contradictory),
        "censored_left": sum(1 for iv in intervals
                             if "left" in iv.censoring),
        "censored_right": sum(1 for iv in intervals
                              if "right" in iv.censoring),
        "timing_bounds": timing,
        "emergence_records": emergence,
        "units": {"frames": "source frames (movie index)",
                  "seconds": ("explicit acquisition timestamps" if clock.timestamps_s else
                              "derived from supplied cadence"
                              if clock.cadence_s is not None
                              else "NULL — no acquisition provenance")},
        "growth_rate_note": "rates require valid geometry AND actual "
                            "acquisition time; gap/context interpolation "
                            "never becomes measurement",
    }
