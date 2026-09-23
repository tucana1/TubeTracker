"""v30 evaluate: ownership-aware probe scoring + oracle-substitution hooks.

rev5 repair: score_probes reports oracle (best-of-candidates) recall AND
model-selected accuracy separately. A wrong 0.99 output plus an unselected
correct 0.001 candidate scores 100% oracle recall but 0% selected — the
selected number is the tracking result; oracle recall is proposal coverage.
Negative/ambiguous gold rows and foreign emissions are scored explicitly.

rev11: the median is the CONVENTIONAL one (average of the two middle
values for even n). The runner used to report the upper-middle value as
"median" — for the six developed events the conventional median is
72.41 px while the upper-middle statistic reports 86.53. Both are kept,
explicitly labelled, so no consumer can conflate them again.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

MEDIAN_CONVENTION = ("average of the two middle values for even n, "
                     "middle value for odd n (conventional median); the "
                     "upper-middle order statistic is reported separately "
                     "as upper_middle_*, never as 'median'")


def conventional_median(values: Sequence[float]) -> float:
    """The documented quantile convention (see MEDIAN_CONVENTION)."""
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return math.inf
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def upper_middle(values: Sequence[float]) -> float:
    """The upper-middle order statistic (the old, mislabelled median)."""
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return math.inf
    return xs[len(xs) // 2]


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def point_to_polyline_dist(q, poly) -> float | None:
    """Min distance from point q to a polyline (None for <2 points)."""
    import numpy as np

    P = np.asarray(poly or [], float).reshape(-1, 2)
    if P.shape[0] < 2:
        return None
    q = np.asarray(q, float)
    seg = P[1:] - P[:-1]
    seglen = np.hypot(seg[:, 0], seg[:, 1]).clip(min=1e-9)
    w = ((q - P[:-1]) * seg).sum(axis=1) / (seglen ** 2)
    proj = P[:-1] + np.clip(w, 0, 1)[:, None] * seg
    return float(np.hypot(proj[:, 0] - q[0], proj[:, 1] - q[1]).min())


def candidate_metrics(gold_xy, support_path, current_path,
                      current_tip) -> dict:
    """The three rev11 quantities for ONE candidate vs the gold tip.

    support_px   distance to the FULL proposal polyline — support
                 proximity only (proposal coverage), never a front.
    current_px   distance to the measured portion (current_path). None
                 when the candidate carries no measured path — reported
                 unavailable, never silently measured on support.
    endpoint_px  distance from the gold tip to the exported endpoint
                 (current_tip). This is the tip-error domain.
    """
    return {
        "support_px": point_to_polyline_dist(gold_xy, support_path),
        "current_px": point_to_polyline_dist(gold_xy, current_path),
        "endpoint_px": (_dist(float(gold_xy[0]), float(gold_xy[1]),
                              float(current_tip[0]), float(current_tip[1]))
                        if current_tip is not None else None),
    }


def substitute_oracle_routes(candidates: Sequence[dict], oracle_routes: dict) -> list[dict]:
    """Hook P3A: replace predicted route geometry with endpoint-blinded oracle routes."""
    out = []
    for c in candidates:
        c = dict(c)
        key = (c.get("movie_id"), c.get("owner_id"))
        if key in oracle_routes:
            c["route"] = oracle_routes[key]
            c["evidence"] = dict(c.get("evidence", {})) | {"route_source": "oracle"}
        else:
            c["evidence"] = dict(c.get("evidence", {})) | {"route_source": "predicted"}
        out.append(c)
    return out


def substitute_oracle_fronts(candidates: Sequence[dict], oracle_fronts: dict) -> list[dict]:
    """Hook P3A: supply gold visible fronts while keeping predicted routes."""
    out = []
    for c in candidates:
        c = dict(c)
        if c.get("candidate_id") in oracle_fronts:
            c["front"] = oracle_fronts[c["candidate_id"]]
            c["evidence"] = dict(c.get("evidence", {})) | {"front_source": "oracle"}
        else:
            c["evidence"] = dict(c.get("evidence", {})) | {"front_source": "predicted"}
        out.append(c)
    return out


def _select(cands: list[dict]) -> dict | None:
    """The model's actual choice.

    rev6: an explicit "selected" flag (the runner's exported winner)
    takes precedence. Falling back to top tip_score scored a different
    candidate than the exported measurement whenever ranking changed.
    Ties -> first.
    """
    scored = [p for p in cands if p.get("x_native") is not None]
    if not scored:
        return None
    flagged = [p for p in scored if p.get("selected") is True]
    if len(flagged) == 1:
        return flagged[0]
    return max(scored, key=lambda p: (float(p.get("tip_score", 0.0)),
                                      str(p.get("candidate_id", ""))))


def confidence_gate(scored: list[dict], min_q: float) -> dict | None:
    """Selection guard: the top-q proposal wins only above min_q.

    Below threshold the event is WITHHELD (no measurement) instead of
    a forced wrong answer. min_q <= 0 disables the guard. scored must
    be sorted best-first with 'q_max' keys; returns winner or None.
    """
    if not scored:
        return None
    winner = scored[0]
    if min_q > 0 and float(winner.get("q_max", 0.0)) < float(min_q):
        return None
    return winner


def score_measurements(measurements: Sequence[dict],
                       gold: Sequence[dict],
                       tolerance_px: float = 5.0) -> dict:
    """Score FINAL exported measurement rows (rev6).

    Unlike score_probes (which scores the candidate set), this scores
    what the system actually emitted: gating, assisted corrections and
    refusals included. A withheld/missing measurement counts as a miss
    in the denominator — never silently dropped.
    """
    by_key: dict[tuple, dict] = {}
    for m in measurements:
        by_key[(m.get("movie"), str(m.get("event", "")),
                int(m.get("source_frame", -1)))] = m
    n = n_hit = n_withheld = 0
    errors: list[float] = []
    for g in gold:
        if g.get("observation", "observed") == "not_observed" \
                or g.get("x_native") is None:
            continue
        n += 1
        m = by_key.get((g.get("movie_id"), str(g.get("owner_id", "")),
                        int(g.get("source_frame", -1))))
        tip = (m or {}).get("tip")
        if m is None or tip is None:
            n_withheld += 1
            errors.append(math.inf)
            continue
        err = _dist(float(tip[0]), float(tip[1]),
                    float(g["x_native"]), float(g["y_native"]))
        errors.append(err)
        n_hit += int(err <= tolerance_px)
    return {
        "n_opportunities": n,
        "n_hit": n_hit,
        "coverage": (n_hit / n) if n else 0.0,
        "n_withheld_or_missing": n_withheld,
        "median_error_px": conventional_median(errors),
        "median_convention": MEDIAN_CONVENTION,
        "upper_middle_error_px": upper_middle(errors),
        "tolerance_px": tolerance_px,
    }


def score_probes(predictions: Sequence[dict], gold: Sequence[dict],
                 tolerance_px: float = 5.0) -> dict:
    """Ownership-aware scoring with selected-vs-oracle separation.

    - oracle coverage: best candidate of the right owner within tol
      (proposal availability, not tracking success).
    - selected coverage: the SELECTED candidate (top tip_score) of the
      right owner within tol. Refusals (no candidates), misses and
      wrong-owner selections stay in the denominator.
    - negative gold rows (observation not_observed / x None): any emitted
      candidate of that owner+frame counts a false emission.
    - foreign emissions: candidates whose (movie, owner, frame) has no
      visible gold row are counted, not silently dropped.
    """
    gold_visible = [g for g in gold if g.get("observation", "observed") != "not_observed"
                    and g.get("x_native") is not None]
    gold_negative = [g for g in gold if g.get("observation") == "not_observed"
                     or g.get("x_native") is None]
    by_key: dict[tuple, list[dict]] = {}
    for p in predictions:
        if p.get("observation") == "not_observed" or p.get("x_native") is None:
            continue
        by_key.setdefault((p.get("movie_id"), p.get("owner_id"), p.get("source_frame")), []).append(p)
    oracle_hits = selected_hits = 0
    errors, selected_errors = [], []
    per_stratum: dict[str, dict] = {}
    for g in gold_visible:
        key = (g.get("movie_id"), g.get("owner_id"), g.get("source_frame"))
        cands = by_key.get(key, [])
        o_err = min((_dist(p["x_native"], p["y_native"], g["x_native"], g["y_native"])
                     for p in cands), default=math.inf)
        sel = _select(cands)
        s_err = (_dist(sel["x_native"], sel["y_native"], g["x_native"], g["y_native"])
                 if sel is not None else math.inf)
        errors.append(o_err)
        selected_errors.append(s_err)
        oracle_hits += int(o_err <= tolerance_px)
        selected_hits += int(s_err <= tolerance_px)
        strat = str(g.get("stratum", "all"))
        s = per_stratum.setdefault(strat, {"n": 0, "oracle": 0, "selected": 0})
        s["n"] += 1
        s["oracle"] += int(o_err <= tolerance_px)
        s["selected"] += int(s_err <= tolerance_px)
    false_emissions = 0
    for g in gold_negative:
        key = (g.get("movie_id"), g.get("owner_id"), g.get("source_frame"))
        false_emissions += len(by_key.get(key, []))
    gold_keys = {(g.get("movie_id"), g.get("owner_id"), g.get("source_frame"))
                 for g in gold_visible}
    foreign_keys = sorted(set(by_key) - gold_keys)
    n = len(gold_visible)
    finite = sorted(e for e in selected_errors if math.isfinite(e))
    p90 = finite[min(len(finite) - 1, int(0.9 * len(finite)))] if finite else math.inf
    return {
        "n_opportunities": n,
        "oracle_hits": oracle_hits,
        "oracle_coverage": (oracle_hits / n) if n else 0.0,
        "selected_hits": selected_hits,
        "correct_owner_coverage": (selected_hits / n) if n else 0.0,
        "correct_owner_hits": selected_hits,
        "median_selected_error_px": conventional_median(selected_errors),
        "median_convention": MEDIAN_CONVENTION,
        "upper_middle_selected_error_px": upper_middle(selected_errors),
        "p90_error_px": p90,
        "tolerance_px": tolerance_px,
        "false_emissions_on_negatives": false_emissions,
        "n_negative_gold": len(gold_negative),
        "foreign_candidate_keys": foreign_keys,
        "strata": {k: {"n": v["n"],
                       "oracle": (v["oracle"] / v["n"]) if v["n"] else 0.0,
                       "selected": (v["selected"] / v["n"]) if v["n"] else 0.0,
                       "coverage": (v["selected"] / v["n"]) if v["n"] else 0.0}
                  for k, v in per_stratum.items()},
    }
