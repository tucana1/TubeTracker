"""Joint offline cap ownership with hard review constraints and memory.

Search keeps the last observed cap through hidden frames. Hypothetical
identity memory is never exported as a current tip or measured length.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .cap_evidence import emit_cap_candidates


class ConstraintConflict(ValueError):
    pass


@dataclass(frozen=True)
class SolverConfig:
    cap_threshold: float = 0.5
    route_threshold: float = 0.35
    speed_px_per_source_frame: float = 0.35
    position_uncertainty_px: float = 3.0
    switch_cost: float = 0.5
    ambiguity_margin: float = 0.2
    candidates_per_owner: int = 4
    beam_width: int = 256
    max_joint_assignments: int = 512
    exact_single_owner: bool = False
    max_exact_states: int = 1000000
    # WO1-solver: scored model no-emergence alternative. None disables the
    # path entirely (previous behaviour preserved). Otherwise a
    # `model_absent` choice is admitted wherever a frame carries a
    # presence_logit at or below this log-odds threshold; its emission is
    # the absence evidence (-logit) minus the margin, competing directly
    # with route candidates. Never `verified_absent`: that state stays human.
    presence_absent_logit: Optional[float] = None
    presence_absent_margin: float = 0.0
    # WO4: installed event-model veto. None disables it. Otherwise a
    # `model_absent` choice is admitted wherever a frame carries an
    # event_score at or below this value; its emission beats observed
    # rim-route totals (~11) while real tubes never face it (~1.0).
    event_veto_threshold: Optional[float] = None
    event_veto_emission: float = 16.0


def _conflict_components(owners, frames, options):
    """Components include every possible shared cap over the whole interval."""
    parent = {o: o for o in owners}
    def root(o):
        while parent[o] != o:
            parent[o] = parent[parent[o]]
            o = parent[o]
        return o
    for frame in frames:
        seen = {}
        for owner in owners:
            for candidate in options[(owner, frame)]:
                cap = candidate.get("cap_id")
                if not cap:
                    continue
                if cap in seen:
                    a, b = root(owner), root(seen[cap])
                    parent[max(a, b)] = min(a, b)
                seen[cap] = owner
    groups = {}
    for owner in owners:
        groups.setdefault(root(owner), []).append(owner)
    return [groups[k] for k in sorted(groups)]


def _joint_choices(owners, frame, options, limit):
    """Generate bounded exclusive assignments without a Cartesian product.

    Hard states reserve their caps first. The retained partial assignments
    are ranked by current evidence; any pruning is explicitly reported.
    Temporal scoring still runs on the retained joint choices below.
    """
    reserved = {}
    for owner in owners:
        choices = options[(owner, frame)]
        if len(choices) == 1 and choices[0].get("human_constraint") and choices[0].get("cap_id"):
            cap = choices[0]["cap_id"]
            if cap in reserved:
                raise ConstraintConflict(f"frame {frame}: human ownership constraints claim the same identified cap")
            reserved[cap] = owner
    partial = [(0.0, (), frozenset())]
    truncated, expanded = False, 0
    for owner in owners:
        next_partial = []
        for score, chosen, used in partial:
            for candidate in options[(owner, frame)]:
                cap = candidate.get("cap_id")
                if cap and (cap in used or reserved.get(cap, owner) != owner):
                    continue
                expanded += 1
                next_partial.append((score + sum(_score(candidate)), chosen + (candidate,),
                                     used | {cap} if cap else used))
        if len(next_partial) > limit:
            truncated = True
            next_partial.sort(key=lambda v: (-v[0], tuple(c["candidate_id"] for c in v[1])))
            next_partial = next_partial[:limit]
        partial = next_partial
    return [v[1] for v in partial], truncated, expanded


def _logit(p):
    p = float(np.clip(p, 1e-5, 1 - 1e-5))
    return math.log(p / (1 - p))


def _score(candidate):
    if candidate.get("human_constraint"):
        return 0.0, 0.0
    if candidate.get("state") == "model_absent":
        # Scored no-emergence: head absence evidence and/or the
        # installed event-model veto compete directly. Neither source
        # present => no evidence => never wins.
        best = None
        logit = candidate.get("presence_logit")
        margin = float(candidate.get("presence_absent_margin", 0.0))
        if logit is not None and np.isfinite(logit):
            best = float(-logit) - margin
        ev = candidate.get("event_score")
        if ev is not None and np.isfinite(ev):
            thr = candidate.get("event_veto_threshold")
            amp = float(candidate.get("event_veto_emission", 16.0))
            if thr is not None and np.isfinite(thr) and float(ev) <= float(thr):
                veto = amp * max(0.0, 1.0 - float(ev) / float(thr)) if thr else amp
                best = veto if best is None else max(best, veto)
        if best is None:
            return 0.0, 0.0
        return float(best), 0.0
    if candidate["state"] != "present":
        return 0.0, 0.0
    return _logit(candidate["cap_probability"]), 2 * _logit(candidate["route_probability"])


def _human_options(owner, frame, constraint, candidates, all_candidates, movie):
    c = copy.deepcopy(constraint)
    state = c.get("state") or c.get("direct_state")
    tip = c.get("tip_xy", c.get("tip", c.get("direct_xy")))
    states = {"not_directly_visible": "occluded", "no_tube_visible": "verified_absent",
              "unreviewed_tip": "unreviewed_tip",
              "visible_imprecise": "imprecise",
              "out_of_field": "out_of_field", "owner_uncertain": "identity_uncertain",
              "lost_identity": "identity_uncertain", "unresolved_overlap": "identity_uncertain"}
    if state in states or state in states.values():
        resolved = states.get(state, state)
        partial = max(candidates, key=lambda r: r.get("route_probability", 0), default={})
        return [{"candidate_id": f"human-state:{owner}:{frame}", "state": resolved,
                 "tip_xy": None, "cap_id": "", "human_constraint": c,
                 "partial_path_xy": c.get("path_xy") or c.get("partial_path_xy")
                                    or partial.get("observed_partial_path_xy") or [],
                 'route_evidence': partial.get('route_evidence'),
                 "path_complete": False, "root_verified": False,
                 "cap_probability": None, "route_probability": None}]
    if tip is None or len(tip) != 2 or not np.isfinite(tip).all():
        raise ConstraintConflict(f"{owner} frame {frame}: visible correction needs an explicit finite tip")
    tip = list(map(float, tip))
    near = [r for r in all_candidates if r.get("tip_xy") is not None
            and np.linalg.norm(np.asarray(r["tip_xy"]) - tip) <= 5]
    cap_id = (min(near, key=lambda r: np.linalg.norm(np.asarray(r["tip_xy"]) - tip))["cap_id"]
              if near and not c.get("distinct_cap_evidence") else
              emit_cap_candidates([1], [tip], [True], movie=movie, source_frame=frame)[0]["cap_id"])
    if c.get("distinct_cap_evidence"):
        cap_id += ":" + str(c["distinct_cap_evidence"])
    # A point is not a traced body. Use only an explicit current path or
    # a separately identified model path already reaching this very cap.
    path = c.get("path_xy") or []
    complete = bool(c.get("path_complete") or c.get("path_completeness") == "full")
    if complete and (len(path) < 2 or np.linalg.norm(np.asarray(path[-1]) - tip) > 5):
        raise ConstraintConflict(f"{owner} frame {frame}: FULL path disagrees with its reviewed tip")
    root_verified = bool(c.get("root_verified") or (complete and c.get("source")))
    root_supported = root_verified
    model_certificate = None
    model_evidence = None
    if path and np.linalg.norm(np.asarray(path[-1]) - tip) <= 5:
        path = [list(p) for p in path]
        path[-1] = tip
    elif path:
        complete = False
    if not path:
        owned = [r for r in candidates if r.get("tip_xy") is not None
                 and np.linalg.norm(np.asarray(r["tip_xy"]) - tip) <= 5
                 and r.get("current_path_xy")]
        if owned:
            best = max(owned, key=lambda r: r.get("route_probability", 0))
            path = copy.deepcopy(best["current_path_xy"])
            model_evidence = copy.deepcopy(best.get('route_evidence'))
            # A certified model route can accompany a reviewed point only
            # if it was evaluated at that exact endpoint. Moving its final
            # segment would invalidate the current-body support check.
            if (best.get('path_complete') and best.get('path_certificate')
                    and np.linalg.norm(np.asarray(path[-1])-tip) <= 1e-5):
                complete = True
                root_verified = bool(best.get('root_verified'))
                root_supported = bool(best.get('root_supported', best.get('root_verified')))
                model_certificate = copy.deepcopy(best['path_certificate'])
                model_certificate['tip_evidence'] = c.get('tip_evidence')
                model_certificate['review_origin'] = c.get('review_origin', 'human')
            path[-1] = tip
    return [{"candidate_id": f"human-point:{owner}:{frame}", "state": "present",
             "tip_xy": tip, "cap_id": cap_id, "human_constraint": c,
             "current_path_xy": path, "path_complete": complete,
             "root_verified": root_verified, "root_supported": root_supported,
             "route_evidence": model_evidence,
             "path_certificate": model_certificate or ({"kind": "reviewed_full_current_path", "owner_id": owner,
                 "source_frame": frame, "geometry_source": c.get("geometry_source") or {
                     "source": c.get("source"), "id": c.get("observation_id"), "revision": c.get("revision")},
                 "tip_evidence": c.get("tip_evidence"), "review_origin": c.get("review_origin", "human")}
                 if complete else None),
             "cap_probability": 1.0, "route_probability": None}]


def solve_joint_states(candidates_by_owner, frames, *, constraints=None,
                       movie="", config=None):
    """Return accepted states and explicit alternate explanations.

    This is a bounded dynamic-programming beam, with truncation reported.
    Scores and search margins are diagnostic objectives, not calibrated
    probabilities. Hard human states cannot lose to a motion preference.
    """
    cfg = config or SolverConfig()
    if min(cfg.beam_width, cfg.candidates_per_owner, cfg.max_joint_assignments, cfg.max_exact_states) < 1:
        raise ValueError("search budgets must be positive")
    if type(cfg.exact_single_owner) is not bool:
        raise ValueError('exact_single_owner must be a boolean')
    constraints = constraints or {}
    owners = sorted(candidates_by_owner)
    frames = sorted(set(map(int, frames)))
    if not owners or not frames:
        return {"rows": [], "objective": 0.0, "search_truncated": False}
    options = {}
    pool_truncated = False
    for frame in frames:
        all_candidates = [r for oid in owners for r in candidates_by_owner[oid].get(frame, [])]
        for oid in owners:
            raw = candidates_by_owner[oid].get(frame, [])
            constraint = constraints.get((oid, frame))
            if constraint is not None:
                choices = _human_options(oid, frame, constraint, raw, all_candidates, movie)
            else:
                eligible = [dict(r, state="present") for r in raw
                            if r.get("tip_xy") is not None
                            and np.isfinite(r["tip_xy"]).all()
                            and r.get('tip_ownership_supported', True)
                            and float(r.get("cap_probability", 0)) >= cfg.cap_threshold
                            and float(r.get("route_probability", 0)) >= cfg.route_threshold]
                eligible.sort(key=lambda r: (-sum(_score(r)), str(r["candidate_id"])))
                pool_truncated |= len(eligible) > cfg.candidates_per_owner
                choices = eligible[:cfg.candidates_per_owner]
                partial = max(raw, key=lambda r: r.get("route_probability", 0), default={})
                choices += [{"candidate_id": f"uncertain:{oid}:{frame}",
                             "state": "identity_uncertain", "tip_xy": None, "cap_id": "",
                             "partial_path_xy": partial.get("observed_partial_path_xy", []),
                             'route_evidence': partial.get('route_evidence'),
                             "path_complete": False, "root_verified": False}]
                # WO1-solver: scored model no-emergence alternative. Admitted
                # only with a configured threshold and a scored frame; never
                # `verified_absent` (human-only). The emission carries the
                # absence evidence, so this choice can only win where the
                # head actually scores no-emergence against weak routes.
                _model_absent = None
                if cfg.presence_absent_logit is not None:
                    if not np.isfinite(cfg.presence_absent_logit):
                        raise ValueError("presence absent threshold must be finite")
                    _plogits = [r.get("presence_logit") for r in raw
                                if r.get("presence_logit") is not None
                                and np.isfinite(r.get("presence_logit"))]
                    if _plogits and min(_plogits) <= cfg.presence_absent_logit:
                        _model_absent = {
                            "candidate_id": f"model_absent:{oid}:{frame}",
                            "state": "model_absent", "tip_xy": None, "cap_id": "",
                            "presence_logit": float(min(_plogits)),
                            "presence_absent_margin": float(cfg.presence_absent_margin),
                            "presence_absent_threshold": float(cfg.presence_absent_logit),
                            "partial_path_xy": [], "path_complete": False,
                            "root_verified": False,
                            "cap_probability": None, "route_probability": None}
                # WO4: installed event-model veto admission. Fires only
                # with a configured threshold and a scored frame.
                if cfg.event_veto_threshold is not None:
                    if not np.isfinite(cfg.event_veto_threshold):
                        raise ValueError("event veto threshold must be finite")
                    _escores = [r.get("event_score") for r in raw
                                if r.get("event_score") is not None
                                and np.isfinite(r.get("event_score"))]
                    if _escores and min(_escores) <= cfg.event_veto_threshold:
                        _model_absent = _model_absent or {
                            "candidate_id": f"model_absent:{oid}:{frame}",
                            "state": "model_absent", "tip_xy": None, "cap_id": "",
                            "partial_path_xy": [], "path_complete": False,
                            "root_verified": False,
                            "cap_probability": None, "route_probability": None}
                        _model_absent["event_score"] = float(min(_escores))
                        _model_absent["event_veto_threshold"] = float(cfg.event_veto_threshold)
                        _model_absent["event_veto_emission"] = float(cfg.event_veto_emission)
                if _model_absent is not None:
                    choices.append(_model_absent)
            options[(oid, frame)] = choices
    groups = _conflict_components(owners, frames, options)
    results = [(_solve_single_owner_exact(group[0], frames, options, cfg)
                if cfg.exact_single_owner and len(group) == 1 else
                _solve_component(group, frames, options, cfg)) for group in groups]
    rows = [row for result in results for row in result["rows"]]
    rows.sort(key=lambda row: (row["source_frame"], row["owner_id"]))
    for row in rows:
        row["candidate_pool_truncated"] = pool_truncated
        row["search_margin_exhaustive"] = not (
            pool_truncated or row["assignment_search_truncated"] or row["temporal_search_truncated"])
    return {
        "rows": rows, "objective": sum(r["objective"] for r in results),
        "n_transitions_evaluated": sum(r["n_transitions_evaluated"] for r in results),
        "n_assignment_expansions": sum(r["n_assignment_expansions"] for r in results),
        "search_truncated": pool_truncated or any(r["search_truncated"] for r in results),
        "candidate_pool_truncated": pool_truncated,
        "assignment_search_truncated": any(r["assignment_search_truncated"] for r in results),
        "temporal_search_truncated": any(r["temporal_search_truncated"] for r in results),
        "components": groups, "beam_width": cfg.beam_width, "config": vars(cfg),
        "component_search": [{'owners': group, 'algorithm': result.get('algorithm', 'temporal_beam'),
            'states': result.get('n_states')} for group, result in zip(groups, results)],
        "score_note": "log-odds evidence plus source-frame-scaled motion; search margins are not calibrated confidence",
        "memory_note": "last visible cap retained through hidden states; never exported as a current measurement"}


@dataclass(frozen=True, slots=True)
class _History:
    previous: object
    choice: tuple
    transition: tuple


def _unroll(history):
    path, transitions = [], []
    while history is not None:
        path.append(history.choice); transitions.append(history.transition)
        history = history.previous
    path.reverse(); transitions.reverse()
    return path, transitions


def _transition(memory, previous_visible, candidate, frame, cfg):
    tip = candidate.get('tip_xy')
    cost = 0.0
    if tip is not None:
        if memory is not None:
            gap = max(1, frame-memory[0])
            sigma = cfg.position_uncertainty_px + cfg.speed_px_per_source_frame*gap
            cost = -.5 * (math.hypot(tip[0]-memory[1], tip[1]-memory[2])/sigma)**2
        memory = frame, float(tip[0]), float(tip[1]), candidate['cap_id']
    elif previous_visible and not candidate.get('human_constraint'):
        cost = -cfg.switch_cost
    return memory, cost


def _result_row(owner, frame, chosen, temporal, margin, path_margin, alternatives,
                filtering_id, cfg, *, assignment_truncated=False, temporal_truncated=False):
    path_ambiguous = (not chosen.get('human_constraint') and path_margin is not None
                      and path_margin < cfg.ambiguity_margin)
    uncertain = (not chosen.get('human_constraint') and chosen['state'] == 'present'
                 and margin is not None and margin < cfg.ambiguity_margin)
    state = 'identity_uncertain' if uncertain else chosen['state']
    alternatives = [r for r in alternatives
                    if r['candidate_id'] != chosen['candidate_id'] or uncertain][:4]
    local, whole = _score(chosen)
    tip = chosen.get('tip_xy') if state == 'present' else None
    path = chosen.get('current_path_xy') or []
    root_supported = bool(chosen.get('root_supported', chosen.get('root_verified')))
    full = bool(tip is not None and len(path) >= 2 and chosen.get('path_complete')
                and root_supported and chosen.get('path_certificate') and not path_ambiguous)
    if full and np.linalg.norm(np.asarray(path[-1])-tip) > 1e-5:
        raise ConstraintConflict('complete current path must end at the accepted cap')
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if full else None
    return {'owner_id': owner, 'source_frame': frame, 'state': state,
        'assignment_search_truncated': assignment_truncated,
        'temporal_search_truncated': temporal_truncated,
        'tip_xy': tip, 'cap_id': chosen.get('cap_id') if tip is not None else None,
        'current_path_xy': path if tip is not None else [],
        'partial_path_xy': (chosen.get('partial_path_xy') or path) if not full else [],
        'path_complete': full, 'root_verified': bool(chosen.get('root_verified')),
        'root_supported': root_supported, 'path_certificate': chosen.get('path_certificate') if full else None,
        'route_evidence': chosen.get('route_evidence'), 'length_px': length,
        'measurement_domain': 'full' if full else 'partial' if path or chosen.get('partial_path_xy') else 'none',
        'provenance': ('workflow-test' if (chosen.get('human_constraint') or {}).get('review_origin') == 'workflow_test'
                       else 'human-corrected' if chosen.get('human_constraint') else 'model-inferred'),
        'constraint': chosen.get('human_constraint'), 'as_inference_constraint': bool(chosen.get('human_constraint')),
        'tip_region_xy': (chosen.get('human_constraint') or {}).get('tip_region_xy', []),
        'selected_candidate_id': chosen['candidate_id'], 'alternatives': alternatives,
        'presence_logit': chosen.get('presence_logit'),
        'presence_absent_threshold': chosen.get('presence_absent_threshold'),
        'event_score': chosen.get('event_score'),
        'event_veto_threshold': chosen.get('event_veto_threshold'),
        'exit_xy': chosen.get('exit_xy'),
        'exit_support': chosen.get('exit_support'),
        'exit_contract': chosen.get('exit_contract'),
        'search_margin': margin, 'score_components': {'cap': local, 'route': whole, 'temporal': temporal},
        'path_search_margin': path_margin, 'path_ambiguous': path_ambiguous,
        'future_assisted_identity': state == 'present' and chosen['candidate_id'] != filtering_id}


def _solve_single_owner_exact(owner, frames, options, cfg):
    """Exact max-sum over last-visible memory, including conditional alternatives.

    A forward best prefix alone loses earlier ambiguity when histories merge.
    Backward scores recover each candidate's complete-movie max-marginal.
    Only the retained candidate pool is exhaustive; pool clipping remains a
    separate flag at the public solver boundary.
    """
    start = (None, None)
    layers = [{start: {'score': 0., 'choice': None, 'memory': None}}]
    filtering, visited, state_count = {}, 0, 1
    for frame in frames:
        layer = {}
        choices = options[(owner, frame)]
        emissions = {c['candidate_id']: sum(_score(c)) for c in choices}
        for previous_key, previous in layers[-1].items():
            visible = previous['choice'] is not None and previous['choice'].get('tip_xy') is not None
            for candidate in choices:
                visited += 1
                memory, transition = _transition(previous['memory'], visible, candidate, frame, cfg)
                key = candidate['candidate_id'], memory
                score = previous['score'] + emissions[candidate['candidate_id']] + transition
                if key not in layer or score > layer[key]['score']:
                    layer[key] = {'score': score, 'choice': candidate, 'memory': memory,
                                  'previous': previous_key, 'transition': transition}
        if not layer:
            raise ConstraintConflict(f'frame {frame}: no permitted identity state')
        state_count += len(layer)
        if state_count > cfg.max_exact_states:
            raise ValueError('exact single-owner state budget exceeded; refine the sample schedule or increase max_exact_states')
        winner = min(layer.values(), key=lambda n: (-n['score'], n['choice']['candidate_id']))
        filtering[frame] = winner['choice']['candidate_id']
        layers.append(layer)
    last_key = min(layers[-1], key=lambda k: (-layers[-1][k]['score'], k[0]))
    total = layers[-1][last_key]['score']
    selected, key = [], last_key
    for layer in reversed(layers[1:]):
        node = layer[key]; selected.append(node); key = node['previous']
    selected.reverse()

    suffix = [{key: 0. for key in layers[-1]}]
    for i in range(len(frames)-1, 0, -1):
        frame = frames[i]
        choices = options[(owner, frame)]
        emissions = {c['candidate_id']: sum(_score(c)) for c in choices}
        values = {}
        for key, node in layers[i].items():
            visible = node['choice'].get('tip_xy') is not None
            scores = []
            for candidate in choices:
                visited += 1
                memory, transition = _transition(node['memory'], visible, candidate, frame, cfg)
                target = candidate['candidate_id'], memory
                scores.append(emissions[candidate['candidate_id']] + transition + suffix[-1][target])
            values[key] = max(scores)
        suffix.append(values)
    suffix.reverse()

    rows = []
    for i, (frame, chosen_node) in enumerate(zip(frames, selected)):
        chosen = chosen_node['choice']
        candidates, scores = {}, {}
        for key, node in layers[i+1].items():
            cid = node['choice']['candidate_id']
            candidates[cid] = node['choice']
            value = node['score'] + suffix[i][key]
            scores[cid] = max(scores.get(cid, -math.inf), value)
        identity = lambda c: (c['state'], c.get('cap_id'))
        alternate = [s for cid, s in scores.items() if identity(candidates[cid]) != identity(chosen)]
        paths = [s for cid, s in scores.items() if cid != chosen['candidate_id']]
        margin = max(0., total-max(alternate)) if alternate else None
        path_margin = max(0., total-max(paths)) if paths else None
        alternatives = [{'candidate_id': cid, 'state': candidates[cid]['state'],
            'tip_xy': candidates[cid].get('tip_xy'), 'objective_gap': max(0., total-scores[cid])}
            for cid in sorted(scores, key=lambda cid: (-scores[cid], cid))]
        rows.append(_result_row(owner, frame, chosen, chosen_node['transition'], margin, path_margin,
                                alternatives, filtering[frame], cfg))
    if not math.isclose(sum(sum(r['score_components'].values()) for r in rows), total,
                        rel_tol=1e-8, abs_tol=1e-8):
        raise AssertionError('recorded score components differ from optimized objective')
    return {'rows': rows, 'objective': total, 'n_transitions_evaluated': visited,
        'n_assignment_expansions': sum(len(options[(owner, f)]) for f in frames),
        'assignment_search_truncated': False, 'temporal_search_truncated': False,
        'search_truncated': False, 'algorithm': 'exact_single_owner_forward_backward',
        'n_states': state_count}


def _solve_component(owners, frames, options, cfg):
    # Histories share their prefixes. Expanding a successor no longer
    # copies every previous frame into every temporary beam candidate.
    beam = [(0.0, None, tuple([None] * len(owners)))]
    filtering = {}
    truncated = False
    visited = 0
    generation_truncated, expanded = False, 0
    for frame in frames:
        joint, cut, work = _joint_choices(owners, frame, options, cfg.max_joint_assignments)
        generation_truncated |= cut
        expanded += work
        if not joint:
            raise ConstraintConflict(f"frame {frame}: human ownership constraints claim the same identified cap")
        next_states = {}
        emissions = [sum(sum(_score(c)) for c in choice) for choice in joint]
        for score, history, memory in beam:
            for choice, emission in zip(joint, emissions):
                visited += 1
                new_memory, transition = list(memory), []
                for j, candidate in enumerate(choice):
                    visible = history is not None and history.choice[j].get('tip_xy') is not None
                    new_memory[j], cost = _transition(memory[j], visible, candidate, frame, cfg)
                    transition.append(cost)
                total = score + emission + sum(transition)
                state_key = (tuple(c["candidate_id"] for c in choice), tuple(new_memory))
                item = (total, _History(history, tuple(choice), tuple(transition)), tuple(new_memory))
                previous = next_states.get(state_key, [])
                # Keep two histories per state so converging future choices
                # do not automatically erase earlier ambiguity evidence.
                previous.append(item)
                previous.sort(key=lambda v: -v[0])
                next_states[state_key] = previous[:2]
        beam = [v for values in next_states.values() for v in values]
        beam.sort(key=lambda v: (-v[0], tuple(c["candidate_id"] for c in v[1].choice)))
        if len(beam) > cfg.beam_width:
            truncated = True
            beam = beam[:cfg.beam_width]
        filtering[frame] = tuple(c["candidate_id"] for c in beam[0][1].choice)
    final = []
    for score, history, memory in beam:
        path, transitions = _unroll(history)
        final.append((score, path, memory, transitions))
    beam = final
    total, path, _, transitions = beam[0]
    rows = []
    component_sum = 0.0
    for fi, frame in enumerate(frames):
        for oi, owner in enumerate(owners):
            chosen = path[fi][oi]
            identity = lambda c: (c["state"], c.get("cap_id"))
            alt_scores = [b[0] for b in beam if identity(b[1][fi][oi]) != identity(chosen)]
            margin = total - max(alt_scores) if alt_scores else None
            path_scores = [b[0] for b in beam if b[1][fi][oi]["candidate_id"] != chosen["candidate_id"]]
            path_margin = total - max(path_scores) if path_scores else None
            alternatives = {}
            for b in beam:
                candidate = b[1][fi][oi]
                alternatives.setdefault(candidate['candidate_id'], {
                    'candidate_id': candidate['candidate_id'], 'state': candidate['state'],
                    'tip_xy': candidate.get('tip_xy'), 'objective_gap': float(total-b[0])})
            row = _result_row(owner, frame, chosen, transitions[fi][oi], margin, path_margin,
                list(alternatives.values()), filtering[frame][oi], cfg,
                assignment_truncated=generation_truncated, temporal_truncated=truncated)
            component_sum += sum(row['score_components'].values())
            rows.append(row)
    if not math.isclose(component_sum, total, rel_tol=1e-8, abs_tol=1e-8):
        raise AssertionError("recorded score components differ from optimized objective")
    return {"rows": rows, "objective": total, "n_transitions_evaluated": visited,
            "n_assignment_expansions": expanded,
            "assignment_search_truncated": generation_truncated,
            "temporal_search_truncated": truncated,
            "search_truncated": truncated or generation_truncated, "beam_width": cfg.beam_width,
            "config": vars(cfg), "score_note": "log-odds evidence plus source-frame-scaled motion; margins are search diagnostics",
            "memory_note": "last visible cap retained through hidden states; never exported as a current measurement"}
