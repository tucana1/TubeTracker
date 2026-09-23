"""rev13 W2.6: ONE proposal-generation callable for fitting,
evaluation and deployment.

The deployment family: FRST fans, curved walkers, v1 tip peaks and
(optional) model-body walks. Every caller — the proposal builder, the
proposal fit, the interval runner and the movie runner — goes through
`generate_route_hypotheses`, so the candidate family is the same object
everywhere instead of divergent copies. `v1`/`body_prob` are used ONLY
when actually passed, and the returned `grain_info` records which
evidence produced the family (never "a model is loaded, therefore it
was used").
"""
from __future__ import annotations

import numpy as np

from prototypes.v30_video_apex import routes as R


def generate_route_hypotheses(gray, anchor_xy, *, v1=None, dev=None,
                              body_prob=None, body_prob_origin=(0, 0),
                              frst_lengths=(120.0,),
                              frst_radii=(8.0, 13.0, 18.0),
                              n_attach: int = 8,
                              max_v1_peaks: int = 3,
                              max_body_walks: int = 4,
                              min_peak_dist_px: float = 26.0,
                              include_frst: bool = True,
                              include_body_walks: bool = True):
    """Deterministic proposal family from one anchor.

    Returns (props, grain_info). Each prop carries: route_id, seed,
    polyline (NATIVE coordinates), root_source, anchor. body_prob is a
    probability map in crop coordinates (body_prob_origin gives its
    native offset); body walks are converted back to native.
    """
    root_xy, root_source = R.auto_root(gray, anchor_xy)
    grain_c, radius_px, grain_source = R.auto_grain(gray, anchor_xy)
    props: list[dict] = []
    n_frst = n_curved = n_v1 = n_body = 0
    if include_frst and root_source == "frst":
        props += R.propose_from_attachments(
            root_xy, radius_px, dirs_per_attach=1,
            lengths=frst_lengths, radii=frst_radii)
        n_frst = len(props)
        try:
            _cw = R.propose_curved_walkers(
                gray, root_xy, radius_px, n_attach=n_attach,
                dirs_per_attach=1, lengths=frst_lengths,
                radii=frst_radii)
            props += _cw
            n_curved = len(_cw)
        except Exception:  # noqa: BLE001
            pass
    if v1 is not None and dev is not None:
        heat = R.v1_tip_heat(v1, dev, gray)
        peaks = [p for p in R.heat_peaks(heat)
                 if np.hypot(p[0] - root_xy[0],
                             p[1] - root_xy[1]) > min_peak_dist_px]
        for i, (x, y, _h) in enumerate(
                sorted(peaks, key=lambda p: -p[2])[:max_v1_peaks]):
            props.append({"route_id": f"p{i}s", "seed": f"v1-peak-{i}",
                          "polyline": [list(root_xy), [x, y]]})
            props.append({"route_id": f"p{i}c+", "seed": f"v1-peak-{i}",
                          "polyline": R._curve(root_xy, (x, y), 12.0)})
            props.append({"route_id": f"p{i}c-", "seed": f"v1-peak-{i}",
                          "polyline": R._curve(root_xy, (x, y), -12.0)})
            n_v1 += 3
    if include_body_walks and body_prob is not None:
        ox, oy = body_prob_origin
        root_c = (float(root_xy[0]) - ox, float(root_xy[1]) - oy)
        for w in R.propose_body_walks(body_prob, root_c,
                                      max_routes=max_body_walks):
            pts_c = np.asarray(w["polyline"], float)
            props.append({
                "route_id": f"walk-{w.get('route_id', 'w')}",
                "seed": "body-walk",
                "polyline": (pts_c + np.array([ox, oy])).tolist(),
                "root_source": "body-walk"})
            n_body += 1
    for pr in props:
        pr.setdefault("root_source", root_source)
        pr["anchor"] = [float(anchor_xy[0]), float(anchor_xy[1])]
    grain_info = {
        "root_xy": [float(root_xy[0]), float(root_xy[1])],
        "root_source": root_source,
        "grain_center": [float(grain_c[0]), float(grain_c[1])],
        "grain_radius_px": float(radius_px),
        "grain_source": grain_source,
        "family_counts": {"frst": n_frst, "curved": n_curved,
                          "v1_peaks": n_v1, "body_walks": n_body},
        "evidence_used": {
            "v1_peaks": bool(v1 is not None and dev is not None),
            "body_walks": bool(include_body_walks
                               and body_prob is not None)},
    }
    return props, grain_info
