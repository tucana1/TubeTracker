"""Where to draw a grain and its tube at a given bin (plain geometry on the grain records of ``model.RunData``).

A grain record holds its census place (``x``, ``y``, ``r``), its drift per bin if the analysis followed it as it
moved (``drift``), the model's tube route in the grain's own frame (``path``, turned per bin by ``rot`` about
``pivot``), the routes a person traced (``human``: bin, points), and its tube length at every bin (``L``).
"""

from __future__ import annotations

import math

EMERGED = ("emerged_within", "emerged_at_start")
MIN_TUBE_PX = 2.0  # shorter is no tube: not drawn, not measured, and confirmed as none (corrections.confirm)
NEAR_BINS = 20     # a person's traced route is drawn at the bins this close to one they traced; elsewhere the model's
JOIN_PX = 10.0     # the model's route is carried on along a person's longer route if its end lies this close to it
EXTEND_PX = 5.0    # a route is carried on straight past its end by at most this to a tube's length (to_length)


def path_length(pts) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def to_length(pts, length: float, max_extend: float = 5.0) -> list:
    """The route cut to ``length`` px of arc, or carried on along its last direction by at most ``max_extend`` px
    (``sparsetrack.review.to_length``)."""
    pts = [list(map(float, p)) for p in pts]
    if len(pts) < 2:
        return pts
    s = [0.0]
    for a, b in zip(pts, pts[1:]):
        s.append(s[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    if length <= s[-1]:
        i = next((k for k, v in enumerate(s) if v >= length), len(pts) - 1)
        i = max(1, i)
        a = (length - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
        p, q = pts[i - 1], pts[i]
        return pts[:i] + [[p[0] + a * (q[0] - p[0]), p[1] + a * (q[1] - p[1])]]
    last = pts[-1]
    back = next((q for q in reversed(pts[:-1]) if math.hypot(last[0] - q[0], last[1] - q[1]) >= 3.0), pts[0])
    d = math.hypot(last[0] - back[0], last[1] - back[1]) or 1.0
    e = min(length - s[-1], max_extend)
    return pts + [[last[0] + (last[0] - back[0]) / d * e, last[1] + (last[1] - back[1]) / d * e]]


def turned(path, deg: float, pivot) -> list:
    """The model's route turned by ``deg`` about ``pivot`` (``sparsetrack.report.turned_path`` for one bin)."""
    if not path or not deg or pivot is None:
        return [list(map(float, p)) for p in path or []]
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    px, py = pivot
    return [[(x - px) * c - (y - py) * s + px, (x - px) * s + (y - py) * c + py] for x, y in path]


def project(route, pt) -> tuple[float, float]:
    """(arc length along ``route`` of the point nearest ``pt``, the distance to it)."""
    best = (0.0, math.inf)
    acc = 0.0
    for (x0, y0), (x1, y1) in zip(route, route[1:]):
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        a = 0.0 if seg < 1e-9 else max(0.0, min(1.0, ((pt[0] - x0) * dx + (pt[1] - y0) * dy) / (seg * seg)))
        d = math.hypot(x0 + a * dx - pt[0], y0 + a * dy - pt[1])
        if d < best[1]:
            best = (acc + a * seg, d)
        acc += seg
    return best


def drift_at(g: dict, b: int) -> tuple[float, float]:
    d = g.get("drift")
    if not d:
        return (0.0, 0.0)
    x, y = d[max(0, min(int(b), len(d) - 1))]
    return (float(x), float(y))


def pos_at(g: dict, b: int) -> tuple[float, float]:
    """Where the grain is at bin ``b`` (its census place, plus its drift if it was followed)."""
    dx, dy = drift_at(g, b)
    return (g["x"] + dx, g["y"] + dy)


def bent_at(g: dict, b: int) -> list:
    """The model's route as the tube lay at bin ``b`` (``sparsetrack.routes.bent``; the record carries the route's
    arc lengths and normals, ``model.RunData``)."""
    own = g.get("by_bin")
    if own and 0 <= b < len(own["index"]) and own["index"][b] >= 0:
        return own["routes"][own["index"][b]]  # the flood read the tube along another route then
    path = g.get("path") or []
    bend = g.get("bend")
    rows = (bend or {}).get("rows") or []
    if not path or not 0 <= b < len(rows) or not rows[b]:
        return path
    import numpy as np
    from sparsetrack.routes import offsets_along
    off = offsets_along(bend["s"], rows[b], bend["step"])
    return (np.asarray(path, float) + off[:, None] * bend["n"]).tolist()


def model_route(g: dict, b: int) -> list:
    """The model's route as the tube lay at bin ``b`` (bent, or the flood's own route then), turned to ``b``."""
    rot = g.get("rot")
    return turned(bent_at(g, b), rot[b] if rot else 0.0, g.get("pivot"))


def route_at(g: dict, b: int) -> tuple[list, bool]:
    """The route the tube is drawn (and a tip click read) along at bin ``b``, in the grain's own frame, and whether it
    is a person's. With no route a person traced, the model's route at ``b``. Within ``NEAR_BINS`` of a bin a person
    traced, their route: the first traced at or after ``b``, else their last one carried on along the model's route
    where it ended on it. Farther, the model's own route at ``b``, which follows the tube's sway and turns, carried on
    along the person's route (the first at or after ``b``, else the last) where the tube is longer than the model's
    route reaches; but the person's route as near where the model had read no tube by ``b`` (its route is then that
    of a tube it saw later) or its route cannot be carried on to the tube's length. (Until 3 Oct 2026 a person's route
    was drawn at every bin; at the same lengths, tips on the model's per-bin route were in tolerance more often:
    prototypes/review_curve/routes.py.)"""
    model = model_route(g, b)
    drawn = [t for t in g.get("human") or [] if t.get("pts")]
    if not drawn:
        return model, False
    near = [t for t in drawn if abs(t["bin"] - b) <= NEAR_BINS]
    if not near and len(model) >= 2 and read_by_model(g, b):
        later = [t for t in drawn if t["bin"] >= b]
        L = g.get("L")
        route = along(model, (later[0] if later else drawn[-1])["pts"],
                      float(L[b]) if L and 0 <= b < len(L) else 0.0)
        if route is not None:
            return route, False
    pool = near or drawn
    later = [t for t in pool if t["bin"] >= b]
    if later:
        return later[0]["pts"], True
    return continued(pool[-1]["pts"], model), True


def read_by_model(g: dict, b: int) -> bool:
    """Whether the model read a tube at bin ``b``: its own length there (``Lm`` once a person's answers changed the
    curve, else ``L``) at least ``MIN_TUBE_PX``."""
    L = g.get("Lm") or g.get("L")
    return bool(L) and 0 <= b < len(L) and float(L[b]) >= MIN_TUBE_PX


def along(model, pts, need: float) -> list | None:
    """The model's route, long enough for a tube of ``need`` px: as it is if it is (up to ``EXTEND_PX`` short); else
    carried on along a person's route ``pts`` beyond where the model's ends (its end within ``JOIN_PX`` of theirs; the
    rest moved to join it); None if neither reaches."""
    model = [list(map(float, p)) for p in model]
    have = path_length(model)
    if have >= need - EXTEND_PX - 0.5:  # long enough, or carried on straight a little (to_length)
        return model
    s_end, d_end = project(pts, model[-1])
    if d_end > JOIN_PX or path_length(pts) - s_end < need - have - 0.5:
        return None
    on = to_length(pts, s_end, 0.0)[-1]
    dx, dy = model[-1][0] - on[0], model[-1][1] - on[1]
    acc, rest = 0.0, []
    for a, q in zip(pts, pts[1:]):
        acc += math.hypot(q[0] - a[0], q[1] - a[1])
        if acc > s_end + 0.5:
            rest.append([q[0] + dx, q[1] + dy])
    return model + rest


def continued(pts, model, tol: float = 2.5) -> list:
    """A traced route carried on along the model's route beyond its end, if it ends on the model's route."""
    if len(model) < 2 or not pts:
        return [list(p) for p in pts]
    s_end, d_end = project(model, pts[-1])
    if d_end > tol:
        return [list(p) for p in pts]
    acc, rest = 0.0, []
    for a, q in zip(model, model[1:]):
        acc += math.hypot(q[0] - a[0], q[1] - a[1])
        if acc > s_end + 0.5:
            rest.append(list(q))
    return [list(p) for p in pts] + rest


def shift(pts, d) -> list:
    return [[p[0] + d[0], p[1] + d[1]] for p in pts] if d[0] or d[1] else [list(p) for p in pts]


def tube_at(g: dict, b: int) -> list | None:
    """The tube at bin ``b`` where it is in the field (the route cut to the tube's length then), or None."""
    L = g["L"][b] if b < len(g["L"]) else 0.0
    if L < MIN_TUBE_PX or g.get("excluded"):
        return None
    route, _ = route_at(g, b)
    if len(route) < 2:
        return None
    return shift(to_length(route, L), drift_at(g, b))


def state_at(g: dict, b: int) -> str:
    """germinated, notyet, never, lost, excluded or unobservable, at bin ``b``."""
    if g.get("excluded"):
        return "excluded"
    if g["status"] == "unobservable":
        return "unobservable"
    if g.get("lost") is not None and b >= g["lost"]:
        return "lost"
    if g["status"] == "emerged_at_start":
        return "germinated"
    if g["status"] == "emerged_within" and g.get("onset") is not None:
        return "germinated" if b >= g["onset"] else "notyet"
    return "never"


def needs_check(g: dict) -> bool:
    return bool(g.get("check")) and not g.get("done") and not g.get("excluded")
