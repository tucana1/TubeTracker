"""Length-and-tip of the chosen curve with the app's tip (the person's route, overlay.route_at) against the tip on
0.8.8's own route at that bin (report.turned_path + drift), cut to the same length."""
import numpy as np

from prototypes.review_curve.curves import curve
from prototypes.review_fill.build import series_index
from prototypes.review_fill.evaluate import load_movie, tip_tol, tol
from sparsetrack.report import turned_path
from tubetracker.app.overlay import to_length

PRM = dict(family="add", W=20, after="decay", stall="tol")
for movie in ("ld", "m2", "m1"):
    out = []
    for proto in ("P1", "P2", "P3"):
        n = hits = app = own = m0 = 0
        for g in load_movie(movie):
            if proto not in g.protocols():
                continue
            anchors = g.anchors(proto)
            la, fv, _ = g.review_onset(min(b for b, _, _ in anchors), "review")
            c = curve(g.model_px, fv, [(b, L) for b, L, _ in anchors], **PRM)
            for b, h, apex, t in g.targets(proto):
                n += 1
                L = float(c[b])
                ok = abs(L - h) <= tol(h)
                hits += ok
                ta = g.m1_tip(b, L, anchors)
                app += bool(ok and ta is not None and np.hypot(*(ta - apex)) <= tip_tol(h))
                to = None
                if g.res and g.res.get("path") and L >= 2.0:
                    i = series_index(g.res, b, g.fpb)
                    to = np.asarray(to_length(turned_path(g.res, i, g.pred).tolist(), L)[-1], float) + g.drift_at(b)
                to = ta if to is None else to
                own += bool(ok and to is not None and np.hypot(*(to - apex)) <= tip_tol(h))
                L0, t0 = g.m0(b, t.get("source_frame") or b * g.fpb + g.fpb // 2)
                m0 += bool(abs(L0 - h) <= tol(h) and t0 is not None and np.hypot(*(t0 - apex)) <= tip_tol(h))
        out.append(f"{proto}: lengths {hits}/{n}, l&t app route {app}, 0.8.8 route {own} (0.8.8 alone {m0})")
    print(movie, " | ".join(out))
