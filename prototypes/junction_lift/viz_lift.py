"""Where a lift variant changed a grain's reading: the final routes of 0.8.8 (magenta) and the variant (orange) over
the tube map at the last bin, the variant's lifted path pixels (red), the human traces (green, apex dot)."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP, movie, pmap  # noqa: E402
import reflood  # noqa: E402

SIDE = 420


def tile(m, gid, kw):
    base = movie(m)["G"][gid]
    st = reflood.run(m, gid, p=reflood.base_params(m, **kw), use_cache=False)
    res = st["res"]
    n = len(res["length"]["frames"])
    i = n - 1
    b = st["rs"] + i
    d = st["drift"][i]
    routes_ = [np.asarray(base.get("path") or [[np.nan, np.nan]], float) + d,
               np.asarray(res.get("path") or [[np.nan, np.nan]], float) + d]
    lab = movie(m)["lab"]["labels"][gid].get("traces") or {}
    traces = [(k, np.asarray(t["path_xy_ref"], float), t["state"]) for k, t in lab.items()
              if t.get("path_xy_ref") and len(t["path_xy_ref"]) >= 2]
    pts = np.concatenate([r for r in routes_ if np.isfinite(r).all()] + [t for _, t, _ in traces]
                         + [np.array([[base["x"] + d[0], base["y"] + d[1]]])])
    x0, y0 = np.floor(pts.min(0) - 15).astype(int)
    x1, y1 = np.ceil(pts.max(0) + 15).astype(int)
    w = max(x1 - x0, y1 - y0)
    P = pmap(m, b, x0, y0, w, w)
    z = SIDE / w
    img = cv2.resize(np.stack([255 * (1 - 0.75 * P)] * 3, -1).astype(np.uint8), (SIDE, SIDE),
                     interpolation=cv2.INTER_NEAREST)
    f = lambda q: tuple(int(round(v)) for v in ((np.asarray(q) - [x0, y0]) * z))
    # the variant's bridged pixels: in its tube but not reachable by the plain flood's claim (tube minus base tube)
    base_st = reflood.run(m, gid)
    extra = st["fl"]["tube"] & ~base_st["fl"]["tube"]
    ey, ex = np.nonzero(extra)
    for y, x in zip(ey, ex):
        q = reflood.to_field(st, np.array([[x, y]], float), i)[0]
        cv2.circle(img, f(q), 1, (60, 60, 230), -1)
    for k, t, s in traces:
        col = (0, 170, 0) if s == "full" else (0, 120, 60)
        for p_, q_ in zip(t[:-1], t[1:]):
            cv2.line(img, f(p_), f(q_), col, 2, cv2.LINE_AA)
        cv2.circle(img, f(t[-1]), 3, col, -1)
        cv2.putText(img, f"{k}{'' if s == 'full' else ' ' + s[0]}", f(t[-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    for r, col in zip(routes_, [(200, 0, 200), (0, 140, 255)]):
        if np.isfinite(r).all():
            for p_, q_ in zip(r[:-1], r[1:]):
                cv2.line(img, f(p_), f(q_), col, 2, cv2.LINE_AA)
    head = np.full((22, SIDE, 3), 30, np.uint8)
    cv2.putText(head, f"{m} {gid} last bin {b}: 0.8.8 {base.get('final_length_px')} -> lift {res.get('final_length_px')} px",
                (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([head, img], 0)


if __name__ == "__main__":
    out, m = sys.argv[1], sys.argv[2]
    kw = {}
    gids = []
    for a in sys.argv[3:]:
        if "=" in a:
            k, v = a.split("=")
            kw[k] = float(v) if "." in v else int(v)
        else:
            gids.append(a)
    tiles = [tile(m, g, kw) for g in gids]
    S = np.concatenate([np.concatenate([t, np.full((t.shape[0], 4, 3), 255, np.uint8)], 1) for t in tiles], 1)
    cv2.imwrite(SP + "jt/" + out, S)
    print(S.shape)
