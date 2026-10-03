"""Draw the offline route test: P map at the trace bin, human trace (green), routes exit->apex (thick) and
exit->other end (thin) for the plain geodesic (blue) and a lifted variant (orange); J (black), branch ends."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP, pmap, trace  # noqa: E402

SIDE = 300


def tile(a, var="ori16 w20"):
    m, rest = a["case"].split()
    gid, b = rest.split("@")
    b = int(b)
    H, tr = trace(m, gid, b)
    pts = [H, np.asarray(a["plain"]["route"]), np.asarray(a[var]["route"]), np.asarray(a["T"])[None]]
    allp = np.concatenate(pts)
    x0, y0 = np.floor(allp.min(0) - 12).astype(int)
    x1, y1 = np.ceil(allp.max(0) + 12).astype(int)
    w = max(x1 - x0, y1 - y0)
    P = pmap(m, b, x0, y0, w, w)
    z = SIDE / w
    img = cv2.resize(np.stack([255 * (1 - 0.75 * P)] * 3, -1).astype(np.uint8), (SIDE, SIDE),
                     interpolation=cv2.INTER_NEAREST)
    f = lambda q: tuple(int(round(v)) for v in ((np.asarray(q) - [x0, y0]) * z))

    def poly(q, col, th):
        q = np.asarray(q)
        for p_, r_ in zip(q[:-1], q[1:]):
            cv2.line(img, f(p_), f(r_), col, th, cv2.LINE_AA)
    poly(H, (0, 170, 0), 3)
    poly(a["plain"]["routeT"], (255, 120, 0), 1)
    poly(a[var]["routeT"], (0, 140, 255), 1)
    poly(a["plain"]["route"], (255, 60, 0), 2)
    poly(a[var]["route"], (0, 120, 255), 2)
    cv2.circle(img, f(a["J"]), 4, (0, 0, 0), -1)
    if a.get("hb_end"):
        cv2.circle(img, f(a["hb_end"]), 4, (0, 170, 0), 1)
    if a.get("ob_end"):
        cv2.circle(img, f(a["ob_end"]), 4, (0, 0, 220), 1)
    cv2.circle(img, f(a["T"]), 4, (0, 0, 220), -1)
    head = np.full((30, SIDE, 3), 30, np.uint8)
    pl, lv = a["plain"], a[var]
    cv2.putText(head, f"{a['case']} {a['kind']} {a['cls']} Lh {a['Lh']:.0f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (255, 255, 255), 1, cv2.LINE_AA)
    dec = lambda v: "-" if v["dec_ok"] is None else ("ok" if v["dec_ok"] else "X")
    cv2.putText(head, f"plain cov {pl['cov']:.2f} dec {dec(pl)} | {var} cov {lv['cov']:.2f} dec {dec(lv)}", (3, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
    return np.concatenate([head, img], 0)


if __name__ == "__main__":
    res = json.load(open(SP + "jt/route_test.json"))
    sel = sys.argv[2:]
    var = "ori16 w20"
    tiles = [tile(a, var) for a in res if not sel or a["case"] in sel]
    cols = 4
    rows = -(-len(tiles) // cols)
    th, tw = tiles[0].shape[:2]
    S = np.full((rows * (th + 4), cols * (tw + 4), 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        S[r * (th + 4):r * (th + 4) + th, c * (tw + 4):c * (tw + 4) + tw] = t
    cv2.imwrite(SP + "jt/" + sys.argv[1], S)
    print(S.shape)
