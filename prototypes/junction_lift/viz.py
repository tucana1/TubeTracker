"""Contact sheet of cases: P map at the trace bin with the flood's claimed tube, the human trace and the drawn route;
and the arrival map (when each pixel became tube)."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP, drawn, idx, length_at, movie, tip_at, trace  # noqa: E402
import reflood  # noqa: E402

SIDE = 330


def panel(m, gid, b, extra=None, title=""):
    st = reflood.run(m, gid)
    g = movie(m)["G"][gid]
    i = idx(g, b)
    H, tr = trace(m, gid, b)
    D = drawn(m, g, b)
    Dw = drawn(m, g, b, cut=False)
    tip = tip_at(g, b)
    Hc = reflood.to_crop(st, H, i)
    Dc = reflood.to_crop(st, D, i) if D is not None else None
    Dwc = reflood.to_crop(st, Dw, i) if Dw is not None else None
    tc = reflood.to_crop(st, tip[None], i)[0] if tip is not None else None
    allp = [Hc] + ([Dc] if Dc is not None else []) + [np.array([[st["centre"], st["centre"]]])]
    allp += [e for e in (extra or {}).values()]
    P = np.concatenate(allp)
    x0, y0 = np.floor(P.min(0) - 18).astype(int)
    x1, y1 = np.ceil(P.max(0) + 18).astype(int)
    w = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    x0, y0 = cx - w // 2, cy - w // 2
    n2 = 2 * st["half"]
    x0, y0 = int(np.clip(x0, 0, n2 - w)), int(np.clip(y0, 0, n2 - w))
    w = min(w, n2)
    z = SIDE / w
    pc = reflood.pcrop(st, b)[y0:y0 + w, x0:x0 + w].astype(np.float32) / 250
    tube = st["fl"]["tube"][y0:y0 + w, x0:x0 + w]
    t_in = st["fl"]["t_in"][y0:y0 + w, x0:x0 + w]
    img = np.stack([255 * (1 - 0.8 * pc)] * 3, -1)
    now = tube & (t_in <= i)
    later = tube & (t_in > i)
    img[now] = 0.5 * img[now] + 0.5 * np.array([255, 120, 0])   # BGR: blue-ish
    img[later] = 0.6 * img[later] + 0.4 * np.array([255, 255, 0])  # cyan
    a = st["arr"][y0:y0 + w, x0:x0 + w].astype(np.float32)
    nb = st["arr"].max()
    em = st["fl"]["emerge"] or 0
    arrimg = np.full((w, w, 3), 255, np.float32)
    m_old = a < em
    m_mid = (a >= em) & (a <= i)
    arrimg[m_old] = (150, 150, 150)
    if m_mid.any():
        frac = ((a - em) / max(i - em, 1))
        cm = cv2.applyColorMap(np.clip(frac * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
        arrimg[m_mid] = cm[m_mid]
    m_late = (a > i) & (a < nb)
    arrimg[m_late] = (225, 225, 225)
    out = []
    for base in (img, arrimg):
        im = cv2.resize(base.astype(np.uint8), (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
        f = lambda q: tuple(int(round(v)) for v in ((np.asarray(q) - [x0, y0] + 0.5) * z))
        cv2.circle(im, f([st["centre"], st["centre"]]), int(st["gr"] * z), (90, 90, 90), 1, cv2.LINE_AA)
        if Dwc is not None:
            for p_, q_ in zip(Dwc[:-1], Dwc[1:]):
                cv2.line(im, f(p_), f(q_), (230, 150, 230), 1, cv2.LINE_AA)
        if Dc is not None:
            for p_, q_ in zip(Dc[:-1], Dc[1:]):
                cv2.line(im, f(p_), f(q_), (200, 0, 200), 2, cv2.LINE_AA)
        for p_, q_ in zip(Hc[:-1], Hc[1:]):
            cv2.line(im, f(p_), f(q_), (0, 170, 0), 2, cv2.LINE_AA)
        cv2.circle(im, f(Hc[-1]), 4, (0, 170, 0), -1)
        if tc is not None:
            cv2.circle(im, f(tc), 4, (0, 0, 255), -1)
        for name, e in (extra or {}).items():
            col = {"bend": (0, 140, 255), "plain": (255, 0, 0)}.get(name.split(":")[0], (0, 0, 0))
            for p_, q_ in zip(e[:-1], e[1:]):
                cv2.line(im, f(p_), f(q_), col, 1, cv2.LINE_AA)
        out.append(im)
    both = np.concatenate([out[0], np.full((SIDE, 4, 3), 255, np.uint8), out[1]], 1)
    head = np.full((34, both.shape[1], 3), 30, np.uint8)
    Lh = tr["length_px"]
    cv2.putText(head, f"{m} {gid}@{b} (i={i})  human {Lh:.0f}  model {length_at(g, b):.0f}  emerge {em}  {title}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(head, f"crop {w}px  contact={tr.get('contact')}  blue=claimed by i, cyan=claimed later; arrivals: grey<emerge, jet emerge..i",
                (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    return np.concatenate([head, both], 0)


def sheet(cases, path, cols=2):
    tiles = [panel(*c) for c in cases]
    hh = max(t.shape[0] for t in tiles)
    ww = max(t.shape[1] for t in tiles)
    rows = -(-len(tiles) // cols)
    S = np.full((rows * (hh + 6), cols * (ww + 6), 3), 255, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        S[r * (hh + 6):r * (hh + 6) + t.shape[0], c * (ww + 6):c * (ww + 6) + t.shape[1]] = t
    cv2.imwrite(path, S)
    print(path, S.shape)


if __name__ == "__main__":
    m = sys.argv[1]
    cases = [(m, a.split("@")[0], int(a.split("@")[1])) for a in sys.argv[3:]]
    sheet(cases, SP + "jt/" + sys.argv[2])
