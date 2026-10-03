"""Step 2: offline route test on the P map at each trace's bin (field frame).

For each case: (M1) route from the human exit to the human apex - share of the human trace within 2 px of it;
(M2) the junction decision - from the junction J on the human route, arriving in the human route's direction, which
branch is cheaper over the same arc length D: the human route on, or the other tube (the takeover route / the
nearby tube); (M3) a global decision from the exit: route to the apex vs to the takeover end, compared by cost per
px above the length (excess cost / length)."""
import json
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SP, drawn, idx, movie, pmap, proj_arc, routes, seg_dists, tip_at, trace  # noqa: E402
import lifted  # noqa: E402

D_BRANCH = 20.0
BLUR = 1.5


def other_branch_from_component(m, b, H, J_s, x0, y0, P, census):
    """For a near-other-tube case: the other tube's piece nearest the route, as a polyline from J to a point
    ~D_BRANCH px away on it (plain geodesic on P), field coords."""
    Hs, sH = routes.resample(H, 1.0)
    h, w = P.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.stack([xx.ravel() + x0 + 0.5, yy.ravel() + y0 + 0.5], 1)
    d = seg_dists(pts, Hs).reshape(P.shape)
    mask = (P >= 0.5) & (d > 6.0)
    for c in census.values():
        if c.get("exclude_reason") == "not_a_grain":
            continue
        mask &= np.hypot(xx + x0 + 0.5 - c["x"], yy + y0 + 0.5 - c["y"]) > c["r"] + 4.0
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    J = Hs[min(int(round(J_s)), len(Hs) - 1)]
    best = None
    for l in range(1, n):
        c = lab == l
        if c.sum() < 25 or d[c].max() < 14 or d[c].min() > 11:
            continue
        dj = np.hypot(xx + x0 + 0.5 - J[0], yy + y0 + 0.5 - J[1])
        if best is None or dj[c].min() < best[0]:
            best = (dj[c].min(), c, dj)
    if best is None:
        return None
    _, c, dj = best
    # the component pixel ~D_BRANCH + 4 px from J, farthest from the route among those
    cand = c & (np.abs(dj - D_BRANCH - 4) <= 3)
    if not cand.any():
        cand = c & (dj >= dj[c].max() - 3)
    k = int(np.argmax(np.where(cand, d, -1)))
    ty, tx = np.unravel_index(k, P.shape)
    return np.array([tx + x0 + 0.5, ty + y0 + 0.5])


def cut_at(path, s0, D):
    q, s = routes.resample(path, 1.0)
    k0 = int(np.clip(round(s0), 0, len(q) - 1))
    k1 = int(np.clip(round(s0 + D), 0, len(q) - 1))
    return q[k0:k1 + 1]


def cases():
    rows = json.load(open(SP + "jt/cases.json"))
    out = []
    for r in rows:
        tk = r["cls"] in ("takeover", "takeover_start", "overshoot_onto_tube") or (
            r["cls"] == "off_route" and (r["m"], r["g"], r["b"]) in {("m2", "g038", 244), ("m2", "g038", 349)})
        extra = (r["m"], r["g"], r["b"]) in {("m2", "g005", 349), ("m2", "g011", 349)}
        if tk or extra or r["near_other"] is not None:
            out.append(r)
    return out


VARIANTS = [
    ("plain", {}),
    ("iso16 w5", dict(K=16, w_bend=5.0, evidence="iso")),
    ("iso16 w20", dict(K=16, w_bend=20.0, evidence="iso")),
    ("ori16 w0", dict(K=16, w_bend=0.0, evidence="oriented")),
    ("ori16 w5", dict(K=16, w_bend=5.0, evidence="oriented")),
    ("ori16 w20", dict(K=16, w_bend=20.0, evidence="oriented")),
]


def analyse(r, variants=VARIANTS):
    m, gid, b = r["m"], r["g"], r["b"]
    mv = movie(m)
    g = mv["G"][gid]
    census = mv["lab"]["grains"]
    H, tr = trace(m, gid, b)
    Hs, sH = routes.resample(H, 1.0)
    Lh = sH[-1]
    tip = tip_at(g, b)
    D = drawn(m, g, b)
    # the other branch: the takeover route (drawn route from its divergence), else the nearby tube
    kind, T, other = None, None, None
    if r.get("s_div") is not None and D is not None and r["cls"] != "hit":
        Ds, sD = routes.resample(D, 1.0)
        s_div = r["s_div"]
        # junction on the human route: projection of the last close point before the divergence
        J_s = float(proj_arc(Ds[max(s_div - 1, 0):max(s_div, 1)], Hs)[0]) if s_div > 0 else 0.0
        T = Ds[-1] if tip is None or seg_dists(tip[None], Ds)[0] > 3 else tip
        other = Ds[max(s_div - 1, 0):]
        kind = "takeover"
    elif r["near_other"] is not None:
        J_s = r["near_other"][1]
        kind = "near"
    else:
        return None
    pts = [H] + ([D] if D is not None else []) + ([T[None]] if T is not None else [])
    allp = np.concatenate(pts)
    x0, y0 = (np.floor(allp.min(0)) - 30).astype(int)
    x1, y1 = (np.ceil(allp.max(0)) + 30).astype(int)
    P = pmap(m, b, x0, y0, x1 - x0, y1 - y0)
    Pc = cv2.GaussianBlur(P, (0, 0), BLUR) if BLUR else P  # routes in the middle of the band, not along a wall
    if kind == "near":
        T = other_branch_from_component(m, b, H, J_s, x0, y0, P, census)
        if T is None:
            return None
    to_pix = lambda q: (int(np.clip(math.floor(q[1]) - y0, 0, P.shape[0] - 1)),
                        int(np.clip(math.floor(q[0]) - x0, 0, P.shape[1] - 1)))
    h, w = P.shape
    N = h * w
    S, A = to_pix(H[0]), to_pix(H[-1])
    J = Hs[int(np.clip(round(J_s), 0, len(Hs) - 1))]
    Jp = to_pix(J)
    back = Hs[int(np.clip(round(J_s) - 10, 0, len(Hs) - 1))]
    th_in = math.atan2(J[1] - back[1], J[0] - back[0]) if np.hypot(*(J - back)) > 2 else math.atan2(
        Hs[min(10, len(Hs) - 1)][1] - Hs[0][1], Hs[min(10, len(Hs) - 1)][0] - Hs[0][0])
    # branch endpoints at equal arc length D beyond J
    Dh = min(D_BRANCH, Lh - J_s)
    res = {"case": f"{m} {gid}@{b}", "kind": kind, "cls": r["cls"], "Lh": round(Lh, 1), "J_s": round(J_s, 1),
           "contact": r["contact"], "len_hit": r["len_hit"], "both": r["both"]}
    hb_end = Hs[int(np.clip(round(J_s + Dh), 0, len(Hs) - 1))] if Dh >= 6 else None
    # the other branch at arc D from J: along the takeover route, or the plain geodesic to T
    if kind == "takeover":
        oq, os_ = routes.resample(other, 1.0)
        ob_end = oq[int(np.clip(round(min(D_BRANCH, os_[-1])), 0, len(oq) - 1))] if os_[-1] >= 6 else None
    else:
        ob_end = T
    res["ends_at_junction"] = hb_end is None
    gc = np.array([g["x"], g["y"]]) + (np.asarray(g["drift"]["xy"][idx(g, b)]) if g.get("drift") else 0)
    out_dir = math.atan2(H[0][1] - gc[1], H[0][0] - gc[0])
    for name, kw in variants:
        t0 = time.time()
        if name == "plain":
            G = lifted.plain_graph(Pc)
            dist, pred = lifted.run(G, [S[0] * w + S[1]])
            nodesA = lifted.backtrack(pred, A[0] * w + A[1])
            route = lifted.path_xy(nodesA, P.shape) + [x0 + 0.5, y0 + 0.5]
            dA = dist[A[0] * w + A[1]]
            Tp = to_pix(T)
            dT = dist[Tp[0] * w + Tp[1]]
            routeT = lifted.path_xy(lifted.backtrack(pred, Tp[0] * w + Tp[1]), P.shape) + [x0 + 0.5, y0 + 0.5]
            dj, _ = lifted.run(G, [Jp[0] * w + Jp[1]], return_pred=False)
            dec = None
            if hb_end is not None and ob_end is not None:
                hp, op = to_pix(hb_end), to_pix(ob_end)
                dec = (float(dj[hp[0] * w + hp[1]]), float(dj[op[0] * w + op[1]]))
        else:
            G, th = lifted.lifted_graph(Pc, **kw)
            K = len(th)
            outs = [k for k in range(K) if abs(math.remainder(th[k] - out_dir, 2 * math.pi)) <= math.pi / 2]
            dist, pred = lifted.run(G, [k * N + S[0] * w + S[1] for k in outs])
            kA, dA = lifted.best_k(dist, A[0] * w + A[1], K, N)
            nodesA = lifted.backtrack(pred, kA * N + A[0] * w + A[1])
            route = lifted.path_xy(nodesA, P.shape) + [x0 + 0.5, y0 + 0.5]
            Tp = to_pix(T)
            kT, dT = lifted.best_k(dist, Tp[0] * w + Tp[1], K, N)
            routeT = lifted.path_xy(lifted.backtrack(pred, kT * N + Tp[0] * w + Tp[1]), P.shape) + [x0 + 0.5, y0 + 0.5]
            kin = int(np.argmin([abs(math.remainder(t - th_in, 2 * math.pi)) for t in th]))
            dj, _ = lifted.run(G, [((kin + o) % K) * N + Jp[0] * w + Jp[1] for o in (-1, 0, 1)], return_pred=False)
            dec = None
            if hb_end is not None and ob_end is not None:
                hp, op = to_pix(hb_end), to_pix(ob_end)
                dec = (lifted.best_k(dj, hp[0] * w + hp[1], K, N)[1], lifted.best_k(dj, op[0] * w + op[1], K, N)[1])
            del G
        dd = seg_dists(Hs, route) if len(route) > 1 else np.full(len(Hs), 99.0)
        cov = float(np.mean(dd <= 2.0))
        cov4 = float(np.mean(dd <= 4.0))
        LA = routes.arc(route)[-1] if len(route) > 1 else 0.0
        LT = routes.arc(routeT)[-1] if len(routeT) > 1 else 0.0
        exA = (dA - LA) / max(LA, 1.0)
        exT = (dT - LT) / max(LT, 1.0)
        res[name] = {"cov": round(cov, 2), "cov4": round(cov4, 2), "dec": None if dec is None else [round(v, 1) for v in dec],
                     "dec_ok": None if dec is None else bool(dec[0] < dec[1]),
                     "global_ok": bool(exA < exT), "exA": round(exA, 2), "exT": round(exT, 2),
                     "LA": round(LA, 1), "sec": round(time.time() - t0, 1), "route": np.round(route, 1).tolist(),
                     "routeT": np.round(routeT, 1).tolist()}
    res["T"] = np.round(T, 1).tolist()
    res["J"] = np.round(J, 1).tolist()
    res["hb_end"] = None if hb_end is None else np.round(hb_end, 1).tolist()
    res["ob_end"] = None if ob_end is None else np.round(ob_end, 1).tolist()
    return res


if __name__ == "__main__":
    sel = sys.argv[1:]
    out = []
    for r in cases():
        key = f"{r['m']} {r['g']}@{r['b']}"
        if sel and key.split()[0] not in sel and key not in sel:
            continue
        a = analyse(r)
        if a is None:
            print("skip", key, r["cls"], flush=True)
            continue
        out.append(a)
        json.dump(out, open(SP + "jt/route_test.json", "w"))
        line = f"{key:14s} {a['kind']:8s} {a['cls']:20s} Lh {a['Lh']:6.1f} J {a['J_s']:6.1f} {'C' if a['contact'] else ' '} " \
               f"{'end@J' if a['ends_at_junction'] else '     '} |"
        for name, _ in VARIANTS:
            v = a[name]
            line += f" {name}: cov {v['cov']:.2f} dec {'-' if v['dec_ok'] is None else ('ok' if v['dec_ok'] else 'X ')} glob {'ok' if v['global_ok'] else 'X '} |"
        print(line, flush=True)
    json.dump(out, open(SP + "jt/route_test.json", "w"))
