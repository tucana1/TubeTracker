"""The global tip-trajectory reader on stored candidates (cands.py): one Viterbi over all bins per grain.

States per bin: not germinated (N), one of the bin's candidates (a tip with its body), or held (H: the tube keeps its
last reading; no candidate explains the bin). Costs (minimised):
- unary, candidate k at bin t: theta - S(t, k), S = w_det x detector at the tip (with ``det_norm`` 2: divided by the
  movie's detector scale, the 99th percentile of the full-frame map's peaks; 1: of the candidates' 90th percentile) +
  w_sup x map support along the body - w_gap x longest unmarked gap (px / 10) - w_ahead x map ahead of the tip -
  w_tan x max(0, 0.5 - radial gain over the first 8 px) - w_carry (a carried "hold" candidate); N: 0; H: w_hold.
- transitions: N -> N 0; N -> k: w_on + w_l0 x max(0, L - l0) / 10 (at most l0_max); k -> N forbidden;
  j -> k: the length change dL = L_k - L_j in [-shrink, vmax x bins apart], the shorter body's tip within cap_tip of
  the longer body and its points within cap_share on average; cost w_shrink x max(0, -dL) + w_cons x max(0, d_tip -
  1.5) + w_share x max(0, d_share - 1.5) (grain frame: jumping onto another tube is ruled out); k -> H, H -> H:
  w_hold; H -> k: w_reacq (+ the same length and tip terms, tip within cap_reacq).
Length = body arc length from the rim - the grain's visible edge offset there (exit_edge) - tip offset c + w_ext x
the map's band beyond the tip (- c_end for map ends); non-decreasing from the onset with ``iso``; zero before it.
sparsetrack/tiptraj.py holds a copy of this Viterbi and of cands.py's candidates (checked identical on ld g008).

    python -m prototypes.tip_trajectory.dp oracle ld
"""
from __future__ import annotations

import copy
import json
import math
import pickle
import sys

import numpy as np

from .common import OUT, REPO, baseline, labels, length_class, scored_grains

sys.path.insert(0, str(REPO / "scripts"))
from sparsetrack.evaluate import score  # noqa: E402

FI = None  # feature index, set on load


def load(movie: str, name: str | None = None) -> dict:
    global FI
    with open(OUT / (name or f"cands_{movie}.pkl"), "rb") as fh:
        doc = pickle.load(fh)
    FI = {k: i for i, k in enumerate(doc["feats"])}
    for G in doc["grains"].values():  # every candidate of the grain in one array: per bin a slice
        Fs, off, n0 = [], {}, 0
        for b in sorted(G["bins"]):
            B = G["bins"][b]
            if B["n"]:
                Fs.append(B["F"])
                off[b] = (n0, n0 + B["n"])
                n0 += B["n"]
        G["_F"] = np.concatenate(Fs) if Fs else np.zeros((0, len(FI)), np.float32)
        G["_off"] = off
    det = np.concatenate([G["_F"][:, FI["det"]] for G in doc["grains"].values()])
    doc["det_q90"] = float(np.percentile(det, 90)) if len(det) else 1.0  # the movie's detector scale (label-free)
    doc["det_frame_q99"] = frame_scale(OUT / "det" / doc["movie"])
    return doc


def frame_scale(det_dir, every: int = 10) -> float:
    """A movie's detector scale without grains or labels: the 99th percentile of the tip map's local maxima (9 x 9,
    >= 0.05) over the whole frame, every ``every``-th bin."""
    import cv2
    from pathlib import Path
    vals = []
    for f in sorted(Path(det_dir).glob("b*.npz"))[::every]:
        t = np.load(f)["tip"].astype(np.float32) / 250.0
        pk = (t == cv2.dilate(t, np.ones((9, 9), np.uint8))) & (t >= 0.05)
        vals.append(t[pk])
    v = np.concatenate(vals) if vals else np.zeros(0)
    return float(np.percentile(v, 99)) if len(v) else 1.0


def traces(lab: dict, gid: str, contact: bool = False) -> list[tuple[int, float, np.ndarray, dict]]:
    out = []
    for b, t in sorted(((int(b), t) for b, t in (lab["labels"][gid].get("traces") or {}).items()), key=lambda x: x[0]):
        if t["state"] == "full" and (contact or not t.get("contact")):
            out.append((b, float(t["length_px"]), np.asarray(t["path_xy_ref"][-1], float), t))
    return out


def lengths_of(F: np.ndarray, c, p: dict | None = None) -> np.ndarray:
    """Candidate lengths: arc length from the rim less the visible edge offset and the tip offset; with ``p``: plus
    ``w_ext`` x how far the map's band goes on past the tip, less ``c_end`` more for map-end candidates."""
    ec = (p or {}).get("edge_clip", 99.0)
    L = F[:, FI["L_arc"]] - np.clip(F[:, FI["edge"]], -ec, ec) - c
    if p is not None:
        if "ext" in FI:
            L = L + p.get("w_ext", 0.0) * F[:, FI["ext"]]
        src = F[:, FI["src"]]
        L = L - p.get("c_end", 0.0) * ((src == 2) | (src == 6))
    return np.maximum(L, 0.0)


def oracle(doc: dict, lab: dict, c: float = 1.5) -> dict:
    """How often a candidate lies at the human apex (<= 4 px) and how often one gives the human length (within
    max(2 px, 10%)) - the ceiling of any choice among the candidates."""
    rows = []
    for gid, G in doc["grains"].items():
        for b, h, apex, _ in traces(lab, gid):
            B = G["bins"].get(b, {"n": 0})
            row = {"gid": gid, "bin": b, "h": h, "cls": length_class(h), "n": B["n"]}
            if B["n"]:
                tips = B["tips"].astype(float) + G["drift"][b]  # reference coordinates
                d = np.hypot(*(tips - apex).T)
                L = lengths_of(B["F"], c)
                tol = max(2.0, 0.1 * h)
                row.update(tip_min=float(d.min()), tip4=bool((d <= 4).any()), len_any=bool((np.abs(L - h) <= tol).any()),
                           both=bool(((d <= max(5, 0.1 * h)) & (np.abs(L - h) <= tol)).any()),
                           near_len_err=float(L[int(np.argmin(d))] - h), near_src=int(B["F"][int(np.argmin(d)), FI["src"]]))
            rows.append(row)
    return rows


def summarize_oracle(rows: list[dict]) -> str:
    out = []
    for cls in ("young", "mid", "long", "all"):
        rr = [r for r in rows if cls == "all" or r["cls"] == cls]
        if not rr:
            continue
        n = len(rr)
        out.append(f"{cls}: n={n} tip<=4 {sum(r.get('tip4', False) for r in rr)} any-length {sum(r.get('len_any', False) for r in rr)}"
                   f" tip&length {sum(r.get('both', False) for r in rr)}; median cands {np.median([r['n'] for r in rr]):.0f}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------- the reader
DEFAULT = {"c": 1.5, "theta": 0.35, "w_det": 1.0, "w_sup": 0.6, "w_gap": 0.3, "w_ahead": 0.3, "w_tan": 0.5,
           "w_carry": 0.05, "w_on": 1.0, "l0": 12.0, "vmax": 4.0, "shrink": 3.0, "w_shrink": 0.2, "w_cons": 0.3,
           "w_share": 0.3, "cap_tip": 6.0, "cap_share": 6.0, "iso": True,
           "w_ext": 0.0, "c_end": 0.0, "w_l0": 1e3, "l0_max": 60.0, "w_hold": 0.3, "w_reacq": 0.5, "cap_reacq": 12.0,
           "w_hug": 0.0, "hug_deg": 20.0}


def unary(F: np.ndarray, p: dict) -> np.ndarray:
    S = (p["w_det"] * np.minimum(F[:, FI["det"]] / p.get("_det_scale", 1.0), 1.5) + p["w_sup"] * F[:, FI["sup"]] - p["w_gap"] * F[:, FI["gap"]] / 10.0
         - p["w_ahead"] * F[:, FI["ahead"]] - p["w_tan"] * np.maximum(0.0, 0.5 - F[:, FI["radial"]])
         - p["w_carry"] * (F[:, FI["src"]] >= 4))
    return p["theta"] - S


def _start_cost(L: np.ndarray, p: dict) -> np.ndarray:
    return np.where(L <= p.get("l0_max", 1e9),
                    p["w_on"] + p.get("w_l0", 1e3) * np.maximum(0.0, L - p["l0"]) / 10.0, np.inf)


def viterbi(G: dict, rs: int, nb: int, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per bin (rs .. nb-1): the chosen candidate (-1: not germinated, -2: the tube held at its last reading) and the
    length. States: N (not germinated), each candidate C_k, and H (the tube is there but no candidate explains this
    bin: it keeps the length, tip and body of the reading it came from, at ``w_hold`` a bin; it may hand over to a
    candidate later at ``w_reacq`` if the length change and the body agree). H keeps the attributes of its best
    predecessor (a greedy choice inside an otherwise exact Viterbi)."""
    bins = G["bins"]
    w_hold, w_reacq = p.get("w_hold", 0.3), p.get("w_reacq", 0.5)
    L_all, U_all = lengths_of(G["_F"], p["c"], p), unary(G["_F"], p)
    S_all = _start_cost(L_all, p)
    off = G["_off"]
    Lof = lambda bb, kk: float(L_all[off[bb][0] + kk])  # noqa: E731
    cN = 0.0
    cH, hold = np.inf, None          # hold: (bin of the reading, index, length)
    prev = None                      # last bin with candidates: (bin, costs, lengths)
    back = {}                        # bin -> (per k: predecessor code, H's origin at that bin)
    for b in range(rs, nb):
        B = bins.get(b, {"n": 0})
        if not B["n"]:
            if prev is not None and np.isfinite(prev[1]).any():
                j = int(np.argmin(prev[1]))
                if prev[1][j] + w_hold < cH + w_hold:
                    cH, hold = prev[1][j] + w_hold, (prev[0], j, float(prev[2][j]))
                else:
                    cH += w_hold
                # the candidates of the last bin stay the predecessors of the next bin's (pairs_with)
                prev = (prev[0], prev[1] + w_hold, prev[2])
            else:
                cH += w_hold
            back[b] = ("empty", hold)
            continue
        s0, s1 = off[b]
        L, U = L_all[s0:s1], U_all[s0:s1]
        n = s1 - s0
        best = cN + S_all[s0:s1]
        code = np.full(n, -1)  # -1 from N, j >= 0 from C_j of prev bin, -3 from H
        if prev is not None and B["pairs_with"] == prev[0] and len(B["pairs"]):
            pb, pc, pL = prev
            gap = b - pb
            pr = B["pairs"]
            j, k = pr[:, 0].astype(int), pr[:, 1].astype(int)
            dL = L[k] - pL[j]
            ok = ((dL >= -p["shrink"]) & (dL <= p["vmax"] * gap) & np.isfinite(pc[j])
                  & (pr[:, 2] <= p["cap_tip"]) & (pr[:, 3] <= p["cap_share"]))
            if ok.any():
                j, k, dL, pr = j[ok], k[ok], dL[ok], pr[ok]
                T = (pc[j] + p["w_shrink"] * np.maximum(0.0, -dL) + p["w_cons"] * np.maximum(0.0, pr[:, 2] - 1.5)
                     + p["w_share"] * np.maximum(0.0, pr[:, 3] - 1.5))
                o = np.lexsort((T, k))  # per k, its cheapest predecessor first
                first = np.ones(len(o), bool)
                first[1:] = k[o][1:] != k[o][:-1]
                kk, tt, jj = k[o][first], T[o][first], j[o][first]
                better = tt < best[kk]
                best[kk[better]] = tt[better]
                code[kk[better]] = jj[better]
        if np.isfinite(cH) and hold is not None:  # H -> C_k: re-acquire
            hb, hj, hL = hold
            dL = L - hL
            ok = (dL >= -p["shrink"]) & (dL <= p["vmax"] * (b - hb))
            if ok.any():
                tip_h = bins[hb]["tips"][hj].astype(float)
                body_h = bins[hb]["bodies"][hj] / 10.0
                tips = B["tips"].astype(float)
                if "_cat" not in B:  # all bodies of the bin in one array (built once)
                    B["_cat"] = (np.concatenate(B["bodies"]) / 10.0,
                                 np.cumsum([0] + [len(q) for q in B["bodies"]])[:-1])
                cat, starts = B["_cat"]
                d_fwd = np.minimum.reduceat(np.hypot(*(cat - tip_h).T), starts)  # held tip -> each body
                d_back = np.sqrt(((tips[:, None, :] - body_h[None]) ** 2).sum(-1)).min(1)  # each tip -> held body
                d = np.where(dL >= 0, d_fwd, d_back)
                T = np.where(ok & (d <= p.get("cap_reacq", p["cap_tip"])), cH + w_reacq + p["w_shrink"] * np.maximum(0.0, -dL)
                             + p["w_cons"] * np.maximum(0.0, d - 1.5), np.inf)
                better = T < best
                best[better] = T[better]
                code[better] = -3
        # H at this bin: from the last bin's candidates or from H
        hold_here = hold
        cH_new = cH + w_hold
        if prev is not None and np.isfinite(prev[1]).any():
            j = int(np.argmin(prev[1]))
            if prev[1][j] + w_hold < cH_new:
                cH_new, hold_here = prev[1][j] + w_hold, (prev[0], j, float(prev[2][j]))
        back[b] = (code, hold)  # hold: the origin H had when entering this bin (for code -3)
        cur = best + U
        prev = (b, cur, L)
        cH, hold = cH_new, hold_here
        back[b] = back[b] + (hold,)  # H's origin after this bin (for an H state at this bin)
    choice = np.full(nb - rs, -1)
    lengths = np.zeros(nb - rs)
    # termination: N, the last candidates or H
    endC = prev[1].min() if prev is not None and len(prev[1]) else np.inf
    if min(endC, cH) >= cN:
        return choice, lengths
    if endC <= cH:
        state = ("C", prev[0], int(np.argmin(prev[1])))
    else:
        state = ("H", nb - 1, hold)
    b = nb - 1
    while b >= rs:
        if state[0] == "C":
            sb, k = state[1], state[2]
            for bb in range(sb + 1, b + 1):  # empty bins after it: held
                choice[bb - rs], lengths[bb - rs] = -2, Lof(sb, k)
            choice[sb - rs], lengths[sb - rs] = k, Lof(sb, k)
            code = back[sb][0][k]
            if code == -1:
                break
            if code == -3:
                state = ("H", sb - 1, back[sb][1])
            else:
                pb = bins[sb]["pairs_with"]
                for bb in range(pb + 1, sb):
                    choice[bb - rs], lengths[bb - rs] = -2, Lof(pb, int(code))
                state = ("C", pb, int(code))
            b = state[1]
        else:  # H from bin state[1] back to its origin reading
            _, hb_end, origin = state
            if origin is None:
                break
            ob, oj, oL = origin
            for bb in range(ob + 1, hb_end + 1):
                choice[bb - rs], lengths[bb - rs] = -2, oL
            state = ("C", ob, oj)
            b = ob
    return choice, lengths


def isotonic(y: np.ndarray) -> np.ndarray:
    vals, wts, cnt = [], [], []
    for v in np.asarray(y, float):
        vals.append(float(v)); wts.append(1.0); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w]
            wts[-2:] = [w]
            cnt[-2:] = [cnt[-2] + cnt[-1]]
    return np.repeat(np.asarray(vals), cnt) if vals else np.zeros(0)


def hug_corrected(G: dict, choice: np.ndarray, L: np.ndarray, rs: int, p: dict, n0: int = 3) -> np.ndarray:
    """Lengths with the arc along the rim added where a reading's body leaves the grain away from where the tube
    first left it (the exit angle of its first ``n0`` readings): a tube that runs along its grain is measured from
    where it emerged, as annotators trace it, while the cheapest route from the rim cuts that part off.
    ``w_hug`` x r x max(0, |angle change| - ``hug_deg``)."""
    th = np.full(len(choice), np.nan)
    for i, k in enumerate(choice):
        if k >= 0:
            th[i] = G["bins"][rs + i]["F"][k, FI["theta"]]
    fin = np.flatnonzero(np.isfinite(th))
    if len(fin) < 1:
        return L
    th0 = float(np.angle(np.mean(np.exp(1j * th[fin[:n0]]))))
    last = np.nan
    out = L.copy()
    for i in range(len(choice)):
        if np.isfinite(th[i]):
            last = th[i]
        if choice[i] != -1 and np.isfinite(last):
            d = abs(float(np.angle(np.exp(1j * (last - th0)))))
            out[i] = L[i] + p["w_hug"] * G["r"] * max(0.0, d - np.radians(p.get("hug_deg", 20.0)))
    return out


def read_grain(G: dict, rs: int, nb: int, fpb: int, p: dict) -> dict:
    choice, L = viterbi(G, rs, nb, p)
    frames = [b * fpb + fpb // 2 for b in range(rs, nb)]
    germ = np.flatnonzero(choice != -1)
    out = {"frames": frames}
    if not len(germ):
        out.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None, px=[0.0] * len(frames), tips=None)
        return out
    t0 = int(germ[0])
    if p.get("w_hug", 0.0) > 0:
        L = hug_corrected(G, choice, L, rs, p)
    if p.get("iso", True):
        L[t0:] = isotonic(L[t0:])
    tips = []
    last = None
    for i, k in enumerate(choice):
        b = rs + i
        if k >= 0:
            last = G["bins"][b]["tips"][k].astype(float)
        tips.append(None if last is None else [round(float(last[0] - G["x"] * 0), 2), round(float(last[1]), 2)])
    # tips are in the grain frame (reference less drift): what the scorer adds the drift to
    first = next(t for t in tips if t is not None)
    tips = [first if t is None else t for t in tips]
    out.update(status="emerged_at_start" if t0 == 0 else "emerged_within", onset_frame=frames[t0],
               onset_interval=None if t0 == 0 else [frames[t0 - 1], frames[t0]],
               px=[round(float(v), 2) for v in L], tips=tips)
    return out


def movie_speed(doc: dict, p: dict, loose: float = 8.0, window: int = 10) -> float:
    """The movie's growth speed without labels: per grain read with a loose speed cap, the 90th percentile of its
    10-bin growth per bin; the median over grains that grew at least 10 px (px/bin)."""
    q = {**p, "vmax": loose, "vmax_factor": 0.0}
    rates = []
    for G in doc["grains"].values():
        _, L = viterbi(G, doc["rs"], doc["nb"], q)
        if len(L) > window and L.max() >= 10:
            adv = (L[window:] - L[:-window]) / window
            adv = adv[adv > 0.05]
            if len(adv) >= 5:
                rates.append(float(np.percentile(adv, 90)))
    return float(np.median(rates)) if len(rates) >= 3 else 1.0


def with_scale(doc: dict, p: dict) -> dict:
    """``p`` with the detector divided by the movie's 90th percentile of candidate detector values (``det_norm``):
    the detector answers more weakly on some movies (movie 2: 0.36 against the sparse movie's 0.63)."""
    norm = int(p.get("det_norm") or 0)
    return {**p, "_det_scale": {0: 1.0, 1: doc["det_q90"], 2: doc["det_frame_q99"]}[norm]}


def with_speed(doc: dict, p: dict) -> dict:
    """``p`` with ``vmax`` = ``vmax_factor`` x the movie's own growth speed (clipped 1.5-8 px/bin), if a factor is set."""
    if not p.get("vmax_factor"):
        return p
    return {**p, "vmax": float(np.clip(p["vmax_factor"] * movie_speed(doc, p), 1.5, 8.0))}


def predictions(doc: dict, base: dict, p: dict, only_flood: bool = False) -> dict:
    """The 0.8.8 baseline's predictions with the stored grains' status, onset, lengths and tips replaced (drift kept:
    the tips are in the grain frame, as the baseline's). ``only_flood``: only grains 0.8.8 read with the flood."""
    p = with_speed(doc, with_scale(doc, p))
    pred = dict(base)  # shallow: only the replaced grains are copied
    pred["grains"] = [dict(g) for g in base["grains"]]
    by = {g["id"]: g for g in pred["grains"]}
    rs, nb, fpb = doc["rs"], doc["nb"], doc["fpb"]
    for gid, G in doc["grains"].items():
        g = by.get(gid)
        if g is None or (only_flood and "reader:flood" not in g["flags"]):
            continue
        r = read_grain(G, rs, nb, fpb, p)
        g["status"], g["onset_frame"], g["onset_interval"] = r["status"], r["onset_frame"], r["onset_interval"]
        g["length"] = {"frames": r["frames"], "px": r["px"]}
        g["final_length_px"] = r["px"][-1]
        if r["tips"] is not None:
            g["tip"] = {"frames": r["frames"], "xy": r["tips"]}
        else:
            g.pop("tip", None)
    return pred


def per_grain(rep: dict) -> dict:
    out = {}
    for r in rep["rows"]:
        on = int(abs(r["onset_error"]) <= rep["onset"]["tolerance_frames"]) if "onset_error" in r else 0
        full = r.get("full", [])
        ln = sum(abs(f["error"]) <= max(2.0, 0.1 * f["human"]) for f in full)
        both = sum(abs(f["error"]) <= max(2.0, 0.1 * f["human"]) and f.get("tip_error", 1e9) <= max(5.0, 0.1 * f["human"])
                   for f in full)
        out[r["grain"]] = (on, ln, both)
    return out


def paired(base: dict, new: dict, n_boot: int = 4000, seed: int = 0) -> list[tuple[int, float, float]]:
    g = sorted(set(base) & set(new))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    out = []
    for k in range(3):
        d = np.array([new[x][k] - base[x][k] for x in g])
        lo, hi = np.percentile(d[idx].sum(axis=1), [2.5, 97.5])
        out.append((int(d.sum()), float(lo), float(hi)))
    return out


_CACHE: dict = {}


def evaluate(movie: str, doc: dict, p: dict, only_flood: bool = False) -> dict:
    if movie not in _CACHE:
        _CACHE[movie] = (labels(movie), baseline(movie))
    lab, base = _CACHE[movie]
    pred = predictions(doc, base, p, only_flood)
    rep = score(lab, pred)
    return {"onsets": rep["onset"]["hits"], "n_on": rep["onset"]["n_timed"], "lengths": rep["length_full"]["within_tolerance"],
            "n": rep["length_full"]["n"], "lt": rep["tips"]["length_and_tip"], "rep": rep, "pred": pred}


if __name__ == "__main__":
    cmd, movie = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else None
    doc = load(movie, name)
    lab = labels(movie)
    if cmd == "oracle":
        rows = oracle(doc, lab)
        print(summarize_oracle(rows))
        for r in rows:
            print(r)
