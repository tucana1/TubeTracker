"""Judge an onset network on a movie it never saw: per-bin probabilities for every scored grain (and the census
debris), onsets by the change-point decoder, scored against the human brackets and paired with SparseTrack 0.8.8 and
with 0.8.8 + the tip detector's `later` onsets (end to end, prototypes/tip_track) on the same grains.

    python -m prototypes.onset_net.evaluate m2 --net runs/research/onset_net/on_ldm1.pt --tag v1

Grain positions: where the 0.8.8 reading put the grain at each bin (census + its drift: what the tracker would give;
'drift', primary) and the labelling tool's following ('follow', what the annotator saw). Probabilities = mean of the
logits over the 8 rotations/flips. Onset = the single switch 0 -> 1 that best explains them (common.changepoint;
first valid bin = visible at the start, no switch = never). Hit = within 600 frames of the human bracket, over the
scorer's grains (isolated, not excluded) the annotator calls 'emerged within' (a grain called never/at start = miss).
Writes OUT/eval_<movie>_<tag>.json (probabilities, onsets, scores).
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np

from sparsetrack import stack
from sparsetrack.evaluate import score
from sparsetrack.render import Renderer

from .common import (MOVIES, NET_HALF, OUT, REPO, TIPDET_LATER, TOL, auroc, baseline, boot_ci, centre_cut,
                     changepoint, dihedral, disc, extract, hit_vector, inputs_at, labels, load_net, onset_error,
                     onset_prediction, positions, scored_grains)


def probs_for(net, crops: np.ndarray, r: float, rs: int, batch: int = 512) -> np.ndarray:
    """Per-bin probability (bins < rs + 1 get the first valid bin's) for one grain's grain-centred crops."""
    import torch
    nb = len(crops)
    E = crops[rs:rs + 3].astype(np.float32).mean(0)
    g = disc(r, NET_HALF)[None]
    xs = []
    for b in range(rs + 1, nb):
        x = np.concatenate([centre_cut(inputs_at(crops, b, rs, E), NET_HALF), g]).astype(np.float32)
        xs.extend(np.ascontiguousarray(dihedral(x, k)) for k in range(8))
    X = np.stack(xs)
    dev = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(net(torch.from_numpy(X[i:i + batch]).to(dev)).float().cpu().numpy())
    z = np.concatenate(out).reshape(-1, 8).mean(1)
    p = 1 / (1 + np.exp(-z))
    return np.concatenate([np.full(rs + 1, p[0]), p])


def run(movie: str, net_path: str, tag: str, log=print) -> dict:
    t0 = time.time()
    net = load_net(net_path)
    L = labels(movie)
    idx = json.loads((OUT / f"index_{movie}.json").read_text())
    crops_f = np.load(OUT / f"crops_{movie}.npy", mmap_mode="r")
    bins, meta = stack.load(REPO / MOVIES[movie][0])
    R = Renderer(bins, meta)
    rs, nb, fpb = idx["rs"], idx["nb"], idx["fpb"]
    base = {g["id"]: g for g in baseline(movie)["grains"]}
    later_doc = baseline(movie, TIPDET_LATER)
    later = {g["id"]: g for g in later_doc["grains"]}
    res = {"movie": movie, "net": str(net_path), "tag": tag, "rs": rs, "nb": nb, "fpb": fpb, "grains": {}}
    for i, it in enumerate(idx["items"]):
        gid = it["id"]
        ent = {"kind": it["kind"], "verdict": it["verdict"], "la": it["la"], "fv": it["fv"], "isolated": it["isolated"],
               "r": it["r"]}
        ent["p_follow"] = np.round(probs_for(net, np.asarray(crops_f[i]), it["r"], rs), 4).tolist()
        if gid in base:
            pos = positions(base[gid], rs, nb)
            ent["p_drift"] = np.round(probs_for(net, extract(R, pos), it["r"], rs), 4).tolist()
            ent["pos_diff_max"] = float(np.max(np.hypot(*(pos - np.asarray(it["pos"])).T)))
        else:
            ent["p_drift"] = ent["p_follow"]  # debris: census place either way
        for src in ("drift", "follow"):
            t, _ = changepoint(np.asarray(ent[f"p_{src}"]), rs + 1)
            ent[f"onset_{src}"] = t
        res["grains"][gid] = ent
    log(f"{movie}: {len(idx['items'])} items, probabilities in {time.time() - t0:.0f} s")
    res["scores"] = {src: summarise(movie, L, res, base, later, later_doc, src, log) for src in ("drift", "follow")}
    (OUT / f"eval_{movie}_{tag}.json").write_text(json.dumps(res))
    return res


def summarise(movie: str, L: dict, res: dict, base: dict, later: dict, later_doc: dict, src: str, log=print) -> dict:
    rs, fpb = res["rs"], res["fpb"]
    G = res["grains"]
    sg = scored_grains(L)
    em = [g for g in sg["emerged_within"] if g in G]
    # predictions: 0.8.8's readings with the onset replaced
    new = {g: onset_prediction(base[g], G[g][f"onset_{src}"], rs + 1, fpb) for g in G if g in base}
    h_new, h_base, h_later = (hit_vector(L, P, em) for P in (new, base, later))
    out = {"src": src, "n_emerged": len(em), "hits": int(h_new.sum()), "hits_088": int(h_base.sum()),
           "hits_later": int(h_later.sum())}
    for name, h in (("088", h_base), ("later", h_later)):
        d = h_new - h
        lo, hi = boot_ci(d)
        out[f"diff_{name}"] = int(d.sum())
        out[f"ci_{name}"] = [lo, hi]
    # errors
    rows = []
    for g in em:
        on = L["labels"][g]["onset"]
        e = {}
        for name, P in (("net", new), ("088", base), ("later", later)):
            p = P.get(g) or {}
            f = p.get("onset_frame") if p.get("status") == "emerged_within" else None
            err = onset_error(L["labels"][g], f)
            e[name] = None if err is None else round(err / fpb, 1)
            e[name + "_status"] = p.get("status")
        rows.append({"grain": g, "fv": on.get("first_visible_bin"), **e})
    out["rows"] = rows
    out["early"] = sum(1 for r in rows if r["net"] is not None and r["net"] * fpb < -TOL)
    out["late"] = sum(1 for r in rows if r["net"] is not None and r["net"] * fpb > TOL)
    out["never_or_start"] = sum(1 for r in rows if r["net"] is None)
    out["early_088"] = sum(1 for r in rows if r["088"] is not None and r["088"] * fpb < -TOL)
    out["late_088"] = sum(1 for r in rows if r["088"] is not None and r["088"] * fpb > TOL)
    # never-germinated / at-start / debris
    out["never_called"] = {g: new[g]["status"] for g in sg["no_emergence_by_end"] if g in new}
    out["start_called"] = {g: new[g]["status"] for g in sg["emerged_at_start"] if g in new}
    out["debris_onset"] = {g: G[g][f"onset_{src}"] for g in G if G[g]["kind"] == "debris"}
    # the scorer and the germination curve on the same readings
    doc = copy.deepcopy(baseline(movie))
    doc["grains"] = [new.get(p["id"], p) for p in doc["grains"]]
    for name, d in (("net", doc), ("088", baseline(movie)), ("later", later_doc)):
        rep = score(L, d)
        out[f"scorer_{name}"] = {"hits": rep["onset"]["hits"], "n_timed": rep["onset"]["n_timed"],
                                 "confusion": rep["germination_confusion"], "population": rep["population"]}
    # AUROC of the per-bin probabilities around the onset
    neg5, pos_young, neg12, pos6 = [], [], [], []
    for g in em:
        p = np.asarray(G[g][f"p_{src}"])
        la, fv = G[g]["la"], G[g]["fv"]
        la = fv - 1 if la is None else la
        for k in (1, 3, 6, 12, 24):
            if la - k >= rs + 1:
                neg5.append(p[la - k])
        neg12.extend(p[max(rs + 1, la - 12):la + 1])
        pos6.extend(p[fv:fv + 7])
    for g in G:
        if G[g]["kind"] != "grain":
            continue
        p = np.asarray(G[g][f"p_{src}"])
        for b, t in (L["labels"][g].get("traces") or {}).items():
            if t["state"] == "full" and t.get("length_px", 99) < 15 and int(b) >= rs + 1:
                pos_young.append(p[int(b)])
    out["auroc_pre5_vs_young"] = auroc(neg5, pos_young)
    out["auroc_12before_vs_6after"] = auroc(neg12, pos6)
    out["n_auroc"] = [len(neg5), len(pos_young), len(neg12), len(pos6)]
    pn, pb_, pl = (out[f"scorer_{k}"]["population"] or {} for k in ("net", "088", "later"))
    log(f"  {movie} [{src}] onset hits {out['hits']}/{len(em)} vs 0.8.8 {out['hits_088']} "
        f"({out['diff_088']:+d}, 95% CI {out['ci_088'][0]:+.0f}..{out['ci_088'][1]:+.0f}) vs tipdet-later "
        f"{out['hits_later']} ({out['diff_later']:+d}, {out['ci_later'][0]:+.0f}..{out['ci_later'][1]:+.0f}); "
        f"early {out['early']} (0.8.8 {out['early_088']}), late {out['late']} ({out['late_088']}), "
        f"never/start {out['never_or_start']}; scorer {out['scorer_net']['hits']}/{out['scorer_net']['n_timed']}; "
        f"AUROC pre5-vs-young {out['auroc_pre5_vs_young']:.3f}, 12-before-vs-6-after "
        f"{out['auroc_12before_vs_6after']:.3f}; T50 human {pn.get('t50_human')} net {pn.get('t50_model')} "
        f"0.8.8 {pb_.get('t50_model')} later {pl.get('t50_model')}; gap net {pn.get('max_gap')} 0.8.8 "
        f"{pb_.get('max_gap')} later {pl.get('max_gap')}; never {out['never_called']} start {out['start_called']} "
        f"debris {out['debris_onset']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("movie")
    ap.add_argument("--net", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run(a.movie, a.net, a.tag)
