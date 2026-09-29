"""Read kymographs with a trained KymoNet and score the result.

    python -m prototypes.kymo_reader.evaluate --model runs/kymo_reader/models/A.pt --movie ld m2 \
        --tag A [--start rest] [--onset-px 2 --min-len 8]

For each movie two predictions files are written (copies of SparseTrack 0.5.3's predictions with the
read grains' length, tip, status and onset replaced): ``{tag}_{movie}_st.json`` (SparseTrack's own
path and rotation track) and ``{tag}_{movie}_route.json`` (the human's latest route, SparseTrack's
rotation track along it: the oracle route). Each is scored with ``sparsetrack.evaluate.score`` and
compared with st053 by a paired bootstrap over grains (as ``scripts/synth_bench.py``); per-trace
rows go to CSV so the files can be re-scored against any later baseline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from . import decode, features, store

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/kymo_reader"
ST053 = {"ld": REPO / "runs/lab_checks_2026-09-29/ld_st053.json",
         "m2": REPO / "runs/lab_checks_2026-09-29/m2_st053_predictions.json"}
LABELS = {"ld": REPO / "benchmark/labels/ld_v1.json", "m2": REPO / "benchmark/labels/m2_v1.json"}


# ----------------------------------------------------------------------------- network
@torch.no_grad()
def logits(net, sm: dict, dev: str) -> np.ndarray:
    net.eval()
    dyn, stat = features.inputs(sm)
    out = net(torch.from_numpy(dyn)[None].to(dev), torch.from_numpy(stat)[None].to(dev))[0]
    return out.float().cpu().numpy()


# ----------------------------------------------------------------------------- synthetic validation
def _within(err: float, truth: float) -> bool:
    return abs(err) <= max(2.0, 0.1 * truth)


def synth_score(net, samples: list[dict], dev: str, vmax: float = 4.0, onset_px: float = 2.0,
                every: int = 12, first: int = 8) -> dict:
    """Lengths at the synthetic truth's trace times (every 12th bin from bin 8), absences, onsets, for the
    reader and for SparseTrack's own front along the same routes (its reported lengths)."""
    rep = {"len_hit": 0, "len_n": 0, "abs_ok": 0, "abs_n": 0, "on_hit": 0, "on_n": 0,
           "st_len_hit": 0, "st_abs_ok": 0, "st_on_hit": 0, "err": []}
    for sm in samples:
        L = sm["target"].astype(float)
        T = len(L)
        F = decode.fronts(logits(net, sm, dev), sm["s"], vmax)
        st = sm["st_len"].astype(float)
        for t in range(first, T - 1, every):
            if L[t] > 0:
                rep["len_n"] += 1
                rep["len_hit"] += _within(F[t] - L[t], L[t])
                rep["st_len_hit"] += _within(st[t] - L[t], L[t])
                rep["err"].append(float(F[t] - L[t]))
            else:
                rep["abs_n"] += 1
                rep["abs_ok"] += F[t] < 2.0
                rep["st_abs_ok"] += st[t] < 2.0
        grown = np.nonzero(L >= 2.0)[0]
        if len(grown) and grown[0] > 0 and L[-1] >= 8.0:
            fv = int(grown[0])
            la = int(np.nonzero(L[:fv] <= 0)[0][-1]) if (L[:fv] <= 0).any() else 0
            rep["on_n"] += 1
            for key, series in (("on_hit", F), ("st_on_hit", st)):
                above = np.nonzero(series >= (onset_px if key == "on_hit" else 1e-6))[0]
                if len(above):
                    b = int(above[0])
                    rep[key] += (la - 2) <= b <= (fv + 2)
    rep["med_abs_err"] = float(np.median(np.abs(rep["err"]))) if rep["err"] else None
    rep["bias"] = float(np.mean(rep["err"])) if rep["err"] else None
    del rep["err"]
    return rep


# ----------------------------------------------------------------------------- real movies
def read_sample(net, sm: dict, dev: str, frames: list, start_mode: str, vmax: float = 4.0, onset_px: float = 2.0,
                min_len: float = 8.0, keep_call: dict | None = None, F: np.ndarray | None = None) -> dict:
    """One grain read along one path. ``keep_call``: a SparseTrack result whose germination call and onset
    are kept (lengths zeroed before its onset), instead of the reader's own."""
    s = sm["s"].astype(float)
    if F is None:
        F = decode.fronts(logits(net, sm, dev), s, vmax)
    if sm["info"]["variant"] == "route":
        a = 0.0  # the route starts at the annotator's own first click
    else:
        a = decode.start(F, start_mode, sm["bb"], s)
    Lraw = np.maximum(F - a, 0.0)
    c = decode.calls(Lraw, frames, onset_px, min_len)
    if keep_call is not None:
        L = Lraw.copy()
        if keep_call["status"] == "no_emergence_by_end":
            L[:] = 0.0
        elif keep_call.get("onset_frame") is not None:
            L[np.asarray(frames) < keep_call["onset_frame"]] = 0.0
        c = {"L": L, "status": keep_call["status"], "onset_frame": keep_call.get("onset_frame"),
             "onset_interval": keep_call.get("onset_interval")}
    rot = sm.get("rot")
    return {"F": F, "a": a, "L": c["L"], "status": c["status"], "onset_frame": c["onset_frame"],
            "onset_interval": c["onset_interval"],
            "tips": decode.tips(F, sm["pts"], s, rot, sm.get("pivot"))}


def write_pred(base: dict, reads: dict, method: str) -> dict:
    """A copy of SparseTrack's predictions with the read grains replaced."""
    pred = copy.deepcopy(base)
    pred["method"] = method
    for g in pred["grains"]:
        r = reads.get(g["id"])
        if r is None:
            continue
        frames = g["length"]["frames"]
        g["length"] = {"frames": frames, "px": [round(float(v), 2) for v in r["L"]]}
        g["tip"] = {"frames": frames, "xy": r["tips"]}
        g["status"], g["onset_frame"], g["onset_interval"] = r["status"], r["onset_frame"], r["onset_interval"]
        g["final_length_px"] = round(float(r["L"][-1]), 2)
        g["flags"] = list(g.get("flags", [])) + ["reader:kymo"]
        g["kymo_start_px"] = round(float(r.get("a", 0.0)), 2)
    return pred


def st_oracle_reads(samples: list[dict], frames: list) -> dict:
    """SparseTrack's own reading along the human route (stored at extraction)."""
    out = {}
    for sm in samples:
        if sm["info"]["variant"] != "route":
            continue
        L = sm["st_len"].astype(float)
        st = sm["info"].get("st_status")
        if st is None:  # no change region at the rim: SparseTrack never reads a path (oracle_route.py: no emergence)
            out[sm["info"]["grain"]] = {"L": np.zeros_like(L), "tips": [[0.0, 0.0]] * len(L),
                                        "status": "no_emergence_by_end", "onset_frame": None, "onset_interval": None}
            continue
        onset = sm["info"].get("st_onset_frame")
        i = frames.index(onset) if onset in frames else None
        tips = sm["st_tip"].tolist() if "st_tip" in sm else [[0.0, 0.0]] * len(L)
        out[sm["info"]["grain"]] = {"L": L, "tips": tips, "status": st, "onset_frame": onset,
                                    "onset_interval": None if not i else [frames[i - 1], frames[i]]}
    return out


def grain_hits(rep: dict) -> dict:
    out = {}
    for r in rep["rows"]:
        full = r.get("full", [])
        out[r["grain"]] = {"onset_hit": (abs(r["onset_error"]) <= 600) if "onset_error" in r else None,
                           "len_hit": sum(_within(f["error"], f["human"]) for f in full), "len_n": len(full),
                           "both_hit": sum(_within(f["error"], f["human"]) and f.get("tip_error", 1e9)
                                           <= max(5.0, 0.1 * f["human"]) for f in full)}
    return out


def paired(base: dict, new: dict, n_boot: int = 4000, seed: int = 0) -> dict:
    """Per-grain differences (new - base) in onset, length and length-and-tip hits with 95% bootstrap
    intervals over grains (the logic of scripts/synth_bench.py ``paired``)."""
    rng = np.random.default_rng(seed)
    g = sorted(set(base) & set(new))
    d = {k: np.array([int(bool(new[x][k])) - int(bool(base[x][k])) if k == "onset_hit"
                      else new[x][k] - base[x][k] for x in g]) for k in ("onset_hit", "len_hit", "both_hit")}
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    out = {"grains": len(g)}
    for k, v in d.items():
        lo, hi = np.percentile(v[idx].sum(axis=1), [2.5, 97.5])
        out[k] = {"delta": int(v.sum()), "ci": [float(lo), float(hi)]}
    return out


def summary(rep: dict) -> dict:
    return {"onset": f"{rep['onset']['hits']}/{rep['onset']['n_timed']}",
            "lengths": f"{rep['length_full']['within_tolerance']}/{rep['length_full']['n']}",
            "length_and_tip": f"{rep['tips']['length_and_tip']}/{rep['length_full']['n']}",
            "median_abs_err": rep["length_full"]["median_abs_error"], "bias": rep["length_full"]["bias"],
            "absences": f"{rep['absences']['correct']}/{rep['absences']['n']}",
            "germination": rep["germination_confusion"]}


def trace_rows(rep: dict) -> list[dict]:
    rows = []
    for r in rep["rows"]:
        for f in r.get("full", []):
            rows.append({"grain": r["grain"], "frame": f["frame"], "human_px": f["human"], "pred_px": f["pred"],
                         "error_px": f["error"], "hit": _within(f["error"], f["human"]),
                         "tip_error_px": f.get("tip_error"),
                         "tip_hit": f.get("tip_error") is not None and f["tip_error"] <= max(5.0, 0.1 * f["human"])})
    return rows


def run_movie(net, movie: str, dev: str, tag: str, start_mode: str, vmax: float, onset_px: float, min_len: float,
              onset_source: str = "kymo", log=print) -> dict:
    from sparsetrack.evaluate import load, score
    labels = load(LABELS[movie])
    base = load(ST053[movie])
    frames = base["grains"][0]["length"]["frames"]
    samples = store.load(OUT / "data" / f"real_{movie}.npz")
    rep_base = score(labels, base)
    hits_base = grain_hits(rep_base)
    out = {"movie": movie, "tag": tag, "start": start_mode, "vmax": vmax, "onset_px": onset_px, "min_len": min_len,
           "onset_source": onset_source, "st053": summary(rep_base)}
    by_id = {g["id"]: g for g in base["grains"]}
    res_dir = OUT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    variants = {"st": {}, "route": {}}
    for sm in samples:
        v = sm["info"]["variant"]
        gid = sm["info"]["grain"]
        keep = by_id.get(gid) if (onset_source == "st053" and v == "st") else None
        variants[v][gid] = read_sample(net, sm, dev, frames, start_mode, vmax, onset_px, min_len, keep_call=keep)
    preds = {f"{tag}_st": write_pred(base, variants["st"], f"kymo_reader {tag} on st053 paths"),
             f"{tag}_route": write_pred(base, variants["route"], f"kymo_reader {tag} on human routes (oracle)")}
    preds["st053_route"] = write_pred(base, st_oracle_reads(samples, frames), "st053 change reader on human routes")
    hits = {}
    for name, pred in preds.items():
        rep = score(labels, pred)
        hits[name] = grain_hits(rep)
        (res_dir / f"{name}_{movie}.json").write_text(json.dumps(pred))
        with open(res_dir / f"{name}_{movie}_traces.csv", "w", newline="") as fh:
            rows = trace_rows(rep)
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["grain"])
            w.writeheader()
            w.writerows(rows)
        out[name] = {**summary(rep), "vs_st053": paired(hits_base, hits[name])}
    # reading quality on identical routes: the reader against SparseTrack's change reader, both on the human route
    out[f"{tag}_route"]["vs_st053_route"] = paired(hits["st053_route"], hits[f"{tag}_route"])
    out["n_read"] = {"st": len(variants["st"]), "route": len(variants["route"])}
    (res_dir / f"{tag}_{movie}_summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def fmt_row(name: str, s: dict) -> str:
    v = s.get("vs_st053")
    txt = f"{name:14s} onset {s['onset']:>6s}  lengths {s['lengths']:>7s}  len+tip {s['length_and_tip']:>7s}"
    if v:
        ci = lambda k: f"{v[k]['delta']:+d} [{v[k]['ci'][0]:+.0f},{v[k]['ci'][1]:+.0f}]"
        txt += f"   vs st053: onset {ci('onset_hit')}, lengths {ci('len_hit')}, len+tip {ci('both_hit')}"
    return txt


def main(argv=None):
    from .model import load as load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--movie", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--start", default="rest", choices=("census", "rest", "edge"))
    ap.add_argument("--vmax", type=float, default=4.0)
    ap.add_argument("--onset-px", type=float, default=2.0)
    ap.add_argument("--min-len", type=float, default=8.0)
    a = ap.parse_args(argv)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = load_model(a.model, dev)
    for mv in a.movie:
        out = run_movie(net, mv, dev, a.tag, a.start, a.vmax, a.onset_px, a.min_len)
        print(f"== {mv} ({a.tag}, start={a.start}, onset_px={a.onset_px}, min_len={a.min_len})")
        print(fmt_row("st053", out["st053"]))
        for k in (f"{a.tag}_st", f"{a.tag}_route", "st053_route"):
            print(fmt_row(k.replace(a.tag + "_", "kymo_"), out[k]))
        v = out[f"{a.tag}_route"]["vs_st053_route"]
        print("  same human routes, kymo vs SparseTrack: " + ", ".join(
            f"{k} {v[k]['delta']:+d} [{v[k]['ci'][0]:+.0f},{v[k]['ci'][1]:+.0f}]" for k in ("onset_hit", "len_hit", "both_hit")))
        print(f"  grains read: {out['n_read']}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- choosing the calls
ONSET_GRID = (1.0, 1.5, 2.0, 3.0, 4.0)
MINLEN_GRID = (4.0, 6.0, 8.0, 10.0)


def choose_on_synth(net, samples: list[dict], dev: str, vmax: float = 4.0, start_mode: str = "edge") -> dict:
    """onset_px and min_len on synthetic validation kymographs (lengths measured from ``start_mode``, as
    they will be on the real movies): onsets within 2 bins of the truth bracket plus correct germination
    calls (controls not called, tubes >= 8 px called)."""
    Fs = [(sm, decode.fronts(logits(net, sm, dev), sm["s"], vmax)) for sm in samples]
    best = None
    for op in ONSET_GRID:
        for ml in MINLEN_GRID:
            hit = calls_ok = 0
            for sm, F in Fs:
                L = sm["target"].astype(float)
                a = decode.start(F, start_mode, sm["bb"], sm["s"].astype(float))
                c = decode.calls(np.maximum(F - a, 0.0), list(range(len(F))), op, ml)
                grows = L[-1] >= 8.0
                calls_ok += (c["status"] != "no_emergence_by_end") == grows
                grown = np.nonzero(L >= 2.0)[0]
                if grows and len(grown) and grown[0] > 0 and c["onset_bin"] is not None:
                    fv = int(grown[0])
                    la = int(np.nonzero(L[:fv] <= 0)[0][-1]) if (L[:fv] <= 0).any() else 0
                    hit += (la - 2) <= c["onset_bin"] <= (fv + 2)
            key = (hit + calls_ok, -abs(op - 2.0), -abs(ml - 8.0))
            if best is None or key > best[0]:
                best = (key, {"onset_px": op, "min_len": ml, "onset_hits": hit, "calls_ok": calls_ok, "n": len(Fs)})
    return best[1]


def restrict(labels: dict, grains) -> dict:
    """Labels with every grain outside ``grains`` excluded from scoring."""
    out = copy.deepcopy(labels)
    for gid, g in out["grains"].items():
        if gid not in grains:
            g["excluded"] = True
    return out


def choose_on_movie(net, movie: str, dev: str, vmax: float = 4.0, starts=("census", "rest", "edge"),
                    grains=None) -> dict:
    """onset_px, min_len and the start convention on a TRAINING movie's labels (SparseTrack's paths):
    most onset + length hits. Never call this with the movie (or the grains, ``grains``) being tested."""
    from sparsetrack.evaluate import load, score
    labels = load(LABELS[movie])
    if grains is not None:
        labels = restrict(labels, set(grains))
    base = load(ST053[movie])
    frames = base["grains"][0]["length"]["frames"]
    samples = [sm for sm in store.load(OUT / "data" / f"real_{movie}.npz") if sm["info"]["variant"] == "st"
               and (grains is None or sm["info"]["grain"] in set(grains))]
    Fs = {sm["info"]["grain"]: (sm, decode.fronts(logits(net, sm, dev), sm["s"].astype(float), vmax)) for sm in samples}
    by_id = {g["id"]: g for g in base["grains"]}
    best = None
    configs = [(st, "st053", 2.0, 8.0) for st in starts] + \
              [(st, "kymo", op, ml) for st in starts for op in ONSET_GRID for ml in MINLEN_GRID]
    for st, src, op, ml in configs:
        reads = {gid: read_sample(net, sm, dev, frames, st, vmax, op, ml, F=F,
                                  keep_call=by_id.get(gid) if src == "st053" else None)
                 for gid, (sm, F) in Fs.items()}
        rep = score(labels, write_pred(base, reads, "choose"))
        k = rep["onset"]["hits"] + rep["length_full"]["within_tolerance"]
        key = (k, src == "kymo", -abs(op - 2.0), -abs(ml - 8.0))
        if best is None or key > best[0]:
            best = (key, {"start": st, "onset_source": src, "onset_px": op, "min_len": ml,
                          "onset_hits": rep["onset"]["hits"], "len_hits": rep["length_full"]["within_tolerance"]})
    return best[1]
