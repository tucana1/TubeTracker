"""The review-confidence function (``sparsetrack.review.trace_confidence``) checked on a third movie and refitted
leave one movie out over the three (30 Sep 2026).

    python -m prototypes.review_check.conf3 ld=PRED.json m2=PRED.json m1=PRED.json [--json OUT]

``trace_confidence`` (fitted 29 Sep on the dev movie and movie 2: z = -0.41 - 0.63 log1p(stall) + 0.50 log1p(L))
orders the review, least sure first. Movie 1's readings were never used to fit it, so its movie-1 result is an
honest check. Per movie, over every scored FULL trace (``sparsetrack.evaluate.score``): AUROC of the confidence
against the reading being within tolerance, and the review effort - the share of traces a reviewer checks, least
confident first, before 79% / 90% of them are within tolerance (checked ones count as right) - against a random
order. Then the same two features refitted on two movies and applied to the third (logistic regression, C = 1),
paired against the fixed function with a bootstrap over grains.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
LABELS = {"ld": "benchmark/labels/ld_v1.json", "m2": "benchmark/labels/m2_v1.json", "m1": "benchmark/labels/m1_v1.json"}
TARGETS = (0.79, 0.90)


def rows(movie: str, pred_path: str) -> list[dict]:
    from sparsetrack.evaluate import load, match_grains, score
    pred = json.loads(Path(pred_path).read_text())
    L = load(REPO / LABELS[movie])
    fpb = int(L["frames_per_bin"])
    grains = {gid: g for gid, g in L["grains"].items() if not g.get("excluded") and g.get("isolated", True)}
    matched = match_grains({"grains": grains}, pred.get("grains", []), radius=float(pred.get("match_radius_px", 12.0)))
    rep = score(L, pred)
    out = []
    for r in rep["rows"]:
        res = matched.get(r["grain"])
        if res is None:
            continue
        px = (res.get("length") or {}).get("px") or []
        for f in r.get("full", []):
            b = int(f["frame"]) // fpb
            if not px:
                continue
            i = min(max(b, 0), len(px) - 1)
            Ls = np.asarray(px, float)
            grew = np.flatnonzero(np.diff(Ls[: i + 1]) > 0.25) if i > 0 else np.array([], int)
            stall = float(i - (grew[-1] + 1)) if len(grew) else float(i)
            out.append({"movie": movie, "grain": r["grain"], "bin": b, "human": float(f["human"]),
                        "error": float(f["error"]), "hit": abs(f["error"]) <= max(2.0, 0.1 * f["human"]),
                        "stall": stall, "L": float(max(Ls[i], 0.0))})
    return out


def fixed_conf(t: dict) -> float:
    z = -0.41 - 0.63 * math.log1p(t["stall"]) + 0.50 * math.log1p(t["L"])
    return 1.0 / (1.0 + math.exp(-z))


def feats(T: list[dict]) -> np.ndarray:
    return np.array([[math.log1p(t["stall"]), math.log1p(t["L"])] for t in T])


def effort_curve(hit: np.ndarray, score: np.ndarray | None) -> np.ndarray:
    """runs/review_triage/analysis.py's: expected share within tolerance after fixing the k least confident."""
    N, H0 = len(hit), hit.sum()
    if score is None:
        k = np.arange(N + 1)
        return (H0 + k * (N - H0) / N) / N
    order = np.argsort(score, kind="stable")
    s, h = score[order], hit[order]
    fixed, i, done = np.zeros(N + 1), 0, 0.0
    while i < N:
        j = i
        while j + 1 < N and s[j + 1] == s[i]:
            j += 1
        n_g, m_g = j - i + 1, (1 - h[i:j + 1]).sum()
        for t in range(1, n_g + 1):
            fixed[i + t] = done + t * m_g / n_g
        done += m_g
        i = j + 1
    return (H0 + fixed) / N


def effort_to(curve: np.ndarray, target: float) -> float:
    ok = np.flatnonzero(curve >= target - 1e-12)
    return float(ok[0] / (len(curve) - 1)) if len(ok) else float("nan")


def boot_effort(T: list[dict], s1: np.ndarray, s0: np.ndarray | None, target: float, n: int = 2000,
                seed: int = 0) -> tuple[float, float, float]:
    """Effort(s1) - effort(s0) with a 95% bootstrap over grains."""
    rng = np.random.default_rng(seed)
    hit = np.array([t["hit"] for t in T], float)
    grains = sorted({t["grain"] for t in T})
    gi = {g: np.array([k for k, t in enumerate(T) if t["grain"] == g]) for g in grains}
    d0 = effort_to(effort_curve(hit, s1), target) - effort_to(effort_curve(hit, s0), target)
    ds = []
    for _ in range(n):
        idx = np.concatenate([gi[g] for g in rng.choice(grains, len(grains))])
        ds.append(effort_to(effort_curve(hit[idx], s1[idx]), target) -
                  effort_to(effort_curve(hit[idx], None if s0 is None else s0[idx]), target))
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return d0, float(lo), float(hi)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="MOVIE=predictions.json")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    T = {mv: rows(mv, p) for mv, p in (kv.split("=", 1) for kv in a.preds)}
    out = {}
    for mv, t in T.items():
        others = [x for m, ts in T.items() if m != mv for x in ts]
        hit = np.array([x["hit"] for x in t], float)
        fx = np.array([fixed_conf(x) for x in t])
        res = {"traces": len(t), "within": int(hit.sum()), "auroc_fixed": float(roc_auc_score(hit, fx))}
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(feats(others), [x["hit"] for x in others])
        rf = lr.predict_proba(feats(t))[:, 1]
        res["auroc_refit"] = float(roc_auc_score(hit, rf))
        res["refit_coef"] = [float(lr.intercept_[0]), *map(float, lr.coef_[0])]
        for tg in TARGETS:
            k = f"{int(100 * tg)}"
            res[f"effort{k}_fixed"] = effort_to(effort_curve(hit, fx), tg)
            res[f"effort{k}_random"] = effort_to(effort_curve(hit, None), tg)
            res[f"effort{k}_refit"] = effort_to(effort_curve(hit, rf), tg)
            res[f"effort{k}_fixed_vs_random"] = boot_effort(t, fx, None, tg)
            res[f"effort{k}_refit_vs_fixed"] = boot_effort(t, rf, fx, tg)
        out[mv] = res
        print(f"{mv}: {len(t)} traces, {int(hit.sum())} within tolerance; AUROC fixed {res['auroc_fixed']:.2f}, refit on "
              f"the other two {res['auroc_refit']:.2f} (coef {', '.join(f'{c:+.2f}' for c in res['refit_coef'])})")
        for tg in TARGETS:
            k = f"{int(100 * tg)}"
            d, lo, hi = res[f"effort{k}_fixed_vs_random"]
            d2, lo2, hi2 = res[f"effort{k}_refit_vs_fixed"]
            print(f"   to {k}%: check {100 * res[f'effort{k}_fixed']:.0f}% least sure first (random "
                  f"{100 * res[f'effort{k}_random']:.0f}%; {100 * d:+.0f} pts, 95% CI {100 * lo:+.0f} to {100 * hi:+.0f}); "
                  f"refit {100 * res[f'effort{k}_refit']:.0f}% ({100 * d2:+.0f} vs fixed, {100 * lo2:+.0f} to {100 * hi2:+.0f})")
    allT = [x for ts in T.values() for x in ts]
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(feats(allT), [x["hit"] for x in allT])
    out["pooled_coef"] = [float(lr.intercept_[0]), *map(float, lr.coef_[0])]
    print("pooled over the three movies: z = %+.2f %+.2f log1p(stall) %+.2f log1p(L)" % tuple(out["pooled_coef"]))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
