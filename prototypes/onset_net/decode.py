"""The change-point decoder's switch level, chosen leave one movie out on the other movies' HELD-OUT probabilities.

    python -m prototypes.onset_net.decode --tags ld=v1 m2=v1 m1=v1 [--src drift] [--out v1tau]

The decoder (common.changepoint) picks the onset t maximising sum_{b >= t} (logit p_b - logit tau): tau = 0.5 is the
plain likelihood change point (pre-registered default). Held-out probabilities on movie 2 rose at the human onset but
stayed at 0.2-0.5 for 10-25 bins (the network under-confident on that movie's young tubes), so a lower tau may be
better; it is chosen for each held-out movie on the other two movies' held-out probabilities (each from a network
that never saw that movie), most onset hits, ties to the tau nearest 0.5, and only then applied to the held-out movie.
Writes OUT/eval_<movie>_<out>.json (the evaluation with the re-decoded onsets, scored as evaluate.py).
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from .common import OUT, TIPDET_LATER, TOL, baseline, labels, onset_error, scored_grains
from .evaluate import summarise

TAUS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7)


def decode(p: np.ndarray, valid_from: int, tau: float, eps: float = 0.02) -> int | None:
    q = np.clip(np.asarray(p, float)[valid_from:], eps, 1 - eps)
    s = np.log(q / (1 - q)) - np.log(tau / (1 - tau))
    tail = np.concatenate([np.cumsum(s[::-1])[::-1], [0.0]])  # tail[t] = sum_{b >= t} s_b; tail[n] = never
    t = int(np.argmax(tail))
    return None if t == len(q) else valid_from + t


def hits_at(ev: dict, L: dict, tau: float, src: str) -> int:
    rs, fpb = ev["rs"], ev["fpb"]
    n = 0
    for g in scored_grains(L)["emerged_within"]:
        if g not in ev["grains"]:
            continue
        t = decode(np.asarray(ev["grains"][g][f"p_{src}"]), rs + 1, tau)
        if t is None or t <= rs + 1:
            continue
        e = onset_error(L["labels"][g], t * fpb + fpb // 2)
        n += int(abs(e) <= TOL)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--src", default="drift")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tags = dict(t.split("=") for t in a.tags)
    evs = {m: json.loads((OUT / f"eval_{m}_{t}.json").read_text()) for m, t in tags.items()}
    Ls = {m: labels(m) for m in tags}
    grid = {m: {tau: hits_at(evs[m], Ls[m], tau, a.src) for tau in TAUS} for m in tags}
    for m in tags:
        print(f"{m}: hits by tau " + " ".join(f"{tau}:{grid[m][tau]}" for tau in TAUS))
    for test in tags:
        train = [m for m in tags if m != test]
        tau = max(TAUS, key=lambda t: (sum(grid[m][t] for m in train), -abs(t - 0.5)))
        print(f"held out {test}: tau {tau} chosen on {train} ({sum(grid[m][tau] for m in train)} hits there); "
              f"{test} {grid[test][tau]} at it (tau 0.5: {grid[test][0.5]})")
        if a.out:
            ev = copy.deepcopy(evs[test])
            ev["tau"] = tau
            ev["tau_chosen_on"] = train
            for g, e in ev["grains"].items():
                for src in ("drift", "follow"):
                    e[f"onset_{src}"] = decode(np.asarray(e[f"p_{src}"]), ev["rs"] + 1, tau)
            base = {g["id"]: g for g in baseline(test)["grains"]}
            later_doc = baseline(test, TIPDET_LATER)
            later = {g["id"]: g for g in later_doc["grains"]}
            ev["scores"] = {src: summarise(test, Ls[test], ev, base, later, later_doc, src)
                            for src in ("drift", "follow")}
            (OUT / f"eval_{test}_{a.out}.json").write_text(json.dumps(ev))


if __name__ == "__main__":
    main()
