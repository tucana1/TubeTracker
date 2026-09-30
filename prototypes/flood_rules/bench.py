"""Flood-rule variants on a human benchmark: one movie and one network per process, many variants each.

    python -m prototypes.flood_rules.bench serve MOVIE MODEL QUEUE [--bg 96]     # MODEL "shipped": tubes_synth_v1
    python -m prototypes.flood_rules.bench submit QUEUE NAME [key=value ...]    # Params overrides, as synth_bench --set
    python -m prototypes.flood_rules.bench compare BASE.json NEW.json [...]     # paired, from QUEUE/done dumps

``serve`` computes the network's probability movie once, in memory (``prototypes.tube_net.pixels.prob_movie``, as
``prototypes/tube_net/bench.py`` does; "shipped" reads tubes_synth_v1's cached one), then runs every variant dropped
in QUEUE/todo, oldest first, and waits for more (a file QUEUE/STOP ends it once the queue is empty). The change
reader's result per grain is computed once (it does not depend on the flood's rules). Each variant is read as
``analyze`` reads it (the same dispatch) and scored as ``synth_bench.score_real`` scores a run: QUEUE/done/NAME.json
holds that dump (so ``synth_bench.paired`` compares any two), the growth agreement, and per grain the flood's own
onset (before the hybrid keeps the change reader's) for the accidental-hit count; predictions go to QUEUE/pred.
"""

from __future__ import annotations

import copy
import json
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
MOVIES = {"ld": ("runs/sparsetrack/ld", "benchmark/labels/ld_v1.json"),
          "m2": ("runs/sparsetrack/m2", "benchmark/labels/m2_v1.json")}
EARLY = 10  # bins: a flood start this far before the annotator's last-absent bin is a start on the rim, not the tube


def _within(err: float, truth_len: float) -> bool:
    return abs(err) <= max(2.0, 0.1 * truth_len)


class Movie:
    def __init__(self, movie: str, model: str, bg: int = 0, log=print):
        from sparsetrack import stack
        from sparsetrack.analyze import focus_changes
        from sparsetrack.render import Renderer
        cache, labels = MOVIES[movie]
        self.movie, self.log = movie, log
        self.bins, self.meta = stack.load(REPO / cache)
        self.renderer = Renderer(self.bins, self.meta)
        if model == "shipped":
            from sparsetrack import learned
            self.model = str(learned.MODEL)
            self.prob = Renderer(*stack.load(REPO / cache / f"prob_{learned.MODEL.stem}"))
        else:
            from prototypes.tube_net.pixels import prob_movie
            self.model = str(Path(model).resolve())
            self.prob = Renderer(*prob_movie(self.model, movie, list(range(int(self.meta["n_bins"]))), bg=bg, log=log))
        self.labels = json.loads((REPO / labels).read_text())
        census = list(self.labels["grains"].values())
        self.grains = [g for g in census if not g.get("excluded")]
        self.physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
        self.focus = focus_changes(self.bins, self.meta)
        self._change: dict = {}   # (base params, grain id) -> the change reader's result
        self._scale: dict = {}    # base params -> growth scale

    @staticmethod
    def _base(p) -> tuple:
        """The params the change reader sees (everything but the flood's and the reader choice)."""
        return tuple(sorted((k, repr(v)) for k, v in asdict(p).items()
                            if not k.startswith("flood_") and k not in ("reader", "model")))

    def read(self, overrides: dict) -> tuple[dict, dict]:
        """(predictions, the flood's own reading per grain it read) for Params(model=..., **overrides)."""
        from sparsetrack import learned
        from sparsetrack.analyze import Params, analyze_grain, growth_scale
        p = Params(**{"model": self.model, **overrides})
        key = self._base(p)
        if p.vmax_auto:
            if key not in self._scale:
                self._scale[key] = growth_scale(self.renderer, self.meta, self.grains, p, lambda *a: None, self.physical)
            if self._scale[key] is not None:
                p = replace(p, vmax_px=float(np.clip(p.vmax_factor * self._scale[key], p.vmax_px, p.vmax_cap)))
        results, own = [], {}
        for g in self.grains:
            others = [o for o in self.physical if o["id"] != g["id"]]
            res = None
            if p.reader != "flood":
                if (key, g["id"]) not in self._change:
                    r = analyze_grain(self.renderer, self.meta, g, others, p)
                    r.pop("_diag", None)
                    self._change[(key, g["id"])] = r
                res = copy.deepcopy(self._change[(key, g["id"])])
            crowded = res is not None and any(f.startswith(("touches:", "shared_change_split")) for f in res["flags"])
            noisy = res is not None and res.get("map_threshold", 0.0) > p.map_floor
            if p.reader == "flood" or (p.reader == "hybrid" and (crowded or noisy)):
                fl = learned.read_grain(self.renderer, self.prob, self.meta, g, others, p)
                fl.pop("_diag", None)
                own[g["id"]] = {"status": fl["status"], "onset_frame": fl.get("onset_frame"),
                                "flags": list(fl["flags"]), "crowded": crowded, "noisy": noisy,
                                "length": fl["length"]["px"]}
                if (p.reader == "hybrid" and p.flood_fallback and res is not None
                        and res["status"] != "no_emergence_by_end" and fl["status"] == "no_emergence_by_end"):
                    res["flags"].append("flood_found_no_tube")
                else:
                    if res is not None and not crowded:
                        fl = learned.with_onset(fl, res)
                    res = fl
            res.pop("_diag", None)
            results.append(res)
        fpb = int(self.meta["frames_per_bin"])
        for res in results:
            of = res.get("onset_frame")
            if of is not None and any(abs(of // fpb - f["bin"]) <= 3 for f in self.focus):
                res["flags"].append("onset_at_focus_change")
        pred = {"schema": "sparsetrack.pred.v1", "method": "flood_rules bench", "params": asdict(p),
                "frames_per_bin": self.meta["frames_per_bin"], "movie": self.meta.get("movie"),
                "focus_changes": self.focus, "grains": results}
        return pred, own

    def score(self, pred: dict, own: dict, pred_path: Path) -> dict:
        """``synth_bench.score_real``'s dump for these predictions, plus the growth agreement and the flood's own
        onsets (bins) with the hits they explain."""
        from sparsetrack.evaluate import score
        rep = score(self.labels, pred)
        fpb = int(self.meta["frames_per_bin"])
        grains = {}
        for r in rep["rows"]:
            full = r.get("full", [])
            grains[r["grain"]] = {"onset_hit": (abs(r["onset_error"]) <= 600) if "onset_error" in r else None,
                                  "len_hit": sum(_within(f["error"], f["human"]) for f in full), "len_n": len(full),
                                  "both_hit": sum(_within(f["error"], f["human"]) and f.get("tip_error", 1e9)
                                                  <= max(5.0, 0.1 * f["human"]) for f in full)}
        early = {}
        for gid, o in own.items():
            on = (self.labels["labels"].get(gid) or {}).get("onset") or {}
            if o["onset_frame"] is None or on.get("verdict") != "emerged_within" or on.get("last_absent_bin") is None:
                continue
            early[gid] = int(o["onset_frame"] // fpb - on["last_absent_bin"])
        acc = sum(grains[g]["len_hit"] for g, e in early.items() if e < -EARLY and g in grains)
        return {"on_hit": rep["onset"]["hits"], "on_n": rep["onset"]["n_timed"],
                "len_hit": rep["length_full"]["within_tolerance"], "len_n": rep["length_full"]["n"],
                "len_med": rep["length_full"]["median_abs_error"], "len_bias": rep["length_full"]["bias"],
                "both": rep["tips"]["length_and_tip"],
                "errs": [(f["error"], f["human"], r["grain"], f["frame"]) for r in rep["rows"] for f in r.get("full", [])],
                "grains": grains, "pred": str(pred_path), "growth": rep["growth"], "population": rep["population"],
                "flood_onset_vs_last_absent": early, "accidental_len_hits": acc,
                "flood_grains": sorted(own), "own": {g: {k: v for k, v in o.items() if k != "length"}
                                                     for g, o in own.items()}}


def serve(movie: str, model: str, queue: Path, bg: int = 0) -> None:
    todo, done, pred_dir = queue / "todo", queue / "done", queue / "pred"
    for d in (todo, done, pred_dir):
        d.mkdir(parents=True, exist_ok=True)
    logf = open(queue / "log.txt", "a", buffering=1)
    log = lambda *a: print(*a, file=logf, flush=True)
    t0 = time.time()
    M = Movie(movie, model, bg, log)
    log(f"{movie} {Path(model).name} bg {bg}: ready in {time.time() - t0:.0f} s")
    while True:
        reqs = sorted(todo.glob("*.json"), key=lambda f: (f.stat().st_mtime, f.name))
        if not reqs:
            if (queue / "STOP").exists():
                log("stopped")
                return
            time.sleep(3)
            continue
        req = reqs[0]
        spec = json.loads(req.read_text())
        name = spec["name"]
        t = time.time()
        try:
            if spec.get("reload"):  # the readers' code changed: read it again (the maps stay in memory)
                import importlib
                import sparsetrack.analyze
                import sparsetrack.learned
                importlib.reload(sparsetrack.learned)
                importlib.reload(sparsetrack.analyze)
                M._change.clear()
                M._scale.clear()
            if spec.get("diag"):  # a diagnostic function run on the maps in memory: "module:function"
                import importlib
                mod, fn = spec["diag"].split(":")
                out = getattr(importlib.reload(importlib.import_module(mod)), fn)(M, **spec.get("args", {}))
                (done / f"{name}.json").write_text(json.dumps(out, default=float))
                log(f"{name} ({time.time() - t:.0f} s): diagnostic {spec['diag']}")
                req.unlink()
                continue
            pred, own = M.read(spec.get("set", {}))
            (pred_dir / f"{name}.json").write_text(json.dumps(pred, default=float))
            res = M.score(pred, own, pred_dir / f"{name}.json")
            res["set"], res["movie"], res["model"], res["bg"] = spec.get("set", {}), movie, M.model, bg
            (done / f"{name}.json").write_text(json.dumps({movie: res}, default=float))
            log(f"{name} ({time.time() - t:.0f} s): {line(movie, res)}")
        except Exception:
            (done / f"{name}.err").write_text(traceback.format_exc())
            log(f"{name}: FAILED\n{traceback.format_exc()}")
        req.unlink()


def line(movie: str, r: dict) -> str:
    g = r.get("growth") or {}
    return (f"{movie} len {r['len_hit']}/{r['len_n']} (med {r['len_med']:.2f}, bias {r['len_bias']:+.2f}), "
            f"len&tip {r['both']}, onsets {r['on_hit']}/{r['on_n']}, accidental {r['accidental_len_hits']}, "
            f"rate within {g.get('rate_within')}/{g.get('grains_with_rate')} (r {g.get('rate_pearson') or float('nan'):.2f})")


def submit(queue: Path, name: str, kvs: list[str]) -> None:
    """A variant (Params overrides key=value), or with diag=module:function a diagnostic run in the server on its
    maps (other key=value its keyword arguments); reload=1 first reads the readers' code again."""
    import synth_bench as sb
    sets, spec = {}, {"name": name}
    for kv in kvs:
        k, v = kv.split("=", 1)
        if k in ("diag", "reload"):
            spec[k] = v
        else:
            sets[k] = sb.parse_value(v)
    if "diag" in spec:
        spec["args"] = sets
    else:
        from sparsetrack.analyze import Params
        Params(**sets)  # unknown keys fail here, not in the server
        spec["set"] = sets
    (queue / "todo").mkdir(parents=True, exist_ok=True)
    if (queue / "done" / f"{name}.json").exists() or (queue / "todo" / f"{name}.json").exists():
        raise SystemExit(f"{name} exists in {queue}")
    (queue / "todo" / f"{name}.json").write_text(json.dumps(spec))


def paired(base: dict, new: dict, key: str, n_boot: int = 4000, seed: int = 0) -> tuple[int, float, float]:
    """Change in a per-grain count (onset_hit, len_hit, both_hit) with a 95% bootstrap interval over grains."""
    rng = np.random.default_rng(seed)
    g = sorted(set(base["grains"]) & set(new["grains"]))
    d = np.array([int(new["grains"][k][key] or 0) - int(base["grains"][k][key] or 0) for k in g])
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    lo, hi = np.percentile(d[idx].sum(axis=1), [2.5, 97.5])
    return int(d.sum()), float(lo), float(hi)


def compare(base: dict, new: dict) -> str:
    out = []
    for mv in new:
        if mv not in base:
            continue
        cells = [f"{lab} {d:+d} ({lo:+.0f} to {hi:+.0f})" for lab, (d, lo, hi) in
                 (("lengths", paired(base[mv], new[mv], "len_hit")), ("len&tip", paired(base[mv], new[mv], "both_hit")),
                  ("onsets", paired(base[mv], new[mv], "onset_hit")))]
        out.append(f"  {mv}: " + ", ".join(cells))
    return "\n".join(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "serve":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("movie", choices=tuple(MOVIES))
        ap.add_argument("model")
        ap.add_argument("queue")
        ap.add_argument("--bg", type=int, default=0)
        a = ap.parse_args()
        serve(a.movie, a.model, Path(a.queue), a.bg)
    elif cmd == "submit":
        submit(Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
    elif cmd == "compare":
        docs = [json.loads(Path(f).read_text()) for f in sys.argv[2:]]
        for f, d in zip(sys.argv[3:], docs[1:]):
            print(Path(f).stem)
            print(compare(docs[0], d))
    else:
        raise SystemExit(__doc__)
