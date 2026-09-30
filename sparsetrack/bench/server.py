"""Local HTTP server for benchmark labelling (standard library only; binds to 127.0.0.1).

Labels use the vocabulary of ``tubetracker.annotation_schema.GerminationEvent``
(verdicts ``emerged_at_start`` / ``emerged_within`` / ``no_emergence_by_end`` /
``unobservable``; bracket endpoints ``last_absent_frame`` / ``first_visible_frame``)
and of the centerline observations (``path_xy``, ``path_complete``, ``direct_state``),
so answers can later be imported into an annotation project.

Every judgement is made on a registered bin average; a bin decision is stored both
as the bin index and as the bin-centre source frame. Grain views follow the grain (``Bench.follow``); traces are
clicked in that grain-following view and stored in reference coordinates (``path_xy_ref``), with the view's offset
at that bin (``view_offset``) and the clicked points (``path_xy_view``). The page draws a saved trace from its
reference coordinates less the view's current offset, so it stays on the tube when the way the tool follows a grain
changes.

How views follow a grain (30 Sep 2026):
- labelling (a file the tool made): wherever ``track.follow`` has the grain (knocked grains found again,
  ``Params.track_refind``), however little it moved; once the tracker has lost it, where it was last seen. The
  annotator can say where the grain is at a trace time (G, then a click on its centre): the tool keeps it in the
  labels file (``labels[gid]["refinds"]``) and follows the grain from there;
- review (a file pre-filled with a model's answers, ``sparsetrack review``): the analysis' own frame, so the model's
  drawn paths stay on its tubes (as before: the followed drift once a grain is off its place, the phase track nearer).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import __version__, stack
from ..grains import annotate_layout
from ..render import Renderer, png

SCHEMA = "sparsetrack.bench.v1"
VERDICTS = ("emerged_at_start", "emerged_within", "no_emergence_by_end", "unobservable")
# "burst": the tube has burst by this time and there is nothing left to trace (the old engine's
# is_bursted/burst_frame, here bracketed by the trace times); the grain's later trace times drop out
TRACE_STATES = ("full", "partial", "no_tube", "unsure", "burst")
EXCLUDE_REASONS = ("not_a_grain", "clump", "edge", "out_of_focus", "other", "not_sampled")
STATIC = Path(__file__).with_name("static")

# display layouts (CSS px): coarse = whole movie, fine = single bins around a transition
COARSE = {"bins_per_tile": 4, "half": 28, "zoom": 2.0, "cols": 11, "header": 16, "gap": 2}
FINE = {"n_tiles": 18, "half": 24, "zoom": 4.0, "cols": 6, "header": 16, "gap": 2}
# trace views, all 640 px across; "far" is for tubes longer than "wide" shows, and near the movie's
# edges it slides inward ("fit") instead of showing the space beyond the frame
TRACE_VIEWS = {"near": {"half": 64, "zoom": 5.0}, "wide": {"half": 128, "zoom": 2.5},
               "far": {"half": 256, "zoom": 1.25, "fit": True}}
ZOOM = {"half": 56, "zoom": 4.5}  # census close-up of the grain in focus
RETEST_SIZE = 8
TRACE_RETEST_SIZE = 15
FOLLOW_HALF = 60        # crop used to measure a grain's own drift
FOLLOW_MAX_STEP = 10.0  # a jump bigger than this between bins means the tracking is unreliable
LABEL_TRACK = {"track_refind": True}  # the tracker's options for labelling views (Params overrides)


def trace_bins(first_visible_bin: int, n_bins: int) -> list[int]:
    """Bins at which a germinated grain's tube is traced: 6 bins after onset, then fixed times.

    Every trace is at least 6 bins after the first visible bin, so no trace asks for a
    tube only a pixel long.
    """
    last = n_bins - 2  # the final bin is usually partial
    early = first_visible_bin + 6
    fixed = [round(0.4 * last), round(0.7 * last), last]
    wanted = sorted({b for b in [early, *fixed] if early <= b <= last})
    merged: list[int] = []
    for b in wanted:
        if not merged or b - merged[-1] >= 8:
            merged.append(b)
        elif b == last:
            merged[-1] = b
    return merged


def grain_trace_plan(first_visible_bin: int, n_bins: int, saved: dict) -> list[int]:
    """``trace_bins``, ending at a burst: nothing is left to trace later (answers given stay listed)."""
    bins = trace_bins(first_visible_bin, n_bins)
    burst = [b for b in bins if (saved.get(str(b)) or {}).get("state") == "burst"]
    return [b for b in bins if b <= burst[0] or str(b) in saved] if burst else bins


class Bench:
    def __init__(self, cache_dir: str | Path, labels_path: str | Path, annotator: str = "investigator",
                 sample: int = 0, seed: int = 20260923, follow_mode: str | None = None):
        """``follow_mode``: how the views follow a grain (module docstring), "label" or "review"; by default
        "review" for a file pre-filled with a model's answers, else "label"."""
        self.cache_dir = Path(cache_dir)
        self.bins, self.meta = stack.load(self.cache_dir)
        self.renderer = Renderer(self.bins, self.meta)
        self.fpb = int(self.meta["frames_per_bin"])
        self.n_bins = int(self.meta["n_bins"])
        self._follow: dict[str, dict] = {}
        self._track_lock = threading.Lock()
        self.labels_path = Path(labels_path)
        self.journal_path = self.labels_path.with_suffix(".journal.jsonl")
        self.annotator = annotator
        self.lock = threading.Lock()
        if self.labels_path.exists():
            self.doc = json.loads(self.labels_path.read_text())
            self._check_movie()
        else:
            self.doc = self._new_doc()
            if sample:
                self._sample(sample, seed)
            self.save("create")
        if follow_mode not in (None, "label", "review"):
            raise ValueError(f"follow_mode {follow_mode!r}: label or review")
        self.follow_mode = follow_mode or ("review" if self.doc.get("prefill") else "label")

    # ---- document -----------------------------------------------------------------
    def _new_doc(self) -> dict:
        grains = json.loads((self.cache_dir / "grains.json").read_text())["grains"]
        return {
            "schema": SCHEMA,
            "sparsetrack_version": __version__,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "movie": self.meta["movie"],
            "frames_per_bin": self.fpb,
            "n_bins": self.n_bins,
            "coordinates": ("reference: registered to the mean of bins 0-2; raw source "
                            "coordinates at a bin = reference + shift[bin]"),
            "bin_semantics": ("bin b averages the keyframes of source frames [b*fpb, (b+1)*fpb); "
                              "*_frame fields hold the bin-centre frame b*fpb + fpb//2"),
            "grains": {g["id"]: g for g in grains},
            "labels": {},
            "retest": {"grains": [], "labels": {}},
        }

    def _sample(self, n: int, seed: int) -> None:
        """Label a random sample of isolated grains away from the edge; the rest start excluded
        as "not_sampled" (they can be included again in the census)."""
        pool = sorted(gid for gid, g in self.doc["grains"].items() if g.get("isolated") and not g.get("border"))
        keep = set(random.Random(seed).sample(pool, min(n, len(pool))))
        for gid, g in self.doc["grains"].items():
            if gid not in keep:
                g["excluded"], g["exclude_reason"] = True, "not_sampled"
        self.doc["sample"] = {"n": len(keep), "seed": seed, "from": "isolated grains away from the edge",
                              "grains": sorted(keep)}

    def _check_movie(self) -> None:
        a, b = self.doc.get("movie", {}), self.meta["movie"]
        if (a.get("name"), a.get("size_bytes")) != (b.get("name"), b.get("size_bytes")):
            raise SystemExit(f"labels {self.labels_path} belong to {a.get('name')!r}, "
                             f"cache is {b.get('name')!r}; refusing to mix them")

    def save(self, event: str, payload: dict | None = None) -> None:
        self.doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.labels_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.doc, indent=1))
        os.replace(tmp, self.labels_path)
        with open(self.journal_path, "a") as fh:
            fh.write(json.dumps({"t": self.doc["updated"], "event": event, "payload": payload}) + "\n")

    def bin_centre(self, b: int) -> int:
        return int(b) * self.fpb + self.fpb // 2

    def grain(self, gid: str) -> dict:
        if gid not in self.doc["grains"]:
            raise KeyError(gid)
        return self.doc["grains"][gid]

    def order(self) -> list[str]:
        """Isolated, included grains first; then other included grains; excluded last. A review (a file
        pre-filled with a model's answers) goes by the model's confidence instead: its least sure grains first."""
        conf = (self.doc.get("prefill") or {}).get("confidence") or {}

        def key(item):
            gid, g = item
            if conf:
                return (bool(g.get("excluded")), gid not in conf, conf.get(gid, 1.0), gid)
            return (bool(g.get("excluded")), not g.get("isolated", False), gid)
        return [gid for gid, _ in sorted(self.doc["grains"].items(), key=key)]

    def state(self) -> dict:
        labels = self.doc["labels"]
        grains = self.doc["grains"]
        todo = [gid for gid in self.order() if not grains[gid].get("excluded")]
        onset_done = sum(1 for gid in todo if labels.get(gid, {}).get("onset"))
        human = lambda rec: (rec or {}).get("review_origin", "human") == "human"  # answered here, not pre-filled
        onset_checked = sum(1 for gid in todo if labels.get(gid, {}).get("onset") and human(labels[gid]["onset"]))
        traces_needed = traces_done = traces_checked = 0
        plan = {}
        for gid in todo:
            onset = labels.get(gid, {}).get("onset") or {}
            if onset.get("verdict") in ("emerged_within", "emerged_at_start"):
                fv = onset.get("first_visible_bin")
                fv = 0 if fv is None else fv
                saved = labels[gid].get("traces", {})
                plan[gid] = grain_trace_plan(fv, self.n_bins, saved)
                traces_needed += len(plan[gid])
                traces_done += sum(1 for b in plan[gid] if str(b) in saved)
                traces_checked += sum(1 for b in plan[gid] if str(b) in saved and human(saved[str(b)]))
        return {
            "movie": self.doc["movie"], "n_bins": self.n_bins, "frames_per_bin": self.fpb,
            "shifts": self.meta["shifts"], "order": self.order(), "grains": grains,
            "labels": labels, "retest": self.doc["retest"], "trace_plan": plan,
            "layout": {"coarse": COARSE, "fine": FINE, "trace": TRACE_VIEWS, "zoom": ZOOM},
            "verdicts": VERDICTS, "trace_states": TRACE_STATES, "exclude_reasons": EXCLUDE_REASONS,
            "progress": {"grains": len(todo), "onset_done": onset_done,
                         "traces_needed": traces_needed, "traces_done": traces_done,
                         "onset_checked": onset_checked, "traces_checked": traces_checked},
            "labels_path": str(self.labels_path),
            # a file pre-filled with a model's answers (sparsetrack review): the tool asks until each is checked
            "review": self.doc.get("prefill"),
            "follow_mode": self.follow_mode,
        }

    # ---- updates --------------------------------------------------------------------
    def set_onset(self, gid: str, body: dict, retest: bool = False) -> dict:
        verdict = body.get("verdict")
        if verdict not in VERDICTS:
            raise ValueError(f"bad verdict {verdict!r}")
        la, fv = body.get("last_absent_bin"), body.get("first_visible_bin")
        la = None if la is None else int(la)
        fv = None if fv is None else int(fv)
        if verdict == "emerged_within":
            if fv is None:
                raise ValueError("emerged_within needs first_visible_bin")
            if la is None:
                la = fv - 1 if fv > 0 else None
            if la is not None and la >= fv:
                raise ValueError("last_absent_bin must precede first_visible_bin")
        elif verdict == "emerged_at_start":
            la, fv = None, 0
        else:
            la = fv = None
        record = {
            "verdict": verdict,
            "last_absent_bin": la, "first_visible_bin": fv,
            "last_absent_frame": None if la is None else self.bin_centre(la),
            "first_visible_frame": None if fv is None else self.bin_centre(fv),
            "window_start": 0, "window_end": int(self.meta["movie"]["n_frames"]) - 1,
            "coarse_tile": body.get("coarse_tile"),
            "consulted_frames": sorted({self.bin_centre(b) for b in body.get("consulted_bins", [])}),
            "annotator": self.annotator, "review_origin": "human",
            "view": "registered bin averages", "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.lock:
            self.grain(gid)
            target = self.doc["retest"]["labels"] if retest else self.doc["labels"]
            entry = target.setdefault(gid, {})
            entry["onset"] = record
            if "time_spent_s" in body:
                entry["time_spent_s"] = round(float(body["time_spent_s"]), 1)
            self.save("retest_onset" if retest else "onset", {"grain": gid, **record})
        return record

    def set_trace(self, gid: str, body: dict, retest: bool = False) -> dict:
        state = body.get("state")
        if state not in TRACE_STATES:
            raise ValueError(f"bad trace state {state!r}")
        b = int(body["bin"])
        pts = [[round(float(x), 2), round(float(y), 2)] for x, y in body.get("points", [])]
        if state in ("full", "partial") and len(pts) < 2:
            raise ValueError("a traced tube needs at least an exit and an apex point")
        if state in ("no_tube", "unsure", "burst"):
            pts = pts if state == "unsure" else []
        dx, dy = self.meta["shifts"][b]
        fx, fy = (float(v) for v in self.follow(gid)[b])  # clicked in the grain-following view
        ref = [[round(x + fx, 2), round(y + fy, 2)] for x, y in pts]
        length = float(np.sum(np.hypot(*np.diff(np.array(pts), axis=0).T))) if len(pts) > 1 else 0.0
        record = {
            "bin": b, "source_frame": self.bin_centre(b), "state": state,
            "path_xy_ref": ref,
            "path_xy": [[round(x + dx, 2), round(y + dy, 2)] for x, y in ref],
            "path_xy_view": pts, "view_offset": [round(fx, 2), round(fy, 2)],
            "path_complete": state == "full",
            "direct_state": {"full": "direct_visible", "partial": "direct_visible",
                             "no_tube": "no_tube_visible", "unsure": "not_directly_visible",
                             "burst": "not_directly_visible"}[state],
            "length_px": round(length, 2),
            "contact": bool(body.get("contact", False)),
            "view": body.get("view", "near"),
            "annotator": self.annotator, "review_origin": "human",
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.lock:
            self.grain(gid)
            if retest:  # the blind length retest: kept apart from the answers it repeats
                self.doc["retest"].setdefault("trace_labels", {})[f"{gid}:{b}"] = record
                self.save("retest_trace", {"grain": gid, **record})
                return record
            entry = self.doc["labels"].setdefault(gid, {})
            entry.setdefault("traces", {})[str(b)] = record
            if "time_spent_s" in body:
                entry["time_spent_s"] = round(float(body["time_spent_s"]), 1)
            self.save("trace", {"grain": gid, **record})
        return record

    def set_exclusion(self, gid: str, body: dict) -> dict:
        """Exclude a grain (or include it again). A grain of the random sample excluded for any reason (not a grain,
        a clump, ...) is replaced by the next grain of the sample's reserve, so the sample keeps its size."""
        excluded = bool(body.get("excluded"))
        reason = body.get("reason") if excluded else None
        if excluded and reason not in EXCLUDE_REASONS:
            raise ValueError(f"bad exclusion reason {reason!r}")
        with self.lock:
            g = self.grain(gid)
            g["excluded"], g["exclude_reason"] = excluded, reason
            replaced_by = self._replace(gid, reason) if excluded else None
            self.save("exclude", {"grain": gid, "excluded": excluded, "reason": reason, "replaced_by": replaced_by})
        return {**g, "replaced_by": replaced_by}

    def _reserve(self) -> list[str]:
        """The sample's replacements in a random order fixed by its seed, drawn once from the rest of its pool
        (isolated grains away from the edge) and kept in the labels file: who replaces whom is chosen by neither
        the annotator nor the model."""
        smp = self.doc["sample"]
        if "reserve" not in smp:
            pool = sorted(gid for gid, g in self.doc["grains"].items()
                          if g.get("isolated") and not g.get("border") and gid not in smp["grains"])
            smp["reserve"] = random.Random(int(smp["seed"]) + 1).sample(pool, len(pool))
        return smp["reserve"]

    def _replace(self, gid: str, reason: str) -> str | None:
        """The reserve grain that joins the sample in place of ``gid`` (None: not a sampled grain, the sample is
        still full, or the reserve is used up)."""
        smp = self.doc.get("sample")
        if not smp or smp.get("closed") or gid not in smp["grains"] or reason == "not_sampled":
            return None  # "closed": the annotator finished the sample; exclusions from then on only shrink it
        if sum(not self.doc["grains"][s].get("excluded") for s in smp["grains"]) >= int(smp["n"]):
            return None  # e.g. a grain excluded, included again and excluded again: its replacement already came
        for cand in self._reserve():
            c = self.doc["grains"].get(cand)
            if c is None or cand in smp["grains"] or (c.get("excluded") and c.get("exclude_reason") != "not_sampled"):
                continue
            c["excluded"], c["exclude_reason"] = False, None
            smp["grains"].append(cand)
            smp.setdefault("replacements", []).append({"out": gid, "in": cand, "reason": reason,
                                                       "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            return cand
        return None

    def add_grain(self, body: dict) -> dict:
        x, y = float(body["x"]), float(body["y"])
        r = float(body.get("r", 13.0))
        with self.lock:
            n = 1 + sum(1 for gid in self.doc["grains"] if gid.startswith("u"))
            gid = f"u{n:03d}"
            grains = list(self.doc["grains"].values()) + [
                {"id": gid, "x": x, "y": y, "r": r, "source": "user", "ring_contrast": None,
                 "body_contrast": None}]
            laid = annotate_layout(grains, (self.renderer.height, self.renderer.width))
            self.doc["grains"] = {g["id"]: {**self.doc["grains"].get(g["id"], {}), **g} for g in laid}
            self.save("add_grain", {"grain": gid, "x": x, "y": y, "r": r})
        return self.doc["grains"][gid]

    def set_refind(self, gid: str, body: dict) -> dict:
        """"The grain is here" at bin ``body["bin"]``: its centre clicked at (``x``, ``y``) in that bin's view (the
        grain-following view, as trace points are), or ``clear`` to take back the answer given at that bin. Kept in
        the labels file (``labels[gid]["refinds"]``, reference coordinates) and followed from there. Labelling only:
        a review shows the analysis' own frame. Returns the grain's ``follow_info``."""
        if self.follow_mode != "label":
            raise ValueError("saying where a grain is belongs to labelling; a review shows the analysis' own frame")
        b = int(body["bin"])
        if not 0 <= b < self.n_bins:
            raise ValueError(f"bin {b} is outside the movie")
        g = self.grain(gid)
        fx, fy = (float(v) for v in self.follow(gid)[b])  # the view the click was made in
        with self.lock:
            refinds = self.doc["labels"].setdefault(gid, {}).setdefault("refinds", {})
            if body.get("clear"):
                gone = refinds.pop(str(b), None)
                if not refinds:
                    self.doc["labels"][gid].pop("refinds")
                self.save("refind_clear", {"grain": gid, "bin": b, "was": gone})
            else:
                x, y = float(body["x"]) + fx, float(body["y"]) + fy
                record = {"bin": b, "source_frame": self.bin_centre(b), "xy_ref": [round(x, 2), round(y, 2)],
                          "offset": [round(x - float(g["x"]), 2), round(y - float(g["y"]), 2)],
                          "clicked_view": [round(float(body["x"]), 2), round(float(body["y"]), 2)],
                          "view_offset": [round(fx, 2), round(fy, 2)], "annotator": self.annotator,
                          "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
                refinds[str(b)] = record
                self.save("refind", {"grain": gid, **record})
            self._follow.pop(gid, None)
            self.renderer._contrast = {k: v for k, v in self.renderer._contrast.items()
                                       if not (k[0] == gid or (isinstance(k[0], tuple) and k[0][0] == gid))}
        return self.follow_info(gid)

    def pick_retest(self) -> list[str]:
        with self.lock:
            if not self.doc["retest"]["grains"]:
                done = sorted(gid for gid, lab in self.doc["labels"].items()
                              if (lab.get("onset") or {}).get("verdict") == "emerged_within")
                random.Random(20260923).shuffle(done)
                self.doc["retest"]["grains"] = done[:RETEST_SIZE]
                self.save("retest_pick", {"grains": self.doc["retest"]["grains"]})
        return self.doc["retest"]["grains"]

    def pick_trace_retest(self, n: int = TRACE_RETEST_SIZE) -> list[dict]:
        """A fixed random set of FULL traces to repeat blind: one per grain, none touching anything."""
        with self.lock:
            if not self.doc["retest"].get("traces"):
                rng = random.Random(20260927)
                picks = []
                for gid in sorted(self.doc["labels"]):
                    g = self.doc["grains"].get(gid, {})
                    if g.get("excluded") or not g.get("isolated", True):
                        continue
                    full = sorted(int(b) for b, t in (self.doc["labels"][gid].get("traces") or {}).items()
                                  if t["state"] == "full" and not t.get("contact"))
                    if full:
                        picks.append({"grain": gid, "bin": rng.choice(full)})
                rng.shuffle(picks)
                self.doc["retest"]["traces"] = picks[:n]
                self.save("trace_retest_pick", {"traces": self.doc["retest"]["traces"]})
        return self.doc["retest"]["traces"]

    # ---- images ---------------------------------------------------------------------
    def set_follow(self, gid: str, offsets) -> None:
        """Use these (n_bins, 2) offsets as ``gid``'s grain-following view instead of measuring them (``follow``
        takes seconds per grain): e.g. the analysis' own drift, as the TubeTracker app draws the grain."""
        offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
        if len(offsets) != self.n_bins:
            raise ValueError(f"{gid}: {len(offsets)} offsets for {self.n_bins} bins")
        self._follow[gid] = offsets

    def follow(self, gid: str) -> np.ndarray:
        """(n_bins, 2) offsets that keep a grain centred in its views (grain registration on top of the field
        registration; the first reference bin's before the reference bins). How, depends on ``follow_mode`` (module
        docstring): labelling follows the tracker wherever it has the grain and from wherever the annotator said the
        grain is; a review shows the analysis' own frame."""
        return self._track(gid)["offsets"]

    def follow_info(self, gid: str) -> dict:
        """What the page needs to follow a grain: the view offsets per bin; ``lost``, the stretches [from, to, why]
        where the tool does not know where the grain is (from a loss by the tracker to the annotator's next "the grain
        is here", or to the end: the view stays where the grain was last seen) and ``lost_from``, the start of such
        a stretch that runs to the end (None: the grain is followed to the end); the bins the tracker found it again
        at by its look after a knock; and the annotator's own answers."""
        tr = self._track(gid)
        return {"grain": gid, "mode": self.follow_mode, "offsets": np.round(tr["offsets"], 2).tolist(),
                "lost": tr["lost"], "lost_from": tr["lost_from"], "lost_reason": tr["lost_reason"],
                "refound": tr["refound"],
                "refinds": sorted((self.doc["labels"].get(gid) or {}).get("refinds", {}).values(),
                                  key=lambda r: r["bin"])}

    def _track(self, gid: str) -> dict:
        with self._track_lock:  # the page asks for a grain's views and images at once: follow it once
            if gid not in self._follow:
                self._follow[gid] = self._label_track(gid) if self.follow_mode == "label" else self._review_track(gid)
            return self._follow[gid]

    def _others(self, gid: str) -> list[dict]:
        return [o for oid, o in self.doc["grains"].items() if oid != gid and o.get("exclude_reason") != "not_a_grain"]

    def _review_track(self, gid: str) -> dict:
        """The analysis' own frame (Params defaults): its followed drift where it reads the grain in its own frame
        ("follow"; "auto" once it moves off its place; after a loss where the grain was last seen), otherwise the
        phase-correlation track, zero if it is erratic."""
        from ..analyze import Params, followed_drift, hold_nan, local_shifts, plausible_drift, reads_in_grain_frame
        g = self.grain(gid)
        rs, half = self.renderer.ref_start, FOLLOW_HALF
        p = Params()
        ls, fd = None, {}
        if p.grain_track in ("follow", "auto"):
            fd = followed_drift(self.renderer, self.meta, g, self._others(gid), p)
            ls = hold_nan(fd["drift"])
            ls = ls if reads_in_grain_frame(ls, g["r"], p) else None  # auto: as the analysis reads it
        if ls is None:
            crops = np.stack([self.renderer.crop(b, g["x"], g["y"], half) for b in range(rs, self.n_bins)])
            if np.isnan(crops).any():
                crops = np.nan_to_num(crops, nan=float(np.nanmedian(crops)))
            ls = local_shifts(crops, half - 0.5, g["r"], 12.0, 3)
            if not plausible_drift(ls, FOLLOW_MAX_STEP):
                ls = np.zeros_like(ls)
        lost_from = fd.get("lost_from")
        return {"offsets": np.vstack([np.repeat(ls[:1], rs, axis=0), ls]), "lost_from": lost_from,
                "lost_reason": fd.get("lost_reason"), "refound": fd.get("refound") or [],
                "lost": [] if lost_from is None else [[int(lost_from), self.n_bins, fd.get("lost_reason")]]}

    def _label_track(self, gid: str) -> dict:
        """Wherever the tracker has the grain (``analyze.followed_drift`` with LABEL_TRACK), however little it moved;
        after a loss, where it was last seen. From each bin the annotator said where the grain is, the tracker starts
        again there (``track.follow`` from that bin, its look there as the reference)."""
        from dataclasses import replace
        from .. import track
        from ..analyze import Params, followed_drift, hold_nan
        g = self.grain(gid)
        rs = self.renderer.ref_start
        p = replace(Params(), **LABEL_TRACK)
        others = self._others(gid)
        fd = followed_drift(self.renderer, self.meta, g, others, p)
        ls = hold_nan(fd["drift"])
        offsets = np.vstack([np.repeat(ls[:1], rs, axis=0), ls])
        marks = sorted((self.doc["labels"].get(gid) or {}).get("refinds", {}).values(), key=lambda r: r["bin"])
        ends = [int(m["bin"]) for m in marks] + [self.n_bins]
        # stretches the tool does not know where the grain is: from a loss to the next answer (or the end)
        lost = [] if fd["lost_from"] is None or fd["lost_from"] >= ends[0] else [
            [int(fd["lost_from"]), ends[0], fd["lost_reason"]]]
        refound = [r for r in fd.get("refound") or [] if r["bin"] < ends[0]]
        near = [(float(o["x"]), float(o["y"]), float(o["r"])) for o in others
                if np.hypot(o["x"] - g["x"], o["y"] - g["y"]) < 200]
        cfg = track.FollowConfig(step_px=p.track_step_px, min_score=p.track_min_score, max_gap=p.track_max_gap,
                                 refind=p.track_refind)
        for m, b1 in zip(marks, ends[1:]):
            b0 = int(m["bin"])
            x, y = (float(v) for v in m["xy_ref"])
            tr = track.follow(self.renderer, x, y, float(g["r"]), b0, self.n_bins, near, cfg)
            offsets[b0:b1] = np.array([x - float(g["x"]), y - float(g["y"])]) + hold_nan(tr["xy"])[:b1 - b0]
            if tr["lost_from"] is not None and tr["lost_from"] < b1:
                lost.append([int(tr["lost_from"]), b1, tr["lost_reason"]])
            refound += [r for r in tr["refound"] if r["bin"] < b1]
        return {"offsets": offsets, "lost": lost, "refound": refound,
                "lost_from": lost[-1][0] if lost and lost[-1][1] == self.n_bins else None,
                "lost_reason": lost[-1][2] if lost and lost[-1][1] == self.n_bins else None}

    def coarse_png(self, gid: str, mode: str) -> bytes:
        g = self.grain(gid)
        k = COARSE["bins_per_tile"]
        ranges = [(b, min(b + k - 1, self.n_bins - 1)) for b in range(0, self.n_bins, k)]
        labels = [f"{b0 * self.fpb}" for b0, _ in ranges]
        return png(self.renderer.strip(gid, g["x"], g["y"], ranges, labels, COARSE["half"], COARSE["zoom"],
                                       COARSE["cols"], mode, COARSE["header"], COARSE["gap"], self.follow(gid), True))

    def fine_png(self, gid: str, start: int, mode: str) -> bytes:
        g = self.grain(gid)
        start = max(0, min(int(start), self.n_bins - FINE["n_tiles"]))
        ranges = [(b, b) for b in range(start, min(self.n_bins, start + FINE["n_tiles"]))]
        labels = [f"bin {b}  f{self.bin_centre(b)}" for b, _ in ranges]
        return png(self.renderer.strip(gid, g["x"], g["y"], ranges, labels, FINE["half"], FINE["zoom"],
                                       FINE["cols"], mode, FINE["header"], FINE["gap"], self.follow(gid), True))

    def frame_png(self, gid: str, b: int, view: str, mode: str, smooth: int,
                  cx: float | str | None = None, cy: float | str | None = None) -> bytes:
        """A trace view centred on the grain, or on (cx, cy): the page slides a "fit" view to stay
        inside the movie and maps clicks back through the same centre."""
        g = self.grain(gid)
        v = TRACE_VIEWS[view]
        cx = g["x"] if cx is None else float(cx)
        cy = g["y"] if cy is None else float(cy)
        b0, b1 = int(b) - smooth, int(b) + smooth
        crop = self.renderer.growth_crop if mode == "g" else self.renderer.mean_crop
        img = crop(b0, b1, cx, cy, v["half"], self.follow(gid), mark_outside=True)
        window = self.renderer.contrast((gid, cx, cy), cx, cy, v["half"], mode, self.follow(gid))
        return png(self.renderer.to_display(img, window, v["zoom"]))

    def zoom_png(self, x: float, y: float, which: str) -> bytes:
        """Census close-up: the early (reference) or late field around a reference point."""
        rs = self.renderer.ref_start
        b0, b1 = (rs, rs + 2) if which == "early" else (self.n_bins - 4, self.n_bins - 2)
        img = self.renderer.mean_crop(b0, b1, x, y, ZOOM["half"], mark_outside=True)
        finite = img[np.isfinite(img)]
        lo, hi = (np.percentile(finite, [0.5, 99.5]) if finite.size else (0.0, 255.0))
        return png(self.renderer.to_display(img, (float(lo), float(hi)), ZOOM["zoom"]))

    def field_png(self, which: str) -> bytes:
        key = f"field_{which}"
        cached = self.cache_dir / f"{key}.png"
        if not cached.exists():
            cached.write_bytes(png(self.renderer.field(which, 0.68)))
        return cached.read_bytes()


def make_handler(bench: Bench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def _send(self, status, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(url.query).items()}
            parts = [p for p in url.path.split("/") if p]
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if parts[:1] == ["static"] and len(parts) == 2:
                    f = STATIC / parts[1]
                    ctype = {"js": "text/javascript", "css": "text/css"}.get(f.suffix[1:], "text/plain")
                    return self._send(200, f.read_bytes(), ctype + "; charset=utf-8")
                if parts == ["api", "state"]:
                    return self._json(bench.state())
                if parts == ["api", "retest"]:
                    return self._json({"grains": bench.pick_retest()})
                if parts == ["api", "trace_retest"]:
                    return self._json({"traces": bench.pick_trace_retest()})
                if parts[:2] == ["api", "follow"] and len(parts) == 3:
                    return self._json(bench.follow_info(parts[2]))
                if parts[:2] == ["api", "img"]:
                    kind, mode = parts[2], q.get("contrast", "n")
                    if kind == "field":
                        return self._send(200, bench.field_png(q.get("which", "early")), "image/png")
                    if kind == "zoom":
                        return self._send(200, bench.zoom_png(float(q["x"]), float(q["y"]), q.get("which", "early")),
                                          "image/png")
                    gid = parts[3]
                    if kind == "coarse":
                        return self._send(200, bench.coarse_png(gid, mode), "image/png")
                    if kind == "fine":
                        return self._send(200, bench.fine_png(gid, int(q.get("start", 0)), mode), "image/png")
                    if kind == "frame":
                        return self._send(200, bench.frame_png(gid, int(q["bin"]), q.get("view", "near"), mode,
                                                               int(q.get("smooth", 0)), q.get("cx"), q.get("cy")),
                                          "image/png")
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError, IndexError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self):
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                if parts[:2] == ["api", "onset"]:
                    return self._json(bench.set_onset(parts[2], body))
                if parts[:2] == ["api", "retest"]:
                    return self._json(bench.set_onset(parts[2], body, retest=True))
                if parts[:2] == ["api", "trace_retest"]:
                    return self._json(bench.set_trace(parts[2], body, retest=True))
                if parts[:2] == ["api", "trace"]:
                    return self._json(bench.set_trace(parts[2], body))
                if parts[:2] == ["api", "exclude"]:
                    return self._json(bench.set_exclusion(parts[2], body))
                if parts[:2] == ["api", "refind"]:
                    return self._json(bench.set_refind(parts[2], body))
                if parts == ["api", "grain"]:
                    return self._json(bench.add_grain(body))
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(cache_dir, labels_path, port: int = 8765, open_browser: bool = True, annotator: str = "investigator",
          sample: int = 0, seed: int = 20260923):
    bench = Bench(cache_dir, labels_path, annotator, sample, seed)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(bench))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Benchmark labelling at {url}\nLabels: {bench.labels_path}\nStop with Ctrl-C.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
