"""Corrections: what a person says about a grain, saved at once to the movie's review labels file.

The store is the labelling tool's own (``sparsetrack.bench.server.Bench`` on ``review/review_labels.json``,
pre-filled with the model's answers by ``sparsetrack.review.prefill``), so the web review tool, the reviewed exports
(``sparsetrack.review.export``) and the scoring all read the app's answers as their own. Every answer is saved
before the call returns (Bench writes the file and appends to its journal).

Answers are given at a bin (the time on screen):
- ``confirm``: the onset and the lengths on screen are right (the model's answers, or the reviewed curve, become
  checked answers at the tool's trace times);
- ``onset`` (the tube is first visible here), ``no_onset`` (it never germinated);
- ``tip`` (the tube's tip is here: its length is read along the tube's route to that point), ``path`` (the tube
  runs along these points, exit first), ``no_tube`` (no tube here);
- ``burst`` (the grain has burst or is gone by here: nothing is measured after it);
- ``exclude`` (not a grain, a clump, ...) and ``include``;
- ``revert`` (back to the model's answers for this grain) and ``undo`` (the last answer taken back).

Points are clicked on the movie in reference coordinates; a grain the analysis followed as it moved is shown at its
census place plus its drift, so its points are stored in its own frame (the tool's grain-following view), where
the model's routes are.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from sparsetrack.bench.server import EXCLUDE_REASONS, Bench, trace_bins
from sparsetrack.review import asked_bins, prefill

from .model import RunData
from .overlay import project, to_length

SNAP_PX = 10.0      # a tip click farther than this from the tube's route (along its length) is not on it
EXTEND_PX = 60.0    # ...but beyond the route's end, the tube may be carried on straight to the click this far
ANNOTATOR = "reviewer (TubeTracker app)"


def ahead(route, p) -> bool:
    """Whether ``p`` lies on ahead of the route's end (within ``EXTEND_PX`` and 60 degrees of its last direction)."""
    end = route[-1]
    back = next((q for q in reversed(route[:-1]) if math.hypot(end[0] - q[0], end[1] - q[1]) >= 3.0), route[0])
    d = (end[0] - back[0], end[1] - back[1])
    v = (p[0] - end[0], p[1] - end[1])
    nd, nv = math.hypot(*d), math.hypot(*v)
    return 0 < nv <= EXTEND_PX and nd > 0 and (d[0] * v[0] + d[1] * v[1]) / (nd * nv) >= 0.5


class ReviewError(ValueError):
    """An answer that cannot be taken as given (the message says why, in words for the window)."""


class Reviewer:
    """Corrections for one analysed movie."""

    def __init__(self, data: RunData, annotator: str = ANNOTATOR):
        self.data = data
        self.folder = data.folder
        self.annotator = annotator
        self.bench: Bench | None = None
        self.error: str | None = None
        self.lock = threading.RLock()  # one answer at a time; the window's reads wait for it
        self._ready = threading.Event()
        self._undo: list[tuple[str, str, object, dict]] = []

    # ---- the labels file ---------------------------------------------------------------
    def follow_offsets(self) -> dict:
        """Every grain's view offsets as the app draws it: the analysis' drift where it followed the grain, else
        none (the grain is read and drawn at its census place)."""
        nb = self.data.n_bins
        out = {}
        for gid in self.data.order:
            d = self.data._static[gid]["drift"]
            out[gid] = np.asarray(d, float) if d else np.zeros((nb, 2))
        return out

    def start(self, background: bool = True) -> None:
        """Open the review labels file, pre-filling it with the model's answers first if there is none yet."""
        if background:
            threading.Thread(target=self._open, name=f"review:{self.folder.name}", daemon=True).start()
        else:
            self._open()

    def _open(self) -> None:
        labels = self.folder.review_labels
        try:
            if not labels.exists():
                prefill(self.folder.cache, self.folder.predictions, labels, log=lambda *a: None,
                        follow=self.follow_offsets())
            bench = Bench(self.folder.cache, labels, annotator=self.annotator)
            for gid, offsets in self.follow_offsets().items():
                bench.set_follow(gid, offsets)
            model = labels.with_suffix(".model.json")
            model_doc = json.loads(model.read_text()) if model.exists() else None
            with self.lock:
                self.bench = bench
                self.data.doc, self.data.model_doc = bench.doc, model_doc
        except Exception as exc:  # noqa: BLE001 - shown in the window; the display works without corrections
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()

    def ready(self) -> bool:
        return self.bench is not None

    def wait(self, timeout: float = 60.0) -> Bench:
        self._ready.wait(timeout)
        if self.bench is None:
            raise ReviewError(self.error or "The review file is still being prepared: try again in a moment.")
        return self.bench

    # ---- bookkeeping -----------------------------------------------------------------------
    def _remember(self, gid: str, action: str) -> None:
        doc = self.bench.doc
        grain = {k: doc["grains"][gid].get(k) for k in ("excluded", "exclude_reason")}
        self._undo.append((gid, action, copy.deepcopy(doc["labels"].get(gid)), grain))
        del self._undo[:-50]

    def _check(self, gid: str, b: int | None = None) -> None:
        if gid not in self.data.model:
            raise ReviewError(f"There is no grain {gid} in this analysis.")
        if b is not None and not 0 <= int(b) < self.data.n_bins:
            raise ReviewError(f"Bin {b} is outside the movie (0-{self.data.n_bins - 1}).")

    def _to_own_frame(self, gid: str, b: int, pts) -> list[list[float]]:
        dx, dy = self.data.drift_at(gid, b)
        return [[round(float(x) - dx, 2), round(float(y) - dy, 2)] for x, y in pts]

    def _trace(self, gid: str, b: int, state: str, pts=None, view: str = "app") -> dict:
        return self.bench.set_trace(gid, {"bin": int(b), "state": state, "points": pts or [], "view": view})

    def act(self, gid: str | None, action: str, **kw) -> dict:
        """Apply one answer (see the module's docstring); returns {"gid": the grain it changed, "message": what was
        saved, in words}."""
        self.wait()
        handlers = {"confirm": self.confirm, "onset": self.onset, "no_onset": self.no_onset, "tip": self.tip,
                    "path": self.path, "no_tube": self.no_tube, "burst": self.burst, "exclude": self.exclude,
                    "include": self.include, "revert": self.revert}
        if action == "undo":
            return self.undo()
        if action not in handlers:
            raise ReviewError(f"Unknown correction {action!r}.")
        self._check(gid, kw.get("b"))
        with self.lock:
            self._remember(gid, action)
            try:
                return {"gid": gid, "message": handlers[action](gid, **kw)}
            except ReviewError:
                self._undo.pop()
                raise
            except (KeyError, ValueError, TypeError) as exc:
                self._undo.pop()
                raise ReviewError(str(exc)) from exc

    # ---- the answers ---------------------------------------------------------------------------
    def confirm(self, gid: str) -> str:
        """The onset and lengths shown are right: the answers at the tool's trace times become checked ones."""
        doc, d = self.bench.doc, self.data
        lab = doc["labels"].get(gid) or {}
        on = lab.get("onset")
        if not on:
            raise ReviewError(f"{gid} has no onset answer to confirm.")
        self.bench.set_onset(gid, {k: on.get(k) for k in ("verdict", "first_visible_bin", "last_absent_bin")})
        if on["verdict"] not in ("emerged_within", "emerged_at_start"):
            return f"{gid}: confirmed as not germinated"
        shown = d.grain(gid)
        traces = (doc["labels"][gid].get("traces") or {})
        plan = asked_bins(doc["labels"][gid]["onset"], traces, d.n_bins) or trace_bins(on.get("first_visible_bin") or 0,
                                                                                          d.n_bins)
        n = 0
        for b in plan:
            t = traces.get(str(b))
            if t and t.get("review_origin", "human") == "human":
                continue
            L = float(shown["L"][b])
            if t and t.get("review_origin") == "model" and abs(float(t.get("length_px") or 0.0) - L) <= 0.05:
                self._trace(gid, b, t["state"], t.get("path_xy_view") or [], view=t.get("view", "model"))
            elif L >= 2.0:
                route, _ = d.route(gid, b)
                self._trace(gid, b, "full", to_length(route, L), view="app-confirm")
            else:
                self._trace(gid, b, "no_tube", view="app-confirm")
            n += 1
        return f"{gid}: onset and {n} length(s) confirmed"

    def onset(self, gid: str, b: int) -> str:
        b = int(b)
        if b <= 0:
            self.bench.set_onset(gid, {"verdict": "emerged_at_start"})
            return f"{gid}: germinated before the movie started"
        self.bench.set_onset(gid, {"verdict": "emerged_within", "first_visible_bin": b, "last_absent_bin": b - 1})
        return f"{gid}: tube first visible at {self.data.when(self.data.frame(b))}"

    def no_onset(self, gid: str) -> str:
        self.bench.set_onset(gid, {"verdict": "no_emergence_by_end"})
        return f"{gid}: never germinated"

    def tip(self, gid: str, b: int, x: float, y: float) -> str:
        """The tube's tip is at (x, y) at bin ``b``: its length is read along the route shown to the nearest point
        (or on from the route's end, straight to the click)."""
        b = int(b)
        d = self.data
        on = (self.bench.doc["labels"].get(gid) or {}).get("onset") or {}
        if on.get("verdict") not in ("emerged_within", "emerged_at_start"):
            raise ReviewError(f"{gid} has not germinated: set its onset first (O at the time the tube appears).")
        fv = 0 if on.get("verdict") == "emerged_at_start" else on.get("first_visible_bin") or 0
        if b < fv:
            raise ReviewError(f"{gid}'s tube only appears at {d.when(d.frame(fv))}: move the onset first.")
        route, _ = d.route(gid, b)
        if len(route) < 2:
            raise ReviewError(f"{gid} has no tube route to measure along: draw the tube instead (D).")
        p = self._to_own_frame(gid, b, [(x, y)])[0]
        total = float(np.sum(np.hypot(*np.diff(np.asarray(route, float), axis=0).T)))
        s, dist = project(route, p)
        if s >= total - 0.5 and dist > 1.0 and ahead(route, p):
            pts = [list(q) for q in route] + [p]  # beyond the route's end: carried on straight to the click
        elif dist <= SNAP_PX:
            pts = to_length(route, s, 0.0)
        else:
            raise ReviewError(f"That point is {dist:.0f} px from {gid}'s tube route: click on the tube, or draw "
                              f"the tube's path (D).")
        L = float(np.sum(np.hypot(*np.diff(np.asarray(pts, float), axis=0).T))) if len(pts) > 1 else 0.0
        if L < 2.0:
            self._trace(gid, b, "no_tube")
            return f"{gid}: no tube at {d.when(d.frame(b))}"
        rec = self._trace(gid, b, "full", pts, view="app-tip")
        return f"{gid}: tube {d.length_words(rec['length_px'])} long at {d.when(d.frame(b))}"

    def path(self, gid: str, b: int, points: list) -> str:
        """The tube runs along ``points`` (reference coordinates, from where it leaves the grain to its tip)."""
        b = int(b)
        if len(points) < 2:
            raise ReviewError("Click at least where the tube leaves the grain and its tip.")
        on = (self.bench.doc["labels"].get(gid) or {}).get("onset") or {}
        if on.get("verdict") not in ("emerged_within", "emerged_at_start"):  # a tube drawn: it germinated by then
            self.onset(gid, b)
        rec = self._trace(gid, b, "full", self._to_own_frame(gid, b, points), view="app-path")
        return f"{gid}: tube drawn, {self.data.length_words(rec['length_px'])} at {self.data.when(self.data.frame(b))}"

    def no_tube(self, gid: str, b: int) -> str:
        self._trace(gid, int(b), "no_tube")
        return f"{gid}: no tube at {self.data.when(self.data.frame(int(b)))}"

    def burst(self, gid: str, b: int) -> str:
        self._trace(gid, int(b), "burst")
        return f"{gid}: burst or gone by {self.data.when(self.data.frame(int(b)))}"

    def exclude(self, gid: str, reason: str = "not_a_grain") -> str:
        if reason not in EXCLUDE_REASONS or reason == "not_sampled":
            raise ReviewError(f"Unknown reason {reason!r}.")
        self.bench.set_exclusion(gid, {"excluded": True, "reason": reason})
        return f"{gid}: excluded ({reason.replace('_', ' ')})"

    def include(self, gid: str) -> str:
        self.bench.set_exclusion(gid, {"excluded": False})
        return f"{gid}: included again"

    def revert(self, gid: str) -> str:
        """Back to the model's answers for this grain (everything a person said about it is dropped)."""
        doc = self.bench.doc
        model = self.data.model_doc or {}
        with self.bench.lock:
            mlab = (model.get("labels") or {}).get(gid)
            if mlab is None:
                doc["labels"].pop(gid, None)
            else:
                doc["labels"][gid] = copy.deepcopy(mlab)
            mg = (model.get("grains") or {}).get(gid) or {}
            doc["grains"][gid]["excluded"] = bool(mg.get("excluded", False))
            doc["grains"][gid]["exclude_reason"] = mg.get("exclude_reason")
            self.bench.save("revert", {"grain": gid, "by": self.annotator})
        return f"{gid}: back to the model's answers"

    def undo(self) -> dict:
        with self.lock:
            if not self._undo:
                raise ReviewError("Nothing to undo.")
            gid, action, label, grain = self._undo.pop()
            doc = self.bench.doc
            with self.bench.lock:
                if label is None:
                    doc["labels"].pop(gid, None)
                else:
                    doc["labels"][gid] = label
                doc["grains"][gid].update(grain)
                self.bench.save("undo", {"grain": gid, "undid": action, "by": self.annotator})
            return {"gid": gid, "message": f"{gid}: took back \"{action.replace('_', ' ')}\""}

    def undo_depth(self) -> int:
        return len(self._undo)

    def last(self) -> tuple[str, str] | None:
        """The grain and the kind of the answer Undo would take back."""
        return (self._undo[-1][0], self._undo[-1][1]) if self._undo else None


def labels_age(folder) -> float | None:
    """When the review labels file was last written (s since the epoch), if there is one."""
    p = Path(folder.review_labels)
    return p.stat().st_mtime if p.exists() else None


def archive_review(folder) -> Path | None:
    """Move a review aside (whole, beside it: ``review_<time>``) before a new analysis replaces the one it checked;
    returns where it went (None if there was none)."""
    review = Path(folder.review_labels).parent
    if not review.exists():
        return None
    keep = review.with_name(f"review_{time.strftime('%Y%m%d-%H%M%S')}")
    review.rename(keep)
    return keep
