"""TubeTracker Annotator workbench controller (P0A v1, napari/Qt).

Runs with a real display (Mac GUI session); napari's Viewer cannot
construct offscreen. All persistence/schema/frame logic lives in the
Qt-free modules so headless CI covers everything except this
controller's viewer wiring. Controller accepts ``viewer=None`` for
headless logic tests (layer calls are skipped).

v1 scope: open a movie, walk a task queue (apex clicks), transactional
save/next with resume, reference mode (predictions never loaded),
training snapshot export. Crossing/centerline/uncertainty-region tasks
reuse the same queue with per-type instructions (full editors follow).

P0A slice: centerline path tasks (click-to-extend + Path-done),
typed missingness states (no-tube / hidden / whose-tube), view-only
neighbor-frame stepping with consulted-frame audit, all headless-safe.
"""

from __future__ import annotations

import copy

import math
from functools import wraps
from pathlib import Path

from tubetracker.annotation_frames import FrameReader
from tubetracker.annotation_schema import (
    DIRECT_VISIBLE,
    NOT_DIRECTLY_VISIBLE,
    NO_TUBE_VISIBLE,
    OWNER_UNCERTAIN,
    UNRESOLVED_OVERLAP,
    VISIBLE_IMPRECISE,
    Observation,
    new_uuid,
)
from tubetracker.annotation_store import AnnotationStore
from tubetracker.annotation_tasks import next_unfinished
from tubetracker.review_region import (disc_bbox_xy, paint_bbox_xy,
                                       review_region_usable, licensed_review_region,
                                       REVIEW_GEOMETRY_SCHEMA)

try:
    from qtpy.QtWidgets import (
        QDockWidget,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPushButton,
        QTableWidget,
        QVBoxLayout,
        QWidget,
    )
    HAVE_QT = True
except ImportError:  # headless logic tests
    HAVE_QT = False


TASK_INSTRUCTIONS = {
    'grain_identity': ('Match the highlighted reference grain to the same physical grain '
                       'in this frame. Click its centre, then Save grain centre and next. '
                       'Use Cannot judge if its identity cannot be established.'),
    "apex": ("Click the far end of the pollen tube growing out of the ball "
             "IN THE CENTER of the view — ignore the other balls around "
             "it. One click saves and moves you on. If the center ball "
             "has no tube, or you cannot tell where it ends, press Can't "
             "tell instead of guessing."),
    "centerline": ("TRACE THE TUBE dot by dot. Click along the middle of "
                   "the tube, starting at the ball and going outward to "
                   "the far end. Each click adds one dot — you stay here. "
                   "Stop where you can't see it any more; never guess. "
                   "Press 'Trace saved — next' when done."),
    "crossing": ("TWO TUBES CROSS HERE. Trace each one separately. "
                 "Start with Tube 1: click dots along it, right through "
                 "the crossing (sharing pixels is fine). Then press "
                 "'Tube 2' and trace the other one the same way. "
                 "Press 'Both tubes drawn — save & next' when both are "
                 "traced, or 'Too tangled' if they can't be told apart."),
    "owner": ("PILE OF BALLS STUCK TOGETHER. Work the table top to "
              "bottom: each row is one ball. LEFT cell picks the ball, "
              "RIGHT cell picks its tube start — click a cell, then "
              "click that spot on the picture. A ball with NO tube? "
              "Press 'No tube' in its row instead. Marking a ball opens "
              "a fresh row below for the next ball. 'Undo last dot' takes "
              "back one dot at a time, and ✕ Remove deletes a whole row "
              "(the rest slide up). When every row is answered, "
              "press Submit. If the pile can't be separated, "
              "press 'Too tangled'."),
    "review_tip": ("RE-CHECK this dim tip. The dashed mark is the earlier "
                   "judgment — shown as a draft, not truth. If the tip is "
                   "precisely localizable, click it (or press Confirm draft "
                   "if the draft is exactly right). If you can only bound "
                   "the area, switch to area mode, click 3+ corners, and "
                   "press Save area. Never upgrade a guess into a point."),
    "review_path": ("CONFIRM this tube's traced route. The yellow line is "
                    "the earlier stroke (draft). Press FULL if it runs from "
                    "the ball's attachment to the far end; press PARTIAL if "
                    "any stretch is hidden or missing. Do not redraw."),
    "review_crossing": ("CHECK these two tubes. Dashed lines are your "
                        "earlier lanes, already loaded. If a lane is right, "
                        "leave it — re-click dots only where it is wrong. "
                        "Then press Lanes confirmed. Nothing here retrains "
                        "today's model; it banks the route geometry and "
                        "proves the batch under the fixed UI."),
    "body_mask": ("PAINT THE TUBE BODY. The yellow line is the confirmed "
                  "tube route (guide only). Paint over the whole visible "
                  "tube — both walls and the far end — with the brush. "
                  "Missed spots stay unknown, so cover the full length. "
                  "Press 'Body saved — next' when the whole tube is "
                  "painted (only then is the background trusted), or "
                  "'Partial' if part is hidden, or Can't tell."),
    "route_duel": ("WHICH CURVE FOLLOWS THE TUBE? Yellow vs blue. Click "
                   "NEAR the better one — one click votes and moves you "
                   "on. If NEITHER follows the tube, press Neither. If "
                   "you can't tell, press Can't tell. Never guess."),
    "germination_event": ("JUDGE THIS GRAIN'S WHOLE INTERVAL — nothing to click. "
                   "Scrub with the navigation row (looking records each "
                   "frame you visit). Watch the ringed grain's CIRCULAR RIM: "
                   "tubes start as a tiny bump breaking the circle, long before "
                   "an obvious tube. Optionally mark the last clearly "
                   "smooth/undisrupted rim frame and the first clearly bumped/"
                   "outgrowth frame with the bracket buttons. Then return to "
                   "this opening frame and press ONE verdict: already "
                   "emerged at start, emerged within the bracket, no "
                   "emergence by end, or unobservable. Judge only the "
                   "ringed grain's OWN rim — neighbours' tubes don't count."),
}


def _task_frame_write(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        self.require_task_frame()
        return method(self, *args, **kwargs)
    return guarded


def _draft_write(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        self.persist_draft()
        return result
    return guarded


class AnnotatorController:
    """Task queue + persistence; viewer wiring is thin by design."""

    def __init__(self, store: AnnotationStore, reader: FrameReader,
                 viewer=None, actor: str = ""):
        self.store = store
        self.reader = reader
        self.viewer = viewer
        self.actor = actor
        # P0A multi-movie: extra readers by task movie key; tasks carry
        # "movie" naming their reader, defaulting to the primary one.
        self._readers: dict[str, FrameReader] = {}
        self._default_movie: str = ""
        self.tasks: list[dict] = []
        self.current: dict | None = None
        self._apex_layer = None
        self._image_layer = None
        self._back: list[str] = []  # visited task uuids, for the Back button
        self._path_layer = None
        self._path_pts: list[list[float]] = []  # in-progress centerline
        self._context_offset: int = 0  # neighbor-frame viewing offset
        self._consulted: list[int] = []  # context frames inspected
        # P0A crossing/clump: multiple labeled lanes/paths per task.
        self._lanes: dict[str, list[list[float]]] = {}
        self._lane: str = "A"
        self._grains: dict[str, dict] = {}  # label -> {xy, root_xy}
        self._mark_mode: str = "grain"  # owner tasks: grain | root
        self._region_mode: bool = False
        self._region_pts: list = []
        self._box_pending: list = []
        self._neg_note: str = ""
        self._look_only: bool = False  # look-only mode: clicks never mark
        self._census_mode: str = 'tip'
        self._drafts_visible: bool = True
        # Every dot placed on a pile/crossing/trace, newest last:
        # ("grain"|"root"|"lane"|"path", label). Undo pops one dot.
        self._dot_history: list[tuple[str, str]] = []
        self.analysis_session = None

    def require_task_frame(self):
        if self._context_offset != 0:
            raise ValueError("return to the task frame before changing annotations")

    def load_tasks(self, tasks: list[dict]) -> None:
        self.tasks = list(tasks)
        for t in self.tasks:
            if not t.get("completed"):
                previous = self.store.load(t['uuid'])
                if previous is None or previous['data'] != t:
                    self.store.save("task", t["uuid"], t, actor=self.actor)

    def _record_dot(self, kind, lab: str = "") -> None:
        """Append to dot history AND persist it on the task.

        Restarts used to orphan saved dots from history, silently
        killing undo. History now survives in the task record and is
        restored by advance()/go_back(). Accepts a ("kind", lab)
        tuple from the click paths.
        """
        if isinstance(kind, (tuple, list)):
            kind, lab = kind[0], kind[1] if len(kind) > 1 else ""
        self._dot_history.append((str(kind), str(lab)))
        if self.current is not None:
            self.current['completed'] = False
        self.persist_draft()

    def persist_draft(self):
        """Save editable work without creating or revising biological truth."""
        if self.current is None:
            return
        if self._dot_history or '_dot_history' in self.current:
            self.current["_dot_history"] = [list(q) for q in self._dot_history]
        # Census evidence lives in the task itself. Merely leaving its view
        # must not create a new scientific revision from unrelated path drafts.
        # Actual census edits save their fields; dot history still persists.
        if self.current.get('task_type') != 'census':
            self._capture_mask_draft()
            self.current["_drawing_draft"] = copy.deepcopy({
                "path_xy": self._path_pts, "lanes": self._lanes,
                "region_xy": self._region_pts, "active_lane": self._lane,
                "context_frames": self._consulted})
        previous = self.store.load(self.current['uuid'])
        if previous is None or previous['data'] != self.current:
            self.store.save("task", self.current["uuid"], self.current, actor=self.actor)
        if callable(getattr(self, 'on_draft_changed', None)):
            self.on_draft_changed()

    def _confirm_review_task(self):
        if self.current is not None:
            self.current.update(review_status='active', review_verdict='confirmed')
            self.current.pop('review_withdrawal', None)

    def _save_review(self, kind, uuid, data, actor=''):
        """Explicit confirmation supersedes a prior withdrawal atomically.

        Every save also appends the state it replaces to an explicit
        logical undo trail (rev14 P1). Undo traverses accepted user edits
        through that trail -- never "the previous numeric revision", which
        would redo its own restorations.
        """
        if (self.current or {}).get('task_type') == 'grain_identity' and kind != 'grain_identity':
            raise ValueError('grain identity tasks save grain centres, not tube observations')
        data = dict(data, review_status='active')
        data.pop('review_withdrawal', None)
        prior = self.store.load(uuid)
        if prior is not None and prior.get('kind') == kind:
            previous = {k: v for k, v in (prior.get('data') or {}).items()
                        if k not in ('undo_trail', 'revision')}
            trail = list((prior.get('data') or {}).get('undo_trail') or [])
            trail = (trail + [previous])[-20:]
            data['undo_trail'] = trail
        self._confirm_review_task()
        records = [(kind, uuid, data)]
        if self.current is not None:
            records.append(('task', self.current['uuid'], self.current))
        return self.store.save_many(records, actor=actor)[0]

    @_task_frame_write
    def withdraw_review(self):
        if self.current is None:
            raise ValueError('Open a task before withdrawing its review')
        self.persist_draft()
        revised = self.store.withdraw_task_review(self.current['uuid'], actor=self.actor)
        self.current.clear()
        self.current.update(revised)
        return revised

    def restore_draft(self):
        if self.current is None:
            return
        self._restore_dots()
        draft = self.current.get("_drawing_draft")
        if draft is not None:
            self._path_pts = copy.deepcopy(draft.get("path_xy", []))
            self._lanes = copy.deepcopy(draft.get("lanes", {}))
            self._region_pts = copy.deepcopy(draft.get("region_xy", []))
            self._lane = draft.get("active_lane", "A")
            self._consulted = list(draft.get("context_frames", []))
        elif self.current.get("task_type") == "centerline":
            saved = self.store.load("obs-" + self.current["uuid"])
            self._path_pts = copy.deepcopy((saved or {}).get("data", {}).get("path_xy", []))
            self._dot_history = [("path", "")] * len(self._path_pts)
        self._sync_path_layer()

    def open_task(self, uuid):
        """Select any stored queue item without losing an unfinished draft."""
        self.persist_draft()
        saved = self.store.load(uuid)
        if saved is None or saved["kind"] != "task":
            raise ValueError("Unknown annotation task")
        task = dict(saved["data"], uuid=uuid)
        self._reader_for(task)  # refuse a missing movie mapping before changing the display
        self.current = task
        self.tasks = [t if t["uuid"] != uuid else task for t in self.tasks]
        if not any(t["uuid"] == uuid for t in self.tasks):
            self.tasks.append(task)
        self._context_offset = 0
        self._consulted = []
        self._path_pts, self._region_pts, self._box_pending = [], [], []
        self._lanes = copy.deepcopy(task.get("draft_lanes", {}))
        self._lane, self._mark_mode = "A", "grain"
        self._grains = copy.deepcopy(task.get("grains", {}))
        self._region_mode, self._drafts_visible = False, True
        self.restore_draft()
        if self.viewer is not None:
            self._show_current()
        return task

    def _restore_dots(self) -> None:
        cur = self.current
        if cur is None:
            self._dot_history = []
            return
        try:
            self._dot_history = [
                (str(k), str(l))
                for k, l in (cur.get("_dot_history", []) or [])]
        except Exception:
            self._dot_history = []

    def advance(self) -> dict | None:
        self.persist_draft()
        pending = [t for t in self.tasks if not t.get("completed")]
        self.current = next_unfinished(pending)
        self._path_pts = []
        self._sync_path_layer()
        self._context_offset = 0
        self._consulted = []
        self._lanes = {}
        self._lane = "A"
        self._grains = {}
        self._mark_mode = "grain"
        self._dot_history = []
        self._restore_dots()
        self._region_mode = False
        self._region_pts: list = []
        self._box_pending: list = []
        self._neg_note: str = ""
        self._census_mode: str = 'tip'
        self._drafts_visible = True
        if self.current is not None:
            # Preload draft lanes on crossing reviews (editable copy).
            if str(self.current.get("task_type")) == "review_crossing":
                for _lab, _pts in (self.current.get("draft_lanes") or {}).items():
                    if isinstance(_pts, list) and len(_pts) >= 1:
                        self._lanes[str(_lab)[:1]] = [
                            [float(q[0]), float(q[1])] for q in _pts]
            # Restore persisted clump grains when (re)visiting a task.
            for k, v in (self.current.get("grains") or {}).items():
                if isinstance(v, dict):
                    self._grains[k] = dict(v)
            self.restore_draft()
        if self.current is not None and self.viewer is not None:
            self._show_current()
        return self.current

    def add_reader(self, movie: str, reader: FrameReader) -> None:
        """Register a movie reader; first registration sets the default."""
        self._readers[str(movie)] = reader
        if not self._default_movie:
            self._default_movie = str(movie)

    def _reader_for(self, task: dict) -> FrameReader:
        key = str(task.get("movie", "") or task.get("movie_uuid", "") or self._default_movie)
        if not key:
            return self.reader
        if key not in self._readers:
            raise ValueError(f"No movie reader registered for {key!r}")
        return self._readers[key]

    def _show_current(self) -> None:
        assert self.current is not None
        fid = int(self.current["query_frames"][0])
        res = self._reader_for(self.current).read(fid)
        data = res.frame
        if self._image_layer is None:
            self._image_layer = self.viewer.add_image(
                data, name="raw", rgb=data.ndim == 3)
        else:
            self._image_layer.data = data
        # Center on the target ball and zoom in (H236: centered, zoomed,
        # easy — camera work, not cropping).
        # frame the annotation (ring + guide) when the task has one;
        # otherwise centre on the declared focus at the declared zoom
        if not self._fit_camera_to_task():
            focus = self.current.get("focus_xy")
            if focus is not None:
                try:
                    self.viewer.camera.center = (float(focus[1]),
                                                 float(focus[0]))
                    self.viewer.camera.zoom = float(
                        self.current.get("view_zoom", 5.0) or 5.0)
                except Exception:
                    pass
        self._show_drafts()
        self._show_mask_task()
        self._sync_path_layer()
        if self._apex_layer is not None:
            prior = self.store.load("obs-" + self.current["uuid"])
            xy = (prior or {}).get("data", {}).get("direct_xy")
            self._apex_layer.data = [[xy[1], xy[0]]] if xy else []
        if self.analysis_session is not None:
            self.analysis_session.show_frame(self, fid)
        self._context_overlay_visibility(True)
        # Restore context visibility first, then replace marks with those of
        # the destination task. Old census/region marks must not survive it.
        self.draw_task_markup()

    def _context_overlay_visibility(self, on_task_frame: bool) -> None:
        """Never display task-frame marks as evidence in a different frame."""
        if self.viewer is None:
            return
        saved = getattr(self, "_context_layer_visibility", {})
        if on_task_frame and not saved:
            return
        for layer in self.viewer.layers:
            if layer is self._image_layer or str(layer.name).startswith("analysis-"):
                continue
            if not on_task_frame:
                saved.setdefault(id(layer), layer.visible)
                layer.visible = False
            elif id(layer) in saved:
                layer.visible = saved[id(layer)]
        self._context_layer_visibility = {} if on_task_frame else saved

    def _fit_camera_to_task(self) -> bool:
        """Frame the ANNOTATION, not a single point (rev8).

        Centring on one point leaves the thing being drawn off to one
        side of the canvas (reported twice against the live app). This
        fits the camera to the union of the target ring and the guide
        path with a margin, so what the task is about sits in the
        middle of the view at a size that shows all of it.
        """
        from math import isfinite
        if self.viewer is None or self.current is None:
            return False
        pts = []
        if self.current.get("review_region_fixed"):
            pts.extend(tuple(q) for q in self.current.get("review_region", []))
        for q in (self.current.get("guide_path") or []):
            try:
                pts.append((float(q[0]), float(q[1])))
            except (TypeError, ValueError, IndexError):
                continue
        for key in ("target_xy", "focus_xy"):
            t = self.current.get(key)
            if t and len(t) == 2:
                try:
                    pts.append((float(t[0]), float(t[1])))
                except (TypeError, ValueError):
                    pass
        ring = self.current.get("target_xy")
        ring_center = None
        if ring is not None and len(ring) == 2:
            try:
                rx, ry = float(ring[0]), float(ring[1])
                radius = max(0., float(self.current.get("target_r", 14.)))
                if all(isfinite(value) for value in (rx, ry, radius)):
                    ring_center = (rx, ry)
                    pts.extend([(rx-radius, ry-radius), (rx+radius, ry+radius)])
            except (TypeError, ValueError):
                pass
        pts = [q for q in pts if all(isfinite(value) for value in q)]
        if not pts:
            return False
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        # rev8: the RING belongs in the middle of the canvas -- the user
        # reads its position as the answer to "which grain am I working
        # on", and twice reported it as off-centre when the view was
        # centred on the ring+guide union instead. The guide still sets
        # the zoom, so the whole tube stays visible.
        if ring_center is not None and not self.current.get("review_region_fixed"):
            cx, cy = ring_center
        # A one-sided tube can extend almost the entire bounding-box width
        # from the centred grain. Fit distances from the chosen centre, not
        # the union's width, and use the actual canvas aspect ratio.
        width = max(24., 2 * max(abs(x-cx) for x in xs)) * 1.12
        height = max(24., 2 * max(abs(y-cy) for y in ys)) * 1.12
        cw, ch = self._canvas_size()
        if cw <= 0 or ch <= 0:
            return False
        zoom = min(8.0, cw / width, ch / height)
        try:
            self.viewer.camera.center = (cy, cx)
            self.viewer.camera.zoom = zoom
        except Exception:
            return False
        return True

    def _canvas_size(self):
        """Return logical (width, height), independent of the canvas wrapper."""
        if self.viewer is None:
            return 0., 0.
        window = getattr(self.viewer, 'window', None)
        if window is None:
            return 0., 0.
        qt_viewer = getattr(window, '_qt_viewer', None) or getattr(window, 'qt_viewer', None)
        if qt_viewer is None:
            return 0., 0.
        canvas = qt_viewer.canvas
        native = getattr(canvas, 'native', None)
        if native is not None:
            # Qt QSize is always width/height. Napari's VispyCanvas.size
            # instead returns height/width, unlike raw Vispy SceneCanvas.
            size = native.size()
            return float(size.width()), float(size.height())
        size = canvas.size
        try:
            return float(size.width()), float(size.height())
        except AttributeError:
            if hasattr(canvas, '_scene_canvas'):
                return float(size[1]), float(size[0])
            return float(size[0]), float(size[1])

    def toggle_drafts(self) -> bool:
        """Show/hide the yellow draft overlays. Returns visibility."""
        self._drafts_visible = not self._drafts_visible
        if self.viewer is not None and self.current is not None:
            try:
                if self._drafts_visible:
                    self._clear_drafts()
                    self._show_drafts()
                else:
                    for ly in list(self.viewer.layers):
                        if ly.name.startswith("draft-"):
                            self.viewer.layers.remove(ly)
            except Exception:
                pass
        return self._drafts_visible

    def _clear_drafts(self) -> None:
        """Remove all draft overlay layers (stale drafts must never
        leak onto the next task's frame). Headless-safe no-op."""
        if self.viewer is None:
            return
        try:
            for ly in list(self.viewer.layers):
                if ly.name.startswith("draft-"):
                    self.viewer.layers.remove(ly)
        except Exception:
            pass

    def _show_drafts(self) -> None:
        """Draw draft geometry (earlier judgments) as faint overlays.

        Drafts are guidance, never truth: dashed yellow path/lanes,
        hollow circle for a draft point. Headless-safe no-op.
        """
        if self.viewer is None or self.current is None:
            return
        if not self._drafts_visible:
            return
        self._clear_drafts()
        try:
            layers = []
            t = self.current
            if t.get("draft_xy"):
                layers.append(("draft-point", [[t["draft_xy"][1],
                                                t["draft_xy"][0]]], "points",
                               None))
            if (t.get("draft_path") or {}).get("path_xy"):
                pts = [[q[1], q[0]] for q in
                       t["draft_path"]["path_xy"]]
                layers.append(("draft-path", [pts], "path", None))
            for lab, pts in (t.get("draft_lanes") or {}).items():
                if isinstance(pts, list) and len(pts) >= 2:
                    layers.append((f"draft-lane-{lab}",
                                   [[[q[1], q[0]] for q in pts]], "path",
                                   None))
            # rev6 duels: lane A yellow, lane B cyan — the vote needs
            # distinguishable curves. Other task types stay all-yellow.
            if str(t.get("task_type", "")) == "route_duel":
                layers = [(n, d, k, ("yellow" if n.endswith("-A")
                                     else "cyan" if n.endswith("-B")
                                     else None))
                          for n, d, k, _ in layers]
            else:
                # rev13 fix: entries are 4-tuples (name, data,
                # kind, color); the old 3-way unpack raised and the
                # swallowed exception meant NO draft overlay was
                # ever drawn for non-route_duel tasks (caught by
                # the live W4 edit receipt).
                layers = [(n, d, k, None) for n, d, k, _c in layers]
            names = {ly.name for ly in self.viewer.layers}
            for name, data, kind, color in layers:
                if name in names:
                    continue
                if kind == "points":
                    # rev13: the draft must be UNMISSABLE — a filled
                    # high-contrast marker (a hollow thin ring was
                    # reported invisible on the live app against the
                    # bright phase-contrast background).
                    native = self.analysis_session is not None
                    self.viewer.add_points(
                        data, name=name, size=9 if native else 18,
                        face_color="transparent" if native else "yellow",
                        border_color="yellow" if native else "black", opacity=0.95)
                else:
                    self.viewer.add_shapes(
                        data, shape_type="path", name=name,
                        edge_color=color or "yellow", edge_width=2)
        except Exception as e:  # noqa: BLE001
            # never swallow silently again: an invisible draft was
            # reported from the live app and the old `pass` hid the
            # cause. Full traceback so the exact line is known.
            import traceback as _tb
            print(f"draft overlay failed: {type(e).__name__}: {e}\n"
                  + _tb.format_exc(), flush=True)

    @_task_frame_write
    def save_apex(self, x: float, y: float) -> Observation:
        """Record an apex click for the current task (transactional)."""
        if self.current is None:
            raise ValueError("no current task")
        obs = Observation(
            # Stable id per task (H237): re-marking after Back overwrites
            # the old click instead of piling up duplicates; the store's
            # revision history keeps every attempt.
            uuid=f"obs-{self.current.get('uuid', new_uuid())}",
            owner_uuid=str(self.current.get("owner_uuid", "")),
            source_frame=int(self.current["query_frames"][0]),
            direct_state=DIRECT_VISIBLE,
            direct_xy=(float(x), float(y)),
            annotator=self.actor or "annotator",
        )
        self._save_review("observation", obs.uuid, {
            "task_uuid": str(self.current.get("uuid", "")),
            "owner_uuid": obs.owner_uuid,
            "source_frame": obs.source_frame,
            "direct_state": obs.direct_state,
            "direct_xy": list(obs.direct_xy or []),
            "tip_source": "explicit_point",
            "distinct_cap_evidence": self.current.get("distinct_cap_evidence"),
            "annotator": obs.annotator,
            "revision": obs.revision,
        }, actor=self.actor)
        if self.viewer is not None:
            pts = [[y, x]]
            if self._apex_layer is None:
                self._apex_layer = self.viewer.add_points(
                    pts, name="apex", size=9, face_color="transparent",
                    border_color="cyan")
                try:
                    self._apex_layer.editable = False  # never steal clicks
                except Exception:
                    pass
            else:
                self._apex_layer.data = pts
        return obs

    @_task_frame_write
    def save_cant_tell(self, task_uuid: str | None = None) -> Observation:
        # P0 repair: task-derived IDs (was random) + explicit task link,
        # so folder/scorer join observations to tasks deterministically.
        if self.current is None:
            raise ValueError("no current task")
        obs = Observation(
            uuid=f"obs-{task_uuid or self.current.get('uuid', new_uuid())}",
            owner_uuid=str(self.current.get("owner_uuid", "")),
            source_frame=int(self.current["query_frames"][0]),
            direct_state=NOT_DIRECTLY_VISIBLE,
            annotator=self.actor or "annotator",
        )
        self._save_review("observation", obs.uuid, {
            "task_uuid": str(self.current.get("uuid", "")),
            "owner_uuid": obs.owner_uuid,
            "source_frame": obs.source_frame,
            "direct_state": obs.direct_state,
            "annotator": obs.annotator,
        }, actor=self.actor)
        return obs

    @_task_frame_write
    def save_germination_event(self, verdict: str, last_absent: int | None = None,
                                first_visible: int | None = None) -> str:
        """Save the current germination-episode verdict (WO2, transactional).

        Bracket endpoints are source frames marked while scrubbing; either
        may be null (censored). Consulted frames come from the task's
        recorded context trail. Validates against the schema before writing;
        a bad verdict never reaches the DB.
        """
        from tubetracker.annotation_schema import (
            GerminationEvent, GERMINATION_VERDICTS)
        if self.current is None:
            raise ValueError("no current task")
        if str(self.current.get("task_type")) != "germination_event":
            raise ValueError("current task is not a germination event")
        if verdict not in GERMINATION_VERDICTS:
            raise ValueError(f"bad verdict {verdict!r}")
        task = self.current
        event = GerminationEvent(
            uuid=f"event-{task.get('uuid', '')}",
            movie_uuid=str(task.get("movie") or task.get("movie_uuid") or ""),
            movie_content_hash=str(task.get("movie_content_hash", "") or ""),
            owner_uuid=str(task.get("owner_uuid", "") or ""),
            task_uuid=str(task.get("uuid", "") or ""),
            window_start=int(task.get("source_start", 0)),
            window_end=int(task.get("source_end", 0)),
            verdict=verdict,
            last_absent_frame=last_absent,
            first_visible_frame=first_visible,
            consulted_frames=sorted({int(q) for q in (self._consulted or [])}),
            annotator=self.actor or "annotator",
            lineage=[f"task {task.get('uuid', '')} "
                     f"[{task.get('source_start')},{task.get('source_end')}]"],
        )
        errs = event.validate()
        if errs:
            raise ValueError(f"bad germination event: {errs}")
        data = {k: getattr(event, k) for k in (
            "uuid", "movie_uuid", "movie_content_hash", "owner_uuid",
            "task_uuid", "window_start", "window_end", "verdict",
            "last_absent_frame", "first_visible_frame", "consulted_frames",
            "absence_region_xy", "grain_mask_uuid", "tube_mask_uuid",
            "exit_xy", "apex_xy", "annotator", "revision", "lineage")}
        data["review_origin"] = "human"
        self.current["completed"] = True  # the verdict is the whole answer
        self._save_review("germination_event", event.uuid, data, actor=self.actor)
        return event.uuid

    @_task_frame_write
    def save_and_next(self) -> dict | None:
        """Mark current complete (only if an observation exists)."""
        if self.current is None:
            return None
        self._back.append(str(self.current["uuid"]))
        self.current["completed"] = True
        self.store.save("task", self.current["uuid"], self.current,
                        actor=self.actor)
        return self.advance()

    def click(self, x: float, y: float) -> dict | None:
        """One canvas click = mark the CENTER ball's tip, move on (H237).

        One tube per ball: the click saves and advances. The Back
        button revisits the previous ball; re-clicking overwrites.
        P0A: on centerline tasks the click extends the drawn path
        instead; on crossing tasks it extends the ACTIVE lane;
        clicks on a context frame refuse to save.
        """
        if self.current is None:
            return None
        if self._look_only:
            return self.current  # look-only mode: never mark anything
        if self._context_offset != 0:
            return None  # look, don't mark, on neighbor frames
        ttype = str(self.current.get("task_type", "apex"))
        if ttype == 'grain_identity':
            width, height = self._reader_for(self.current).native_size
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                raise ValueError('grain centre must be inside the native image')
            self._region_pts = [[float(x), float(y)]]
            self._dot_history = [('region', '')]
            self.current['completed'] = False
            self.persist_draft()
            return self.current
        if ttype == "review_tip":
            if self._region_mode:
                self.region_click(x, y)
                return self.current  # stay: Save-area finishes
            self.save_apex(x, y)
            return self.current  # stay: confirm/next button moves on
        if ttype == "review_path":
            return self.current  # buttons confirm the draft; no drawing
        if ttype == "neg_region":
            if self.current.get("negative_geometry") == "polygon":
                self.region_click(x, y)
                return self.current
            self.neg_click(x, y)
            return self.current  # single click acts + usually advances
        if ttype == "neg_draft":
            self.draft_click(x, y)
            return self.current  # inside a draft: confirm + advance
        if ttype == "census":
            self.census_click(x, y)
            return self.current  # stay: Tile-complete finishes
        if ttype == "germination_event":
            # Verdict tasks take no canvas marks: scrubbing only consults.
            # The dock's verdict controls save the answer; clicks stay.
            return self.current
        if ttype == "body_mask":
            # Painting uses the napari brush (mask-paint layer), not
            # clicks. A click here is guidance, never paint.
            self._neg_note = ("Use the brush on the mask-paint layer, "
                              "not clicks.")
            return self.current
        if ttype == "route_duel":
            # Click near a curve votes for it and advances. Far clicks
            # guide; Neither/Can't-tell are buttons.
            try:
                winner = self.duel_vote(x, y)
            except ValueError:
                self._neg_note = ("Click NEAR one of the two curves "
                                  "(yellow/blue), or Neither below.")
                return self.current
            self.save_duel(winner)
            return self.save_and_next()
        if ttype == "review_crossing":
            self.lane_click(self._lane, x, y)
            return self.current  # stay: confirm button finishes
        if ttype == "centerline":
            self.path_click(x, y)
            return self.current  # stay: double-click/Path-done finishes
        if ttype == "crossing":
            self.lane_click(self._lane, x, y)
            return self.current  # stay: lanes + continuation finish
        if ttype == "owner":
            # Clump work: mark grains and their roots under labels.
            if self._mark_mode == "root":
                self.root_click(self._lane, x, y)
            else:
                self.grain_click(self._lane, x, y)
            return self.current  # stay: Clump-done finishes
        self.save_apex(x, y)
        return self.save_and_next()

    # ---- P0A: crossing lanes ----
    # Display names: lanes are Tube 1/2 (crossings) or Ball 1/2/3...
    # (piles) on screen; A/B/C... stay as the stored keys. Piles grow
    # one row per ball, up to 8.
    BALL_LANES = ["A", "B", "C", "D", "E", "F", "G", "H"]
    LANE_DISPLAY = {lab: str(i + 1) for i, lab in enumerate(BALL_LANES)}

    def new_dots(self, label: str) -> int:
        """Dots the user placed this visit (excludes preloaded drafts)."""
        lab = str(label).upper()[:1] or "A"
        return sum(1 for kind, hist_lab in self._dot_history
                   if kind == "lane" and hist_lab == lab)

    def lane_number(self, label: str) -> str:
        return self.LANE_DISPLAY.get(str(label).upper()[:1] or "A",
                                     str(label).upper()[:1])

    def select_lane(self, label: str) -> str:
        """Activate a lane/ball (A/B/C...). Creates it if new.

        Auto-picks what the next click should mark: the ball itself
        when its dot is missing, else its tube start.
        """
        self._lane = str(label).upper()[:1] or "A"
        self._lanes.setdefault(self._lane, [])
        g = self._grains.get(self._lane, {})
        if not g.get("xy"):
            self._mark_mode = "grain"
        elif not g.get("root_xy") and not g.get("no_tube"):
            self._mark_mode = "root"
        else:
            self._mark_mode = "grain"
        return self._lane

    def lane_click(self, label: str, x: float, y: float) -> int:
        """Append a vertex to one crossing lane. Returns vertex count."""
        if self.current is None:
            raise ValueError("no current task")
        if self._context_offset != 0:
            raise ValueError("return to the task frame to draw")
        lane = self.select_lane(label)
        self._lanes[lane].append([float(x), float(y)])
        self._record_dot(("lane", lane))
        return len(self._lanes[lane])

    @_task_frame_write
    def save_crossing(self, continuation: dict[str, str] | None = None,
                      unresolved: bool = True) -> str:
        """Save lanes + continuation decision as a Crossing record.

        Each lane needs >=2 vertices; shared pixels are legal (lanes
        are separate lists, never merged). Returns the crossing uuid.
        """
        if self.current is None:
            raise ValueError("no current task")
        good = {k: v for k, v in self._lanes.items() if len(v) >= 2}
        if len(good) < 2:
            raise ValueError("a crossing needs >=2 lanes with >=2 points")
        from tubetracker.annotation_schema import Crossing
        mapping = {str(k): str(v) for k, v in (continuation or {}).items()
                   if k in good and v}
        allowed = {str(o['id']) for o in self.current.get('owner_choices', [])}
        if allowed and not set(mapping.values()).issubset(allowed):
            raise ValueError('Choose a known grain for each assigned lane')
        rec = Crossing(
            uuid=f"cross-{self.current.get('uuid', new_uuid())}",
            movie_uuid=str(self.current.get("movie_uuid") or self.current.get("movie", "")),
            owner_uuids=sorted(set(mapping.values())) if mapping else sorted(good),
            entry_segments={k: [(px, py) for px, py in v[:2]]
                            for k, v in good.items()},
            exit_segments={k: [(px, py) for px, py in v[-2:]]
                           for k, v in good.items()},
            continuation=mapping,
            unresolved=not (mapping and set(mapping) == set(good)) or unresolved,
        )
        errs = rec.validate()
        if errs:
            raise ValueError(f"invalid crossing: {errs}")
        self._save_review("crossing", rec.uuid, {
            "task_uuid": str(self.current.get("uuid", "")),
            "movie_uuid": rec.movie_uuid,
            "lanes": {k: [[px, py] for px, py in v]
                      for k, v in good.items()},
            "owner_uuids": rec.owner_uuids,
            "entry_segments": {k: [list(p) for p in v]
                               for k, v in rec.entry_segments.items()},
            "exit_segments": {k: [list(p) for p in v]
                              for k, v in rec.exit_segments.items()},
            "continuation": rec.continuation,
            "unresolved": rec.unresolved,
            "source_frame": int(self.current["query_frames"][0]),
            "context_frames": list(self._consulted),
            "annotator": self.actor or "annotator",
            "review_origin": self.current.get("review_origin", "human"),
        }, actor=self.actor)
        return rec.uuid

    @_task_frame_write
    def save_grain_identity(self, state='confirmed'):
        from .grain_identity import validate_identity_observation
        if not self.current or self.current.get('task_type') != 'grain_identity':
            raise ValueError('open a grain identity task first')
        if state == 'confirmed' and len(self._region_pts) != 1:
            raise ValueError('click the centre of this physical grain before saving')
        origin = self.current.get('review_origin', 'human')
        if self.actor == 'workflow-test':
            origin = 'workflow_test'
        data = {'task_uuid': self.current['uuid'], 'movie': self.current['movie'],
            'owner_uuid': self.current['owner_uuid'],
            'source_frame': int(self.current['query_frames'][0]),
            'identity_state': state,
            'grain_native': list(self._region_pts[0]) if state == 'confirmed' else None,
            'context_frames': list(self._consulted), 'annotator': self.actor,
            'review_origin': origin}
        validate_identity_observation(data)
        self.current['completed'] = True
        if state != 'confirmed':
            self._region_pts = []
            self._dot_history = []
        uuid = 'grain-identity-' + self.current['uuid']
        self._save_review('grain_identity', uuid, data, actor=self.actor)
        self.persist_draft()
        return uuid

    # ---- P0A: clump grain/root pairs ----
    # No-mode-needed flow: just click ball, tube-start, next ball,
    # next tube-start... grain_click auto-arms "tube start" next;
    # root_click auto-moves to the next unfinished ball and re-arms
    # "ball". The Ball 1/2/3 + clicks-mark buttons are only for
    # fixing an earlier ball.
    @_task_frame_write
    def grain_click(self, label: str, x: float, y: float) -> dict:
        """Mark one ball itself inside a pile (Ball 1/2/3...)."""
        if self.current is None:
            raise ValueError("no current task")
        lab = str(label).upper()[:1] or "A"
        self._lane = lab
        g = self._grains.setdefault(lab, {})
        g["xy"] = [float(x), float(y)]
        self._mark_mode = "root"  # next click wants its tube start
        self._record_dot(("grain", lab))
        self._persist_grains()
        return dict(g)

    @_task_frame_write
    def root_click(self, label: str, x: float, y: float) -> dict:
        """Mark where that same ball's tube comes out."""
        if self.current is None:
            raise ValueError("no current task")
        lab = str(label).upper()[:1] or "A"
        self._lane = lab
        g = self._grains.setdefault(lab, {})
        g["root_xy"] = [float(x), float(y)]
        g.pop("no_tube", None)  # a drawn tube overrides a stale flag
        self._record_dot(("root", lab))
        self._persist_grains()
        # Auto-advance: first ball missing a dot wins, else a fresh
        # next ball, so plain clicking walks the pile with no buttons.
        nxt = self._next_unfinished_ball(exclude=lab)
        if nxt is not None:
            self._lane = nxt
        elif not g.get("xy"):
            pass
        else:
            order = self.BALL_LANES
            try:
                nxt2 = order[order.index(lab) + 1]
            except (ValueError, IndexError):
                nxt2 = None
            if nxt2 is not None and nxt2 not in self._grains:
                self._lane = nxt2
        self._mark_mode = "grain"
        return dict(g)

    def _next_unfinished_ball(self, exclude: str | None = None) -> str | None:
        for lab in self.BALL_LANES:
            if exclude is not None and lab == exclude:
                continue
            g = self._grains.get(lab)
            if g is not None and not (g.get("xy") and (
                    g.get("root_xy") or g.get("no_tube"))):
                return lab
        return None

    def ordered_balls(self) -> list[str]:
        """Ball labels in use, top-to-bottom table order."""
        return [lab for lab in self.BALL_LANES if lab in self._grains]

    def next_fresh_ball(self) -> str | None:
        """First unused ball label, or None when the table is full."""
        for lab in self.BALL_LANES:
            if lab not in self._grains:
                return lab
        return None

    def arm(self, label: str, what: str) -> str:
        """Point the next picture-click at one table cell.

        what="grain" arms the ball itself, anything else arms its
        tube start. Used by the pile table; canvas clicks keep
        working exactly as before.
        """
        lab = str(label).upper()[:1] or "A"
        self._lane = lab
        self._lanes.setdefault(lab, [])
        self._mark_mode = "root" if what == "root" else "grain"
        return lab

    @_task_frame_write
    def remove_ball(self, label: str) -> bool:
        """Delete one pile row (ball + its tube dot), then compact.

        Remaining balls slide up so the table stays Ball 1, 2, 3...
        with no gaps. Returns False when the row was already empty.
        """
        lab = str(label).upper()[:1] or "A"
        if lab not in self._grains:
            return False
        remaining = [l for l in self.BALL_LANES
                     if l in self._grains and l != lab]
        mapping = {old: new for new, old in
                   zip(self.BALL_LANES, remaining)}
        self._grains = {mapping[old]: dict(self._grains[old])
                        for old in remaining}
        self._lanes = {mapping.get(old, old): pts
                       for old, pts in self._lanes.items()
                       if old != lab}
        kept: list[tuple[str, str]] = []
        for kind, hist_lab in self._dot_history:
            if hist_lab == lab:
                continue  # dots of the deleted row vanish
            kept.append((kind, mapping.get(hist_lab, hist_lab)))
        self._dot_history = kept
        nxt = self._next_unfinished_ball()
        self._lane = nxt if nxt is not None else (
            self.next_fresh_ball() or "A")
        g = self._grains.get(self._lane, {})
        self._mark_mode = ("root" if g.get("xy") and not g.get("root_xy")
                           else "grain")
        self._persist_grains()
        return True

    def clump_next_needed(self) -> tuple[str, str]:
        """(ball label, 'ball'|'tube start'|'done') — what to click next."""
        g = self._grains.get(self._lane, {})
        if not g.get("xy"):
            return self._lane, "ball"
        if not g.get("root_xy") and not g.get("no_tube"):
            return self._lane, "tube start"
        nxt = self._next_unfinished_ball()
        if nxt is not None:
            return nxt, "ball" if not self._grains[nxt].get("xy") else "tube start"
        for lab in self.BALL_LANES:
            if lab not in self._grains:
                return lab, "ball"
        return self._lane, "done"

    @_task_frame_write
    def _persist_grains(self) -> None:
        assert self.current is not None
        self.current["grains"] = {k: dict(v)
                                  for k, v in self._grains.items()}
        self.store.save("task", str(self.current["uuid"]), self.current,
                        actor=self.actor)

    def clump_complete(self) -> bool:
        """A pile is done when every marked ball has its tube answer.

        Each ball needs its ball dot plus EITHER a tube-start dot OR
        an explicit no-tube flag (some balls genuinely grow no tube).
        A missing tube dot without the flag is unfinished, not negative.
        """
        return bool(self._grains) and all(
            g.get("xy") and (g.get("root_xy") or g.get("no_tube"))
            for g in self._grains.values())

    @_task_frame_write
    def set_no_tube(self, label: str, val: bool = True) -> dict:
        """Flag one ball as genuinely tube-less (or lift the flag).

        Requires the ball dot; refuses when a tube dot is present
        (remove that first — the two are mutually exclusive). The
        flag is its own undo: call with False to lift it. Drawing a
        tube dot later clears a stale flag.
        """
        if self.current is None:
            raise ValueError("no current task")
        lab = str(label).upper()[:1] or "A"
        g = self._grains.get(lab)
        if g is None or not g.get("xy"):
            raise ValueError("mark the ball first")
        if val and g.get("root_xy"):
            raise ValueError("this ball already has a tube dot")
        if val:
            g["no_tube"] = True
        else:
            g.pop("no_tube", None)
        self._persist_grains()
        if val:  # ball answered -> walk on like after a tube dot
            nxt = self._next_unfinished_ball(exclude=lab)
            if nxt is not None:
                self._lane = nxt
            else:
                fresh = self.next_fresh_ball()
                if fresh is not None and fresh != lab:
                    self._lane = fresh
            self._mark_mode = "grain"
        else:
            self._lane = lab
            self._mark_mode = "root"
        return dict(g)

    @_task_frame_write
    def finish_path_and_next(self) -> dict | None:
        """Save the drawn path (complete iff it reaches the far end)."""
        if self.current is None:
            return None
        ttype = str(self.current.get("task_type", "apex"))
        if ttype == "owner":
            # Clump-done: every marked grain needs its root; completing
            # A preserves B/C markers (separate labels, never merged).
            if not self.clump_complete():
                raise ValueError("each marked grain still needs its root")
            return self.save_and_next()
        if ttype == "crossing":
            # Lanes saved unresolved by default; explicit continuation
            # assignment UI follows. Unresolved-but-drawn beats unmarked.
            self.save_crossing(unresolved=True)
            return self.save_and_next()
        if ttype != "centerline":
            return self.current
        self.save_path(complete=True)
        return self.save_and_next()

    def go_back(self) -> dict | None:
        """Revisit the previously completed task (H237)."""
        self.persist_draft()
        while self._back:
            uuid = self._back.pop()
            prev = next((t for t in self.tasks if t.get("uuid") == uuid),
                        None)
            if prev is None:
                continue
            prev["completed"] = False
            self.store.save("task", uuid, prev, actor=self.actor)
            self.current = prev
            self._path_pts = []
            self._context_offset = 0
            self._consulted = []
            self._lanes = {}
            self._lane = "A"
            self._grains = {}
            self._mark_mode = "grain"
            self._dot_history = []
            self._restore_dots()
            self._region_mode = False
            self._region_pts = []
            self._box_pending = []
            self._census_mode = 'tip'
            if str(prev.get("task_type")) == "review_crossing":
                for _lab, _pts in (prev.get("draft_lanes") or {}).items():
                    if isinstance(_pts, list) and len(_pts) >= 1:
                        self._lanes[str(_lab)[:1]] = [
                            [float(q[0]), float(q[1])] for q in _pts]
            for k, v in (prev.get("grains") or {}).items():
                if isinstance(v, dict):
                    self._grains[k] = dict(v)
            self.restore_draft()
            if self.viewer is not None:
                self._show_current()
            return prev
        return self.current

    # ---- P0A: temporal context ----
    def step_frame(self, delta: int) -> int | None:
        """View-only step to a neighboring source frame (context).

        Marks must be made on the task frame: click() while offset
        refuses to save. Consulted frames attach to the next save.
        Headless-safe (no viewer needed for bookkeeping).
        """
        if self.current is None:
            return None
        n = len(self._reader_for(self.current))
        fid = int(self.current["query_frames"][0]) + self._context_offset + delta
        fid = max(0, min(n - 1, fid))
        base = int(self.current["query_frames"][0])
        self._context_offset = fid - base
        if self._context_offset != 0 and fid not in self._consulted:
            self._consulted.append(fid)
        if self.viewer is not None:
            try:
                res = self._reader_for(self.current).read(fid)
                self._image_layer.data = res.frame
                self._context_overlay_visibility(self._context_offset == 0)
                if self.analysis_session is not None:
                    self.analysis_session.show_frame(self, fid)
            except Exception:
                pass
        return self._context_offset

    def back_to_task_frame(self) -> None:
        """Return the view to the task frame (consultation kept)."""
        self._context_offset = 0
        if self.viewer is not None and self.current is not None:
            try:
                res = self._reader_for(self.current).read(
                    int(self.current["query_frames"][0]))
                self._image_layer.data = res.frame
                self._context_overlay_visibility(True)
                if self.analysis_session is not None:
                    self.analysis_session.show_frame(self, int(self.current["query_frames"][0]))
            except Exception:
                pass

    # ---- P0A: typed missingness ----
    @_task_frame_write
    def save_state(self, state: str) -> Observation:
        """Record a typed no-observation state (never invents coords)."""
        if self.current is None:
            raise ValueError("no current task")
        obs = Observation(
            uuid=f"obs-{self.current.get('uuid', new_uuid())}",
            owner_uuid=str(self.current.get("owner_uuid", "")),
            source_frame=int(self.current["query_frames"][0]),
            direct_state=state,
            context_frames=list(self._consulted),
            annotator=self.actor or "annotator",
        )
        errs = obs.validate()
        if errs:
            raise ValueError(f"invalid observation: {errs}")
        previous = self.store.load(obs.uuid)
        saved_path = {k: previous["data"][k] for k in
                      ("path_xy", "path_visible", "tube_uuid", "path_source")
                      if previous and k in previous["data"]}
        if saved_path.get("path_xy"):
            saved_path["path_complete"] = False
            obs.path_xy = saved_path["path_xy"]
            obs.path_visible = saved_path.get("path_visible", [])
            obs.tube_uuid = saved_path.get("tube_uuid", "")
        self._save_review("observation", obs.uuid, {
            **saved_path,
            "task_uuid": str(self.current.get("uuid", "")),
            "owner_uuid": obs.owner_uuid,
            "source_frame": obs.source_frame,
            "direct_state": obs.direct_state,
            "context_frames": obs.context_frames,
            "annotator": obs.annotator,
            "tip_source": "visibility_review",
            "review_origin": self.current.get("review_origin", "human"),
        }, actor=self.actor)
        return obs

    # ---- P0A: centerline paths ----
    def _sync_path_layer(self) -> None:
        """Repaint the in-progress centerline from _path_pts.

        Clicks draw incrementally, but undo/advance/Back mutate the
        points without clicking — without this the yellow line on
        screen lies about the data (undo looked dead, stale lines
        leaked across tasks). Headless-safe no-op.
        """
        if self.viewer is None:
            return
        try:
            pts = [[py, px] for px, py in self._path_pts]
            if len(pts) >= 2:
                if self._path_layer is None:
                    self._path_layer = self.viewer.add_shapes(
                        [pts], shape_type="path", name="centerline",
                        edge_color="yellow", edge_width=2,
                        blending="translucent_no_depth")
                else:
                    self._path_layer.data = [pts]
            elif self._path_layer is not None:
                try:
                    self.viewer.layers.remove(self._path_layer)
                except Exception:
                    pass
                self._path_layer = None
        except Exception:
            pass

    def path_click(self, x: float, y: float) -> int:
        """Append one vertex to the in-progress centerline. Returns count."""
        if self.current is None:
            raise ValueError("no current task")
        if self._context_offset != 0:
            raise ValueError("return to the task frame to draw")
        self._path_pts.append([float(x), float(y)])
        self._record_dot(("path", ""))
        self._sync_path_layer()
        return len(self._path_pts)

    def _path_tip_evidence(self, complete, endpoint):
        if complete:
            return DIRECT_VISIBLE, tuple(endpoint), "reviewed_full_path"
        prior = self.store.load(f"obs-{self.current['uuid']}")
        data = prior["data"] if prior else {}
        if data.get("tip_source") == "explicit_point":
            from .review_semantics import precise_tip
            tip = precise_tip(data)
            if tip is not None:
                return DIRECT_VISIBLE, tuple(tip), "explicit_point"
        return NOT_DIRECTLY_VISIBLE, None, "unobserved"

    @_task_frame_write
    def save_path(self, complete: bool = False) -> Observation:
        """Save the drawn polyline (partial spans allowed, never closed)."""
        if self.current is None:
            raise ValueError("no current task")
        if len(self._path_pts) < 2:
            raise ValueError("a path needs at least 2 points")
        apex = self._path_pts[-1]
        state, tip, tip_source = self._path_tip_evidence(complete, apex)
        obs = Observation(
            uuid=f"obs-{self.current.get('uuid', new_uuid())}",
            owner_uuid=str(self.current.get("owner_uuid", "")),
            source_frame=int(self.current["query_frames"][0]),
            direct_state=state,
            direct_xy=tip,
            context_frames=list(self._consulted),
            path_xy=[(px, py) for px, py in self._path_pts],
            path_visible=[True] * len(self._path_pts),
            path_complete=complete,
            annotator=self.actor or "annotator",
        )
        errs = obs.validate()
        if errs:
            raise ValueError(f"invalid observation: {errs}")
        self._save_review("observation", obs.uuid, {
            "task_uuid": str(self.current.get("uuid", "")),
            "owner_uuid": obs.owner_uuid,
            # rev7: task-scoped tube identity. Every observation from
            # one trace task is one tube-time of one tube; cross-time
            # linking (same tube at another frame) is UNBUILT — join
            # on tube_uuid only within a task until then.
            "tube_uuid": f"tube-{self.current.get('uuid', new_uuid())}",
            "source_frame": obs.source_frame,
            "direct_state": obs.direct_state,
            "direct_xy": list(tip) if tip is not None else None,
            "tip_source": tip_source,
            "context_frames": obs.context_frames,
            "path_xy": [[px, py] for px, py in self._path_pts],
            "path_visible": [True] * len(self._path_pts),
            "path_complete": complete,
            "annotator": obs.annotator,
            "review_origin": self.current.get("review_origin", "human"),
            "annotation_role": self.current.get("annotation_role", "unspecified"),
        }, actor=self.actor)
        return obs

    @_task_frame_write
    def save_region(self, points: list) -> Observation:
        """Save an imprecise apex as a bounded region, never an exact point.

        Repair batch (P0A): the DT dim tips may prove unresolvable to a
        pixel. A region judgment carries no direct_xy, so fold/export
        can never promote it into an exact target. Requires >=3 vertices.
        """
        if self.current is None:
            raise ValueError("no current task")
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) < 3:
            raise ValueError("a region needs at least 3 points")
        obs = Observation(
            uuid=f"obs-{self.current.get('uuid', new_uuid())}",
            owner_uuid=str(self.current.get("owner_uuid", "")),
            source_frame=int(self.current["query_frames"][0]),
            direct_state=VISIBLE_IMPRECISE,
            direct_region=pts,
            tube_uuid=str(self.current.get("tube_uuid", "")),
            annotator=self.actor or "annotator",
        )
        errs = obs.validate()
        if errs:
            raise ValueError(f"invalid observation: {errs}")
        self._save_review("observation", obs.uuid, {
            "task_uuid": str(self.current.get("uuid", "")),
            "owner_uuid": obs.owner_uuid,
            "source_frame": obs.source_frame,
            "direct_state": obs.direct_state,
            "direct_region": [list(q) for q in pts],
            "tube_uuid": obs.tube_uuid,
            "annotator": obs.annotator,
        }, actor=self.actor)
        return obs

    @_task_frame_write
    def confirm_path_span(self, complete: bool) -> Observation:
        """Confirm a draft path's span without redrawing it (repair batch).

        The current task must carry task["draft_path"] (path_xy from an
        earlier stroke, shown as an editable draft). Saves a confirming
        observation with path_complete set True (full root-to-tip) or
        False (partial, rest hidden) — then advances.
        """
        if self.current is None:
            raise ValueError("no current task")
        draft = self.current.get("draft_path") or {}
        path = [tuple(q) for q in draft.get("path_xy", [])]
        if len(path) < 2:
            raise ValueError("draft path needs at least 2 points")
        if complete and draft.get("path_visible") and not all(draft["path_visible"]):
            raise ValueError("FULL requires the whole current path to be visible")
        state, tip, tip_source = self._path_tip_evidence(complete, path[-1])
        obs = Observation(
            uuid=f"obs-{self.current.get('uuid', new_uuid())}",
            owner_uuid=str(self.current.get("owner_uuid", "")),
            source_frame=int(self.current["query_frames"][0]),
            direct_state=state,
            direct_xy=tip,
            path_xy=list(path),
            path_visible=[bool(v) for v in
                          draft.get("path_visible", [])] or [True] * len(path),
            path_complete=bool(complete),
            tube_uuid=str(self.current.get("tube_uuid", "")),
            annotator=self.actor or "annotator",
        )
        errs = obs.validate()
        if errs:
            raise ValueError(f"invalid observation: {errs}")
        self._save_review("observation", obs.uuid, {
            "task_uuid": str(self.current.get("uuid", "")),
            "owner_uuid": obs.owner_uuid,
            "source_frame": obs.source_frame,
            "direct_state": obs.direct_state,
            "direct_xy": list(tip) if tip is not None else None,
            "tip_source": tip_source,
            "path_xy": [list(q) for q in path],
            "path_visible": obs.path_visible,
            "path_complete": bool(complete),
            "tube_uuid": obs.tube_uuid,
            "annotator": obs.annotator,
        }, actor=self.actor)
        return obs

    @_task_frame_write
    def assign_continuation(self, mapping: dict) -> str:
        """Link lanes to stable tubes on a draft crossing (repair batch).

        The current task must carry task["draft_crossing"] (a crossing
        uuid whose lanes are shown as drafts). mapping = {lane: tube}.
        Empty mapping keeps the crossing unresolved (lanes preserved).
        Returns the crossing uuid.
        """
        if self.current is None:
            raise ValueError("no current task")
        xid = str(self.current.get("draft_crossing", ""))
        if not xid:
            raise ValueError("task carries no draft crossing")
        rec = self.store.load(xid)
        if rec is None:
            raise ValueError(f"draft crossing {xid} not found")
        data = dict(rec["data"])
        lanes = data.get("lanes", {})
        full_lanes = {str(k) for k, v in lanes.items()
                      if isinstance(v, list) and len(v) >= 2}
        clean = {str(k): str(v) for k, v in (mapping or {}).items()
                 if str(k) in full_lanes}
        data["continuation"] = clean
        # Resolved only when EVERY traced lane is assigned; a partial
        # mapping stays explicitly unresolved (test-caught, H261).
        data["unresolved"] = not (bool(clean) and set(clean) == full_lanes)
        data["adjudicated"] = True
        self._save_review("crossing", xid, data, actor=self.actor)
        return xid

    def region_click(self, x: float, y: float) -> int:
        """Append one area corner for an imprecise apex. Returns count."""
        if self.current is None:
            raise ValueError("no current task")
        if self._context_offset != 0:
            raise ValueError("return to the task frame to draw")
        self._region_pts.append([float(x), float(y)])
        self._record_dot(("region", ""))
        return len(self._region_pts)

    @_task_frame_write
    def save_negative_polygon(self):
        """Confirm only the drawn polygon; all unmarked pixels remain unknown."""
        import numpy as np
        from prototypes.v30_video_apex.cap_evidence import inside_polygon
        from .annotation_schema import SupervisionRegion
        if self.current is None or self.current.get("task_type") != "neg_region":
            raise ValueError("Select a negative-region task")
        pts = np.asarray(self._region_pts, float)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2 or not np.isfinite(pts).all():
            raise ValueError("Draw at least three finite polygon vertices")
        area = abs(float(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1))))/2
        if area < 1:
            raise ValueError("The negative polygon must enclose an area")
        scope = self.current.get("review_region")
        if scope and not inside_polygon(pts, scope).all():
            raise ValueError("Keep the polygon inside this task's reviewed field")
        tips = self.current.get("known_caps", [])
        if self.current.get("known_tip_xy"):
            tips = list(tips) + [self.current["known_tip_xy"]]
        if tips and inside_polygon(tips, pts).any():
            raise ValueError("This polygon contains a reviewed cap; leave it outside the negative region")
        polygons = self.current.setdefault("negative_polygons", [])
        reg = SupervisionRegion(uuid=f"neg-{self.current['uuid']}-poly-{len(polygons)+1}",
            movie_uuid=self.current["movie"], source_frame=int(self.current["query_frames"][0]),
            polygon_xy=[tuple(p) for p in pts.tolist()], kind="verified_negative",
            class_scope=self.current.get("neg_class", "cap"), confirmed=True)
        errors = reg.validate()
        if errors:
            raise ValueError(f"Invalid region: {errors}")
        data = {"task_uuid": self.current["uuid"], "movie_uuid": reg.movie_uuid,
                "source_frame": reg.source_frame, "polygon_xy": pts.tolist(),
                "kind": reg.kind, "class_scope": reg.class_scope, "confirmed": True,
                "review_origin": self.current.get("review_origin", "human"),
                "annotation_role": self.current.get("annotation_role", "training"),
                "annotator": self.actor or "annotator"}
        self._save_review("region", reg.uuid, data, actor=self.actor)
        polygons.append({"uuid": reg.uuid, "polygon_xy": pts.tolist()})
        self._region_pts = []
        self._dot_history = []
        self.persist_draft()
        self.draw_task_markup()
        return reg.uuid

    def finish_region_and_next(self) -> dict | None:
        """Save the drawn area as the apex judgment, then advance."""
        if self.current is None:
            return None
        self.save_region(self._region_pts)
        self._region_pts = []
        self._region_mode = False
        return self.save_and_next()

    def confirm_draft_tip(self) -> dict | None:
        """Accept the draft point as precisely right, then advance."""
        if self.current is None:
            return None
        draft = self.current.get("draft_xy") or []
        if len(draft) != 2:
            raise ValueError("task carries no draft point to confirm")
        self.save_apex(float(draft[0]), float(draft[1]))
        return self.save_and_next()

    def finish_review_tip(self) -> dict | None:
        """Advance a re-marked review tip (an observation must exist)."""
        if self.current is None:
            return None
        uuid = str(self.current["uuid"])
        if self.store.load(f"obs-{uuid}") is None:
            raise ValueError("mark the tip (or an area) before moving on")
        return self.save_and_next()

    def confirm_review_crossing(self) -> dict | None:
        """Save edited lanes back to the draft crossing, still unresolved."""
        if self.current is None:
            return None
        xid = str(self.current.get("draft_crossing", ""))
        if not xid:
            raise ValueError("task carries no draft crossing")
        rec = self.store.load(xid)
        if rec is None:
            raise ValueError(f"draft crossing {xid} not found")
        good = {k: v for k, v in self._lanes.items() if len(v) >= 2}
        if len(good) < 2:
            raise ValueError("trace both tubes (>=2 dots each) first")
        data = dict(rec["data"])
        data["lanes"] = {k: [[px, py] for px, py in v]
                         for k, v in good.items()}
        data["unresolved"] = True
        data["adjudicated"] = True
        self._save_review("crossing", xid, data, actor=self.actor)
        return self.save_and_next()

    # Half-side of a single-click negative box (56px across ≈ a few
    # tube widths: covers a bend/rim/shaft with context, never precise).
    NEG_BOX_HALF = 28.0
    # Tip-confirm radius / box-exclusion ring around the known tip.
    NEG_TIP_R = 12.0
    NEG_RING_R = 30.0

    def neg_click(self, x: float, y: float) -> int:
        # ONE click finishes the task — no modes, no corner pairing:
        #   on the known tip -> confirm it, move on;
        #   in the ring around it -> refuse with guidance (a box here
        #     would swallow the tip and poison training);
        #   elsewhere -> save a fixed box centered here, move on.
        # Returns the number of saved boxes.
        if self.current is None:
            raise ValueError("no current task")
        if self._context_offset != 0:
            raise ValueError("return to the task frame to draw")
        known = self.current.get("known_tip_xy")
        task = self.current
        if known is not None:
            import math
            d = math.dist([float(x), float(y)],
                          [float(known[0]), float(known[1])])
            if d < self.NEG_TIP_R:
                self._neg_note = ""
                self.save_apex(float(x), float(y))
                self.save_and_next()
                return len(task.get("neg_boxes", []))
            if d < self.NEG_RING_R:
                # Too close: a box here would include the tip itself.
                self._neg_note = ("Too close to the tip — click the "
                                  "look-alike itself (further out), or "
                                  "the tip to confirm it.")
                return len(task.get("neg_boxes", []))
        self._neg_note = ""
        h = self.NEG_BOX_HALF
        self._save_neg_box(float(x) - h, float(y) - h,
                           float(x) + h, float(y) + h)
        n = len(task.get("neg_boxes", []))
        self.save_and_next()
        return n

    def draft_click(self, x: float, y: float) -> int:
        # Adjudicate ONE pre-drawn candidate box: a click inside an
        # unconfirmed draft confirms it as a verified negative and
        # advances. Anything else stays with guidance. Returns the
        # saved-box count.
        if self.current is None:
            raise ValueError("no current task")
        if self._context_offset != 0:
            raise ValueError("return to the task frame to draw")
        done = {tuple(float(v) for v in b)
                for b in self.current.get("neg_boxes", []) or []}
        task = self.current
        for b in self.current.get("draft_boxes", []) or []:
            x0, y0, x1, y1 = (float(v) for v in b)
            lo_x, hi_x = min(x0, x1), max(x0, x1)
            lo_y, hi_y = min(y0, y1), max(y0, y1)
            if not (lo_x <= float(x) <= hi_x and lo_y <= float(y) <= hi_y):
                continue
            if (x0, y0, x1, y1) in done:
                continue  # already confirmed (revisit)
            self._neg_note = ""
            self._save_neg_box(x0, y0, x1, y1)
            n = len(task.get("neg_boxes", []))
            self.save_and_next()
            return n
        self._neg_note = ("Click INSIDE a yellow box to confirm it, or "
                          "Nothing-box-worthy to move on.")
        return len(task.get("neg_boxes", []))

    @_task_frame_write
    def _save_neg_box(self, x0: float, y0: float,
                      x1: float, y1: float) -> int:
        # Persist one box as a SupervisionRegion entity plus a
        # task-level record. Returns the saved box count.
        if self.current is None:
            raise ValueError("no current task")
        boxes = list(self.current.get("neg_boxes", []))
        boxes.append([float(x0), float(y0), float(x1), float(y1)])
        self.current["neg_boxes"] = boxes
        self.store.save("task", str(self.current["uuid"]), self.current,
                        actor=self.actor)
        from tubetracker.annotation_schema import SupervisionRegion
        reg = SupervisionRegion(
            uuid=f"neg-{self.current.get('uuid')}-{len(boxes)}",
            movie_uuid=str(self.current.get("movie_uuid",
                                            self.current.get("movie", ""))),
            source_frame=int(self.current["query_frames"][0]),
            polygon_xy=[(float(x0), float(y0)), (float(x1), float(y0)),
                        (float(x1), float(y1)), (float(x0), float(y1))],
            kind="verified_negative",
            class_scope=str(self.current.get("neg_class", "apex")),
            confirmed=True,
        )
        errs = reg.validate()
        if errs:
            raise ValueError(f"invalid region: {errs}")
        self._save_review("region", reg.uuid, {
            "task_uuid": str(self.current.get("uuid", "")),
            "movie_uuid": reg.movie_uuid,
            "source_frame": reg.source_frame,
            "polygon_xy": [list(q) for q in reg.polygon_xy],
            "kind": reg.kind,
            "class_scope": reg.class_scope,
            "confirmed": True,
            "annotator": self.actor or "annotator",
        }, actor=self.actor)
        self._record_dot(("negbox", ""))
        return len(boxes)

    def census_click(self, x: float, y: float) -> dict:
        # Add one instance to an exhaustive tile (tip or ball mode).
        if self.current is None:
            raise ValueError("no current task")
        if self._context_offset != 0:
            raise ValueError("return to the task frame to draw")
        if self.current.get("analysis_census"):
            region = self.current["review_region"]
            x0, y0 = region[0]; x1, y1 = region[2]
            if not (x0 <= x < x1 and y0 <= y < y1):
                raise ValueError("Mark a grain whose centre is inside the yellow rectangle")
            instances = self.current.setdefault("census_instances", [])
            if any(math.dist(g["xy"], [x, y]) < 4 for g in instances):
                raise ValueError("A grain is already marked here; select its row to confirm or remove it")
            instances.append({"grain_id": f"{self.current['movie']}|review-{new_uuid()}",
                              "xy": [float(x), float(y)], "confirmed": True})
            self.current["census_complete"] = False
        key = "census_grains" if self._census_mode == "grain" else "census_tips"
        items = list(self.current.get(key, []))
        items.append([float(x), float(y)])
        self.current[key] = items
        self.store.save("task", str(self.current["uuid"]), self.current,
                        actor=self.actor)
        # Record WHICH list (tip vs ball): undo must pop the newest
        # dot placed, not always tips first.
        self._record_dot((
            "census_grain" if self._census_mode == "grain"
            else "census_tip", ""))
        return {"tip": len(self.current.get("census_tips", [])),
                "grain": len(self.current.get("census_grains", []))}

    @_task_frame_write
    def finish_census(self) -> dict | None:
        # Assert tile exhaustiveness (even zero = explicitly empty).
        if self.current is None:
            return None
        if self.current.get("analysis_census"):
            if any(not g.get("confirmed") for g in self.current.get("census_instances", [])):
                raise ValueError("Confirm or remove every proposed grain first")
            if not self.current.get("census_membership_resolved"):
                raise ValueError("Confirm that you checked the whole field and separated touching grains")
            self._confirm_review_task()
            self.current.update(census_complete=True, completed=True)
            self.store.save("task", self.current["uuid"], self.current, actor=self.actor)
            return self.current
        self._confirm_review_task()
        self.current["census_complete"] = True
        return self.save_and_next()

    @_task_frame_write
    def edit_census_grain(self, index, *, remove=False):
        if not self.current or not self.current.get("analysis_census"):
            raise ValueError("Open the field census first")
        items = self.current["census_instances"]
        if index < 0 or index >= len(items):
            raise ValueError("Select a grain row")
        if remove:
            self.current.setdefault("census_rejected", []).append(items.pop(index))
        else:
            items[index]["confirmed"] = True
        self.current["census_grains"] = [g["xy"] for g in items]
        self.current["census_complete"] = False
        self.store.save("task", self.current["uuid"], self.current, actor=self.actor)
        self.draw_task_markup()

    @_task_frame_write
    def record_census_ruling(self, index, classification, rule, frames):
        if classification not in ("germinated", "nongerminating", "unresolved") or not rule.strip():
            raise ValueError("A biological ruling requires an explicit class and rule")
        if not frames or any(int(f) != f or f < 0 for f in frames):
            raise ValueError("List the nonnegative source frames actually reviewed")
        if classification == "nongerminating" and len(set(frames)) < 2:
            raise ValueError("One no-tube frame cannot establish nongermination over an interval")
        items = (self.current or {}).get("census_instances", [])
        if index < 0 or index >= len(items) or not items[index].get("confirmed"):
            raise ValueError("Confirm the physical grain before classifying it")
        items[index]["classification"] = {"class": classification, "rule": rule.strip(),
                                            "reviewed_frames": sorted(set(frames))}
        self.store.save("task", self.current["uuid"], self.current, actor=self.actor)

    @_task_frame_write
    def finish_neg(self) -> dict | None:
        # Close a negative-region task: needs >= 1 drawn box, because
        # an empty review must not become supervision (unknown stays
        # unknown). Boxes are already saved as SupervisionRegions.
        if self.current is None:
            return None
        if not self.current.get("neg_boxes"):
            raise ValueError("draw at least one box first")
        return self.save_and_next()

    @_task_frame_write
    def finish_neg_empty(self) -> dict | None:
        # Explicit "nothing box-worthy in this view": completes queue
        # hygiene WITHOUT creating supervision. A forced box would be
        # junk supervision — worse than an honest empty verdict.
        if self.current is None:
            return None
        self.current["reviewed_empty"] = True
        return self.save_and_next()

    # ---- rev6: tube-body mask painting ----
    # Points are native brush-stamp centers + brush size; the loader
    # rasterizes the union of discs. Crop-independent, JSON-safe, no
    # codec. The napari Labels layer is the brush; these methods are
    # the headless-testable commit path it calls.

    def _current_review_region(self) -> list:
        """The extent the annotator is actually looking at (rev8).

        A 'complete' mask may claim background only inside the extent
        that was reviewed. napari 0.9.1 has no `camera.rect`, so the
        extent is computed from camera center + zoom and the canvas
        size; without a live canvas it falls back to a region the task
        declares, then to nothing. Never invented.
        """
        if (self.current or {}).get("review_region_fixed"):
            return copy.deepcopy(self.current.get("review_region", []))
        try:
            if self.viewer is not None:
                cam = self.viewer.camera
                # rev8 bug (measured): napari 0.9.1 hands back a
                # 3-COMPONENT camera centre (z, y, x) here, so reading
                # [0] and [1] as (y, x) recorded z (always 0) as the y
                # centre and the real y as the x centre. Every extent
                # saved before this fix has y symmetric about 0 in all
                # three projects, which is the signature: the reviewed
                # region then fell outside the mask's crop, so
                # 'complete' masks silently degraded to band-only
                # validity. Take the LAST two components instead --
                # correct for both the 2- and 3-component cases.
                ctr = [float(q) for q in cam.center]
                if len(ctr) < 2:
                    return []
                cy, cx = ctr[-2], ctr[-1]
                zoom = float(cam.zoom)
                cw, ch = self._canvas_size()
                if zoom > 0 and cw > 0 and ch > 0:
                    hw, hh = cw / (2.0 * zoom), ch / (2.0 * zoom)
                    return [[cx - hw, cy - hh], [cx + hw, cy + hh]]
        except Exception:
            pass
        declared = (self.current or {}).get("review_region")
        if declared:
            return [[float(q[0]), float(q[1])] for q in declared]
        return []

    def _review_region_provenance(self, region):
        """Record enough geometry to verify future background licences."""
        if not region:
            return None
        task = self.current or {}
        common = {'schema': REVIEW_GEOMETRY_SCHEMA,
                  'movie': task.get('movie') or task.get('movie_uuid'),
                  'source_frame': (task.get('query_frames') or [None])[0]}
        if task.get('review_region_fixed'):
            capture = dict(common, kind='declared_field', region_xy=copy.deepcopy(region))
        elif self.viewer is not None:
            try:
                centre = [float(v) for v in self.viewer.camera.center]
                cw, ch = self._canvas_size()
                capture = dict(common, kind='native_canvas', canvas_size_wh=[cw,ch],
                    camera_center_xy=[centre[-1],centre[-2]], camera_zoom=float(self.viewer.camera.zoom))
            except (AttributeError, IndexError, TypeError, ValueError):
                return None
        else:
            return None
        _licensed, policy = licensed_review_region(region, capture)
        return capture if policy in ('declared_field', 'native_canvas') else None

    def _validated_review_region(self, paint_xy=None, center_xy=None,
                                 radius: float | None = None
                                 ) -> tuple[list, str]:
        """The reviewed extent, but only if it may licence supervision.

        rev9 WP-A.1 / H318: a stale app process once stored extents
        whose y pair was symmetric about 0, which the loader then
        rasterised into crops as reviewed background. Fail closed here:
        an extent that is unreadable, empty, or does not overlap the
        object under review (the paint, or the ball for `no tube`
        verdicts) is dropped and the reason recorded instead.
        """
        region = self._current_review_region()
        if not region:
            return [], "missing"
        if paint_xy:
            bbox = paint_bbox_xy(paint_xy)
        else:
            bbox = disc_bbox_xy(center_xy, radius or 14.0)
        if bbox is None:
            return [], "no-scope"
        ok, reason = review_region_usable(region, bbox)
        if not ok:
            return [], reason
        return region, "ok"

    @_task_frame_write
    def mask_paint(self, pts: list) -> int:
        """Add brush-stamp centers (native [x, y]) to the mask draft."""
        if self.current is None:
            raise ValueError("no current task")
        acc = self.current.setdefault("mask_points", [])
        for q in pts:
            acc.append([float(q[0]), float(q[1])])
        return len(acc)

    @_task_frame_write
    def mask_erase_at(self, x: float, y: float, radius_px: float) -> int:
        """Remove stamps within radius_px of (x, y). Returns survivors."""
        if self.current is None:
            raise ValueError("no current task")
        acc = self.current.get("mask_points", []) or []
        acc = [q for q in acc
               if (q[0] - float(x)) ** 2 + (q[1] - float(y)) ** 2
               > float(radius_px) ** 2]
        self.current["mask_points"] = acc
        return len(acc)

    def mask_coverage(self) -> int:
        if self.current is None:
            return 0
        return len(self.current.get("mask_points", []) or [])

    @_task_frame_write
    def save_mask(self, complete: bool) -> str:
        """Commit the painted mask as a mask entity (transactional).

        complete=True asserts the whole view was checked: background
        outside the paint supervises as non-tube. False leaves distant
        background unknown. Empty paint never supervises (refuses).
        Returns the mask uuid.
        """
        if self.current is None:
            raise ValueError("no current task")
        pts = self.current.get("mask_points", []) or []
        if not pts:
            raise ValueError("paint the tube body first (or Can't tell)")
        uuid = f"mask-{self.current.get('uuid', new_uuid())}"
        review_region, region_note = self._validated_review_region(pts)
        self.current["review_region"] = review_region
        self.current["review_region_note"] = region_note
        self.current['review_region_provenance'] = self._review_region_provenance(review_region)
        self._save_review("mask", uuid, {
            "task_uuid": str(self.current.get("uuid", "")),
            "movie_uuid": str(self.current.get("movie", "")),
            "source_frame": int(self.current["query_frames"][0]),
            "painted_xy": [[float(q[0]), float(q[1])] for q in pts],
            "brush_px": float(self.current.get("brush_px", 9.0)),
            "complete": bool(complete),
            # rev7: exact raster when the UI captured one (pixel-exact;
            # holes stay holes) + explicit source-tube link. Legacy
            # stamp-only records carry neither; the loader flags them.
            "mask_raster": self.current.get("mask_raster"),
            "mask_unknown_raster": self.current.get("mask_unknown_raster"),
            "review_origin": self.current.get("review_origin", "human"),
            "annotation_role": self.current.get("annotation_role", "training"),
            "source_obs_uuid": str(
                self.current.get("source_obs_uuid", "")),
            # rev8: the reviewed extent licenses (or withholds) a
            # background claim; identity travels with the mask.
            "review_region": review_region,
            "review_region_note": region_note,
            'review_region_provenance': self.current['review_region_provenance'],
            "owner_uuid": str(self.current.get("owner_uuid", "") or ""),
            "tube_uuid": str(self.current.get("tube_uuid", "") or ""),
            # rev8: the exact object this mask is about. The query
            # point must travel with the mask (deployment prompts with
            # the detected grain disc; the task is not always present).
            "target_xy": ([float(q) for q in (
                self.current.get("target_xy")
                or self.current.get("focus_xy") or [])]
                if (self.current.get("target_xy")
                    or self.current.get("focus_xy")) else []),
            "target_r": float(self.current.get("target_r", 14.0)),
            "annotator": self.actor or "annotator",
        }, actor=self.actor)
        # persist the extent on the task too (survives mask-link loss)
        self.store.save("task", str(self.current.get("uuid", "")),
                        self.current, actor=self.actor)
        return uuid

    @_task_frame_write
    def finish_mask(self, complete: bool) -> dict | None:
        self.save_mask(complete)
        return self.save_and_next()

    @_task_frame_write
    def save_mask_none(self) -> str:
        """Ball-scoped "no tube at this grain" verdict (rev8).

        H242: a scoped "can't tell" is NEVER a full-frame negative.
        This records an empty mask with no_tube=True and the reviewed
        extent; the snapshot turns it into a CAP-scoped negative disc
        at the queried grain — the grain is the scope, nothing more.
        """
        if self.current is None:
            raise ValueError("no current task")
        uuid = f"mask-{self.current.get('uuid', new_uuid())}"
        tgt = self.current.get("target_xy") or self.current.get("focus_xy")
        review_region, region_note = self._validated_review_region(
            center_xy=tgt,
            radius=float(self.current.get("target_r", 14.0) or 14.0))
        self.current["review_region"] = review_region
        self.current["review_region_note"] = region_note
        self.current['review_region_provenance'] = self._review_region_provenance(review_region)
        self.current["no_tube"] = True
        tgt = self.current.get("target_xy") or self.current.get("focus_xy")
        self._save_review("mask", uuid, {
            "task_uuid": str(self.current.get("uuid", "")),
            "movie_uuid": str(self.current.get("movie", "")),
            "source_frame": int(self.current["query_frames"][0]),
            "painted_xy": [],
            "brush_px": float(self.current.get("brush_px", 9.0)),
            "complete": False,
            "mask_raster": None,
            "source_obs_uuid": str(
                self.current.get("source_obs_uuid", "")),
            "review_region": review_region,
            'review_region_provenance': self.current['review_region_provenance'],
            "owner_uuid": str(self.current.get("owner_uuid", "") or ""),
            "tube_uuid": str(self.current.get("tube_uuid", "") or ""),
            "no_tube": True,
            "scope": "ball",
            "target_xy": ([float(tgt[0]), float(tgt[1])]
                          if tgt and len(tgt) == 2 else []),
            "target_r": float(self.current.get("target_r", 14.0)),
            "annotator": self.actor or "annotator",
        }, actor=self.actor)
        self.store.save("task", str(self.current.get("uuid", "")),
                        self.current, actor=self.actor)
        return uuid

    # ---- rev6: route-duel judgment ----
    # Two candidate curves (A/B) + click-near-to-vote. A vote advances;
    # Neither/Can't-tell stay honest (neither IS supervision: both bad).

    def duel_vote(self, x: float, y: float) -> str:
        """Vote for the nearer lane within 25px. Returns 'A'/'B'."""
        if self.current is None:
            raise ValueError("no current task")
        lanes = self.current.get("draft_lanes", {}) or {}
        best, bestd = None, 1e18
        for lab, pts in lanes.items():
            if not pts or len(pts) < 2:
                continue
            d = min(math.hypot(q[0] - float(x), q[1] - float(y))
                    for q in pts)
            if d < bestd:
                best, bestd = lab, d
        if best is None or bestd > 25.0:
            raise ValueError("click near one of the two curves")
        return best

    @_task_frame_write
    def save_duel(self, winner: str) -> str:
        """Commit a duel verdict (winner 'A'/'B'/'neither')."""
        if self.current is None:
            raise ValueError("no current task")
        if winner not in ("A", "B", "neither"):
            raise ValueError("winner must be A, B, or neither")
        uuid = f"duel-{self.current.get('uuid', new_uuid())}"
        lanes = self.current.get("draft_lanes", {}) or {}
        truth = self.current.get("lane_truth", {}) or {}
        self._save_review("duel", uuid, {
            "task_uuid": str(self.current.get("uuid", "")),
            "movie_uuid": str(self.current.get("movie", "")),
            "source_frame": int(self.current["query_frames"][0]),
            "winner": winner,
            "lane_a": [[float(q[0]), float(q[1])]
                       for q in lanes.get("A", [])],
            "lane_b": [[float(q[0]), float(q[1])]
                       for q in lanes.get("B", [])],
            "truth_a": str(truth.get("A", "")),
            "truth_b": str(truth.get("B", "")),
            "annotator": self.actor or "annotator",
        }, actor=self.actor)
        return uuid

    @_task_frame_write
    def finish_duel(self, winner: str) -> dict | None:
        self.save_duel(winner)
        return self.save_and_next()

    def _clear_mask_layers(self) -> None:
        """Remove mask paint/guide layers (never leak across tasks)."""
        if self.viewer is None:
            return
        try:
            for ly in list(self.viewer.layers):
                if ly.name.startswith("mask-"):
                    self.viewer.layers.remove(ly)
        except Exception:
            pass

    def _show_mask_task(self) -> None:
        """Body-mask task overlays: yellow guide path + paintable Labels.

        Headless-safe no-op. The Labels layer is napari's native brush;
        commit reads its nonzero pixels back to native points.
        """
        if self.viewer is None or self.current is None:
            return
        self._clear_mask_layers()
        try:
            if str(self.current.get("task_type", "")) != "body_mask":
                return
            guide = self.current.get("guide_path", []) or []
            if len(guide) >= 2:
                self.viewer.add_shapes(
                    [[[q[1], q[0]] for q in guide]], shape_type="path",
                    name="mask-guide", edge_color="yellow", edge_width=3)
                try:
                    self.viewer.layers["mask-guide"].editable = False
                except Exception:
                    pass
            import numpy as _np
            # rev8: mark the exact object this task is about. A mask
            # query must say WHICH grain's tube to paint — and for a
            # paired crop (two neighbouring grains) the target cannot
            # be inferred from the guide, because the second grain has
            # no traced path. Magenta ring, non-editable.
            tgt = self.current.get("target_xy") \
                or self.current.get("focus_xy")
            if tgt and len(tgt) == 2:
                try:
                    import math as _m
                    tr = float(self.current.get("target_r", 14.0))
                    tx, ty = float(tgt[0]), float(tgt[1])
                    # A closed vertex RING, not a 2-corner ellipse:
                    # napari's ellipse vertex convention varies by
                    # version and a mis-read bbox covers the view.
                    ring = [[ty + tr * _m.sin(2 * _m.pi * i / 32),
                             tx + tr * _m.cos(2 * _m.pi * i / 32)]
                            for i in range(32)]
                    self.viewer.add_shapes(
                        [ring], shape_type="path", name="mask-target",
                        edge_color="magenta", edge_width=3,
                        blending="translucent_no_depth")
                    self.viewer.layers[
                        "mask-target"].editable = False
                except Exception:
                    pass
            shape = self._image_layer.data.shape[:2] \
                if self._image_layer is not None else None
            if shape is None:
                return
            from napari.utils.colormaps import DirectLabelColormap
            lab = self.viewer.add_labels(
                _np.zeros(shape, dtype=_np.uint8), name="mask-paint",
                metadata={"task_uuid": self.current["uuid"]},
                colormap=DirectLabelColormap(color_dict={
                    0: (0, 0, 0, 0), 1: (0, 1, .3, 1),
                    2: (1, .65, 0, 1), None: (0, 0, 0, 0)}),
                blending="translucent_no_depth")
            try:
                lab.brush_size = int(
                    self.current.get("brush_px", 9.0))
            except Exception:
                pass
            # rev8: restore already-committed paint so reopening a mask
            # task for correction is one click, not a re-paint (rev7:
            # saved masks could not be revisited).
            try:
                ras = self.current.get("mask_draft_raster", self.current.get("mask_raster"))
                if ras:
                    from prototypes.v30_video_apex.targets import (
                        decode_mask_raster)
                    prev = decode_mask_raster(
                        int(shape[0]), int(shape[1]), (0.0, 0.0), ras)
                    if prev.any():
                        lab.data = (_np.asarray(lab.data).astype(
                            _np.uint8) | prev.astype(_np.uint8))
                unknown_raster = self.current.get("mask_unknown_draft_raster", self.current.get("mask_unknown_raster"))
                if unknown_raster:
                    from prototypes.v30_video_apex.targets import decode_mask_raster
                    unknown = decode_mask_raster(int(shape[0]), int(shape[1]), (0., 0.), unknown_raster)
                    lab.data[unknown] = 2
            except Exception as error:
                raise ValueError(f'Could not restore saved mask: {error}') from error
            lab.selected_label = 1
            lab.mode = "paint"
            self.viewer.layers.selection.active = lab
            self._watch_mask_draft(lab)
        except Exception as error:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Could not initialize mask editor: {error}") from error

    def _watch_mask_draft(self, layer):
        # Native strokes and undo modify Labels data in place. They emit
        # labels_update after the pixels change, not the data-assignment event.
        layer.events.data.connect(lambda _event: self.persist_mask_draft())
        layer.events.labels_update.connect(lambda _event: self.persist_mask_draft())
        # Undo/redo reload the slice after replaying array edits. Cover both
        # synchronous and asynchronous napari slice settings.
        layer.events.set_data.connect(lambda _event: self.persist_mask_draft())
        layer.events.reload.connect(lambda _event: self.persist_mask_draft())

    def _capture_mask_draft(self):
        if self.viewer is None or not self.current or self.current.get("task_type") != "body_mask" or self._context_offset:
            return False
        try:
            layer = self.viewer.layers["mask-paint"]
        except (KeyError, TypeError):
            return False
        if getattr(layer, 'metadata', {}).get('task_uuid') != self.current['uuid']:
            return False
        from prototypes.v30_video_apex.targets import encode_mask_raster
        import numpy as np
        labels = np.asarray(layer.data)
        painted, unknown = encode_mask_raster(labels == 1), encode_mask_raster(labels == 2)
        if (painted == self.current.get("mask_draft_raster") and
                unknown == self.current.get("mask_unknown_draft_raster")):
            return False
        self.current['completed'] = False
        self.current["mask_draft_raster"], self.current["mask_unknown_draft_raster"] = painted, unknown
        return True

    def persist_mask_draft(self):
        if self._capture_mask_draft():
            self.persist_draft()

    @_task_frame_write
    def commit_mask_labels(self, complete: bool) -> str:
        """Read the Labels paint layer into mask stamps and save."""
        if self.viewer is None:
            raise ValueError("no viewer: use mask_paint headless")
        if self.current is None:
            raise ValueError("no current task")
        try:
            lab = self.viewer.layers["mask-paint"]
            import numpy as _np
            paint = (_np.asarray(lab.data) == 1)
            unknown = (_np.asarray(lab.data) == 2)
            ys, xs = _np.nonzero(paint)
        except Exception as e:
            raise ValueError(f"no paint layer ({e})")
        # Stride-3 subsample: a filled tube is thousands of pixels;
        # stamps every 3px fully cover a 9px brush. Stamps stay for
        # back-compat, but the EXACT raster is what supervises now
        # (rev7: stamp redraw expanded paint 364px -> 766px and
        # refilled erased holes).
        pts = [[float(x), float(y)]
               for y, x in zip(ys[::3].tolist(), xs[::3].tolist())]
        # The editable raster is authoritative; erased paint must not survive as stale stamps.
        self.current["mask_points"] = pts
        from prototypes.v30_video_apex.targets import encode_mask_raster
        self.current["mask_raster"] = encode_mask_raster(paint)
        self.current["mask_unknown_raster"] = encode_mask_raster(unknown)
        self.current["mask_draft_raster"] = self.current["mask_raster"]
        self.current["mask_unknown_draft_raster"] = self.current["mask_unknown_raster"]
        return self.save_mask(complete)

    def draw_task_markup(self) -> None:
        """Draw saved neg boxes / census dots for the current task.

        Viewer-only feedback so every click is visible immediately.
        No-op headless. Never steals clicks (layers are uneditable).
        Each layer is guarded separately and failures are logged to a
        file — a silent catch-all here once hid every overlay.
        """
        if self.viewer is None:
            return
        try:
            t = self.current or {}
            ttype = str(t.get("task_type", ""))
            boxes = []
            if ttype in ("neg_region", "neg_draft"):
                for b in t.get("neg_boxes", []) or []:
                    x0, y0, x1, y1 = (float(v) for v in b)
                    boxes.append([[y0, x0], [y0, x1],
                                  [y1, x1], [y1, x0]])
                boxes.extend([[q[1], q[0]] for q in poly["polygon_xy"]]
                             for poly in t.get("negative_polygons", []))
            drafts = []
            if ttype == "neg_draft":
                done = {tuple(float(v) for v in b)
                        for b in t.get("neg_boxes", []) or []}
                for b in t.get("draft_boxes", []) or []:
                    x0, y0, x1, y1 = (float(v) for v in b)
                    if (x0, y0, x1, y1) in done:
                        continue
                    drafts.append([[y0, x0], [y0, x1],
                                   [y1, x1], [y1, x0]])
            pend = []
            if ttype == "neg_region" and self._box_pending:
                x, y = self._box_pending[0]
                pend = [[y, x]]
            known_tip = []
            if ttype == "neg_region" and t.get("known_tip_xy"):
                kx, ky = (float(v) for v in t["known_tip_xy"])
                known_tip = [[ky, kx]]
            tips, balls = [], []
            tile = []
            grain_ring = []
            # Highlight the grain under review (the "ball"): a ring at the
            # task's target location, for every task type that carries one.
            # Camera framing alone leaves Josh asking which ball is his.
            if ttype in ("germination_event", "body_mask", "review_tip",
                         "centerline", "apex", "grain_identity"):
                g = t.get("target_xy") or t.get("focus_xy")
                try:
                    gx, gy = float(g[0]), float(g[1])
                    gr = max(1.0, float(t.get("target_r", 13.0)))
                    import math as _math
                    grain_ring = [[[gy + gr * _math.sin(aa),
                                    gx + gr * _math.cos(aa)]
                                   for aa in [i * _math.pi / 16 for i in range(32)]]]
                except (TypeError, ValueError, IndexError):
                    grain_ring = []
            if ttype == "census":
                tips = [[q[1], q[0]]
                        for q in t.get("census_tips", []) or []]
                balls = [[q[1], q[0]]
                         for q in t.get("census_grains", []) or []]
                # Exact reviewed extent; old tasks retain their historical viewport fallback.
                fx, fy = (t.get("focus_xy") or (None, None))
                if fx is not None and fy is not None:
                    x0, y0 = float(fx) - 160.0, float(fy) - 160.0
                    x1, y1 = float(fx) + 160.0, float(fy) + 160.0
                    tile = [[[y0, x0], [y0, x1], [y1, x1], [y1, x0]]]
                if t.get("review_region"):
                    tile = [[[q[1], q[0]] for q in t["review_region"]]]
            # MULTI-GRAIN windows: circle the competing grains too -- the
            # identity question is only decidable when the neighbors are
            # named as candidates (Josh: "nothing is outlined/highlighted").
            mg = []
            mg_src = (t.get("multigrain_window") or {}).get("grains", []) or []
            tgt = t.get("target_xy") or t.get("focus_xy")
            for g in mg_src:
                try:
                    gx, gy = float(g["xy"][0]), float(g["xy"][1])
                    gr = max(8.0, float(g.get("r") or 13.0))
                    if tgt is not None and (gx - float(tgt[0])) ** 2 \
                            + (gy - float(tgt[1])) ** 2 <= 36:
                        continue   # that is the target ring, not a rival
                    import math as _math
                    mg.append([[gy + gr * _math.sin(aa),
                                gx + gr * _math.cos(aa)]
                               for aa in [i * _math.pi / 16 for i in range(32)]])
                except (TypeError, ValueError, KeyError, IndexError):
                    continue
        except Exception:
            return
        for name, kind, data, show, color in (
                ("neg-boxes", "shapes", boxes, bool(boxes), "red"),
                ("neg-drafts", "shapes", drafts, bool(drafts), "yellow"),
                ("neg-pending", "points", pend, bool(pend), "yellow"),
                ("known-tip", "points", known_tip, bool(known_tip),
                 "white"),
                ("census-tips", "points", tips, bool(tips), "cyan"),
                ("census-balls", "points", balls, bool(balls),
                 "magenta"),
                ("census-tile", "shapes", tile, bool(tile), "yellow"),
                ("mg-competitors", "shapes", mg, bool(mg), "yellow"),
                ("target-grain", "shapes", grain_ring, bool(grain_ring),
                 "lime")):
            try:
                self._set_markup_layer(name, kind, data, show,
                                       edge_color=color)
            except Exception as e:
                try:
                    with open("/tmp/annotator_markup_errors.log",
                              "a") as f:
                        f.write(f"{name}: {type(e).__name__}: {e}\n")
                except Exception:
                    pass

    def _set_markup_layer(self, name: str, kind: str, data: list,
                          show: bool, edge_color: str = "red") -> None:
        layers = self.viewer.layers
        lay = None
        try:
            lay = layers[name]
        except KeyError:
            pass
        if not show:
            if lay is not None:
                try:
                    lay.visible = False
                except Exception:
                    pass
            return
        if lay is None:
            if kind == "shapes":
                lay = self.viewer.add_shapes(
                    data, shape_type="polygon", name=name,
                    edge_color=edge_color, face_color="transparent",
                    edge_width=2, blending="translucent_no_depth")
            else:
                lay = self.viewer.add_points(
                    data, name=name, size=9, face_color="transparent",
                    border_color=edge_color, blending="translucent_no_depth")
            try:
                lay.editable = False  # never steal clicks (H237)
            except Exception:
                pass
        else:
            try:
                lay.data = data
                lay.visible = True
            except Exception:
                pass

    @_task_frame_write
    def mint_tube_id(self, lane: str | None = None) -> dict:
        # Mint stable tube IDs for crossing lanes (idempotent).
        # With no lane: mints for every traced lane missing one. IDs are
        # stable manual identities; lane-to-exit mapping comes later.
        if self.current is None:
            raise ValueError("no current task")
        ids = dict(self.current.get("tube_ids", {}))
        lanes = [lane] if lane else sorted(self._lanes)
        for lab in lanes:
            lab = str(lab).upper()[:1]
            if lab not in ids:
                ids[lab] = new_uuid()[:8]
        self.current["tube_ids"] = ids
        self.store.save("task", str(self.current["uuid"]), self.current,
                        actor=self.actor)
        return dict(ids)

    @_task_frame_write
    @_draft_write
    def undo_last_dot(self) -> bool:
        """Remove the newest pile/crossing/trace dot. Returns False if none."""
        while self._dot_history:
            kind, lab = self._dot_history.pop()
            if kind == "grain":
                g = self._grains.get(lab)
                if g is None or not g.get("xy"):
                    continue  # stale entry, keep popping
                g.pop("xy", None)
                if not g:
                    self._grains.pop(lab, None)
                self._lane = lab
                self._mark_mode = "grain"
                self._persist_grains()
                return True
            if kind == "root":
                g = self._grains.get(lab)
                if g is None or not g.get("root_xy"):
                    continue
                g.pop("root_xy", None)
                if not g:
                    self._grains.pop(lab, None)
                self._lane = lab
                self._mark_mode = "root"
                self._persist_grains()
                return True
            if kind == "lane":
                pts = self._lanes.get(lab)
                if not pts:
                    continue
                pts.pop()
                self._lane = lab
                return True
            if kind == "path":
                if not self._path_pts:
                    continue
                self._path_pts.pop()
                self._sync_path_layer()
                return True
            if kind == "region":
                if not self._region_pts:
                    continue
                self._region_pts.pop()
                return True
            if kind == "negbox":
                boxes = self.current.get("neg_boxes", []) if self.current else []
                if not boxes:
                    continue
                boxes.pop()
                self.current["neg_boxes"] = boxes
                self.store.save("task", str(self.current["uuid"]), self.current,
                                actor=self.actor)
                try:
                    self.store.delete("region",
                                      f"neg-{self.current.get('uuid')}-{len(boxes) + 1}",
                                      actor=self.actor)
                except Exception:
                    pass
                return True
            if kind in ("census", "census_tip", "census_grain"):
                if not self.current:
                    continue
                # LIFO across both lists: pop the newest dot of the
                # recorded kind (tips-first always was the bug — ball
                # dots were uneatable while any tip existed).
                keys = (("census_tips", "census_grains")
                        if kind == "census" else
                        ("census_tips",) if kind == "census_tip" else
                        ("census_grains",))
                popped = False
                for _k in keys:
                    if self.current.get(_k):
                        self.current[_k].pop()
                        popped = True
                        break
                if popped:
                    if _k == "census_grains" and self.current.get("analysis_census"):
                        self.current["census_instances"].pop()
                        self.current["census_complete"] = False
                    self.store.save("task", str(self.current["uuid"]),
                                    self.current, actor=self.actor)
                    return True
                continue
        # Revisited pile (Back button) has persisted dots but no
        # history: fall back to the highest-numbered unfinished ball.
        for lab in reversed(self.BALL_LANES):
            g = self._grains.get(lab)
            if g is None:
                continue
            if g.get("root_xy"):
                g.pop("root_xy", None)
                self._lane = lab
                self._mark_mode = "root"
                self._persist_grains()
                return True
            if g.get("xy"):
                g.pop("xy", None)
                if not g:
                    self._grains.pop(lab, None)
                self._lane = lab
                self._mark_mode = "grain"
                self._persist_grains()
                return True
        for lab in reversed(self.BALL_LANES):
            pts = self._lanes.get(lab)
            if pts:
                pts.pop()
                self._lane = lab
                return True
        # Persisted census dots with no history (app restart orphans
        # in-DB dots from _dot_history): pop the newest of either
        # list so undo keeps working across restarts. Order across
        # kinds is approximate here — exact LIFO needs live history.
        if self.current:
            for _k in ("census_tips", "census_grains"):
                if self.current.get(_k):
                    self.current[_k].pop()
                    self.store.save("task", str(self.current["uuid"]),
                                    self.current, actor=self.actor)
                    return True
        if self._path_pts:
            self._path_pts.pop()
            self._sync_path_layer()
            return True
        return False

    @_task_frame_write
    @_draft_write
    def clear_task_marks(self) -> bool:
        """Remove ALL dots I put on the current task (start it over).

        Clears in-progress + persisted dots (trace points, lanes,
        pile grains, tile tips/balls, region drafts) and the dot
        history. Saved observation entities stay in revision history;
        the task stays open. Returns False when there was nothing.
        """
        if self.current is None:
            return False
        had = bool(self._path_pts or self._lanes or self._grains or
                   self._region_pts or self._box_pending or
                   self._dot_history or
                   self.current.get("census_tips") or
                   self.current.get("census_grains"))
        self._path_pts = []
        self._lanes = {}
        self._lane = "A"
        self._grains = {}
        self._mark_mode = "grain"
        self._dot_history = []
        self._region_pts = []
        self._box_pending = []
        self.current["census_tips"] = []
        self.current["census_grains"] = []
        self.current["_dot_history"] = []
        try:
            self.store.save("task", str(self.current["uuid"]),
                            self.current, actor=self.actor)
        except Exception:
            pass
        self._sync_path_layer()
        return had

    def undo_mark(self) -> bool:
        """Take back one dot (pile/crossing/trace) or reopen the task.
        On pile/crossing/trace tasks each press removes the newest dot
        placed (ball dot, tube-start dot, crossing dot, trace dot) and
        stays on the task. On single-click tasks it deletes the saved
        mark and reopens the task, as before. Returns False when there
        is nothing to undo. Revision history retains everything.
        """
        if self.current is None:
            return False
        if self.undo_last_dot():
            try:
                self.current["_dot_history"] = [
                    [k, l] for k, l in self._dot_history]
                self.store.save("task", str(self.current["uuid"]),
                                self.current, actor=self.actor)
            except Exception:
                pass
            if self.viewer is not None:
                try:
                    self._show_current()
                except Exception:
                    pass
            return True
        uuid = str(self.current["uuid"])
        removed = False
        for candidate in (f"obs-{uuid}", f"obs-{uuid}-0"):
            if self.store.delete("observation", candidate,
                                 actor=self.actor):
                removed = True
        if not removed:
            return False
        self.current["completed"] = False
        self.current.pop("marks", None)
        self.store.save("task", uuid, self.current, actor=self.actor)
        if self.viewer is not None:
            try:
                if self._apex_layer is not None:
                    self._apex_layer.data = []
            except Exception:
                pass
            self._show_current()
        return True

    def undo_saved_correction(self) -> str:
        """Undo the most recent SAVED correction for the current selection.

        Walks the explicit logical undo trail: the state the undone save
        replaced is restored as a NEW audited revision; when the trail is
        exhausted the saved correction is removed with an audited history
        entry (the store keeps every revision). Never touches draft points.
        Raises ValueError when nothing is saved.
        """
        if self.current is None:
            raise ValueError("no current task")
        uuid = str(self.current.get("uuid", ""))
        candidates = ([('grain_identity', f'grain-identity-{uuid}')]
                      if self.current.get('task_type') == 'grain_identity' else
                      [('observation', f'obs-{uuid}'), ('observation', f'obs-{uuid}-0')])
        for kind, candidate in candidates:
            saved = self.store.load(candidate)
            if saved is None or saved.get("kind") != kind:
                continue
            data = dict(saved.get("data") or {})
            trail = list(data.get("undo_trail") or [])
            revision = int(saved.get("revision", 1))
            if trail:
                restored = dict(trail[-1])
                restored["undo_trail"] = trail[:-1]
                restored["undo_restored_from"] = revision
                self.store.save(kind, candidate, restored,
                                actor=f"{self.actor} (undo saved correction)")
                note = (f"Restored the previous accepted state as a new "
                        f"audited revision (the undone save was revision "
                        f"{revision}).")
            else:
                self.store.delete(kind, candidate,
                                  actor=f"{self.actor} (undo saved correction)")
                note = ("Removed the saved correction; its revision history "
                        "remains in the store.")
            self.current["completed"] = False
            if kind == 'grain_identity':
                point = restored.get('grain_native') if trail else None
                self._region_pts = [list(point)] if point else []
                self._dot_history = [('region', '')] if point else []
                draft = self.current.setdefault('_drawing_draft', {})
                draft['region_xy'] = copy.deepcopy(self._region_pts)
                self.current['_dot_history'] = [list(x) for x in self._dot_history]
            self.store.save("task", uuid, self.current, actor=self.actor)
            if self.viewer is not None:
                try:
                    self._show_current()
                except Exception:
                    pass
            if callable(getattr(self, "on_draft_changed", None)):
                try:
                    self.on_draft_changed()
                except Exception:
                    pass
            return note
        raise ValueError(
            "There is no saved correction to undo for this selection.")

    def toggle_zoom(self) -> None:
        """Full-field view <-> zoomed ball (P0A second-view slice)."""
        if self.viewer is None or self.current is None:
            return
        try:
            if float(self.viewer.camera.zoom) > 2.0:
                self.viewer.camera.zoom = 1.0
            else:
                focus = self.current.get("focus_xy")
                if focus is not None:
                    self.viewer.camera.center = (float(focus[1]),
                                                 float(focus[0]))
                self.viewer.camera.zoom = float(
                    self.current.get("view_zoom", 5.0) or 5.0)
        except Exception:
            pass

    @_task_frame_write
    def skip_unresolvable(self, reason: str = "") -> dict | None:
        """Skip an impossible clump/crossing WITHOUT labeling it.

        Marks the task unresolvable and completes it with NO observation
        entity. Fold/export/audit must exclude unresolvable tasks: no
        forced guess may ever reach training data.
        """
        if self.current is None:
            return None
        self.current["resolution"] = "unresolvable"
        if reason:
            self.current["resolution_reason"] = reason
        return self.save_and_next()

    @_task_frame_write
    def save_not_a_ball(self, reason: str = "") -> dict | None:
        """The premise was wrong: this isn't a ball (debris, artifact).

        Completes the task with NO observation and a not_a_ball tag.
        Like unresolvable it is fenced off from training entirely: the
        reviewer only saw the zoomed crop, so promoting it to a
        full-frame "no grains" negative would invent background
        negatives (P0 audit). The tag keeps the judgment queryable for
        detector-precision analytics and P2 scoped negatives.
        """
        if self.current is None:
            return None
        self.current["resolution"] = "not_a_ball"
        if reason:
            self.current["resolution_reason"] = reason
        return self.save_and_next()

    def progress(self) -> tuple[int, int]:
        """(balls done, balls total) tally for the top of the panel."""
        total = len(self.tasks)
        done = sum(1 for t in self.tasks if t.get("completed"))
        return done, total

    def instruction(self) -> str:
        if self.current is None:
            return "Queue complete."
        return self.current.get("why") or TASK_INSTRUCTIONS.get(
            str(self.current.get("task_type", "apex")), TASK_INSTRUCTIONS["apex"])


class TaskDock(QDockWidget if HAVE_QT else object):
    """Right-side task panel (constructed only with a display)."""

    def __init__(self, controller: AnnotatorController):
        if not HAVE_QT:
            raise RuntimeError("Qt required for the task dock")
        super().__init__("Tasks")
        self.controller = controller
        root = QWidget()
        layout = QVBoxLayout(root)
        self.tally = QLabel("Marked 0 of 0 balls")
        layout.addWidget(self.tally)
        self.info = QLabel("No task loaded.")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        def header(text: str) -> QLabel:
            lab = QLabel(text)
            layout.addWidget(lab)
            return lab

        def btn(text: str, slot) -> QPushButton:
            b = QPushButton(text)
            b.clicked.connect(slot)
            return b

        # 1. Save answers — every button here SAVES and moves to the
        # next ball. Clicking the tube tip does the same.
        header("1. Save your answer (moves to the next ball)")
        save_row = QHBoxLayout()
        self.cant_btn = btn("Tip hidden — save & next", self._on_cant)
        self.notaball_btn = btn("Not a ball — save & next",
                                self._on_not_a_ball)
        self.notube_btn = btn("No tube — save & next",
                              lambda: self._on_state(NO_TUBE_VISIBLE))
        self.whose_btn = btn("Can't tell whose tube — save & next",
                             lambda: self._on_state(OWNER_UNCERTAIN))
        save_row.addWidget(self.cant_btn)
        save_row.addWidget(self.notaball_btn)
        save_row.addWidget(self.notube_btn)
        save_row.addWidget(self.whose_btn)
        layout.addLayout(save_row)
        self.notaball_btn.setVisible(False)
        # 2. Look around — these NEVER save, they only change the view.
        header("2. Look around (never saves, view only)")
        nav = QHBoxLayout()
        self.prev_btn = btn("◀ Earlier frame",
                            lambda: self._on_step(-1))
        self.taskframe_btn = btn("Back to task frame", self._on_taskframe)
        self.next_btn = btn("Later frame ▶",
                            lambda: self._on_step(1))
        self.zoom_btn = btn("Zoom out / to ball", self._on_zoom)
        self.look_btn = btn("Look only: OFF", self._on_look_toggle)
        self.drafts_btn = btn("Hide yellow drafts", self._on_drafts)
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.taskframe_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.zoom_btn)
        nav.addWidget(self.look_btn)
        nav.addWidget(self.drafts_btn)
        layout.addLayout(nav)
        # 3. Bigger jobs — tracing a whole tube, untangling a pile or
        # a crossing. These buttons STAY here; only the bottom "done"
        # button in this section saves and moves on. Hidden entirely on
        # single-ball click tasks: those need only sections 1, 2 and 4.
        self.sec_paths_header = header("3. Bigger job (stays here)")
        self.sec_paths = QWidget()
        sec_layout = QVBoxLayout(self.sec_paths)
        sec_layout.setContentsMargins(0, 0, 0, 0)
        # "What do I click next?" line — the whole point of the
        # redesign: never wonder what a click will do.
        self.next_label = QLabel("")
        self.next_label.setWordWrap(True)
        sec_layout.addWidget(self.next_label)
        self.pathdone_btn = btn("Done — save & next",
                                self._on_pathdone)
        sec_layout.addWidget(self.pathdone_btn)
        lanes = QHBoxLayout()
        self.lane_btns: dict = {}
        for lab in ("A", "B", "C", "D", "E"):
            b = btn(f"Ball {lab}",
                    lambda _checked=False, lab=lab: self._on_lane(lab))
            self.lane_btns[lab] = b
            lanes.addWidget(b)
        self.mode_btn = btn("Next click marks: THE BALL", self._on_mode)
        lanes.addWidget(self.mode_btn)
        sec_layout.addLayout(lanes)
        # Pile table: one row per ball, LEFT cell = the ball, RIGHT
        # cell = its tube start. Click a cell to aim the next
        # picture-click there; marking a ball opens a fresh row below.
        # Submit sits on the right of the table.
        self.clump_box = QWidget()
        clump_wrap = QHBoxLayout(self.clump_box)
        clump_wrap.setContentsMargins(0, 0, 0, 0)
        self.clump_table = QTableWidget(0, 4)
        self.clump_table.setHorizontalHeaderLabels(
            ["Ball — click, then click the ball",
             "Tube start — click, then click where its tube comes out",
             "",
             "No tube?"])
        try:
            hdr = self.clump_table.horizontalHeader()
            hdr.setSectionResizeMode(0, QHeaderView.Stretch)
            hdr.setSectionResizeMode(1, QHeaderView.Stretch)
            hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        except Exception:
            pass
        self.clump_table.verticalHeader().setVisible(False)
        self.clump_table.setAlternatingRowColors(True)
        try:
            self.clump_table.setSelectionMode(self.clump_table.NoSelection)
        except Exception:
            pass
        clump_wrap.addWidget(self.clump_table, 1)
        clump_right = QVBoxLayout()
        self.clump_submit = btn("Submit pile — save & next",
                                self._on_pathdone)
        clump_right.addWidget(self.clump_submit)
        clump_right.addStretch(1)
        clump_wrap.addLayout(clump_right)
        sec_layout.addWidget(self.clump_box)
        self.clump_box.setVisible(False)
        # Repair batch: re-check a dim tip (point or bounded area).
        self.rev_tip_box = QWidget()
        _rt = QVBoxLayout(self.rev_tip_box)
        _rt.setContentsMargins(0, 0, 0, 0)
        _rtrow = QHBoxLayout()
        self.region_btn = btn("Clicks mark: precise point",
                              self._on_region_mode)
        self.confirmdraft_btn = btn("Draft is exactly right — next",
                                    self._on_confirm_draft)
        _rtrow.addWidget(self.region_btn)
        _rtrow.addWidget(self.confirmdraft_btn)
        _rt.addLayout(_rtrow)
        _rtrow2 = QHBoxLayout()
        self.tipdone_btn = btn("Tip marked — next", self._on_tip_done)
        self.savearea_btn = btn("Save area — next", self._on_save_area)
        _rtrow2.addWidget(self.tipdone_btn)
        _rtrow2.addWidget(self.savearea_btn)
        _rt.addLayout(_rtrow2)
        # rev13: the Can't-tell control lives in the SAME always-
        # visible row as the other review controls (it was only in
        # a container that can be hidden for this task type, and a
        # live report said it was not reachable).
        _rtrow3 = QHBoxLayout()
        self.canttip_btn = btn("Can't tell — next", self._on_cant)
        _rtrow3.addWidget(self.canttip_btn)
        _rt.addLayout(_rtrow3)
        sec_layout.addWidget(self.rev_tip_box)
        self.rev_tip_box.setVisible(False)
        # Repair batch: confirm a draft route's span.
        self.rev_path_box = QWidget()
        _rp = QHBoxLayout(self.rev_path_box)
        _rp.setContentsMargins(0, 0, 0, 0)
        self.full_btn = btn("FULL root-to-tip — next",
                            lambda: self._on_span(True))
        self.partial_btn = btn("PARTIAL — rest hidden — next",
                               lambda: self._on_span(False))
        _rp.addWidget(self.full_btn)
        _rp.addWidget(self.partial_btn)
        sec_layout.addWidget(self.rev_path_box)
        self.rev_path_box.setVisible(False)
        # Verified negatives: two clicks close one box (stays here).
        self.neg_box = QWidget()
        _nb = QVBoxLayout(self.neg_box)
        _nb.setContentsMargins(0, 0, 0, 0)
        self.neg_info = QLabel("")
        self.neg_info.setWordWrap(True)
        _nb.addWidget(self.neg_info)
        self.negdone_btn = btn("Regions done — next",
                               self._on_neg_done)
        _nb.addWidget(self.negdone_btn)
        self.negempty_btn = btn("Nothing box-worthy here — next",
                                self._on_neg_empty)
        _nb.addWidget(self.negempty_btn)
        sec_layout.addWidget(self.neg_box)
        self.neg_box.setVisible(False)
        # Census tile: click every visible tip/ball inside the box.
        self.census_box = QWidget()
        _cb = QVBoxLayout(self.census_box)
        _cb.setContentsMargins(0, 0, 0, 0)
        self.census_info = QLabel("")
        self.census_info.setWordWrap(True)
        _cb.addWidget(self.census_info)
        _cbrow = QHBoxLayout()
        self.censusmode_btn = btn("Clicks mark: TIPS",
                                  self._on_census_mode)
        self.censusdone_btn = btn("Tile complete — next",
                                  self._on_census_done)
        _cbrow.addWidget(self.censusmode_btn)
        _cbrow.addWidget(self.censusdone_btn)
        _cb.addLayout(_cbrow)
        sec_layout.addWidget(self.census_box)
        self.census_box.setVisible(False)
        # Tube-body mask: paint with the brush, commit via Labels layer.
        self.mask_box = QWidget()
        _mb = QVBoxLayout(self.mask_box)
        _mb.setContentsMargins(0, 0, 0, 0)
        self.mask_info = QLabel("")
        self.mask_info.setWordWrap(True)
        _mb.addWidget(self.mask_info)
        _mbrow = QHBoxLayout()
        self.maskdone_btn = btn("Body saved — next",
                                lambda: self._on_mask_done(True))
        self.maskpartial_btn = btn("Partial — rest hidden — next",
                                   lambda: self._on_mask_done(False))
        self.masknotube_btn = btn("No tube at this grain — next",
                                  self._on_mask_none)
        _mbrow.addWidget(self.maskdone_btn)
        _mbrow.addWidget(self.maskpartial_btn)
        _mbrow.addWidget(self.masknotube_btn)
        _mb.addLayout(_mbrow)
        sec_layout.addWidget(self.mask_box)
        self.mask_box.setVisible(False)
        # Germination verdict: no canvas marks — one verdict saves.
        self.event_box = QWidget()
        _eb = QVBoxLayout(self.event_box)
        _eb.setContentsMargins(0, 0, 0, 0)
        self.event_info = QLabel("")
        self.event_info.setWordWrap(True)
        _eb.addWidget(self.event_info)
        _ebbtn = QVBoxLayout()
        self.event_within_btn = btn("Emerged within bracket — save",
                                    lambda: self._on_event_verdict("emerged_within"))
        self.event_start_btn = btn("Already emerged at start — save",
                                   lambda: self._on_event_verdict("emerged_at_start"))
        self.event_none_btn = btn("No emergence by end — save",
                                  lambda: self._on_event_verdict("no_emergence_by_end"))
        self.event_unobs_btn = btn("Unobservable — save",
                                   lambda: self._on_event_verdict("unobservable"))
        _ebbtn.addWidget(self.event_within_btn)
        _ebbtn.addWidget(self.event_start_btn)
        _ebbtn.addWidget(self.event_none_btn)
        _ebbtn.addWidget(self.event_unobs_btn)
        _eb.addLayout(_ebbtn)
        _ebbracket = QHBoxLayout()
        self.event_absent_btn = btn("Mark last-absent HERE",
                                    lambda: self._on_event_bracket("last_absent_candidate"))
        self.event_visible_btn = btn("Mark first-visible HERE",
                                     lambda: self._on_event_bracket("first_visible_candidate"))
        _ebbracket.addWidget(self.event_absent_btn)
        _ebbracket.addWidget(self.event_visible_btn)
        _eb.addLayout(_ebbracket)
        sec_layout.addWidget(self.event_box)
        self.event_box.setVisible(False)
        # Route duel: click near the better curve, or Neither.
        self.duel_box = QWidget()
        _db = QVBoxLayout(self.duel_box)
        _db.setContentsMargins(0, 0, 0, 0)
        self.duel_info = QLabel("")
        self.duel_info.setWordWrap(True)
        _db.addWidget(self.duel_info)
        self.duelneither_btn = btn("Neither follows it — next",
                                   self._on_duel_neither)
        _db.addWidget(self.duelneither_btn)
        sec_layout.addWidget(self.duel_box)
        self.duel_box.setVisible(False)
        # Stable tube IDs for crossings (mint once, shown, exported).
        self.tubeid_btn = btn("Give Tube 1 + Tube 2 their own IDs",
                              self._on_tubeids)
        sec_layout.addWidget(self.tubeid_btn)
        self.tubeid_btn.setVisible(False)
        self.rev_path_box.setVisible(False)
        # Fix-helpers live in section 4; the lanes row doubles as the
        # "pick which ball to fix" row, so no extra widgets needed.
        layout.addWidget(self.sec_paths)
        # 4. Fix mistakes — Back reopens the last ball for editing,
        # Undo deletes your mark on THIS ball, skip parks the
        # impossible pile with NOTHING saved for training.
        # None of these advances on their own.
        header("4. Fix mistakes (stays, never advances)")
        fix = QHBoxLayout()
        self.back_btn = btn("← Back (edit last ball)", self._on_back)
        self.undo_btn = btn("Undo my mark on this ball", self._on_undo)
        self.clear_btn = btn("Clear my dots on this task",
                             self._on_clear_marks)
        self.skip_btn = btn("Too tangled — skip (keeps out of training)",
                            self._on_unresolvable)
        fix.addWidget(self.back_btn)
        fix.addWidget(self.undo_btn)
        fix.addWidget(self.clear_btn)
        fix.addWidget(self.skip_btn)
        layout.addLayout(fix)
        self.setWidget(root)
        self.refresh()

    def refresh(self) -> None:
        done, total = self.controller.progress()
        cur0 = self.controller.current
        # Task counter: position in the queue (1-based) + task id, so a
        # stuck annotator can name the task instead of describing it.
        pos = ""
        if cur0 is not None:
            try:
                idx = [t.get("uuid") for t in self.controller.tasks
                       ].index(cur0.get("uuid"))
                pos = f"Task {idx + 1} of {len(self.controller.tasks)} " \
                      f"({cur0.get('uuid')}) — "
            except ValueError:
                pass
        self.tally.setText(f"{pos}Marked {done} of {total} balls")
        cur = self.controller.current
        if cur is None:
            self.info.setText(
                "Done — every ball is marked. You can press Back to "
                "review, or close the app.")
            return
        left = total - done
        extra = ""
        if self.controller._context_offset != 0:
            extra = ("You are looking at a neighboring frame — look, "
                     "don't mark. Press Back to task frame, then mark.\n")
        ttype = str(cur.get("task_type", "apex"))
        show_paths = ttype in ("crossing", "owner", "centerline",
                                     "review_tip", "review_path",
                                     "review_crossing", "neg_region",
                                     "neg_draft", "census", "body_mask",
                                     "route_duel", "germination_event")
        try:
            self.sec_paths.setVisible(show_paths)
            if self.sec_paths_header is not None:
                self.sec_paths_header.setVisible(show_paths)
        except Exception:
            pass
        c = self.controller
        is_review = ttype in ("review_tip", "review_path", "review_crossing")
        try:
            self.notaball_btn.setVisible(not is_review)
            self.notube_btn.setVisible(not is_review)
            self.whose_btn.setVisible(not is_review)
        except Exception:
            pass
        try:
            self.rev_tip_box.setVisible(ttype == "review_tip")
            self.rev_path_box.setVisible(ttype == "review_path")
            self.neg_box.setVisible(ttype in ("neg_region", "neg_draft"))
            self.census_box.setVisible(ttype == "census")
            self.mask_box.setVisible(ttype == "body_mask")
            self.duel_box.setVisible(ttype == "route_duel")
            self.event_box.setVisible(ttype == "germination_event")
            if ttype == "germination_event":
                lo = cur.get("last_absent_candidate")
                hi = cur.get("first_visible_candidate")
                self.event_info.setText(
                    f"Grain {cur.get('owner_uuid', '?')} · frames "
                    f"{cur.get('source_start')}–{cur.get('source_end')}. "
                    f"Scrub with section 2 (looking logs each visit). "
                    f"Bracket: last-absent={lo}, first-visible={hi}.")
            is_cross = ttype in ("crossing", "review_crossing")
            self.tubeid_btn.setVisible(is_cross)
        except Exception:
            pass
        try:
            is_pile = (ttype == "owner")
            self.cant_btn.setVisible(not is_pile)
            # No-ball premise rejection: piles AND fresh traces (a
            # mined focus can be debris or empty — forcing dots makes
            # junk; the verdict is fenced from training like not_a_ball).
            self.notaball_btn.setVisible(
                ttype in ("owner", "centerline"))
        except Exception:
            pass
        try:
            self.clump_box.setVisible(ttype == "owner")
            self.pathdone_btn.setVisible(ttype != "owner")
        except Exception:
            pass
        try:
            c.draw_task_markup()
        except Exception:
            pass
        if ttype in ("neg_region", "neg_draft"):
            self.sec_paths_header.setText(
                "3. Check the yellow box" if ttype == "neg_draft"
                else "3. Mark the NOT-tip spots (stays here)")
            for _lab, _b in self.lane_btns.items():
                try:
                    _b.setVisible(False)
                except Exception:
                    pass
            try:
                self.mode_btn.setVisible(False)
                self.pathdone_btn.setVisible(False)
                self.clump_box.setVisible(False)
                n = len(cur.get("neg_boxes", []))
                if ttype == "neg_draft":
                    confirmed = {tuple(float(v) for v in b)
                                 for b in cur.get("neg_boxes", []) or []}
                    nd = sum(1 for b in cur.get("draft_boxes", []) or []
                             if tuple(float(v) for v in b) not in confirmed)
                    self.neg_info.setText(
                        f"Look at the YELLOW BOX ({nd} left). Is there "
                        f"a growing tip inside it? NO tip inside → "
                        f"CLICK INSIDE THE BOX. YES, a tip is inside → "
                        f"press 'This box has a tip' below.")
                    self.next_label.setText(
                        "No tip in the yellow box? CLICK INSIDE THE BOX. "
                        "Tip in it? 'This box has a tip' button.")
                    try:
                        self.negempty_btn.setText(
                            "This box has a tip — skip it")
                    except Exception:
                        pass
                    self.next_label.setVisible(True)
                else:
                    note = (f" {c._neg_note}" if c._neg_note else "")
                    cls = str(cur.get("neg_class", "apex"))
                    self.neg_info.setText(
                        f"One click finishes: click the WHITE-DOTTED tip to "
                        f"confirm it, or click a {cls} look-alike to box it. "
                        f"{n} box(es) saved.{note}")
                    self.next_label.setText(
                        "NEXT CLICK moves you on — the tip, or a look-alike.")
                    self.next_label.setVisible(True)
                    try:
                        self.negempty_btn.setText(
                            "Nothing box-worthy here — next")
                    except Exception:
                        pass
            except Exception:
                pass
        elif ttype == "census":
            self.sec_paths_header.setText(
                "3. Mark EVERY tip/ball INSIDE THE YELLOW BOX (stays "
                "here) — the middle is just the box center, not a "
                "subject")
            for _lab, _b in self.lane_btns.items():
                try:
                    _b.setVisible(False)
                except Exception:
                    pass
            try:
                self.mode_btn.setVisible(False)
                self.pathdone_btn.setVisible(False)
                self.clump_box.setVisible(False)
                nt = len(cur.get("census_tips", []))
                ng = len(cur.get("census_grains", []))
                tile = cur.get("tile_xywh")
                self.censusmode_btn.setText(
                    "Clicks mark: BALLS" if c._census_mode == "grain"
                    else "Clicks mark: TIPS")
                self.census_info.setText(
                    f"Tile {list(tile) if tile else '(whole view)'}: "
                    f"{nt} tip(s), {ng} ball(s) marked. Miss nothing — "
                    f"press Tile-complete even when the tile is empty.")
                self.next_label.setText(
                    "NEXT CLICK marks one "
                    f"{'BALL' if c._census_mode == 'grain' else 'TIP'} "
                    "inside the tile — click the picture.")
                self.next_label.setVisible(True)
            except Exception:
                pass
        elif ttype == "body_mask":
            self.sec_paths_header.setText(
                "3. Paint the tube body (stays here)")
            for _lab, _b in self.lane_btns.items():
                try:
                    _b.setVisible(False)
                except Exception:
                    pass
            try:
                self.mode_btn.setVisible(False)
                self.pathdone_btn.setVisible(False)
                self.clump_box.setVisible(False)
                n = c.mask_coverage()
                self.mask_info.setText(
                    f"Paint the tube along the yellow guide with the "
                    f"brush ({n} stamps). 'Body saved' = whole tube "
                    f"painted (background trusted). 'Partial' = part "
                    f"hidden (background stays unknown). 'No tube at "
                    f"this grain' = nothing grows from the ringed "
                    f"grain (scoped to that grain only).")
            except Exception:
                pass
        elif ttype == "route_duel":
            self.sec_paths_header.setText(
                "3. Pick the curve that follows the tube (stays here)")
            for _lab, _b in self.lane_btns.items():
                try:
                    _b.setVisible(False)
                except Exception:
                    pass
            try:
                self.mode_btn.setVisible(False)
                self.pathdone_btn.setVisible(False)
                self.clump_box.setVisible(False)
                self.duel_info.setText(
                    "YELLOW vs BLUE: click NEAR the curve that follows "
                    "the tube (one click votes + advances). Neither "
                    "good? Neither button. Can't tell? Can't tell.")
                self.next_label.setText(
                    "NEXT CLICK votes for the nearer curve.")
                self.next_label.setVisible(True)
            except Exception:
                pass
        elif ttype == "review_tip":
            self.sec_paths_header.setText(
                "3. Re-check this tip — point or bounded area (stays here)")
            for _lab, _b in self.lane_btns.items():
                try:
                    _b.setVisible(False)
                except Exception:
                    pass
            try:
                self.mode_btn.setVisible(False)
                self.pathdone_btn.setVisible(False)
                self.clump_box.setVisible(False)
                self.region_btn.setText(
                    "Clicks mark: AREA corner (3+ clicks)"
                    if c._region_mode else "Clicks mark: precise point")
                n = len(c._region_pts)
                self.next_label.setText(
                    f"AREA mode: click corner {n + 1} (needs 3+, has {n})."
                    if c._region_mode else
                    "Click the tip precisely, or press Draft-is-right.")
                self.next_label.setVisible(True)
            except Exception:
                pass
        elif ttype == "review_path":
            self.sec_paths_header.setText(
                "3. Confirm the yellow draft route (stays here)")
            for _lab, _b in self.lane_btns.items():
                try:
                    _b.setVisible(False)
                except Exception:
                    pass
            try:
                self.mode_btn.setVisible(False)
                self.pathdone_btn.setVisible(False)
                self.clump_box.setVisible(False)
                n = len((c.current.get("draft_path") or {}).get("path_xy", []))
                self.next_label.setText(
                    f"Draft has {n} dots. FULL = ball attachment to far "
                    f"end visible; else PARTIAL.")
                self.next_label.setVisible(True)
            except Exception:
                pass
            extra += ("Yellow line = earlier stroke (draft, not truth).\n")
        elif ttype == "review_crossing":
            self.sec_paths_header.setText(
                "3. Check the dashed lanes — fix only what is wrong "
                "(stays here)")
            names = {"A": "Tube 1", "B": "Tube 2"}
            for lab, b in self.lane_btns.items():
                if lab not in names:
                    try:
                        b.setVisible(False)
                    except Exception:
                        pass
                    continue
                npts = len(c._lanes.get(lab, []))
                new = c.new_dots(lab)
                prefix = "→ " if lab == c._lane else ""
                if new > 0:
                    tick = f" ({npts - new} draft + {new} new)"
                elif npts:
                    tick = f" ({npts} draft)"
                else:
                    tick = ""
                try:
                    b.setText(f"{prefix}{names.get(lab, lab)}{tick}")
                except Exception:
                    pass
                b.setVisible(True)
            try:
                self.mode_btn.setVisible(False)
                self.clump_box.setVisible(False)
                self.pathdone_btn.setVisible(True)
            except Exception:
                pass
            self.pathdone_btn.setText("Lanes confirmed — still unresolved")
            try:
                self.next_label.setText(
                    "Lane right? Touch nothing. Lane wrong? Click Tube 1 "
                    "or 2, then click the right dots.")
                self.next_label.setVisible(True)
            except Exception:
                pass
        elif ttype == "owner":
            self.sec_paths_header.setText(
                "3. Pile of balls — work the table top to bottom "
                "(stays here; Submit moves on)")
            for _lab, _b in self.lane_btns.items():
                try:
                    _b.setVisible(False)
                except Exception:
                    pass
            try:
                self.mode_btn.setVisible(False)
                self.pathdone_btn.setVisible(False)
                self.clump_box.setVisible(True)
            except Exception:
                pass
            try:
                self._rebuild_clump_table()
            except Exception:
                pass
            lab, need = c.clump_next_needed()
            n = c.lane_number(lab)
            if c.clump_complete():
                nxt = ("Every row has both dots ✓ — press Submit "
                       "on the right, or keep marking if the pile "
                       "has more balls.")
            elif need == "ball":
                nxt = (f"NEXT CLICK marks Ball {n} ITSELF — "
                       f"click the round middle of ball {n}.")
            else:
                nxt = (f"NEXT CLICK marks Ball {n}'s TUBE START — "
                       f"click where the tube comes out of ball {n}.")
            try:
                self.next_label.setText(nxt)
                self.next_label.setVisible(True)
            except Exception:
                pass
            try:
                self.clump_submit.setEnabled(bool(c.clump_complete()))
            except Exception:
                pass
            extra += ("Click a table cell to aim there — LEFT for the "
                      "ball, RIGHT for its tube start. Only ONE ball in "
                      "view? Mark a single row and Submit — that counts. "
                      "Not a ball at all (debris)? 'Not a ball' above.\n")
        elif ttype == "crossing":
            self.sec_paths_header.setText(
                "3. Crossing — trace Tube 1, then Tube 2 (stays here)")
            names = {"A": "Tube 1", "B": "Tube 2"}
            for lab, b in self.lane_btns.items():
                if lab not in names:
                    try:
                        b.setVisible(False)
                    except Exception:
                        pass
                    continue
                npts = len(c._lanes.get(lab, []))
                new = c.new_dots(lab)
                prefix = "→ " if lab == c._lane else ""
                if new > 0:
                    tick = f" ({npts - new} draft + {new} new)"
                elif npts:
                    tick = f" ({npts} draft)"
                else:
                    tick = ""
                try:
                    b.setText(f"{prefix}{names.get(lab, lab)}{tick}")
                except Exception:
                    pass
                b.setVisible(True)
            self.mode_btn.setVisible(False)
            npts = len(c._lanes.get(c._lane, []))
            nm = names.get(c._lane, c._lane)
            try:
                self.next_label.setText(
                    f"NOW TRACING {nm} — click dots along that one tube, "
                    f"right through the crossing ({npts} dots so far). "
                    f"To trace the other tube, press its button.")
                self.next_label.setVisible(True)
            except Exception:
                pass
            self.pathdone_btn.setText("Both tubes drawn — save & next")
        elif ttype == "centerline":
            self.sec_paths_header.setText(
                "3. Trace — each click adds a dot (stays here)")
            for _lab, _b in self.lane_btns.items():
                try:
                    _b.setVisible(False)
                except Exception:
                    pass
            try:
                self.mode_btn.setVisible(False)
            except Exception:
                pass
            try:
                self.next_label.setText(
                    f"TRACE: click the next dot along the tube "
                    f"({len(c._path_pts)} dots so far).")
                self.next_label.setVisible(True)
            except Exception:
                pass
            self.pathdone_btn.setText("Trace saved — save & next")
        else:
            try:
                self.next_label.setVisible(False)
            except Exception:
                pass
            self.pathdone_btn.setText("Path finished — save & next")
        try:
            self.undo_btn.setText(
                "Undo last dot"
                if ttype in ("crossing", "owner", "centerline",
                             "review_tip", "review_crossing")
                else "Undo my mark on this ball")
        except Exception:
            pass
        self.info.setText(
            f"Ball {done + 1} of {total} ({left} left)\n"
            f"{extra}{self.controller.instruction()}")
        if self.controller._look_only:
            try:
                self.info.setText(
                    "LOOK-ONLY ON — clicks move the camera, nothing "
                    "marks. Toggle it off to mark.\n" + self.info.text())
            except Exception:
                pass

    def _on_step(self, delta: int) -> None:
        self.controller.step_frame(delta)
        self.refresh()

    def _on_taskframe(self) -> None:
        self.controller.back_to_task_frame()
        self.refresh()

    def _on_state(self, state: str) -> None:
        if self.controller.current is None:
            self.refresh()
            return
        self.controller.save_state(state)
        self.controller.save_and_next()
        self.refresh()

    def _on_pathdone(self) -> None:
        cur = self.controller.current
        if cur is not None and str(cur.get("task_type")) == "review_crossing":
            self._on_confirm_crossing()
            return
        note = ""
        try:
            self.controller.finish_path_and_next()
        except ValueError:
            # Stay on the task, but say what's missing instead of
            # silently nothing — e.g. which balls lack their tube dot.
            try:
                c = self.controller
                if (c.current is not None
                        and str(c.current.get("task_type")) == "owner"):
                    missing = [c.lane_number(lab) for lab in c.ordered_balls()
                               if not (c._grains[lab].get("xy")
                                       and (c._grains[lab].get("root_xy")
                                            or c._grains[lab].get("no_tube")))]
                    if missing:
                        note = ("Not yet — still needs an answer: "
                                + ", ".join(f"Ball {m}" for m in missing)
                                + ". Give each its tube dot or flag "
                                  "'No tube'.")
            except Exception:
                pass
        self.refresh()
        if note:
            try:
                self.next_label.setText(note)
            except Exception:
                pass

    def _on_unresolvable(self) -> None:
        # P0A escape hatch: an impossible clump/crossing is flagged
        # unresolvable with NO observation — fold/export/audit exclude
        # it, so no forced guess ever reaches training data.
        self.controller.skip_unresolvable()
        self.refresh()

    def _on_not_a_ball(self) -> None:
        # Premise rejection: debris/artifact, not a ball. Tagged and
        # fenced off from training (crop-scoped judgment, never a
        # full-frame negative).
        self.controller.save_not_a_ball()
        self.refresh()

    def _on_undo(self) -> None:
        self.controller.undo_mark()
        self.refresh()

    def _on_clear_marks(self) -> None:
        self.controller.clear_task_marks()
        self.refresh()

    def _on_arm_ball(self, label: str) -> None:
        self.controller.arm(label, "grain")
        self.refresh()

    def _on_arm_root(self, label: str) -> None:
        self.controller.arm(label, "root")
        self.refresh()

    def _on_remove_ball(self, label: str) -> None:
        self.controller.remove_ball(label)
        self.refresh()

    def _on_flag_no_tube(self, label: str) -> None:
        c = self.controller
        g = c._grains.get(str(label).upper()[:1] or "A", {})
        try:
            c.set_no_tube(label, not bool(g.get("no_tube")))
        except ValueError:
            pass  # e.g. tube dot present — remove that first
        self.refresh()

    def _rebuild_clump_table(self) -> None:
        """One row per ball: LEFT aims at the ball, RIGHT at its tube."""
        from qtpy.QtWidgets import QPushButton as _PB
        from qtpy.QtWidgets import QTableWidgetItem as _TWI
        c = self.controller
        rows = c.ordered_balls()
        fresh = c.next_fresh_ball()
        if fresh is not None and fresh not in rows:
            rows = rows + [fresh]
        tbl = self.clump_table
        try:
            tbl.setRowCount(len(rows))
        except Exception:
            return
        for i, lab in enumerate(rows):
            n = c.lane_number(lab)
            g = c._grains.get(lab, {})
            has_ball = bool(g.get("xy"))
            has_root = bool(g.get("root_xy"))
            flagged = bool(g.get("no_tube"))
            is_fresh = lab not in c._grains
            armed_here = (c._lane == lab)
            if not has_ball and lab not in c._grains:
                left_txt = f"+ Ball {n} — click, then click the ball"
            elif has_ball and not has_root:
                left_txt = f"Ball {n} ✓"
            else:
                left_txt = f"Ball {n} ✓ — redo"
            if armed_here and c._mark_mode == "grain":
                left_txt = "→ " + left_txt
            left = _PB(left_txt)
            left.clicked.connect(
                lambda _checked=False, lab=lab: self._on_arm_ball(lab))
            if has_root:
                right_txt = "Tube start ✓ — redo"
            elif flagged:
                right_txt = "No tube ✓"
            elif has_ball:
                right_txt = "Mark tube start"
            else:
                right_txt = "Mark tube start (needs the ball first)"
            if armed_here and c._mark_mode == "root" and not flagged:
                right_txt = "→ " + right_txt
            right = _PB(right_txt)
            right.setEnabled(has_ball and not flagged)
            right.clicked.connect(
                lambda _checked=False, lab=lab: self._on_arm_root(lab))
            if flagged:
                flag_txt, flag_on = "No tube ✓ — undo", True
            elif has_ball and not has_root:
                flag_txt, flag_on = "No tube", True
            elif has_root:
                flag_txt, flag_on = "Has tube ✓", False
            else:
                flag_txt, flag_on = "—", False
            flag = _PB(flag_txt)
            flag.setEnabled(flag_on)
            flag.clicked.connect(
                lambda _checked=False, lab=lab:
                self._on_flag_no_tube(lab))
            try:
                tbl.setCellWidget(i, 0, left)
                tbl.setCellWidget(i, 1, right)
                tbl.setCellWidget(i, 3, flag)
                if is_fresh:
                    tbl.setItem(i, 2, _TWI(""))
                else:
                    rm = _PB("✕ Remove")
                    rm.clicked.connect(
                        lambda _checked=False, lab=lab:
                        self._on_remove_ball(lab))
                    tbl.setCellWidget(i, 2, rm)
            except Exception:
                pass

    def _on_region_mode(self) -> None:
        c = self.controller
        c._region_mode = not c._region_mode
        self.region_btn.setText(
            "Clicks mark: AREA corner (3+ clicks)"
            if c._region_mode else "Clicks mark: precise point")
        self.refresh()

    def _on_confirm_draft(self) -> None:
        try:
            self.controller.confirm_draft_tip()
        except ValueError:
            pass
        self.refresh()

    def _on_tip_done(self) -> None:
        try:
            self.controller.finish_review_tip()
        except ValueError:
            pass
        self.refresh()

    def _on_save_area(self) -> None:
        try:
            self.controller.finish_region_and_next()
        except ValueError:
            pass
        self.refresh()

    def _on_neg_done(self) -> None:
        try:
            self.controller.finish_neg()
        except ValueError:
            self.neg_info.setText(
                "Draw at least one box first: click two opposite "
                "corners. An empty review cannot become supervision.")
        self.refresh()

    def _on_neg_empty(self) -> None:
        try:
            self.controller.finish_neg_empty()
        except ValueError:
            pass
        self.refresh()

    def _on_census_done(self) -> None:
        try:
            self.controller.finish_census()
        except (ValueError, AttributeError):
            pass
        self.refresh()

    def _on_mask_done(self, complete: bool) -> None:
        try:
            self.controller.commit_mask_labels(complete)
            self.controller.save_and_next()
        except ValueError:
            self.mask_info.setText(
                "Paint the tube body first with the brush (mask-paint "
                "layer). An empty review cannot become supervision.")
        self.refresh()

    def _on_mask_none(self) -> None:
        """Ball-scoped 'no tube at this grain' verdict (rev8)."""
        try:
            self.controller.save_mask_none()
            self.controller.save_and_next()
        except ValueError as e:
            self.mask_info.setText(f"Cannot record: {e}")
        self.refresh()

    def _on_event_verdict(self, verdict: str) -> None:
        try:
            self.controller.save_germination_event(
                verdict, self.controller.current.get("last_absent_candidate"),
                self.controller.current.get("first_visible_candidate"))
            self.controller.save_and_next()
        except ValueError as e:
            self.event_info.setText(f"Cannot record: {e}")
        self.refresh()

    def _on_event_bracket(self, slot: str) -> None:
        try:
            frame = int(self.controller.current["query_frames"][0]) + int(
                getattr(self.controller, "_context_offset", 0))
            self.controller.current[slot] = frame
            self.controller.persist_draft()
        except (ValueError, KeyError, TypeError) as e:
            self.event_info.setText(f"Cannot mark: {e}")
        self.refresh()

    def _on_duel_neither(self) -> None:
        try:
            self.controller.finish_duel("neither")
        except ValueError:
            pass
        self.refresh()

    def _on_census_mode(self) -> None:
        cur = self.controller._census_mode
        self.controller._census_mode = (
            "grain" if cur == "tip" else "tip")
        self.refresh()

    def _on_tubeids(self) -> None:
        try:
            self.controller.mint_tube_id()
        except ValueError:
            pass
        self.refresh()

    def _on_span(self, complete: bool) -> None:
        try:
            self.controller.confirm_path_span(complete)
            self.controller.save_and_next()
        except ValueError:
            pass
        self.refresh()

    def _on_confirm_crossing(self) -> None:
        try:
            self.controller.confirm_review_crossing()
        except ValueError:
            pass
        self.refresh()

    def _on_lane(self, label: str) -> None:
        self.controller.select_lane(label)
        self.refresh()

    def _on_mode(self) -> None:
        cur = self.controller._mark_mode
        self.controller._mark_mode = (
            "root" if cur == "grain" else "grain")
        self.mode_btn.setText(
            "Next click marks: TUBE START (where tube leaves the ball)"
            if self.controller._mark_mode == "root"
            else "Next click marks: THE BALL (the round ball itself)")
        self.refresh()

    def _on_zoom(self) -> None:
        self.controller.toggle_zoom()
        self.refresh()

    def _on_look_toggle(self) -> None:
        c = self.controller
        c._look_only = not c._look_only
        try:
            self.look_btn.setText(
                "Look only: ON (clicks safe)" if c._look_only
                else "Look only: OFF")
        except Exception:
            pass
        self.refresh()
    def _on_drafts(self) -> None:
        vis = self.controller.toggle_drafts()
        try:
            self.drafts_btn.setText(
                "Hide yellow drafts" if vis else "Show yellow drafts")
        except Exception:
            pass

    def _on_back(self) -> None:
        self.controller.go_back()
        self.refresh()

    def _on_cant(self) -> None:
        # Guard (H239): the Done screen leaves current=None; pressing
        # the button there crashed the app instead of no-opping.
        if self.controller.current is None:
            self.refresh()
            return
        self.controller.save_cant_tell()
        self.controller.save_and_next()
        self.refresh()


def wire_canvas_click(controller: AnnotatorController,
                      dock: "TaskDock | None" = None) -> None:
    """One mouse click on the image = mark the tip, move on (H237).

    Uses a drag callback that only fires when the pointer barely moved
    (a real click, not a pan), so dragging to look around never marks.
    Bound at VIEWER level (H237 fix): the pin layer sits on top of the
    image after the first mark and swallows layer-level clicks.
    Napari-only; no-op headless.
    """
    if controller.viewer is None or controller._image_layer is None:
        return
    viewer = controller.viewer

    def _on_drag(_viewer, event):
        import math

        try:
            start = tuple(float(v) for v in event.position)
        except Exception:
            return
        yield
        try:
            end = tuple(float(v) for v in event.position)
        except Exception:
            return
        if len(start) >= 2 and len(end) >= 2 and \
                math.dist(start[-2:], end[-2:]) < 5:
            controller.click(float(end[-1]), float(end[-2:][0]))
            if dock is not None:
                try:
                    dock.refresh()
                except Exception as error:
                    import traceback
                    traceback.print_exc()
                    if hasattr(dock, 'show_error'):
                        dock.show_error(f'Could not display annotation: {error}')

    viewer.mouse_drag_callbacks.append(_on_drag)
