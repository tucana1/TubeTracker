"""Launch the TubeTracker Annotator workbench (P0A v1, needs a display).

Usage:
    .venv-annotator/bin/python scripts/run_annotation_app.py \
        --movie "/path/to/movie.mp4" --project-dir /path/to/project

Builds a napari viewer and resumes saved annotation tasks. An explicit
native analysis configuration also enables movie inference and review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tubetracker.annotation_app import (AnnotatorController, TaskDock,
                                         wire_canvas_click)  # noqa: E402
from tubetracker.annotation_frames import FrameReader  # noqa: E402
from tubetracker.annotation_store import AnnotationStore  # noqa: E402


def reopen_tasks(store: AnnotationStore, uuids, actor: str = ""
                 ) -> list[str]:
    """Mark completed tasks unfinished so they can be revisited.

    rev9 WP-A.1: the reviewer asks for the surviving masks to be reopened
    and re-confirmed (an enclosing reviewed region, unknown patches,
    ownership, completeness). Re-saving a mask writes a NEW REVISION of
    the same mask uuid, so the original bytes stay in the history.
    """
    done = []
    for raw in uuids:
        uuid = str(raw).strip()
        if not uuid:
            continue
        rec = store.load(uuid)
        if not rec or rec.get("kind") != "task":
            print(f"reopen: no task {uuid}", flush=True)
            continue
        data = dict(rec["data"])
        data["completed"] = False
        data["reopened_for_review"] = True
        store.save("task", uuid, data, actor=actor or "reopen")
        done.append(uuid)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--actor", default="")
    ap.add_argument("--reopen", default="",
                    help="comma-separated TASK uuids to reopen (mark "
                         "unfinished) before loading; for review rounds "
                         "such as re-confirming a reviewed extent")
    ap.add_argument("--task-ids", default="",
                    help="comma-separated TASK uuids to present in this "
                         "session only. Other unfinished tasks stay "
                         "untouched in the DB (stale programme tasks are "
                         "never auto-completed or deleted); use to show "
                         "just the current programme.")
    ap.add_argument("--extra-movies", action="append", default=[],
                    help="additional movies as key=path (repeatable); "
                    "tasks name their movie, default is --movie")
    ap.add_argument("--pipeline", default="v29-legacy",
                    choices=["v29-legacy", "v30-strict", "v30-native"],
                    help="rev12 P1.3: explicit inference backend for "
                         "this session. v30-strict loads the strict "
                         "factory checkpoint (--pipeline-checkpoint) "
                         "and exposes it to the session; unavailable "
                         "backends are reported, never silently "
                         "substituted.")
    ap.add_argument("--pipeline-checkpoint", default="",
                    help="checkpoint for --pipeline v30-strict")
    ap.add_argument("--analysis-config", help="Native movie-analysis JSON; enables the analysis dock")
    a = ap.parse_args()

    import napari

    proj = Path(a.project_dir)
    proj.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(proj / "annotations.db")
    saved_analysis = store.load("analysis-configuration")
    analysis_config = (json.loads(Path(a.analysis_config).read_text()) if a.analysis_config else
                       saved_analysis["data"] if saved_analysis else None)
    if a.analysis_config or (analysis_config and a.pipeline == "v29-legacy"):
        a.pipeline = "v30-native"
    if a.pipeline == "v30-native" and analysis_config is None:
        raise ValueError("Native analysis needs a saved configuration or --analysis-config. "
                         "Use Start_TubeTracker_Analysis.command to prepare the supported project.")
    if analysis_config and Path(analysis_config["request"]["movie_path"]).resolve() != Path(a.movie).resolve():
        raise ValueError("Analysis configuration belongs to a different movie")
    if analysis_config:
        analysis_config.setdefault("inference_python", str(REPO/".venv/bin/python"))
    # rev12 P1.3: pipeline selection is recorded in the session and
    # the app's task metadata; the legacy path stays the default.
    from tubetracker.pipeline_backends import load_backend
    _predict, _binfo = load_backend(
        a.pipeline, checkpoint=a.pipeline_checkpoint, repo_root=str(REPO),
        analysis_config=analysis_config, cache_dir=proj/"analysis_cache")
    print(f"pipeline backend: {a.pipeline} -> {_binfo}", flush=True)
    # the selection is recorded as a session entity (same store the
    # app already writes; reloadable, append-only)
    store.save("session", "pipeline-backend",
               {"backend": a.pipeline, "info": _binfo},
               actor=a.actor or "session")
    if a.reopen.strip():
        done = reopen_tasks(store, a.reopen.split(","), actor=a.actor)
        print(f"reopened {len(done)} task(s): {','.join(done)}", flush=True)
    reader = FrameReader(a.movie)
    controller = AnnotatorController(store, reader, viewer=None,
                                     actor=a.actor)
    if a.pipeline == "v30-native":
        from tubetracker.analysis_workbench import AnalysisSession
        controller.analysis_session = AnalysisSession(store, analysis_config, service=_predict)
    controller.add_reader("default", reader)
    if analysis_config:
        controller.add_reader(analysis_config['request']['movie_id'], reader)
        for key, path in analysis_config.get('extra_movies', {}).items():
            controller.add_reader(key, FrameReader(path))
    for spec in a.extra_movies:
        key, _, path = spec.partition("=")
        if key and path:
            controller.add_reader(key, FrameReader(path))

    pending = store.unfinished_tasks(limit=500)
    if not analysis_config:
        primary_keys = {str(t.get('movie') or t.get('movie_uuid') or '') for t in pending}
        primary_keys -= set(controller._readers) | {''}
        if len(primary_keys) == 1:
            controller.add_reader(primary_keys.pop(), reader)
        elif len(primary_keys) > 1:
            raise ValueError('Register every task movie with --extra-movies key=path')
    if not pending and store.count_tasks() == 0 and a.pipeline != "v30-native":
        # Starter batch (H236): only on a TRULY new project. A fully
        # completed project must reopen as complete — silently
        # regenerating starters on empty-pending was a P0A acceptance
        # failure (user's marks would be buried under a fresh queue).
        # Starter batch (H236): 12 evenly spaced frames, each centered
        # on a real detected ball (FRST) spread across the field, so
        # the annotator opens zoomed onto a target, not a blank field.
        import cv2

        sys.path.insert(0, str(REPO / "prototypes" / "timesfm_tip_forecast"))
        from grain_detect import detect_grains

        n = len(reader)
        step = max(1, n // 12)
        tasks = []
        taken: list[tuple[float, float]] = []
        for i in range(12):
            fid = min(i * step, n - 1)
            res = reader.read(fid)
            gray = (res.frame if res.frame.ndim == 2 else
                    cv2.cvtColor(res.frame, cv2.COLOR_BGR2GRAY))
            dets = [d for d in detect_grains(gray)
                    if all(abs(d[0] - x) > 120 or abs(d[1] - y) > 120
                           for x, y in taken)]
            if not dets:
                continue
            x, y, _s = dets[0]
            taken.append((x, y))
            tasks.append({
                "uuid": f"starter-{i:02d}",
                "owner_uuid": "unassigned",
                "query_frames": [fid],
                "focus_xy": [float(x), float(y)],
                "task_type": "apex",
                "stratum": "starter",
                "completed": False,
            })
        controller.load_tasks(tasks)
        pending = store.unfinished_tasks(limit=500)
    else:
        pending = store.unfinished_tasks(limit=500)
    only = [u.strip() for u in a.task_ids.split(",") if u.strip()]
    if only:
        missing = [u for u in only if u not in {t.get("uuid") for t in pending}]
        if missing:
            raise ValueError(f"task-ids not unfinished in this project: {missing}")
        controller.tasks = [t for t in pending if t.get("uuid") in set(only)]
        print(f"session task scope: {len(controller.tasks)} task(s)", flush=True)
    else:
        controller.tasks = pending
    print(f"resuming {len(pending)} pending tasks", flush=True)

    viewer = napari.Viewer(show=False)
    controller.viewer = viewer
    if controller.analysis_session is not None:
        from tubetracker.analysis_dock import AnalysisDock
        from tubetracker.analysis_queue import ReviewWorkspace
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication, QScrollArea, QFrame

        def scrollable(panel):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(panel)
            scroll.setMinimumWidth(panel.minimumWidth()+16)
            scroll.setMaximumWidth(panel.maximumWidth()+16)
            return scroll

        dock = ReviewWorkspace(controller)
        QApplication.instance().aboutToQuit.connect(controller.persist_draft)
        dock.container = viewer.window.add_dock_widget(scrollable(dock), name="Review selection", area="right")
        analysis_dock = AnalysisDock(controller, dock)
        viewer.window.add_dock_widget(scrollable(analysis_dock), name="Movie analysis", area="left")
        controller.analysis_session.preview(controller)
        dock.container.hide()
        viewer.title = "TubeTracker — Movie analysis"
        viewer.window._qt_viewer.dockLayerControls.hide()
        viewer.window._qt_viewer.dockLayerList.hide()
        window = viewer.window._qt_window
        available = window.screen().availableGeometry()
        window.resize(min(1280, available.width()-40), min(820, available.height()-50))
        window.move(available.x()+20, available.y()+20)
    else:
        dock = TaskDock(controller)
        viewer.window.add_dock_widget(dock, name="Tasks")
    controller.advance()
    dock.refresh()
    wire_canvas_click(controller, dock)
    if controller.analysis_session is not None and controller.analysis_session.result:
        selection = controller.analysis_session.selection
        if selection.get("kind") == "queue":
            dock.open_queue()
        elif selection.get("kind") == "census":
            controller.analysis_session.open_census(controller, selection["source_frame"])
            dock.container.show()
            dock.refresh()
        else:
            analysis_dock.select_row(next((i for i, r in enumerate(analysis_dock.rows)
                if r["owner_id"] == selection.get("owner_id")
                and r["source_frame"] == selection.get("source_frame")), 0))
            if selection.get("kind") == "path":
                analysis_dock.start_path()
    elif controller.analysis_session is not None:
        dock.open_queue()
    viewer.show()
    # Dock layout determines the real canvas size only after showing the
    # window. Refit the restored selection once the initial events finish.
    from qtpy.QtCore import QTimer
    QTimer.singleShot(0, controller._fit_camera_to_task)
    napari.run()


if __name__ == "__main__":
    main()
