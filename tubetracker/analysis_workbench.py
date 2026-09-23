"""Persistent analysis sessions shared by the GUI and headless receipts."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

from .analysis_contracts import AnalysisRequest, stable_hash, store_review_records, file_hash
from .analysis_dependencies import FileFingerprinter, dependency_fingerprint, require_same_dependencies
from .movie_analysis import MovieAnalysisService, export_analysis


class AnalysisSession:
    def __init__(self, store, config=None, *, service=None):
        self.store = store
        self.directory = store.path.parent
        saved = store.load("analysis-configuration")
        self.config = copy.deepcopy(config if config is not None else saved["data"] if saved else {})
        self.service = service
        self.result = None
        self.saved_signature = ""
        self._fingerprinter = FileFingerprinter()
        self._layers = []
        selected = store.load("analysis-selection")
        self.selection = selected["data"] if selected else {}
        self.configure(self.config, persist=config is not None)
        last = store.load("analysis-latest")
        if last:
            path = Path(last["data"]["analysis_path"])
            if path.exists():
                if file_hash(path) != last["data"]["sha256"]:
                    raise ValueError("saved analysis checksum does not match its receipt")
                self.result = json.loads(path.read_text())
                self.saved_signature = last["data"]["review_signature"]

    def configure(self, config, *, persist=True):
        request = AnalysisRequest(**config["request"])
        request.validate()
        for name in ("cap_checkpoint", "body_checkpoint"):
            if not Path(config[name]).is_file():
                raise ValueError(f"{name} does not exist")
        def backend_identity(settings):
            return (str(Path(settings["cap_checkpoint"]).resolve()),
                    str(Path(settings["body_checkpoint"]).resolve()), settings.get("device", "cpu"),
                    str(Path(settings.get("cache_dir") or self.directory/"analysis_cache").resolve()),
                    str(Path(settings.get("registry_path") or self.directory/"grain_registry.sqlite").resolve()))
        old_backend = backend_identity(self.config)
        new_backend = backend_identity(config)
        if self.service is not None and old_backend != new_backend:
            self.service = None
        self.config = copy.deepcopy(config)
        if persist:
            self.store.save("session", "analysis-configuration", self.config, actor="analysis-settings")
        if self.service is None:
            self.service = MovieAnalysisService(self.config["cap_checkpoint"],
                body_checkpoint=self.config["body_checkpoint"],
                cache_dir=self.config.get("cache_dir") or self.directory/"analysis_cache",
                registry_path=self.config.get("registry_path") or self.directory/"grain_registry.sqlite",
                device=self.config.get("device", "cpu"))

    def review_signature(self, entities=None, *, dependencies=None, tombstones=None):
        from .population import census_review_records
        entities = self.store.entities() if entities is None else entities
        records = store_review_records(entities,
                    movie_id=self.config["request"]["movie_id"], source=str(self.store.path))
        # Exact owned masks now contribute presence bounds. Their accepted
        # revisions and lineage status must stale an export just like tips.
        # Exclude drawing drafts so navigation/draft edits remain harmless.
        lineage_fields = ('task_uuid', 'source_task_uuid', 'source_obs_uuid',
                          'movie', 'movie_uuid', 'owner_uuid', 'review_origin',
                          'review_status', 'review_verdict', 'training_eligible', 'annotator')
        masks = [e for e in entities if e['kind'] == 'mask']
        mask_records = [e['data'] for e in masks]
        snapshot = self.config['request'].get('snapshot')
        if snapshot:
            path = Path(snapshot)/'body_masks.json'
            if path.exists():
                mask_records += json.loads(path.read_text())
        links = ('task_uuid', 'source_task_uuid', 'source_obs_uuid')
        linked = {m[k] for m in mask_records for k in links if m.get(k)}
        by_id = {e['uuid']:e['data'] for e in entities}
        pending = list(linked)
        while pending:
            item = by_id.get(pending.pop(), {})
            for key in links:
                if item.get(key) and item[key] not in linked:
                    linked.add(item[key]); pending.append(item[key])
        lineage = [{'uuid': e['uuid'], 'kind': e['kind'],
                    'data': {k: e['data'].get(k) for k in lineage_fields}}
                   for e in entities if e['uuid'] in linked and e['kind'] in ('task', 'observation')]
        return stable_hash({"configuration": self.config, "reviews": records,
                            "body_masks": masks,
                            'grain_identities': [e for e in entities if e['kind'] == 'grain_identity'],
                            "review_lineage": lineage,
                            "tombstones": self.store.tombstones() if tombstones is None else tombstones,
                            "census": census_review_records(entities),
                            "dependencies": dependencies if dependencies is not None else
                                dependency_fingerprint(self.config, self._fingerprinter)})

    @property
    def stale(self):
        try:
            return self.result is None or self.review_signature() != self.saved_signature
        except (OSError, RuntimeError):
            # A file being replaced during refresh is never evidence that
            # an existing export is current. Recompute reports the cause.
            return True

    def prepare(self):
        """Capture committed reviews on the SQLite connection's owner thread."""
        entities = self.store.entities()
        tombstones = self.store.tombstones()
        dependencies = dependency_fingerprint(self.config, self._fingerprinter)
        return {"request": copy.deepcopy(self.config["request"]), "entities": entities,
                "tombstones": tombstones,
                "configuration": copy.deepcopy(self.config),
                "dependencies": dependencies,
                "review_signature": self.review_signature(entities, dependencies=dependencies,
                                                           tombstones=tombstones)}

    def _validate_packet(self, packet, result=None):
        require_same_dependencies(packet["dependencies"],
            dependency_fingerprint(packet["configuration"], self._fingerprinter))
        if result is not None:
            require_same_dependencies(packet["dependencies"], result.get("dependencies"))

    def calculate(self, packet, progress=None):
        """Worker-safe: no access to the store/Qt objects."""
        self._validate_packet(packet)
        interpreter = packet["configuration"].get("inference_python")
        if interpreter and Path(interpreter).resolve() != Path(sys.executable).resolve():
            job_dir = self.directory/"analysis_jobs"/uuid.uuid4().hex
            job_dir.mkdir(parents=True)
            payload = copy.deepcopy(packet)
            payload["configuration"]["cache_dir"] = str(self.service.cache_dir.resolve())
            payload["configuration"]["registry_path"] = str(self.service.registry_path.resolve())
            payload["review_source"] = str(self.store.path.resolve())
            job = job_dir/"request.json"
            job.write_text(json.dumps(payload, indent=2)+"\n")
            command = [str(interpreter), "-u", str(Path(__file__).resolve().parents[1]/"scripts/analyze_movie.py"),
                       "--job", str(job.resolve()), "--out", str((job_dir/"result").resolve())]
            environment = dict(os.environ)
            environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=environment, cwd=Path(__file__).resolve().parents[1])
            try:
                with (job_dir/"worker.log").open("w") as log:
                    for line in process.stdout:
                        log.write(line); log.flush()
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if progress:
                            progress(event if "stage" in event else {"stage":"saving analysis","completed":1,"total":1})
                code = process.wait()
                if code:
                    tail = (job_dir/"worker.log").read_text()[-1500:]
                    raise RuntimeError(f"Analysis worker failed ({code}): {tail}")
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=5)
                raise
            result = json.loads((job_dir/"result/analysis.json").read_text())
        else:
            result = self.service.analyze(packet["request"], review_entities=packet["entities"],
                        review_source=str(self.store.path),
                        review_tombstones=packet.get("tombstones", ()), progress=progress)
        self._validate_packet(packet, result)
        return result

    def accept(self, result, packet):
        """Persist a finished run on the main thread; never patch export rows."""
        self._validate_packet(packet, result)
        directory = self.directory/"analysis_runs"/result["cache"].get("export_key", result["cache"]["measurement_key"])[:20]
        paths = export_analysis(result, directory)
        self.store.save("session", "analysis-latest", {
            "analysis_path": paths["json"], "sha256": file_hash(paths["json"]),
            "review_signature": packet["review_signature"], "dependencies": packet["dependencies"],
            "artifacts": paths}, actor="analysis")
        self.result, self.saved_signature = result, packet["review_signature"]
        return paths

    def recompute(self, progress=None):
        packet = self.prepare()
        result = self.calculate(packet, progress)
        self.accept(result, packet)
        return result

    def export(self, directory=None):
        if self.stale:
            raise ValueError("Reviews, settings or analysis inputs changed. Recompute before exporting.")
        return export_analysis(self.result, directory or self.directory/"exports")

    def open_row(self, controller, owner_id, frame):
        controller.persist_draft()
        if self.result is None:
            raise ValueError("Run analysis first")
        row = next(r for r in self.result["rows"]
                   if r["owner_id"] == owner_id and r["source_frame"] == int(frame))
        owner = next(o for o in self.result["owners"] if o["id"] == owner_id)
        from .frame_geometry import result_owner_at_frame
        owner = result_owner_at_frame(self.result, owner, frame)
        request = self.config["request"]
        task_id = "analysis-review-" + stable_hash([request["movie_id"], owner_id, int(frame)])[:18]
        previous = self.store.load(task_id)
        task = copy.deepcopy(previous["data"]) if previous else {
            "uuid": task_id, "movie": request["movie_id"], "owner_uuid": owner_id,
            "query_frames": [int(frame)], "task_type": "review_tip", "completed": False,
            "target_xy": owner["grain_native"], "target_r": owner.get("grain_radius_px", 13),
            "focus_xy": row.get("tip_xy") or owner["grain_native"],
            "guide_path": row.get("current_path_xy") or row.get("partial_path_xy") or [],
            "analysis_measurement_key": self.result["cache"]["measurement_key"],
            "draft_xy": row.get("tip_xy"), "view_zoom": 4,
            "review_origin": "workflow_test" if self.config.get("verification_only") else "human",
        }
        # Refresh result-derived context without changing saved human marks or
        # unfinished drawing drafts. Reusing an old guide can crop a new route.
        task.update(target_xy=copy.deepcopy(owner['grain_native']),
                    target_r=owner.get('grain_radius_px', 13),
                    focus_xy=copy.deepcopy(row.get('tip_xy') or owner['grain_native']),
                    guide_path=copy.deepcopy(row.get('current_path_xy') or row.get('partial_path_xy') or []),
                    draft_xy=copy.deepcopy(row.get('tip_xy')),
                    grain_pose=copy.deepcopy(row.get('grain_pose')),
                    analysis_measurement_key=self.result['cache']['measurement_key'])
        self.store.save("task", task_id, task, actor="open-analysis-review")
        existing = next((i for i, t in enumerate(controller.tasks) if t["uuid"] == task_id), None)
        if existing is None:
            controller.tasks.append(task)
        else:
            controller.tasks[existing] = task
        controller.current = task
        self.selection = {"owner_id": owner_id, "source_frame": int(frame)}
        self.store.save("session", "analysis-selection", self.selection, actor="selection")
        controller._context_offset = 0
        controller._path_pts = []
        controller._consulted = []
        controller._restore_dots()
        if controller.viewer is not None:
            controller._show_current()
            self.show_frame(controller, int(frame))
        return task

    def start_path(self, controller):
        controller.persist_draft()
        current = controller.current
        if current is None or not current["uuid"].startswith("analysis-review-"):
            raise ValueError("Select a result row first")
        uid = current["uuid"].replace("analysis-review-", "analysis-path-")
        previous = self.store.load(uid)
        prior_observation = self.store.load("obs-" + uid)
        saved_path = (prior_observation or {}).get("data", {})
        task = copy.deepcopy(previous["data"] if previous else current)
        if not previous:
            task.pop('_drawing_draft', None)
            task.pop('_dot_history', None)
        task.update(uuid=uid,
                    task_type="centerline", completed=bool(previous and task.get("completed")),
                    why="Trace the whole currently visible tube from its grain exit to its current tip. "
                        "If any portion is hidden, save Partial. Yellow geometry is only a proposal.")
        self.store.save("task", task["uuid"], task, actor="open-path-review")
        controller.tasks = [t for t in controller.tasks if t["uuid"] != task["uuid"]] + [task]
        controller.current = task
        controller._path_pts = copy.deepcopy(saved_path.get("path_xy") or [])
        controller._dot_history = [("path", "")] * len(controller._path_pts)
        controller._context_offset = 0
        controller._consulted = list(saved_path.get("context_frames") or [])
        if task.get('_drawing_draft') is not None:
            controller.restore_draft()
        self.selection = {"kind": "path", "owner_id": task["owner_uuid"],
                          "source_frame": int(task["query_frames"][0])}
        self.store.save("session", "analysis-selection", self.selection, actor="selection")
        if controller.viewer is not None:
            controller._show_current()
        return task

    def reassign(self, controller, new_owner):
        controller.require_task_frame()
        task = controller.current
        if task is None or not task["uuid"].startswith("analysis-review-"):
            raise ValueError("Select a result row first")
        old_owner, frame = task["owner_uuid"], int(task["query_frames"][0])
        if old_owner == new_owner:
            raise ValueError("Select the grain that actually owns this tip")
        prior = self.store.load("obs-" + task["uuid"])
        point = (prior["data"].get("direct_xy") if prior else None) or next(
            r["tip_xy"] for r in self.result["rows"]
            if r["owner_id"] == old_owner and r["source_frame"] == frame)
        if point is None:
            raise ValueError("Mark the visible tip before assigning it to another grain")
        target = self.open_row(controller, new_owner, frame)
        target["reassign_from_owner"] = old_owner
        self.store.save("task", target["uuid"], target, actor=controller.actor or "owner-review")
        controller.save_apex(*point)
        return target

    def open_census(self, controller, frame=None):
        request = self.config["request"]
        frame = int(frame if frame is not None else request["frames"][0])
        if frame not in request["frames"]:
            raise ValueError("Choose a frame in the analysis interval")
        roi = request.get("roi_xyxy") or [0, 0, *controller.reader.native_size]
        x0, y0, x1, y1 = roi
        uid = "analysis-census-" + stable_hash([str(self.store.path), request["movie_id"], roi, frame])[:18]
        existing = self.store.load(uid)
        if existing:
            task = copy.deepcopy(existing["data"])
        else:
            owners = (self.result or {}).get("owners", request.get("owners", []))
            if self.result and self.result.get('grain_geometry'):
                from .frame_geometry import result_owner_at_frame
                owners = [result_owner_at_frame(self.result, o, frame) for o in owners]
            instances = [{"grain_id": o["id"], "xy": o["grain_native"], "confirmed": False}
                         for o in owners if x0 <= o["grain_native"][0] < x1
                         and y0 <= o["grain_native"][1] < y1]
            task = {"uuid": uid, "task_type": "census", "analysis_census": True,
                "movie": request["movie_id"], "query_frames": [frame],
                "focus_xy": [(x0+x1)/2, (y0+y1)/2], "view_zoom": 1.5,
                "review_region": [[x0,y0], [x1,y0], [x1,y1], [x0,y1]],
                "census_instances": instances, "census_grains": [g["xy"] for g in instances],
                "census_class_scopes": ["grains"], "census_border_policy": "centre-inside",
                "census_clump_policy": "individual-physical-grains",
                "census_membership_resolved": False, "census_complete": False, "completed": False,
                "review_origin": "workflow_test" if self.config.get("verification_only") else "human",
                "why": "Review every physical grain whose centre is inside the yellow rectangle. "
                       "Confirm or remove each proposal; click any missed grain. Separate touching grains."}
            self.store.save("task", uid, task, actor="open-census-review")
        config = copy.deepcopy(self.config)
        config["request"]["census_scope"] = {"movie": request["movie_id"],
            "roi_xywh": [x0,y0,x1-x0,y1-y0], "frames": [frame], "class_scopes": ["grains"]}
        self.configure(config)
        controller.tasks = [t for t in controller.tasks if t["uuid"] != uid] + [task]
        controller.current = task
        controller._context_offset = 0
        controller._census_mode = "grain"
        controller._dot_history = []
        self.selection = {"kind": "census", "source_frame": frame}
        self.store.save("session", "analysis-selection", self.selection, actor="analysis-selection")
        if controller.viewer is not None:
            controller._show_current()
            controller.draw_task_markup()
        return task

    def show_frame(self, controller, frame):
        viewer = controller.viewer
        if viewer is None or self.result is None:
            return
        for layer in self._layers:
            if layer in viewer.layers:
                viewer.layers.remove(layer)
        self._layers = []
        task = controller.current or {}
        if task.get("analysis_census") or task.get("suppress_analysis_overlays"):
            return
        if (task.get("movie") or task.get("movie_uuid") or self.config["request"]["movie_id"]) != self.result["movie_id"]:
            return
        owners = {o["id"]: o for o in self.result["owners"]}
        palette = ["cyan", "magenta", "orange", "lime", "deepskyblue", "violet"]
        for n, oid in enumerate(sorted(owners)):
            color = palette[n % len(palette)]
            row = next((r for r in self.result["rows"] if r["owner_id"] == oid and r["source_frame"] == frame), None)
            if row is None:
                continue
            from .frame_geometry import result_owner_at_frame
            current_owner = result_owner_at_frame(self.result, owners[oid], frame)
            grain = current_owner["grain_native"]
            uncertain = bool(current_owner.get('grain_pose')
                             and not current_owner['grain_pose'].get('usable_for_model_geometry'))
            selected = (controller.current or {}).get("owner_uuid") == oid
            self._layers.append(viewer.add_points([[grain[1], grain[0]]],
                name=f"analysis-{oid}-grain", size=2*current_owner.get("grain_radius_px", 13),
                face_color="transparent", border_color="white" if selected else 'gray' if uncertain else color,
                blending="translucent_no_depth",
                text={"string": [oid.rsplit('-', 1)[-1] + (' ?' if uncertain else '')],
                      "color": "white" if selected else color, "size": 10,
                      "translation": [-18, 0], "blending": "translucent_no_depth"}))
            path = row.get("current_path_xy") or row.get("partial_path_xy") or []
            if len(path) >= 2:
                self._layers.append(viewer.add_shapes([np.asarray(path)[:, ::-1]],
                    name=f"analysis-{oid}-{'full' if row['path_complete'] else 'partial'}",
                    shape_type="path", edge_color=color, edge_width=2,
                    opacity=.8 if row["path_complete"] else .45, blending="translucent_no_depth"))
            xy = row.get("tip_xy")
            if xy:
                layer = viewer.add_points([[xy[1], xy[0]]], name=f"analysis-{oid}-tip",
                    size=10, face_color="transparent", border_color=color, blending="translucent_no_depth")
                layer.editable = False
                self._layers.append(layer)
        # Drawing remains attached to the raw image, not an inference overlay.
        for layer in self._layers:
            layer.editable = False
        if controller._image_layer is not None:
            viewer.layers.selection.active = controller._image_layer

    def preview(self, controller):
        """Load real pixels before any task or result exists."""
        if controller.viewer is None:
            return
        frame = int(self.config["request"]["frames"][0])
        data = controller.reader.read(frame).frame
        if controller._image_layer is None:
            controller._image_layer = controller.viewer.add_image(data, name="raw", rgb=data.ndim == 3)
        else:
            controller._image_layer.data = data
        controller.viewer.reset_view()
