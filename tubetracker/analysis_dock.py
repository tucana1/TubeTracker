"""Movie analysis controls; inference runs outside the GUI thread."""
from __future__ import annotations

import copy
import traceback

from qtpy.QtCore import QThread, Signal, QTimer
from qtpy.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFormLayout, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QToolButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QInputDialog, QListWidget, QTabWidget)


def grain_summary_text(row):
    """Describe bounds and measurements without turning proposals into truth."""
    gid = row['grain_id']
    label = gid if len(gid) <= 28 else '…' + gid[-12:]
    lines = [f"{label} · ({row['grain_x']:.0f}, {row['grain_y']:.0f})"]
    after, by = row.get('onset_after_frame'), row.get('onset_by_frame')
    if row.get('timing_status') == 'contradictory':
        timing = 'conflicting reviews'
    elif after is not None and by is not None:
        timing = f'after frame {after}, by {by}'
    elif by is not None:
        timing = f'by frame {by}; start not observed'
    elif after is not None:
        timing = f'after frame {after}; not yet observed'
    else:
        timing = 'not determined'
    lines.append('Visible emergence: ' + timing)
    a, b = row.get('onset_after_time_s'), row.get('onset_by_time_s')
    if a is not None or b is not None:
        lines.append('Acquisition time: ' + ', '.join(
            text for text in (f'after {a:g} s' if a is not None else '',
                             f'by {b:g} s' if b is not None else '') if text))
    length = row.get('last_complete_length_px')
    if length is not None:
        physical = row.get('last_complete_length_um')
        units = f'{physical:.2f} µm' if physical is not None else f'{length:.2f} px'
        source = 'reviewed' if row.get('last_complete_length_provenance') == 'human-corrected' else 'model'
        lines.append(f"Last complete length: {units} at frame {row['last_complete_length_frame']} · {source}")
    else:
        lines.append('Complete length: withheld')
    estimate = row.get('model_last_supported_path_length_px')
    if estimate is not None:
        lines.append(f"Provisional model path: {estimate:.2f} px at frame {row['model_last_supported_path_frame']}")
    if row.get('biological_class'):
        lines.append('Reviewed class: ' + row['biological_class'].replace('_', ' '))
    return '\n'.join(lines)


class AnalysisWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(object)

    def __init__(self, session, packet):
        super().__init__()
        self.session, self.packet = session, packet

    def run(self):
        def progress(event):
            if self.isInterruptionRequested():
                raise InterruptedError("Analysis stopped")
            self.progress.emit(event)
        try:
            self.completed.emit(self.session.calculate(self.packet, progress))
        except Exception as error:
            print(traceback.format_exc(), flush=True)
            self.failed.emit(f"{type(error).__name__}: {error}")


class AnalysisReviewPanel(QWidget):
    """Compact native-analysis review controls over the same annotation store."""
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.container = None
        self._review_context = None
        layout = QVBoxLayout(self)
        self.info = QLabel("Select a result row to review.")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.context = QLabel()
        layout.addWidget(self.context)
        navigation = QHBoxLayout()
        for label, delta in (("−30", -30), ("−1", -1), ("Task frame", 0), ("+1", 1), ("+30", 30)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, d=delta: self.step(d))
            navigation.addWidget(button)
        layout.addLayout(navigation)
        self.look = QCheckBox("Look only — clicking does not mark")
        self.look.setToolTip("Result inspection starts in Look only. Uncheck to edit: an image click then saves a tip immediately.")
        self.look.setChecked(True)
        controller._look_only = True
        self.look.toggled.connect(self.look_changed)
        layout.addWidget(self.look)
        self.actions = QWidget()
        grid = QGridLayout(self.actions)
        self.tip_mode = QCheckBox("Mark a tip region instead of a point")
        self.tip_mode.toggled.connect(self.region_mode)
        grid.addWidget(self.tip_mode, 0, 0, 1, 2)
        actions = [
            ("Accept proposed tip", self.accept_tip),
            ("Save tip region", lambda: controller.save_region(controller._region_pts)),
            ("Tip hidden", lambda: controller.save_state("not_directly_visible")),
            ("No owned tube visible", lambda: controller.save_state("no_tube_visible")),
            ("Owner uncertain", lambda: controller.save_state("owner_uncertain")),
            ("Out of field", lambda: controller.save_state("out_of_field")),
            ("Save FULL current path", lambda: controller.save_path(True)),
            ("Save PARTIAL path", lambda: controller.save_path(False)),
        ]
        self.action_buttons = []
        for n, (label, action) in enumerate(actions):
            button = QPushButton(label)
            if label == "Save tip region":
                button.setToolTip("Outline a tip area with at least three vertices in region mode, then save. Point clicks are saved immediately and need no extra save.")
            button.clicked.connect(lambda _=False, fn=action: self.save(fn))
            grid.addWidget(button, 1+n//2, n%2)
            self.action_buttons.append(button)
        layout.addWidget(self.actions)
        self.census_actions = QWidget()
        census_layout = QVBoxLayout(self.census_actions)
        self.census_table = QListWidget()
        self.census_table.setAccessibleName("Grains in the census")
        self.census_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.census_table.setMaximumHeight(180)
        self.census_table.itemSelectionChanged.connect(self.highlight_census)
        census_layout.addWidget(self.census_table)
        census_navigation = QHBoxLayout()
        for label, delta in (("Previous grain", -1), ("Next grain", 1)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, d=delta: self.move_census(d))
            census_navigation.addWidget(button)
        census_layout.addLayout(census_navigation)
        edit_row = QHBoxLayout()
        for label, remove in (("Confirm grain", False), ("Remove false proposal", True)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, r=remove: self.edit_census(r))
            edit_row.addWidget(button)
        census_layout.addLayout(edit_row)
        ruling = QPushButton("Record biological ruling for selected grain")
        ruling.clicked.connect(self.classify_census)
        census_layout.addWidget(ruling)
        self.census_checked = QCheckBox("I checked the whole field, including every touching grain")
        self.census_checked.setToolTip("Count individual physical grains. Grain centres on the top/left edge are included; bottom/right are excluded.")
        self.census_checked.toggled.connect(self.check_census)
        census_layout.addWidget(self.census_checked)
        finish = QPushButton("Save exhaustive grain census")
        finish.clicked.connect(lambda: self.save(controller.finish_census))
        census_layout.addWidget(finish)
        layout.addWidget(self.census_actions)
        self.event_actions = QWidget()
        event_layout = QVBoxLayout(self.event_actions)
        event_nav = QHBoxLayout()
        for label, slot in (("Mark displayed as last smooth rim", "last_absent_candidate"),
                            ("Mark displayed as first bump/outgrowth", "first_visible_candidate")):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _=False, s=slot: self.mark_event_bracket(s))
            event_nav.addWidget(button)
        event_layout.addLayout(event_nav)
        for label, verdict in (("Emerged within bracket", "emerged_within"),
                               ("Already emerged at start", "emerged_at_start"),
                               ("No emergence by end", "no_emergence_by_end"),
                               ("Unobservable", "unobservable")):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _=False, v=verdict: self.save(lambda: controller.save_germination_event(
                    v, controller.current.get("last_absent_candidate"),
                    controller.current.get("first_visible_candidate"))))
            event_layout.addWidget(button)
        clear_bracket = QPushButton("Clear bracket marks")
        clear_bracket.clicked.connect(self.clear_event_bracket)
        event_layout.addWidget(clear_bracket)
        layout.addWidget(self.event_actions)
        # Verdict block first: in event mode this is the whole decision, and
        # Josh should never have to hunt below the fold for it.
        layout.removeWidget(self.event_actions)
        layout.insertWidget(0, self.event_actions)
        undo_saved = QPushButton("Undo saved correction")
        undo_saved.setToolTip("Undo the most recent saved tip point, region, visibility state or path for this selection. Restores the prior accepted revision when one exists.")
        undo_saved.clicked.connect(self.undo_saved)
        layout.addWidget(undo_saved)
        undo = QPushButton("Undo draft point")
        undo.setToolTip("Remove the newest draft vertex (path or region). Draft undo never touches saved evidence.")
        undo.clicked.connect(self.undo)
        layout.addWidget(undo)
        self.message = QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        layout.addStretch(1)
        self.setMinimumWidth(290)
        self.setMaximumWidth(370)
        self.refresh()

    def refresh(self):
        c, task = self.controller, self.controller.current
        if task is None:
            self._review_context = None
            self.info.setText("Select a result row to review.")
            self.actions.setEnabled(False)
            self.census_actions.hide()
            return
        # rev14 P1: one authoritative edit mode. Result inspection defaults
        # to Look only; the checkbox always mirrors the controller flag and
        # display updates never emit toggles (signals blocked).
        from .analysis_project import QUEUE
        context = (task.get('uuid'), task.get('movie') or task.get('movie_uuid'),
                   task.get('owner_uuid'), tuple(task.get('query_frames') or []), task.get('task_type'))
        if task.get('review_queue') != QUEUE and context != self._review_context:
            c._look_only = True
        self._review_context = context
        self.look.blockSignals(True)
        self.look.setChecked(bool(getattr(c, '_look_only', True)))
        self.look.blockSignals(False)
        frame = int(task["query_frames"][0])
        census_mode = bool(task.get("analysis_census"))
        event_mode = str(task.get("task_type", "")) == "germination_event"
        self.actions.setVisible(not census_mode and not event_mode)
        self.census_actions.setVisible(census_mode)
        self.event_actions.setVisible(event_mode)
        self.event_actions.setEnabled(c._context_offset == 0)
        if event_mode:
            self.show_event_task(task, frame)
            return
        self.context.setText(f"Displayed frame: {frame+c._context_offset}" +
            (" · context only; return to Task frame to mark" if c._context_offset else ""))
        if census_mode:
            self.info.setText(f"Grain census · frame {frame}\n{task['why']}")
            self.census_actions.setEnabled(c._context_offset == 0)
            selected = max(0, self.census_table.currentRow())
            self.census_table.blockSignals(True)
            items = task.get("census_instances", [])
            self.census_table.clear()
            for i, item in enumerate(items):
                values = [str(i+1), "Yes" if item.get("confirmed") else "Proposed",
                          (item.get("classification") or {}).get("class", "Unclassified")]
                self.census_table.addItem(f"Grain {values[0]} · {'Reviewed' if item.get('confirmed') else 'Proposed'} · {values[2]}")
                self.census_table.item(i).setToolTip(item["grain_id"])
            if items:
                self.census_table.setCurrentRow(min(selected, len(items)-1))
            self.census_table.blockSignals(False)
            self.census_checked.blockSignals(True)
            self.census_checked.setChecked(bool(task.get("census_membership_resolved")))
            self.census_checked.blockSignals(False)
            self.message.setText("Census withdrawn. Review the stored marks and confirm the whole field to restore it."
                if task.get('review_status') == 'withdrawn' else
                "Exhaustive census saved. Recompute to apply it." if task.get("census_complete")
                else f"{sum(bool(g.get('confirmed')) for g in items)}/{len(items)} grains reviewed. Click missed grain centres in the field.")
            c.draw_task_markup()
            self.highlight_census()
            return
        path_mode = task.get("task_type") == "centerline"
        instruction = ("Click the actual grain exit, then follow the current tube to its tip. "
                       "Save PARTIAL if any portion cannot be followed." if path_mode else
                       "Outline the visible tip area with at least three vertices, then Save tip region. "
                       "This records an imprecise visible tip, without a precise point." if c._region_mode else
                       "Click the current visible tip to save a precise correction. "
                       "Use a visibility state when no precise tip is supported.")
        self.info.setText(f"{task.get('owner_uuid', '')} · frame {frame}\n{instruction}")
        self.context.setText(f"Displayed frame: {frame+c._context_offset}" +
            (" · context only; return to Task frame to mark" if c._context_offset else ""))
        self.actions.setEnabled(c._context_offset == 0)
        self.tip_mode.setEnabled(not path_mode)
        for n, button in enumerate(self.action_buttons):
            button.setVisible(n >= 6 if path_mode else n < 6)
        self.action_buttons[1].setEnabled(not path_mode and c._region_mode and len(c._region_pts) >= 3)
        self.tip_mode.blockSignals(True)
        self.tip_mode.setChecked(c._region_mode)
        self.tip_mode.blockSignals(False)
        self.message.setToolTip('')
        prior = c.store.load("obs-" + task["uuid"])
        if prior:
            data = prior["data"]
            self.message.setText(("Saved " + ("FULL" if data.get("path_complete") else "PARTIAL") +
                f" path ({len(data.get('path_xy') or [])} points)." if path_mode else
                "Saved review: " + data["direct_state"].replace("_", " ") + ".") +
                " Recompute to apply it to tracking.")
        elif self.show_applied_review(task, frame):
            pass
        elif path_mode:
            self.message.setText(f"{len(c._path_pts)} path points; no path saved yet.")
        else:
            self.message.setText("No local correction saved for this selection.")

    def show_applied_review(self, task, frame):
        session = getattr(self.controller, 'analysis_session', None)
        result = getattr(session, 'result', None) or {}
        movie = task.get('movie') or task.get('movie_uuid')
        if movie != result.get('movie_id'):
            return False
        row = next((r for r in result.get('rows', [])
                    if r.get('owner_id') == task.get('owner_uuid')
                    and r.get('source_frame') == frame), None)
        if not row or row.get('provenance') != 'human-corrected':
            return False
        constraint = row.get('constraint') or {}
        origin = constraint.get('review_origin')
        reviewer = 'human reviews' if origin == 'human' else 'workflow-test reviews' if origin == 'workflow_test' else 'recorded reviews'
        if row.get('path_complete') and row.get('length_px') is not None:
            detail = f"FULL path, length {row['length_px']:.2f} px"
        elif row.get('current_path_xy'):
            detail = 'PARTIAL path, length withheld'
        else:
            detail = str(constraint.get('state') or row['state']).replace('_', ' ')
        self.message.setText(f"Last computed result uses {reviewer}: {detail}. No additional edit saved here.")
        self.message.setToolTip('\n'.join(
            f"{source.get('id', '')} · revision {source.get('revision', '?')}"
            for source in constraint.get('lineage', [])))
        return True

    def save(self, action):
        try:
            if self.controller._context_offset:
                raise ValueError("Return to the task frame before saving a review")
            action()
            task = self.controller.current
            task["completed"] = True
            self.controller.store.save("task", task["uuid"], task, actor=self.controller.actor)
            if callable(getattr(self.controller, 'on_draft_changed', None)):
                self.controller.on_draft_changed()
            self.refresh()
        except Exception as error:
            self.message.setText(str(error))

    def accept_tip(self):
        point = self.controller.current.get("draft_xy")
        if not point:
            raise ValueError("There is no proposed point. Mark the visible tip or its visibility state.")
        self.controller.save_apex(*point)

    def region_mode(self, checked):
        self.controller._region_mode = checked
        self.controller._region_pts = []
        self.refresh()

    def step(self, delta):
        if delta:
            self.controller.step_frame(delta)
        else:
            self.controller.back_to_task_frame()
        self.refresh()

    def look_changed(self, checked):
        self.controller._look_only = checked
        self.refresh()
        if checked:
            message = "Look only — image clicks do not write."
        elif (self.controller.current or {}).get('task_type') == 'centerline':
            message = "Editing enabled — trace the tube, then Save FULL or Save PARTIAL."
        elif self.controller._region_mode:
            message = "Editing enabled — outline the tip area, then Save tip region."
        else:
            message = "Editing enabled — an image click now saves a tip immediately."
        self.message.setText(message)

    def undo(self):
        try:
            removed = self.controller.undo_last_dot()
            self.refresh()
            if not removed:
                self.message.setText(
                    "No draft point to undo. This button removes draft vertices only; "
                    "use 'Undo saved correction' for saved evidence.")
        except ValueError as error:
            self.refresh()
            self.message.setText(str(error))

    def show_event_task(self, task, frame):
        """Verdict-task display: bracket marks, consulted trail, saved answer."""
        c = self.controller
        lo = task.get("last_absent_candidate")
        hi = task.get("first_visible_candidate")
        self.info.setText(
            f"{task.get('owner_uuid', '')} · frames {task.get('source_start')}"
            f"–{task.get('source_end')}\nScrub with the navigation row (coarse "
            "first, refine around first emergence). Mark the last clearly "
            "absent frame and the first clearly visible owned outgrowth, "
            "then save one verdict. Blur/burst/disappearance can make "
            "visibility non-monotonic: review the intervening sequence, "
            "never paint a later tube backward.")
        prior = c.store.load("event-" + task["uuid"])
        if prior:
            data = prior["data"]
            self.message.setText(
                f"Saved verdict: {data.get('verdict', '?').replace('_', ' ')} "
                f"[{data.get('last_absent_frame')},{data.get('first_visible_frame')}] "
                f"({len(data.get('consulted_frames') or [])} consulted frames). "
                "Recompute to apply it to tracking.")
        else:
            self.message.setText(
                f"Bracket: last-absent={lo}, first-visible={hi}. "
                f"Consulted {len(getattr(c, '_consulted', None) or [])} frames. "
                "No verdict saved for this episode yet.")

    def mark_event_bracket(self, slot):
        c = self.controller
        if not c.current or str(c.current.get("task_type")) != "germination_event":
            return
        frame = int(c.current["query_frames"][0]) + int(getattr(c, "_context_offset", 0))
        c.current[slot] = frame
        c.persist_draft()
        self.refresh()

    def clear_event_bracket(self):
        c = self.controller
        if not c.current:
            return
        c.current.pop("last_absent_candidate", None)
        c.current.pop("first_visible_candidate", None)
        c.persist_draft()
        self.refresh()

    def undo_saved(self):
        try:
            note = self.controller.undo_saved_correction()
            self.refresh()
            self.message.setText(note)
        except ValueError as error:
            self.refresh()
            self.message.setText(str(error))

    def highlight_census(self):
        c = self.controller
        if c.viewer is None or not (c.current or {}).get("analysis_census"):
            return
        items = c.current.get("census_instances", [])
        if not items or "census-balls" not in c.viewer.layers:
            return
        layer = c.viewer.layers["census-balls"]
        layer.size = 24
        layer.border_color = ["white" if i == self.census_table.currentRow() else
            "lime" if g.get("confirmed") else "orange" for i, g in enumerate(items)]
        layer.text = {"string": [str(i+1) for i in range(len(items))], "color": "white",
                      "size": 11, "translation": [-17, 0], "blending": "translucent_no_depth"}

    def edit_census(self, remove):
        try:
            selected = self.census_table.currentRow()
            self.controller.edit_census_grain(selected, remove=remove)
            self.refresh()
            if not remove:
                pending = [i for i, g in enumerate(self.controller.current.get("census_instances", []))
                           if not g.get("confirmed")]
                if pending:
                    self.census_table.setCurrentRow(next((i for i in pending if i > selected), pending[0]))
        except ValueError as error:
            self.message.setText(str(error))

    def move_census(self, delta):
        if self.census_table.count():
            self.census_table.setCurrentRow(min(max(0, self.census_table.currentRow()+delta),
                                                 self.census_table.count()-1))

    def check_census(self, checked):
        c = self.controller
        if not (c.current or {}).get("analysis_census"):
            return
        c.require_task_frame()
        c.current["census_membership_resolved"] = bool(checked)
        if not checked:
            c.current["census_complete"] = False
        c.store.save("task", c.current["uuid"], c.current, actor=c.actor)

    def classify_census(self):
        try:
            c = self.controller
            c.require_task_frame()
            index = self.census_table.currentRow()
            if index < 0:
                raise ValueError("Select a grain row")
            choices = ["germinated", "nongerminating", "unresolved"]
            cls, ok = QInputDialog.getItem(self, "Biological ruling", "Class for this individual grain:", choices, 2, False)
            if not ok:
                return
            rule, ok = QInputDialog.getMultiLineText(self, "Rule and evidence",
                "State the biological rule and the evidence you reviewed. A hidden tip or one no-tube frame does not establish nongermination.")
            if not ok:
                return
            frames, ok = QInputDialog.getText(self, "Reviewed frames",
                "Source frames actually reviewed, separated by commas:", text=str(c.current["query_frames"][0]))
            if not ok:
                return
            c.record_census_ruling(index, cls, rule, [int(f.strip()) for f in frames.split(",")])
            self.refresh()
        except ValueError as error:
            self.message.setText(str(error))


class AnalysisDock(QWidget):
    def __init__(self, controller, task_dock):
        super().__init__()
        self.controller = controller
        self.session = controller.analysis_session
        self.task_dock = task_dock
        self.worker = None
        layout = QVBoxLayout(self)
        intro = QLabel("Analyze the selected movie interval, then select a result to review its tip, owner or full tube path.")
        if self.session.config.get("verification_only"):
            intro.setText("Workflow verification project. These edits are excluded from biological results.")
        elif self.session.config.get("experimental_note"):
            intro.setText(self.session.config["experimental_note"])
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.settings = QWidget()
        form = QFormLayout(self.settings)
        request = self.session.config["request"]
        self.first, self.last, self.stride = QSpinBox(), QSpinBox(), QSpinBox()
        for spin in (self.first, self.last, self.stride):
            spin.setRange(0, max(1, len(controller.reader)-1))
            # Source-frame numbers need their full width; frame stepping is
            # already available beside the image in the review panel.
            spin.setButtonSymbols(spin.ButtonSymbols.NoButtons)
            spin.setMinimumWidth(spin.fontMetrics().horizontalAdvance(str(len(controller.reader))) + 30)
            spin.setKeyboardTracking(False)
        self.stride.setMinimum(1)
        frames = request["frames"]
        self.first.setValue(frames[0])
        self.last.setValue(frames[-1])
        self.stride.setValue(frames[1]-frames[0] if len(frames)>1 else 30)
        self._loaded_frames = tuple(frames)
        self._loaded_frame_controls = (self.first.value(), self.last.value(), self.stride.value())
        interval = QGridLayout()
        for col, (label, control) in enumerate((("First frame", self.first), ("Last frame", self.last), ("Step", self.stride))):
            control.setAccessibleName(label)
            interval.addWidget(QLabel(label), 0, col)
            interval.addWidget(control, 1, col)
        form.addRow(interval)
        self.sampling_summary = QLabel()
        form.addRow(self.sampling_summary)
        for spin in (self.first, self.last, self.stride):
            spin.valueChanged.connect(self.refresh_sampling_summary)
        self.refresh_sampling_summary()
        self.roi = QLineEdit(",".join(map(str, request.get("roi_xyxy") or [])))
        self.roi.setPlaceholderText("Whole frame, or x0,y0,x1,y1")
        form.addRow("Field rectangle", self.roi)
        self.discover = QCheckBox("Find additional grains in this field")
        self.discover.setChecked(request.get("discover_grains", False))
        form.addRow(self.discover)
        self.grain_backend = QComboBox()
        self.grain_backend.addItem("Detailed grains (slower)", 'cpdino')
        self.grain_backend.addItem("Fast grains", 'radial')
        self.grain_backend.setCurrentIndex(self.grain_backend.findData(
            request.get('grain_detection', {}).get('backend', 'radial')))
        self.grain_backend.setEnabled(self.discover.isChecked())
        self.discover.toggled.connect(self.grain_backend.setEnabled)
        form.addRow("Grain detection", self.grain_backend)
        acq = request.get("acquisition", {})
        self.cadence = QLineEdit(str(acq.get("seconds_per_source_frame", "")))
        self.cadence_source = QLineEdit(acq.get("cadence_source", ""))
        self.scale = QLineEdit(str(acq.get("micrometres_per_pixel", "")))
        self.scale_source = QLineEdit(acq.get("calibration_source", ""))
        self.cadence.setPlaceholderText("Optional; actual acquisition interval")
        self.cadence_source.setPlaceholderText("Experiment log / timestamp source")
        self.scale.setPlaceholderText("Optional")
        self.scale_source.setPlaceholderText("Calibration source")
        self.metadata = QWidget()
        metadata_form = QFormLayout(self.metadata)
        metadata_form.addRow("Seconds/frame", self.cadence)
        metadata_form.addRow("Timing source", self.cadence_source)
        metadata_form.addRow("µm/pixel", self.scale)
        metadata_form.addRow("Scale source", self.scale_source)
        metadata_toggle = QToolButton()
        metadata_toggle.setText("Timing and scale (optional)")
        metadata_toggle.setCheckable(True)
        metadata_toggle.toggled.connect(self.metadata.setVisible)
        form.addRow(metadata_toggle)
        form.addRow(self.metadata)
        self.metadata.hide()
        layout.addWidget(self.settings)
        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run / Recompute")
        self.run_button.clicked.connect(self.start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.export_button = QPushButton("Export results")
        self.export_button.clicked.connect(self.export)
        for button in (self.run_button, self.stop_button, self.export_button):
            run_row.addWidget(button)
        layout.addLayout(run_row)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.population_status = QLabel("Run analysis to see grain-level results.")
        self.population_status.setWordWrap(True)
        layout.addWidget(self.population_status)
        self.result_tabs = QTabWidget()
        self.grain_table = QListWidget()
        self.grain_table.setAccessibleName("Grains, emergence bounds and tube lengths")
        self.grain_table.setWordWrap(True)
        self.grain_table.itemSelectionChanged.connect(self.grain_selected)
        self.result_tabs.addTab(self.grain_table, "Grains & timing")
        self.table = QListWidget()
        self.table.setAccessibleName("Tube results by grain and source frame")
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        self.table.setMinimumHeight(140)
        self.result_tabs.addTab(self.table, "Frame results")
        layout.addWidget(self.result_tabs)
        result_navigation = QHBoxLayout()
        self.previous_result = QPushButton("Previous result")
        self.next_result = QPushButton("Next result")
        self.previous_result.clicked.connect(lambda: self.move_result(-1))
        self.next_result.clicked.connect(lambda: self.move_result(1))
        result_navigation.addWidget(self.previous_result)
        result_navigation.addWidget(self.next_result)
        layout.addLayout(result_navigation)
        self.owner = QComboBox()
        self.owner_label = QLabel("Owner from the selected result row.")
        self.owner_label.setWordWrap(True)
        layout.addWidget(self.owner_label)
        reassign = QHBoxLayout()
        reassign.addWidget(self.owner)
        self.reassign_button = QPushButton("Assign tip to grain")
        self.reassign_button.clicked.connect(self.reassign)
        reassign.addWidget(self.reassign_button)
        layout.addLayout(reassign)
        self.distinct = QCheckBox("This is a distinct cap overlapping another cap")
        self.distinct.setToolTip("Use only when separate tube continuations establish two distinct visible caps.")
        self.distinct.toggled.connect(self.set_distinct)
        layout.addWidget(self.distinct)
        paths = QHBoxLayout()
        self.path_button = QPushButton("Trace whole tube")
        self.path_button.clicked.connect(self.start_path)
        self.partial_button = QPushButton("Save partial trace")
        self.partial_button.clicked.connect(self.save_partial)
        paths.addWidget(self.path_button)
        paths.addWidget(self.partial_button)
        layout.addLayout(paths)
        self.census_button = QPushButton("Review every grain in this field")
        self.census_button.clicked.connect(self.start_census)
        layout.addWidget(self.census_button)
        if hasattr(task_dock, 'open_queue'):
            self.queue_button = QPushButton('Open requested annotations')
            self.queue_button.clicked.connect(task_dock.open_queue)
            layout.addWidget(self.queue_button)
        self.setMinimumWidth(345)
        self.setMaximumWidth(440)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_stale)
        self.timer.start(500)
        QApplication.instance().aboutToQuit.connect(self.shutdown)
        self.populate()

    def requested_frames(self):
        controls = (self.first.value(), self.last.value(), self.stride.value())
        if controls == self._loaded_frame_controls:
            return list(self._loaded_frames)
        first, last, stride = controls
        if last < first:
            return []
        return sorted(set(range(first, last + 1, stride)) | {last})

    def refresh_sampling_summary(self):
        self.sampling_summary.setText(f"{len(self.requested_frames())} source frames")

    def settings_config(self):
        config = copy.deepcopy(self.session.config)
        if self.last.value() < self.first.value():
            raise ValueError("Last source frame must follow the first")
        request = config["request"]
        request["frames"] = self.requested_frames()
        request["roi_xyxy"] = [int(v.strip()) for v in self.roi.text().split(",")] if self.roi.text().strip() else None
        request["discover_grains"] = self.discover.isChecked()
        old_backend = request.get('grain_detection', {}).get('backend', 'radial')
        if self.grain_backend.currentData() != old_backend:
            request['grain_detection'] = dict(request.get('grain_detection', {}),
                                             backend=self.grain_backend.currentData())
        acq = dict(request.get("acquisition", {}))
        for field, source, value, note in (
            ("seconds_per_source_frame", "cadence_source", self.cadence, self.cadence_source),
            ("micrometres_per_pixel", "calibration_source", self.scale, self.scale_source)):
            if value.text().strip():
                acq[field], acq[source] = float(value.text()), note.text().strip()
            else:
                acq.pop(field, None)
                if field != "seconds_per_source_frame" or not acq.get("source_times_s"):
                    acq.pop(source, None)
        request["acquisition"] = acq
        return config

    def start(self):
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            self.session.configure(self.settings_config())
            self.packet = self.session.prepare()
            self.worker = AnalysisWorker(self.session, self.packet)
            self.worker.progress.connect(self.on_progress)
            self.worker.completed.connect(self.on_completed)
            self.worker.failed.connect(self.on_failed)
            self.worker.finished.connect(lambda: self.set_busy(False))
            self.set_busy(True)
            self.status.setText("Starting analysis…")
            self.worker.start()
        except Exception as error:
            self.on_failed(str(error))

    def set_busy(self, busy):
        for widget in (self.settings, self.result_tabs, self.owner, self.reassign_button,
                       self.path_button, self.partial_button, self.distinct, self.census_button,
                       self.previous_result, self.next_result, self.task_dock):
            widget.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.export_button.setEnabled(not busy and not self.session.stale)

    def on_progress(self, item):
        self.status.setText(f"{item['stage'].capitalize()} — {item.get('completed', 0)}/{item.get('total', 0)}")

    def on_completed(self, result):
        try:
            current = self.controller.current or {}
            selected = (current.get("owner_uuid"), (current.get("query_frames") or [None])[0])
            path_selected = current.get("task_type") == "centerline"
            self.session.accept(result, self.packet)
            self.populate()
            if self.rows:
                self.select_row(next((i for i, r in enumerate(self.rows)
                    if (r["owner_id"], r["source_frame"]) == selected), 0))
                if path_selected:
                    self.start_path()
            self.status.setText(f"Analysis saved: {len(result['rows'])} grain/frame observations in {result['elapsed_wall_s']:.1f} s.")
        except Exception as error:
            self.on_failed(str(error))

    def on_failed(self, message):
        self.status.setText(message)

    def stop(self):
        if self.worker:
            self.worker.requestInterruption()
            self.status.setText("Stopping after the current operation…")

    def shutdown(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(30000)

    def populate(self):
        self.rows = self.session.result["rows"] if self.session.result else []
        population = (self.session.result or {}).get('population') or {}
        self.grain_rows = population.get('grain_summaries') or []
        self.grain_table.blockSignals(True)
        self.grain_table.clear()
        for i, grain in enumerate(self.grain_rows):
            self.grain_table.addItem(grain_summary_text(grain))
            self.grain_table.item(i).setToolTip(
                grain['grain_id'] + '\n' + grain.get('model_timing_note', '') + '\n' +
                'Selecting a grain opens its source-frame result for inspection.')
        self.grain_table.blockSignals(False)
        census = population.get('census') or {}
        if self.grain_rows:
            coverage = 'reviewed census for the selected scope' if census.get('census_completeness_certified') else 'inventory may be incomplete'
            assistance = population.get('model_body_assistance')
            assistance_note = (
                f"\nTracking uses reviewed tube masks for {len(assistance['seeds'])} grains; propagated shapes are model estimates."
                if assistance else '')
            geometry = (self.session.result or {}).get('grain_geometry')
            pose_note = ''
            if geometry:
                unresolved = sum(not p['usable_for_model_geometry'] for p in geometry['poses'])
                pose_note = ('\nGrain positions follow the movie as model estimates.' +
                             (f' {unresolved} grain/frame positions need review.' if unresolved else ''))
            self.population_status.setText(
                f"{len(self.grain_rows)} grains · {coverage}.\nEmergence bounds use reviews; model path estimates are provisional."
                + assistance_note + pose_note)
        else:
            self.population_status.setText("Run / Recompute to see grain-level results.")
        self.table.blockSignals(True)
        self.table.clear()
        for i, row in enumerate(self.rows):
            values = [str(row["source_frame"]), row["owner_id"], row["state"].replace("_", " "),
                      f"{row['length_px']:.2f} px" if row.get("length_px") is not None else "withheld"]
            self.table.addItem(f"{values[0]} · {values[1]}\n{values[2]} · length {values[3]}")
            provider = (row.get('route_evidence') or {}).get('body_provider') or {}
            assistance_note = (f"\nModel body guided by a reviewed mask at source frame {provider['seed']['frame']}."
                               if provider.get('backend') == 'sam2_video' else '')
            pose = row.get('grain_pose')
            pose_note = (f"\nGrain position: {pose['pose_status']}; identity: {pose['identity_status']} (model estimate)."
                         if pose else '')
            self.table.item(i).setToolTip(" · ".join(values) + assistance_note + pose_note)
        self.table.blockSignals(False)
        self.owner.clear()
        for owner in (self.session.result or {}).get("owners", []):
            self.owner.addItem(owner["id"])
        self.refresh_stale()

    def grain_selected(self):
        index = self.grain_table.currentRow()
        if not 0 <= index < len(self.grain_rows):
            return
        grain = self.grain_rows[index]
        target = next((grain.get(k) for k in ('onset_by_frame', 'model_first_sustained_presence_frame',
                      'last_complete_length_frame', 'latest_observation_frame') if grain.get(k) is not None), 0)
        choices = [(abs(row['source_frame'] - target), i) for i, row in enumerate(self.rows)
                   if row['owner_id'] == grain['grain_id']]
        if choices:
            self.select_row(min(choices)[1])

    def apply_queue_mode(self):
        """Queue items own their geometry (rev14 W5).

        While a queue task is active the Result review reassignment,
        distinct-cap and path controls are disabled, and one authoritative
        owner label matching the magenta grain is shown."""
        from .analysis_project import QUEUE
        task = self.controller.current or {}
        queue_mode = task.get('review_queue') == QUEUE
        busy = self.worker is not None and self.worker.isRunning()
        for widget in (self.owner, self.reassign_button, self.distinct,
                       self.path_button, self.partial_button):
            widget.setEnabled(not queue_mode and not busy)
            widget.setVisible(not queue_mode)
        if queue_mode:
            identity_task = task.get('task_type') == 'grain_identity'
            self.owner_label.setText(
                f"Owner: {task.get('owner_uuid') or 'field'} — " +
                ("view its magenta highlight on the Reference grain frame, then mark "
                 "the same grain's centre on the Task frame." if identity_task else
                 "the magenta ring in the field. Reassignment, distinct-cap and path "
                 "controls are handled by the queue task."))
        else:
            self.owner_label.setText("Owner from the selected result row.")

    def selection_changed(self):
        row = self.table.currentRow()
        if row >= 0:
            self.select_row(row)

    def move_result(self, delta):
        if self.rows:
            self.result_tabs.setCurrentWidget(self.table)
            self.select_row(min(max(0, self.table.currentRow() + delta), len(self.rows)-1))

    def start_census(self):
        try:
            current = self.controller.current or {}
            frame = (current.get("query_frames") or self.session.config["request"]["frames"])[0]
            self.session.open_census(self.controller, frame)
            self.task_dock.container.show()
            self.task_dock.refresh()
        except Exception as error:
            self.status.setText(str(error))

    def select_row(self, row, _column=0):
        try:
            self.table.blockSignals(True)
            self.table.setCurrentRow(row)
            self.table.blockSignals(False)
            data = self.rows[row]
            self.session.open_row(self.controller, data["owner_id"], data["source_frame"])
            self.owner.setCurrentText(data["owner_id"])
            self.distinct.blockSignals(True)
            self.distinct.setChecked(bool(self.controller.current.get("distinct_cap_evidence")))
            self.distinct.blockSignals(False)
            self.task_dock.refresh()
            self.apply_queue_mode()
            if self.task_dock.container is not None:
                self.task_dock.container.show()
        except Exception as error:
            self.on_failed(str(error))

    def refresh_stale(self):
        busy = self.worker is not None and self.worker.isRunning()
        stale = self.session.stale
        try:
            settings_changed = self.settings_config() != self.session.config
        except (ValueError, TypeError):
            settings_changed = True
        self.apply_queue_mode()
        self.export_button.setEnabled(not busy and not stale and not settings_changed)
        self.previous_result.setEnabled(not busy and self.table.currentRow() > 0)
        self.next_result.setEnabled(not busy and bool(self.rows) and self.table.currentRow() < len(self.rows)-1)
        if not busy and self.session.result and (stale or settings_changed):
            self.status.setText("Reviews, settings or inputs changed. Recompute to update tracking and exports.")

    def export(self):
        try:
            paths = self.session.export()
            self.status.setText("Exported grain summaries, measurements and analysis to " + str(self.session.directory/"exports"))
        except Exception as error:
            self.on_failed(str(error))

    def reassign(self):
        try:
            self.session.reassign(self.controller, self.owner.currentText())
            self.task_dock.refresh()
            self.refresh_stale()
        except Exception as error:
            self.on_failed(str(error))

    def set_distinct(self, checked):
        task = self.controller.current
        if task is None:
            return
        try:
            self.controller.require_task_frame()
            task["distinct_cap_evidence"] = f"review:{task['uuid']}" if checked else None
            self.session.store.save("task", task["uuid"], task, actor="distinct-cap-review")
        except ValueError as error:
            self.distinct.blockSignals(True)
            self.distinct.setChecked(bool(task.get("distinct_cap_evidence")))
            self.distinct.blockSignals(False)
            self.on_failed(str(error))

    def start_path(self):
        try:
            self.session.start_path(self.controller)
            self.task_dock.refresh()
        except Exception as error:
            self.on_failed(str(error))

    def save_partial(self):
        try:
            if not (self.controller.current or {}).get("task_type") == "centerline":
                raise ValueError("Choose Trace whole tube before saving a partial path")
            self.controller.save_path(False)
            self.controller.save_and_next()
            self.task_dock.refresh()
        except Exception as error:
            self.on_failed(str(error))
