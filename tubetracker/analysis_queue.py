"""Native annotation queue in the same project as movie analysis."""
from __future__ import annotations

from collections import Counter

from qtpy.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel,
    QListWidget, QPushButton, QRadioButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from .analysis_dock import AnalysisReviewPanel
from .analysis_project import QUEUE


class ReviewWorkspace(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.container = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.review = AnalysisReviewPanel(controller)
        self.queue = ReviewQueuePanel(controller, self)
        controller.on_draft_changed = self.draft_changed
        self.tabs.addTab(self.review, 'Result review')
        self.tabs.addTab(self.queue, 'Annotation queue')
        layout.addWidget(self.tabs)
        self.setMinimumWidth(320)
        self.setMaximumWidth(390)

    def refresh(self):
        if (self.controller.current or {}).get('review_queue') == QUEUE:
            self.tabs.setCurrentWidget(self.queue)
            self.queue.refresh()
        else:
            self.tabs.setCurrentWidget(self.review)
            self.review.refresh()
            self.queue.clear_overlay()

    def open_queue(self):
        self.queue.populate()
        tasks = self.queue.tasks
        if not tasks:
            self.queue.refresh()
            self.tabs.setCurrentWidget(self.queue)
        else:
            saved = self.controller.store.load('annotation-queue-selection')
            uid = (saved or {}).get('data', {}).get('task_uuid')
            if uid not in {t['uuid'] for t in tasks}:
                uid = next((t['uuid'] for t in tasks if not t.get('completed')), tasks[0]['uuid'])
            self.queue.open(uid)
        if self.container is not None:
            self.container.show()

    def show_error(self, message):
        self.queue.message.setText(message)

    def draft_changed(self):
        if (self.controller.current or {}).get('review_queue') == QUEUE:
            self.queue.populate()
            self.queue.update_message()
            if self.controller.current.get('task_type') == 'grain_identity':
                self.queue.draw_overlay()


class ReviewQueuePanel(QWidget):
    def __init__(self, controller, workspace):
        super().__init__()
        self.controller, self.workspace = controller, workspace
        self.tasks = []
        self.batch_complete = False
        layout = QVBoxLayout(self)
        self.picker = QListWidget()
        self.picker.setAccessibleName('Requested annotation task')
        self.picker.setMaximumHeight(90)
        self.picker.currentRowChanged.connect(self.pick)
        layout.addWidget(self.picker)
        self.navigation = QWidget()
        navigation = QHBoxLayout(self.navigation)
        navigation.setContentsMargins(0, 0, 0, 0)
        for label, delta in [('Previous task', -1), ('Next task', 1)]:
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, d=delta: self.move(d))
            navigation.addWidget(button)
        layout.addWidget(self.navigation)
        self.history_toggle = QCheckBox('Show completed history (never deleted)')
        self.history_toggle.toggled.connect(self.toggle_history)
        layout.addWidget(self.history_toggle)
        self.history = QListWidget()
        self.history.setAccessibleName('Completed annotation history')
        self.history.setMaximumHeight(90)
        self.history.setVisible(False)
        self.history.currentRowChanged.connect(self.pick_history)
        layout.addWidget(self.history)
        self.info = QLabel()
        self.info.setWordWrap(True)
        layout.addWidget(self.info)
        self.progress = QLabel()
        layout.addWidget(self.progress)
        self.context = QLabel()
        layout.addWidget(self.context)
        self.context_navigation = QWidget()
        context = QHBoxLayout(self.context_navigation)
        context.setContentsMargins(0, 0, 0, 0)
        for label, delta in [('−300', -300), ('−30', -30), ('−1', -1),
                                 ('Task frame', 0),
                                 ('+1', 1), ('+30', 30), ('+300', 300)]:
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, d=delta: self.action(lambda: self.step(d)))
            context.addWidget(button)
        layout.addWidget(self.context_navigation)
        from qtpy.QtWidgets import QLineEdit as _QLineEdit
        frame_row = QHBoxLayout()
        self.frame_jump = _QLineEdit()
        self.frame_jump.setPlaceholderText('Go to frame #…')
        self.frame_jump.setAccessibleName('Go to frame number')
        self.frame_jump.returnPressed.connect(
            lambda: self.action(lambda: self.jump_frame_number()))
        frame_row.addWidget(self.frame_jump)
        jump_btn = QPushButton('Go')
        jump_btn.clicked.connect(
            lambda _=False: self.action(lambda: self.jump_frame_number()))
        frame_row.addWidget(jump_btn)
        layout.addLayout(frame_row)
        self.context_choices = QComboBox()
        self.context_choices.setAccessibleName('Task context frame')
        self.context_choices.activated.connect(lambda index: self.action(
            lambda: self.jump_context(self.context_choices.itemData(index))))
        layout.addWidget(self.context_choices)
        self.look = QCheckBox('Look only — clicking does not mark')
        self.look.toggled.connect(self.look_changed)
        layout.addWidget(self.look)
        self.actions = QWidget()
        controls = QVBoxLayout(self.actions)
        controls.setContentsMargins(0, 0, 0, 0)
        self.identity_actions = QWidget()
        identity = QVBoxLayout(self.identity_actions)
        self.identity_save = self.button(identity, 'Save grain centre and next',
                    lambda: self.finish(controller.save_grain_identity))
        self.button(identity, 'Grain out of field — save and next',
                    lambda: self.finish(lambda: controller.save_grain_identity('out_of_field')))
        controls.addWidget(self.identity_actions)
        self.tip_actions = QWidget()
        tip = QVBoxLayout(self.tip_actions)
        self.tip_region = QCheckBox('Mark a tip area instead of a point')
        self.tip_region.toggled.connect(self.tip_region_mode)
        tip.addWidget(self.tip_region)
        self.point_next = self.button(tip, 'Continue after saved point', lambda: self.finish(self.require_saved_tip))
        self.area_save = self.button(tip, 'Save tip area and next',
                    lambda: self.finish(lambda: controller.save_region(controller._region_pts)))
        self.button(tip, 'Tip hidden — save and next',
                    lambda: self.finish(lambda: controller.save_state('not_directly_visible')))
        self.button(tip, 'No owned tube visible — save and next',
                    lambda: self.finish(lambda: controller.save_state('no_tube_visible')))
        controls.addWidget(self.tip_actions)
        self.path_actions = QWidget()
        path = QVBoxLayout(self.path_actions)
        self.button(path, 'Save FULL current path', lambda: self.finish(lambda: controller.save_path(True)))
        self.button(path, 'Save PARTIAL current path', lambda: self.finish(lambda: controller.save_path(False)))
        self.button(path, 'No tube at this frame — save and next',
                    lambda: self.finish(lambda: controller.save_state('no_tube_visible')))
        controls.addWidget(self.path_actions)
        self.crossing_actions = QWidget()
        crossing = QFormLayout(self.crossing_actions)
        self.lane = QButtonGroup(self)
        lane_row = QHBoxLayout()
        for index, label in enumerate(('A', 'B', 'C')):
            button = QRadioButton('Draw lane '+label)
            self.lane.addButton(button, index)
            lane_row.addWidget(button)
        self.lane.button(0).setChecked(True)
        self.lane.idToggled.connect(lambda index, checked: self.choose_lane(index) if checked else None)
        crossing.addRow(lane_row)
        self.assignments = {}
        self.assignment_rows = {}
        for label in ('A', 'B', 'C'):
            group = QButtonGroup(self)
            group.idToggled.connect(lambda _index, checked, lab=label: self.assign(lab) if checked else None)
            self.assignments[label] = group
            self.assignment_rows[label] = QHBoxLayout()
            crossing.addRow(QLabel('Lane '+label+' belongs to'))
            crossing.addRow(self.assignment_rows[label])
        save = QPushButton('Save lanes and assignments')
        save.clicked.connect(lambda: self.action(lambda: self.finish(self.save_crossing)))
        crossing.addRow(save)
        controls.addWidget(self.crossing_actions)
        self.negative_actions = QWidget()
        negative = QVBoxLayout(self.negative_actions)
        self.button(negative, 'Save exact negative polygon', controller.save_negative_polygon)
        self.button(negative, 'Remove last saved polygon', self.remove_polygon)
        self.button(negative, 'Finish regions', lambda: self.finish(self.require_polygon))
        controls.addWidget(self.negative_actions)
        self.event_actions = QWidget()
        event = QVBoxLayout(self.event_actions)
        self.event_info = QLabel("No bracket marks yet on this episode.")
        self.event_info.setWordWrap(True)
        event.addWidget(self.event_info)
        event_buttons = QVBoxLayout()
        for label, verdict in (("Emerged within bracket", "emerged_within"),
                               ("Already emerged at start", "emerged_at_start"),
                               ("No emergence by end", "no_emergence_by_end"),
                               ("Unobservable", "unobservable")):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _=False, v=verdict: self.finish(
                    lambda v=v: self.save_event(v)))
            event_buttons.addWidget(button)
        event.addLayout(event_buttons)
        event_bracket = QVBoxLayout()
        for label, slot in (("Mark last smooth rim HERE", "last_absent_candidate"),
                            ("Mark first bump/outgrowth HERE", "first_visible_candidate")):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _=False, s=slot: self.action(lambda: self.mark_bracket(s)))
            event_bracket.addWidget(button)
        event.addLayout(event_bracket)
        controls.addWidget(self.event_actions)
        self.mask_actions = QWidget()
        mask = QVBoxLayout(self.mask_actions)
        self.brush_mode = QButtonGroup(self)
        for label, value in [('Owned tube', 1), ('Unknown / uncertain area', 2), ('Erase paint', 0)]:
            button = QRadioButton(label)
            self.brush_mode.addButton(button, value)
            mask.addWidget(button)
        self.brush_mode.button(1).setChecked(True)
        self.brush_mode.idToggled.connect(lambda _index, checked: self.brush() if checked else None)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 40)
        self.brush_size.setValue(5)
        self.brush_size.setPrefix('Brush size: ')
        self.brush_size.setSuffix(' px')
        self.brush_size.valueChanged.connect(self.brush)
        mask.addWidget(self.brush_size)
        self.field_checked = QCheckBox('I reviewed all pixels inside the yellow field')
        mask.addWidget(self.field_checked)
        self.button(mask, 'Save reviewed-field mask', lambda: self.finish(lambda: self.save_mask(True)))
        self.button(mask, 'Save partial paint', lambda: self.finish(lambda: self.save_mask(False)))
        controls.addWidget(self.mask_actions)
        self.census_review = AnalysisReviewPanel(controller)
        self.census_review.hide()
        controls.addWidget(self.census_review.census_actions)
        tools = QHBoxLayout()
        self.undo_draft = self.button(tools, 'Undo point', self.undo)
        self.undo_saved = self.button(tools, 'Undo saved correction', self.undo_saved)
        self.clear_draft = self.button(tools, 'Clear draft points', controller.clear_task_marks)
        controls.addLayout(tools)
        self.button(controls, 'Cannot judge this task', self.unresolved)
        layout.addWidget(self.actions)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.census_review.message = self.message
        layout.addWidget(self.message)
        layout.addStretch(1)
        self.populate()
        self.show_task_controls(False)

    def button(self, layout, label, function):
        button = QPushButton(label)
        button.clicked.connect(lambda _=False: self.action(function))
        layout.addWidget(button)
        return button

    def action(self, function):
        try:
            function()
            self.refresh()
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.message.setText(str(error))

    def populate(self):
        every = sorted([dict(e['data'], uuid=e['uuid']) for e in self.controller.store.entities('task')
                        if e['data'].get('review_queue') == QUEUE], key=lambda t: t['queue_order'])
        def archived(t):
            archive = t.get('queue_archive') or {}
            if archive.get('previous_review_queue') == QUEUE:
                return True
            return str(t.get('review_queue') or '').startswith(QUEUE + '-completed-')
        active = [t for t in every if t.get('review_queue') == QUEUE]
        archived_tasks = [dict(e['data'], uuid=e['uuid'])
                          for e in self.controller.store.entities('task')
                          if e['data'].get('review_queue') != QUEUE
                          and archived(dict(e['data']))]
        self.history_tasks = [t for t in active
                              if t.get('completed') or t.get('review_status') == 'withdrawn']
        self.history_tasks += [t for t in archived_tasks
                               if t['uuid'] not in {h['uuid'] for h in self.history_tasks}]
        history_ids = {t['uuid'] for t in self.history_tasks}
        self.tasks = [t for t in active if t['uuid'] not in history_ids]
        batch = (self.controller.current or {}).get('queue_batch')
        if not batch and self.tasks:
            batch = self.tasks[0].get('queue_batch')
        batch_tasks = [t for t in active if batch and t.get('queue_batch') == batch]
        self.batch_size = len(batch_tasks)
        answered = sum(t['uuid'] in history_ids for t in batch_tasks)
        self.progress.setText(f'{answered}/{len(batch_tasks)} answered · '
                              f'{len(batch_tasks)-answered} remaining in this batch' if batch_tasks else '')
        current = (self.controller.current or {}).get('uuid')
        self.picker.blockSignals(True)
        batch_sizes = Counter(t.get('queue_batch') for t in active)
        labels = []
        for n, t in enumerate(self.tasks):
            position = t.get('queue_position', n + 1)
            total = batch_sizes[t['queue_batch']] if t.get('queue_batch') and t.get('queue_position') else len(self.tasks)
            labels.append(f"{position}/{total} · {t['task_type']} · {t['query_frames'][0]} · pending")
        if labels != [self.picker.item(i).text() for i in range(self.picker.count())]:
            self.picker.clear()
            self.picker.addItems(labels)
        index = next((i for i, t in enumerate(self.tasks) if t['uuid'] == current), 0)
        if self.tasks:
            self.picker.setCurrentRow(index)
        self.picker.blockSignals(False)
        self.picker.setEnabled(bool(self.tasks))
        self.navigation.setEnabled(bool(self.tasks))
        self.history_toggle.setVisible(bool(self.history_tasks))
        self.history_toggle.setText(f'Show completed history ({len(self.history_tasks)}, never deleted)')
        if self.history_toggle.isChecked():
            self.refresh_history()
        elif not self.history_tasks:
            self.history.setVisible(False)

    def refresh_history(self):
        self.history.blockSignals(True)
        self.history.clear()
        for t in self.history_tasks:
            state = 'unresolved' if t.get('review_status') == 'withdrawn' else 'saved'
            self.history.addItem(f"{t['task_type']} · {t['query_frames'][0]} · {state}")
        self.history.blockSignals(False)
        self.history.setVisible(self.history_toggle.isChecked())

    def toggle_history(self, checked):
        self.history.setVisible(bool(checked))
        if checked:
            self.refresh_history()

    def pick_history(self, index):
        if 0 <= index < len(self.history_tasks):
            self.action(lambda: self.open(self.history_tasks[index]['uuid']))

    def advance(self):
        """Save-and-next: move to the next pending task, never skipping one.

        The just-completed task has already left the pending list, so the
        first remaining pending task is the next unvisited one."""
        self.populate()
        if self.tasks:
            self.batch_complete = False
            self.action(lambda: self.open(self.tasks[0]['uuid']))
        else:
            self.batch_complete = True
            self.show_task_controls(False)

    def show_task_controls(self, active):
        self.actions.setVisible(active)
        self.actions.setEnabled(active and self.controller._context_offset == 0)
        self.look.setVisible(active)
        self.context_navigation.setVisible(active)
        self.context_choices.setVisible(active and self.context_choices.count() > 1)
        if not active:
            self.info.clear()
            self.context.clear()
            self.message.setText(
                'Select a requested annotation.' if self.tasks else
                ('Batch complete — ' + (f'all {self.batch_size} tasks in this batch answered. '
                 if self.batch_size else 'all requested tasks answered. ') +
                'Completed history is above; nothing was deleted.' if self.history_tasks else
                'No requested annotations in this project.'))

    def pick(self, index):
        if 0 <= index < len(self.tasks):
            self.action(lambda: self.open(self.tasks[index]['uuid']))

    def open(self, uuid):
        self.batch_complete = False
        self.controller.open_task(uuid)
        self.controller._census_mode = 'grain'
        self.controller._look_only = False
        self.look.setChecked(False)
        self.field_checked.setChecked(False)
        task = self.controller.current
        self.context_choices.clear()
        base = task['query_frames'][0]
        reference = task.get('reference_frame')
        contexts = list(dict.fromkeys([base] + ([reference] if reference is not None else [])
                                     + list(task.get('context_frames', []))))
        for frame in contexts:
            label = 'Task frame' if frame == base else 'Reference grain' if frame == reference else 'Context'
            self.context_choices.addItem(f'{label}: {frame}', frame)
        self.tip_region.blockSignals(True)
        self.tip_region.setChecked(False)
        self.tip_region.blockSignals(False)
        if task['task_type'] == 'body_mask':
            self.brush_mode.button(1).setChecked(True)
        self.controller.store.save('session', 'annotation-queue-selection', {'task_uuid': uuid}, actor='selection')
        self.controller.store.save('session', 'analysis-selection', {'kind': 'queue', 'task_uuid': uuid}, actor='selection')
        self.controller.analysis_session.selection = {'kind': 'queue', 'task_uuid': uuid}
        for label, group in self.assignments.items():
            group.blockSignals(True)
            row = self.assignment_rows[label]
            for button in group.buttons():
                group.removeButton(button)
                row.removeWidget(button)
                button.deleteLater()
            choices = [('Unresolved', '')] + [(o['id'][-4:], o['id']) for o in task.get('owner_choices', [])]
            for index, (text, owner_id) in enumerate(choices):
                button = QRadioButton(text)
                button.setProperty('owner_id', owner_id)
                button.setAccessibleName(f'Lane {label}: '+('Grain '+text if owner_id else text))
                group.addButton(button, index)
                row.addWidget(button)
                button.setChecked(owner_id == task.get('lane_assignments', {}).get(label, ''))
            group.blockSignals(False)
        self.lane.blockSignals(True)
        self.lane.button(('A', 'B', 'C').index(self.controller._lane)).setChecked(True)
        self.lane.blockSignals(False)
        self.workspace.tabs.setCurrentWidget(self)
        self.populate()
        self.refresh()

    def move(self, delta):
        current = (self.controller.current or {}).get('uuid')
        self.populate()
        if self.tasks:
            index = (0 if delta > 0 and current not in {t['uuid'] for t in self.tasks} else
                     min(max(0, self.picker.currentRow()+delta), len(self.tasks)-1))
            self.action(lambda: self.open(self.tasks[index]['uuid']))
        elif delta > 0:
            self.advance()

    def jump_frame_number(self):
        c = self.controller
        if c.current is None:
            raise ValueError('no task open')
        try:
            target = int(self.frame_jump.text().strip())
        except ValueError:
            raise ValueError('type a source frame number, then Go')
        base = c.current['query_frames'][0]
        c.step_frame(target - base - c._context_offset)

    def jump_context(self, frame):
        c = self.controller
        if c.current is None or frame is None:
            return
        base = c.current['query_frames'][0]
        c.step_frame(int(frame) - base - c._context_offset)

    def tip_region_mode(self, checked):
        self.controller._region_mode = bool(checked)
        self.update_message()

    def require_saved_tip(self):
        c = self.controller
        saved = c.store.load('obs-' + c.current['uuid'])
        from .review_semantics import is_withdrawn, precise_tip
        if saved is None or is_withdrawn(saved['data']) or precise_tip(saved['data']) is None:
            raise ValueError('click the current apex to save a point, or use a visibility verdict')

    def step(self, delta):
        if delta:
            self.controller.step_frame(delta)
        else:
            self.controller.back_to_task_frame()

    def choose_lane(self, index):
        if (self.controller.current or {}).get('task_type') != 'crossing':
            return
        self.controller.select_lane(('A', 'B', 'C')[index])
        self.controller.persist_draft()
        self.refresh()

    def assign(self, label):
        task = self.controller.current
        if not task or task.get('task_type') != 'crossing':
            return
        task.setdefault('lane_assignments', {})[label] = self.assignments[label].checkedButton().property('owner_id')
        self.controller.persist_draft()

    def save_crossing(self):
        task = self.controller.current
        mapping = {k:v for k,v in task.get('lane_assignments', {}).items() if v}
        return self.controller.save_crossing(mapping, unresolved=False)

    def require_polygon(self):
        if self.controller._region_pts:
            raise ValueError('Save or clear the unfinished polygon first')
        if not self.controller.current.get('negative_polygons'):
            raise ValueError('Save a reviewed polygon or choose Cannot judge')
        for polygon in self.controller.current['negative_polygons']:
            saved = self.controller.store.load(polygon['uuid'])
            if not saved or saved['kind'] != 'region':
                raise ValueError('A saved polygon is missing; redraw it before confirming')
            self.controller._save_review('region', polygon['uuid'], saved['data'], actor=self.controller.actor)

    def remove_polygon(self):
        self.controller.require_task_frame()
        task = self.controller.current
        polygons = task.get('negative_polygons', [])
        if polygons:
            last = polygons.pop()
            self.controller.store.delete('region', last['uuid'], actor=self.controller.actor)
            task['completed'] = False
            self.controller.persist_draft()

    def look_changed(self, checked):
        self.controller._look_only = checked
        self.brush()
        self.update_message()

    def brush(self, *_):
        c = self.controller
        if c.viewer is None or not c.current or c.current.get('task_type') != 'body_mask':
            return
        layer = c.viewer.layers['mask-paint']
        layer.selected_label = self.brush_mode.checkedId()
        layer.brush_size = self.brush_size.value()
        layer.mode = 'pan_zoom' if c._look_only or c._context_offset else 'paint'
        c.viewer.layers.selection.active = layer

    def save_mask(self, complete):
        if complete and not self.field_checked.isChecked():
            raise ValueError('Review the whole yellow field and check the box, or save partial paint')
        return self.controller.commit_mask_labels(complete)

    def undo(self):
        c = self.controller
        c.require_task_frame()
        if (c.current or {}).get('task_type') == 'body_mask' and c.viewer is not None:
            c.viewer.layers['mask-paint'].undo()
            c.persist_mask_draft()
        else:
            c.undo_last_dot()

    def unresolved(self):
        if (self.controller.current or {}).get('task_type') == 'grain_identity':
            self.controller.save_grain_identity('ambiguous')
            self.controller.current['review_verdict'] = 'identity_uncertain'
            self.controller.persist_draft()
        else:
            self.controller.withdraw_review()
        self.field_checked.setChecked(False)
        self.populate()
        # rev14 P1: an unresolved verdict is an answer -- advance to the
        # next pending task exactly like a save, preserving the verdict.
        self.advance()

    def save_event(self, verdict):
        c = self.controller
        c.save_germination_event(
            verdict, c.current.get("last_absent_candidate"),
            c.current.get("first_visible_candidate"))

    def mark_bracket(self, slot):
        c = self.controller
        if not c.current or str(c.current.get("task_type")) != "germination_event":
            raise ValueError("bracket marks belong on a germination episode")
        frame = int(c.current["query_frames"][0]) + int(getattr(c, "_context_offset", 0))
        c.current[slot] = frame
        c.persist_draft()
        lo = c.current.get("last_absent_candidate")
        hi = c.current.get("first_visible_candidate")
        self.event_info.setText(
            f"Bracket marked: last-absent={lo}, first-visible={hi} (viewing {frame}). "
            f"Return to the task frame and press one verdict to save.")

    def finish(self, action):
        c = self.controller
        c.require_task_frame()
        action()
        c._confirm_review_task()
        c.current['completed'] = True
        c.persist_draft()
        self.advance()

    def undo_saved(self):
        note = self.controller.undo_saved_correction()
        self.populate()
        self.refresh()
        self.message.setText(note)

    def refresh(self):
        c, task = self.controller, self.controller.current
        self.populate()
        if not self.tasks and self.batch_complete:
            self.show_task_controls(False)
            self.clear_overlay()
            return
        if not task or task.get('review_queue') != QUEUE:
            self.show_task_controls(False)
            self.clear_overlay()
            return
        self.look.blockSignals(True)
        self.look.setChecked(bool(getattr(c, '_look_only', False)))
        self.look.blockSignals(False)
        self.show_task_controls(True)
        kind = task['task_type']
        self.info.setText(f"{task['movie']} · {task.get('owner_uuid', 'field')}\n{c.instruction()}")
        frame = task['query_frames'][0] + c._context_offset
        self.context.setText(f"Frame {frame}" +
            (' · context only; return to Task frame to mark' if c._context_offset else ''))
        self.context_choices.blockSignals(True)
        index = self.context_choices.findData(frame)
        if index < 0:
            self.context_choices.addItem(f'Context: {frame}', frame)
            index = self.context_choices.count() - 1
        self.context_choices.setCurrentIndex(index)
        self.context_choices.blockSignals(False)
        # Bracket marks are ABOUT the viewed frame, not edits to the task
        # frame — they must stay live while scrubbing on episode tasks.
        # The verdict still requires the task frame (finish() enforces it).
        if kind == 'germination_event':
            self.actions.setEnabled(True)
            self.event_actions.setEnabled(True)
        else:
            self.actions.setEnabled(c._context_offset == 0)
        self.undo_draft.setText('Undo paint' if kind == 'body_mask' else 'Undo point')
        self.clear_draft.setVisible(kind != 'body_mask')
        for widget, types in [(self.path_actions, ['centerline']), (self.crossing_actions, ['crossing']),
            (self.negative_actions, ['neg_region']), (self.mask_actions, ['body_mask']),
            (self.identity_actions, ['grain_identity']), (self.tip_actions, ['review_tip']),
            (self.event_actions, ['germination_event']),
            (self.census_review.census_actions, ['census'])]:
            widget.setVisible(kind in types)
        if kind == 'census':
            self.census_review.refresh()
        else:
            self.update_message()
        self.draw_overlay()

    def update_message(self):
        c, task = self.controller, self.controller.current
        if not task or task.get('task_type') == 'census':
            return
        kind = task['task_type']
        from .review_semantics import is_withdrawn, precise_tip
        saved = c.store.load('obs-' + task['uuid']) if kind == 'review_tip' else None
        point = (precise_tip(saved['data']) if saved and not is_withdrawn(saved['data']) else None)
        can_mark = not c._context_offset and not c._look_only
        self.identity_save.setEnabled(can_mark and len(c._region_pts) == 1)
        self.point_next.setEnabled(can_mark and not c._region_mode and point is not None)
        self.area_save.setEnabled(can_mark and c._region_mode and len(c._region_pts) >= 3)
        has_draft = bool(c._region_pts or c._path_pts or any(c._lanes.values())
                         or task.get('mask_draft_raster') or task.get('mask_unknown_draft_raster'))
        state = ('Cannot judge. Previous confirmed evidence is withdrawn; the editable draft remains.'
                 if task.get('review_status') == 'withdrawn' else
                 'Saved; you can move to the next task or revise this one.' if task.get('completed') else
                 'Point saved. Continue after saved point advances; Undo saved correction undoes this save.'
                 if point is not None and not c._region_mode else
                 'Draft saved; confirm the annotation when ready.' if has_draft else
                 'Ready for your answer.')
        count = (('Grain centre drafted; Save grain centre and next confirms identity.' if c._region_pts else
                  'View the highlighted reference grain, then mark its centre on the task frame.') if kind == 'grain_identity' else
                 (f'{len(c._region_pts)} area vertices; save an area after outlining it.' if c._region_mode else
                  'A click saves a precise tip immediately.' if point is not None else
                  'No tip point saved. Click the current apex once to save it.') if kind == 'review_tip' else
                 f'{len(c._path_pts)} path points' if kind == 'centerline' else
                 f"{len(c._region_pts)} polygon points; {len(task.get('negative_polygons', []))} saved polygons" if kind == 'neg_region' else
                 ', '.join(f'{k}: {len(v)} points' for k,v in c._lanes.items()) if kind == 'crossing' else
                 'Green: owned tube. Orange: unknown. Unpainted pixels remain unknown unless the whole field is reviewed.')
        self.message.setText(count+'\n'+state)

    def clear_overlay(self):
        viewer = self.controller.viewer
        if viewer is not None:
            for layer in list(viewer.layers):
                if layer.name.startswith('queue-'):
                    viewer.layers.remove(layer)

    def draw_overlay(self):
        c = self.controller
        self.clear_overlay()
        if c.viewer is None or not c.current:
            return
        import numpy as np
        task, viewer = c.current, c.viewer
        if c._context_offset:
            frame = task['query_frames'][0] + c._context_offset
            if frame == task.get('reference_frame') and task.get('reference_owners'):
                target = next((o for o in task['reference_owners'] if o['id'] == task.get('owner_uuid')), None)
                if target:
                    layer = viewer.add_points([target['grain_native'][::-1]], name='queue-reference-grain',
                        size=18, face_color='transparent', border_color='magenta', border_width=2,
                        border_width_is_relative=False, blending='translucent_no_depth')
                    layer.editable = False
            return
        if task.get('review_region'):
            # An invisible polygon face must not write depth over the whole field.
            layer = viewer.add_shapes([np.array(task['review_region'])[:, ::-1]], shape_type='polygon',
                name='queue-field', face_color='transparent', edge_color='yellow', edge_width=1,
                blending='translucent_no_depth')
            layer.editable = False
        owners = task.get('owner_choices', [])
        if owners and task['task_type'] != 'census':
            angles = np.linspace(0, 2*np.pi, 33)
            rings = [np.column_stack((o['grain_native'][1]+13*np.sin(angles),
                                      o['grain_native'][0]+13*np.cos(angles))) for o in owners]
            layer = viewer.add_shapes(rings, shape_type='path', name='queue-grains', edge_width=2,
                face_color='transparent', blending='translucent_no_depth',
                edge_color=['magenta' if o['id'] == task.get('owner_uuid') else 'cyan' for o in owners],
                text={'string': [o['id'][-4:] for o in owners], 'color': 'white',
                      'translation': [-18,0], 'size': 11, 'blending': 'translucent_no_depth'})
            layer.editable = False
        if task['task_type'] == 'centerline' and c._path_pts:
            layer = viewer.add_points(np.asarray(c._path_pts)[:, ::-1], name='queue-path-vertices',
                size=5, face_color='yellow', blending='translucent_no_depth')
            layer.editable = False
        for label, pts in c._lanes.items():
            if len(pts) >= 2 and task['task_type'] == 'crossing':
                layer = viewer.add_shapes([np.asarray(pts)[:, ::-1]], shape_type='path',
                    name='queue-lane-'+label, edge_width=3,
                    edge_color={'A':'cyan','B':'magenta','C':'orange'}.get(label,'lime'),
                    blending='translucent_no_depth')
                layer.editable = False
        if c._region_pts:
            layer = viewer.add_points(np.asarray(c._region_pts)[:, ::-1], name='queue-polygon-vertices',
                size=6, face_color='yellow', blending='translucent_no_depth')
            layer.editable = False
            if len(c._region_pts) >= 3:
                layer = viewer.add_shapes([np.asarray(c._region_pts)[:, ::-1]], shape_type='polygon',
                    name='queue-polygon-draft', face_color='transparent', edge_color='yellow', edge_width=2,
                    blending='translucent_no_depth')
                layer.editable = False
        c.draw_task_markup()
        if task['task_type'] == 'body_mask':
            self.brush()
        elif c._image_layer is not None:
            viewer.layers.selection.active = c._image_layer
        if getattr(c.analysis_session, 'config', {}).get('verification_only'):
            import json
            from pathlib import Path
            details = {'task': task['uuid'], 'active_lane': c._lane,
                       'camera': {'center': list(viewer.camera.center), 'zoom': float(viewer.camera.zoom)},
                       'layers': [{'name': l.name, 'visible': bool(l.visible), 'opacity': float(l.opacity),
                                   'ndim': int(l.ndim), 'extent': np.asarray(l.extent.world).tolist()}
                                  for l in viewer.layers if l is not c._image_layer]}
            (Path(c.store.path).parent/'queue-display-state.json').write_text(json.dumps(details, indent=2)+'\n')
