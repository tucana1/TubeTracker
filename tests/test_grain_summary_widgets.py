import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from types import SimpleNamespace
import pytest

pytest.importorskip('qtpy.QtWidgets', exc_type=ImportError)
pytest.importorskip('napari.layers', exc_type=ImportError)
from qtpy.QtWidgets import QApplication
from test_rev14_p1_controls import rig
from tubetracker.analysis_dock import grain_summary_text


def test_grain_selection_opens_its_frame_without_saving_an_observation(rig):
    workspace, dock, controller, store = rig
    opened = []
    controller.analysis_session.open_row = lambda c, g, f: opened.append((g, f))
    summary = dict(grain_id='grain', grain_x=20, grain_y=30, timing_status='left_censored',
        onset_after_frame=None, onset_by_frame=60, complete_measurement_count=0,
        last_complete_length_px=None, model_last_supported_path_length_px=12.,
        model_last_supported_path_frame=60)
    controller.analysis_session.result = {
        'owners': [{'id': 'grain'}],
        'rows': [{'owner_id': 'grain', 'source_frame': f, 'state': 'present'} for f in (30, 60)],
        'population': {'grain_summaries': [summary], 'census': {'census_completeness_certified': False}}}
    dock.populate()
    before = store.entities('observation')
    dock.grain_table.setCurrentRow(0)  # the connected Qt selection signal
    QApplication.instance().processEvents()
    assert opened == [('grain', 60)]
    assert dock.table.currentRow() == 1
    assert store.entities('observation') == before
    assert 'start not observed' in dock.grain_table.item(0).text()
    assert 'Complete length: withheld' in dock.grain_table.item(0).text()
    assert 'Provisional model path: 12.00 px' in dock.grain_table.item(0).text()
    assert 'may be incomplete' in dock.population_status.text()
    assert workspace.review.look.isChecked() and controller._look_only


def test_stale_timer_cannot_reenable_review_controls_during_analysis(rig):
    _, dock, _, _ = rig
    dock.worker = SimpleNamespace(isRunning=lambda: True)
    dock.set_busy(True)
    dock.refresh_stale()
    assert not dock.owner.isEnabled()
    assert not dock.reassign_button.isEnabled()
    assert not dock.grain_table.isEnabled()
    dock.worker = None


def test_summary_names_reviewed_and_provisional_lengths():
    text = grain_summary_text(dict(grain_id='grain', grain_x=20, grain_y=30,
        timing_status='contradictory', onset_after_frame=None, onset_by_frame=None,
        last_complete_length_px=40., last_complete_length_um=20.,
        last_complete_length_frame=60, last_complete_length_provenance='human-corrected',
        model_last_supported_path_length_px=38., model_last_supported_path_frame=60))
    assert 'conflicting reviews' in text
    assert '20.00 µm at frame 60 · reviewed' in text
    assert 'Provisional model path: 38.00 px' in text


def test_irregular_source_schedule_survives_settings_edits_and_restart(rig):
    from tubetracker.analysis_dock import AnalysisDock
    w, _, c, _ = rig
    class MovieReader:
        native_size = (1280, 1024)
        def __len__(self):
            return 52583
    c.reader = MovieReader()
    frames = sorted(set(range(0,52583,300)) | set(range(51030,51451,30)) | {52582})
    assert len(frames) == 191
    c.analysis_session.config['request']['frames'] = frames
    dock = AnalysisDock(c, w)
    dock.timer.stop()
    try:
        assert dock.sampling_summary.text() == '191 source frames'
        dock.roi.setText('830,420,975,525')
        dock.cadence.setText('2')
        dock.cadence_source.setText('test acquisition log')
        configured = dock.settings_config()
        assert configured['request']['frames'] == frames
        assert c.analysis_session.config['request']['frames'] == frames
        c.analysis_session.config = configured
        restarted = AnalysisDock(c, w)
        restarted.timer.stop()
        try:
            assert restarted.settings_config()['request']['frames'] == frames
            restarted.first.setValue(51000)
            restarted.last.setValue(51451)
            assert restarted.settings_config()['request']['frames'] == [51000,51300,51451]
            assert restarted.sampling_summary.text() == '3 source frames'
            restarted.last.setValue(50999)
            with pytest.raises(ValueError, match='Last source frame'):
                restarted.settings_config()
        finally:
            restarted.close()
    finally:
        dock.close()
