import copy
import csv
import json

import pytest

from tubetracker.population import analysis_population
from tubetracker.movie_analysis import export_analysis


OWNER = {'id': 'g', 'grain_native': [20., 30.], 'identity_verified': True}


def model_row(frame, **extra):
    return dict(owner_id='g', source_frame=frame, state='present',
                route_evidence={'geometry': {'geometrically_supported': True,
                                            'proposal_length_px': 42.},
                                'withheld_reasons': ['full_route_validation_missing']}, **extra)


def summarize(rows, constraints=None, model=None):
    return analysis_population([OWNER], rows, constraints or {}, [], movie='m',
                               model_rows=rows if model is None else model)


@pytest.mark.parametrize('evidence', [None, {}, {'geometry': None}, 'invalid'])
def test_unknown_or_null_route_does_not_crash_or_supply_a_length(evidence):
    row = model_row(1)
    row.update(state='identity_uncertain', route_evidence=evidence)
    summary = summarize([row])['grain_summaries'][0]
    assert summary['model_first_supported_presence_frame'] is None
    assert summary['model_last_supported_path_length_px'] is None
    assert summary['last_complete_length_px'] is None
    assert summary['timing_status'] == 'unknown'


def test_model_streak_is_separate_from_reviewed_bounds_and_resets():
    rows = [model_row(f) for f in range(7)]
    rows[2]['route_evidence'] = None
    rows[3]['assignment_search_truncated'] = True
    constraints = {
        ('g', 1): {'state': 'no_tube_visible', 'source': 'human', 'revision': 1},
        ('g', 6): {'state': 'direct_visible', 'tip_xy': [40, 30], 'source': 'human', 'revision': 1}}
    summary = summarize(rows, constraints)['grain_summaries'][0]
    assert (summary['onset_after_frame'], summary['onset_by_frame']) == (1, 6)
    assert summary['model_first_supported_presence_frame'] == 0
    assert summary['model_first_sustained_presence_frame'] == 4
    assert summary['model_first_supported_presence_time_s'] is None
    assert summary['biological_class'] is None
    assert summary['complete_measurement_count'] == 0
    constraints.pop(('g', 1))  # withdrawing absence must restore one-sided bounds
    revised = summarize(rows, constraints)['grain_summaries'][0]
    assert revised['timing_status'] == 'left_censored'
    assert revised['onset_after_frame'] is None


def test_empty_results_still_report_each_grain_and_do_not_invent_zero_length():
    summary = summarize([])['grain_summaries'][0]
    assert summary['grain_id'] == 'g'
    assert summary['last_complete_length_px'] is None
    assert summary['latest_observation_frame'] is None
    assert summary['latest_length_withheld_reasons'] == ['no_observation']


@pytest.mark.parametrize('constraint,reason', [
    ({'tip_xy': [40., 30.]}, 'tip_review_does_not_establish_complete_path'),
    ({'tip_region_xy': [[39., 29.], [41., 29.], [40., 31.]]},
     'tip_review_does_not_establish_complete_path'),
    ({'path_xy': [[20., 30.], [30., 30.]], 'path_complete': False},
     'reviewed_path_partial'),
])
def test_reviewed_presence_explains_why_full_length_is_withheld(constraint, reason):
    row = model_row(4)
    row.update(route_evidence=None, path_complete=False, length_px=None,
               as_inference_constraint=True, constraint=constraint)
    summary = summarize([row])['grain_summaries'][0]
    assert summary['latest_length_withheld_reasons'] == [reason]
    assert summary['last_complete_length_px'] is None


def test_exports_keep_reviewed_length_and_model_estimate_distinct(tmp_path):
    baseline = [model_row(4)]
    corrected = copy.deepcopy(baseline)
    corrected[0].update(path_complete=True, length_px=43.5, length_um=None,
                        provenance='human-corrected', as_inference_constraint=True,
                        route_evidence=None, path_certificate={'kind': 'reviewed_full_current_path'})
    report = summarize(corrected, model=baseline)
    result = {'population': report, 'rows': corrected, 'inputs': {}, 'cache': {},
              'inference': {}, 'elapsed_wall_s': 1.}
    paths = export_analysis(result, tmp_path)
    row = list(csv.DictReader(open(paths['grains_csv'])))[0]
    assert row['last_complete_length_px'] == '43.5'
    assert row['last_complete_length_provenance'] == 'human-corrected'
    assert row['model_last_supported_path_length_px'] == '42.0'
    assert json.loads(row['last_complete_length_certificate'])['kind'] == 'reviewed_full_current_path'
    assert row['onset_by_frame'] == ''  # an output path alone is not a reviewed onset
    assert 'not verified onset' in row['model_presence_support_rule']
    assert json.load(open(paths['population_json']))['grain_summaries'][0]['grain_id'] == 'g'


def test_summary_rejects_workflow_lengths_even_from_old_measurement_rows():
    baseline = [model_row(frame) for frame in (0,30,60)]
    rows = copy.deepcopy(baseline)
    rows[0].update(path_complete=True, length_px=40., provenance='human-corrected')
    rows[1].update(state='absent', provenance='workflow-test')
    rows[2].update(path_complete=True, length_px=1000., provenance='human-corrected',
                   path_certificate={'geometry_source':{'review_origin':'workflow_test'}})
    report = summarize(rows, model=baseline)
    s = report['grain_summaries'][0]
    assert s['complete_measurement_count'] == 1 and s['last_complete_length_frame'] == 0
    assert s['last_complete_length_px'] == 40.
    assert s['latest_length_withheld_reasons'] == ['workflow_verification']
    assert s['model_last_supported_path_frame'] == 60
    assert report['model_observations'] == {'present': 3}
