import pytest

from prototypes.v30_video_apex.census import EmergenceInterval, germination_report
from tubetracker.acquisition import AcquisitionClock, validate_acquisition
from tubetracker.analysis_contracts import measurement_rows
from tubetracker.population import build_population_report
from prototypes.v30_video_apex.census import GrainRecord


def test_irregular_timestamps_are_shared_by_lengths_and_emergence():
    metadata = {'source_times_s': {'100': 4., '200': 7.5}, 'cadence_source': 'microscope log',
                'seconds_per_source_frame': 2.}
    rows = [dict(owner_id='g', source_frame=f, state='present', path_complete=True, length_px=l)
            for f, l in [(100, 10.), (200, 17.)]]
    measured = measurement_rows(rows, metadata, first_frame=100)
    assert measured[1]['growth_interval_s'] == 3.5
    assert measured[1]['growth_px_per_s'] == 2.
    grain = GrainRecord('g', 'm', (20, 20), observed_frames=[100, 200],
                        no_tube_observed_at=[100], present_observed_at=[200])
    report = build_population_report([grain], [], movie='m', acquisition=metadata)
    timing = report['germination']['timing_bounds'][0]
    assert timing['interval_seconds'] == 3.5
    assert timing['absent_time_s'] == 4. and timing['present_time_s'] == 7.5
    assert report['germination']['numerator_germinated'] == 0  # timing is not a biological ruling


def test_missing_timestamp_never_uses_frame_rate_or_interpolation():
    clock = AcquisitionClock.from_metadata({'source_times_s': {'0': 0., '20': 7.},
        'cadence_source': 'log', 'seconds_per_source_frame': 1.})
    assert clock.at(10) is None
    iv = EmergenceInterval('g', {'frame': 10}, {'frame': 20})
    report = germination_report([iv], acquisition_cadence_s=1., cadence_source='log',
                                source_times_s={'0': 0., '20': 7.})
    assert report['timing_bounds'][0]['interval_seconds'] is None
    assert report['emergence_records'][0]['onset_by_time_s'] == 7.


def test_censoring_retains_one_sided_bounds_and_conflicts_never_become_onsets():
    records = [EmergenceInterval('left', None, {'frame': 5}),
               EmergenceInterval('right', {'frame': 8}, None),
               EmergenceInterval('unknown', None, None),
               EmergenceInterval('bad', {'frame': 8}, {'frame': 5}, contradictory=True)]
    rows = {r['grain_id']: r for r in germination_report(records)['emergence_records']}
    assert rows['left']['timing_status'] == 'left_censored'
    assert rows['left']['onset_by_frame'] == 5 and rows['left']['onset_after_frame'] is None
    assert rows['right']['timing_status'] == 'right_censored'
    assert rows['right']['onset_after_frame'] == 8
    assert rows['unknown']['timing_status'] == 'unknown'
    assert rows['bad']['timing_status'] == 'contradictory'
    assert rows['bad']['onset_after_frame'] is rows['bad']['onset_by_frame'] is None
    assert rows['bad']['last_verified_absent_frame'] == 8  # evidence retained


@pytest.mark.parametrize('times', [{'1': 2., '01': 3.}, {'1.5': 1.}, {'-1': 0.}, {'1': 2., '2': 1.}])
def test_invalid_acquisition_indices_or_time_order_are_rejected(times):
    with pytest.raises(ValueError):
        validate_acquisition({'source_times_s': times, 'cadence_source': 'log'})
