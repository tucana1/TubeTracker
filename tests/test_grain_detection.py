import cv2
import numpy as np
import pytest

from tubetracker.grain_detection import GrainDetectionConfig, grain_proposals_from_labels, GrainDetector
from tubetracker.analysis_dependencies import FileFingerprinter, dependency_fingerprint


def test_compact_grains_are_separate_and_an_elongated_tube_is_not_a_grain():
    labels = np.zeros((90, 120), np.int32)
    cv2.circle(labels, (35, 30), 10, 1, -1)
    cv2.circle(labels, (56, 30), 10, 2, -1)
    cv2.rectangle(labels, (15, 65), (100, 76), 3, -1)
    proposals, report = grain_proposals_from_labels(labels, [200, 300], [200, 300, 320, 390])
    assert len(proposals) == 2
    assert [p['xy'] for p in proposals] == [[235., 330.], [256., 330.]]
    assert [p['instance_label'] for p in proposals] == [1, 2]
    assert report['instances'][2]['decision'] == 'elongated_or_merged_instance'
    assert not report['completeness_certified']


def test_truncated_instance_is_not_given_an_unbounded_radius_or_certified_identity():
    labels = np.ones((40, 50), np.int32)
    proposals, report = grain_proposals_from_labels(labels, [0, 0], [0, 0, 50, 40])
    assert len(proposals) == 1 and not proposals[0]['geometry_complete']
    instance = report['unresolved_instances'][0]
    assert instance['inscribed_radius_px'] <= 20
    assert instance['decision'] == 'grain_proposal_truncated'


def test_context_and_roi_are_in_native_coordinates_without_an_image_resize(tmp_path):
    checkpoint = tmp_path/'grain-model'
    checkpoint.write_bytes(b'fixture weights')
    seen = []
    class Model:
        def eval(self, image, **kwargs):
            seen.append(image.copy())
            labels = np.zeros_like(image, dtype=np.int32)
            cv2.circle(labels, (40, 40), 9, 1, -1)
            cv2.circle(labels, (15, 15), 8, 2, -1)  # center outside requested ROI
            return labels, None, None
    image = np.arange(120*130, dtype=np.uint8).reshape(120, 130)
    detector = GrainDetector({'backend': 'cpdino', 'checkpoint': str(checkpoint), 'context_px': 20,
                              'whole_frame_context': False}, model=Model())
    proposals, report = detector.detect(image, [40, 30, 100, 90])
    np.testing.assert_array_equal(seen[0], image[10:110, 20:120])
    assert [p['xy'] for p in proposals] == [[60., 50.]]
    assert report['input_origin'] == [20, 10]


def test_reporting_roi_cannot_change_default_microscopy_input_or_detected_centers(tmp_path):
    checkpoint = tmp_path/'grain-model'
    checkpoint.write_bytes(b'fixture')
    seen = []
    class Model:
        def eval(self, image, **kwargs):
            seen.append(image.copy())
            labels = np.zeros_like(image, dtype=np.int32)
            cv2.circle(labels, (50, 50), 10, 1, -1)
            return labels, None, None
    image = np.arange(120*130, dtype=np.uint8).reshape(120, 130)
    detector = GrainDetector({'backend': 'cpdino', 'checkpoint': str(checkpoint)}, model=Model())
    a, _ = detector.detect(image, [20, 20, 80, 80])
    b, _ = detector.detect(image, [40, 40, 70, 70])
    np.testing.assert_array_equal(seen[0], image)
    np.testing.assert_array_equal(seen[0], seen[1])
    assert a == b


def test_missing_checkpoint_never_silently_falls_back_or_downloads(tmp_path):
    with pytest.raises(FileNotFoundError, match='not installed'):
        GrainDetectionConfig.from_dict({'backend': 'cpdino', 'checkpoint': str(tmp_path/'missing')})


def test_grain_checkpoint_content_participates_in_dependency_invalidation(tmp_path):
    movie, weights = tmp_path/'movie', tmp_path/'weights'
    movie.write_bytes(b'movie'); weights.write_bytes(b'weights-1')
    config = {'request': {'movie_path': str(movie), 'discover_grains': True,
        'grain_detection': {'backend': 'cpdino', 'checkpoint': str(weights)}}}
    fp = FileFingerprinter()
    first = dependency_fingerprint(config, fp)
    weights.write_bytes(b'weights-2')
    second = dependency_fingerprint(config, fp)
    assert first['files']['grain_checkpoint']['sha256'] != second['files']['grain_checkpoint']['sha256']


def test_truncated_grain_rim_never_licenses_a_model_root():
    from prototypes.v30_video_apex.route_quality import inspect_route, geometry_fingerprint
    owner = {'id': 'g', 'grain_native': [20, 20], 'grain_radius_px': 10,
             'attachment_native': None, 'grain_geometry_complete': False}
    cap = {'tip_xy': [60, 20], 'probability': .99, 'source_frame': 1}
    body = np.full((80, 80), .99, np.float32)
    result = inspect_route(body, [0, 0], [[30, 20], [60, 20]], owner, cap)
    assert not result['geometrically_supported']
    assert 'grain_geometry_incomplete' in result['failures']
    complete = dict(owner, grain_geometry_complete=True)
    assert geometry_fingerprint([owner], None, 'grain_crop') != geometry_fingerprint([complete], None, 'grain_crop')
