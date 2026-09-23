"""Actual checkpoint loading and pixel inference across the review/cache boundary."""
from contextlib import closing
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prototypes.v30_video_apex import native_caps
from prototypes.v30_video_apex.native_body import NativeBodyNet, save_body_checkpoint
from tubetracker.analysis_contracts import AnalysisRequest
from tubetracker.annotation_store import AnnotationStore
from tubetracker.evidence_cache import BodyArrayStore
from tubetracker.movie_analysis import MovieAnalysisService, body_root_for


@pytest.mark.parametrize('root_channel', [False, True])
def test_real_root_capability_is_stable_and_review_pixels_never_enter_baseline(tmp_path, monkeypatch, root_channel):
    from tubetracker import annotation_frames
    torch.manual_seed(29)
    cap, body, movie = [tmp_path/n for n in ('cap.pt', 'body.pt', 'movie.bin')]
    movie.write_bytes(b'frame-reader-fixture')
    native_caps.save_native_checkpoint(cap, native_caps.NativeCapNet(base=4).eval(), manifest={})
    save_body_checkpoint(body, NativeBodyNet(base=4, root_channel=root_channel).eval(), {})
    yy, xx = np.mgrid[:64, :64]
    image = np.repeat(((xx+yy*2) % 256).astype(np.uint8)[..., None], 3, axis=2)
    class Reader:
        native_size = (64, 64)
        def __init__(self, path):
            pass
        def __len__(self):
            return 64
        def read(self, frame):
            return SimpleNamespace(frame=image, exact=True)
        def close(self):
            pass
    monkeypatch.setattr(annotation_frames, 'FrameReader', Reader)
    monkeypatch.setattr(native_caps, 'detect_caps', lambda *a, **k: {'caps': [], 'tile_hashes': []})
    request = AnalysisRequest(str(movie), 'm', [0, 30], owners=[{
        'id': 'a', 'grain_native': [8, 32], 'grain_radius_px': 3,
        'attachment_native': [10, 32], 'attachment_verified': True}])
    service = MovieAnalysisService(cap, body_checkpoint=body, cache_dir=tmp_path/'cache')
    first = service.analyze(request)
    repeated = service.analyze(request)
    assert first['cache']['pixel_key'] == repeated['cache']['pixel_key']
    assert repeated['cache']['measurement_hit']
    assert first['inputs']['pixel']['body_prompt_contract']['uses_root'] == root_channel

    def pixels(result):
        key = result['cache']['pixel_key']
        doc = json.loads((service.cache_dir/f'pixels-{key}.json').read_text())
        arrays = BodyArrayStore(service.cache_dir/f'pixels-{key}.arrays', doc['body_arrays']['entries'])
        return doc['frames']['30']['owners']['a'], arrays

    old_info, old_arrays = pixels(first)
    old_body = old_arrays[old_info['body_key']].copy()
    db = tmp_path/'annotations.db'
    with closing(AnnotationStore(db)) as store:
        store.save('task', 'full', {'movie': 'm', 'owner_uuid': 'a', 'task_type': 'centerline'})
        store.save('observation', 'full-path', {
            'task_uuid': 'full', 'owner_uuid': 'a', 'source_frame': 30,
            'direct_state': 'direct_visible', 'direct_xy': [50., 32.],
            'path_xy': [[14., 35.], [30., 32.], [50., 32.]], 'path_complete': True,
            'tip_source': 'reviewed_full_path', 'review_origin': 'human'})
        revised = service.analyze(request, review_entities=store.entities(), review_source=str(db))
        info, arrays = pixels(revised)
        np.testing.assert_array_equal(arrays[info['body_key']], old_body)
        assert first['model_without_reviews'] == revised['model_without_reviews']
        assert revised['cache']['pixel_hit'] is (not root_channel)
        assert ('root_assisted' in info) is root_channel
        if root_channel:
            assisted = info['root_assisted']
            assert info['prompt']['root_xy'] == [10., 32.]
            assert assisted['prompt']['root_xy'] == [14., 35.]
            assert assisted['attachment_source']['observation_id'] == 'full-path'
            assert not np.array_equal(arrays[assisted['body_key']], old_body)
        else:
            assert info['prompt']['root_xy'] is None
        store.delete('observation', 'full-path')
        withdrawn = service.analyze(request, review_entities=store.entities(), review_source=str(db),
                                    review_tombstones=store.tombstones())
    assert withdrawn['cache']['pixel_key'] == first['cache']['pixel_key']
    assert all(not row['path_complete'] for row in withdrawn['rows'])


def test_unverified_declared_attachment_does_not_become_a_body_root():
    assert body_root_for({'attachment_native': [4, 7]}) == (None, None)
    assert body_root_for({'root_native': [4, 7]}) == (None, None)
    assert body_root_for({'attachment_native': [4, 7], 'attachment_verified': True})[0] == [4., 7.]
    with pytest.raises(ValueError, match='finite native point'):
        body_root_for({'attachment_native': [4, float('nan')], 'attachment_verified': True})
