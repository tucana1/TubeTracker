import numpy as np
import pytest

from tubetracker.evidence_cache import BodyArrayStore


def test_body_cache_reopens_lazily_and_rejects_corruption(tmp_path):
    store = BodyArrayStore(tmp_path)
    for i in range(32):
        store[str(i)] = np.full((288, 288), i/32, dtype=np.float32)
    reopened = BodyArrayStore(tmp_path, store.manifest()["entries"])
    assert not reopened._verified
    value = reopened["17"]
    assert isinstance(value, np.memmap) and not value.flags.writeable
    np.testing.assert_array_equal(value, np.full((288, 288), 17/32, np.float32))
    assert reopened._verified == {"17"}
    entry = store.entries["18"]
    (tmp_path / entry["file"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        reopened["18"]
    with pytest.raises(ValueError, match="immutable"):
        store["17"] = value


def test_concurrent_same_key_writers_keep_each_manifest_correct(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(8)
    def publish(i):
        store = BodyArrayStore(tmp_path)
        expected = np.full((288, 288), i//2, np.float32)
        barrier.wait()
        store['shared-owner-frame'] = expected
        return store.manifest(), expected
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(publish, range(8)))
    for manifest, expected in results:
        reopened = BodyArrayStore(tmp_path, manifest['entries'])
        np.testing.assert_array_equal(reopened['shared-owner-frame'], expected)
    assert len(list(tmp_path.glob('*.npy'))) == 4  # identical bytes are shared
    assert not list(tmp_path.glob('*.tmp'))


def test_publication_never_replaces_corrupt_content_at_an_immutable_name(tmp_path):
    store = BodyArrayStore(tmp_path)
    value = np.ones((8, 8), np.float32)
    store['key'] = value
    path = tmp_path / store.entries['key']['file']
    path.write_bytes(b'broken')
    other = BodyArrayStore(tmp_path)
    with pytest.raises(ValueError, match='corrupt'):
        other['key'] = value
    assert path.read_bytes() == b'broken' and not other.entries
    assert not list(tmp_path.glob('*.tmp'))


def test_concurrent_json_publishers_never_share_a_temporary_file(tmp_path):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from tubetracker.movie_analysis import _write_json
    barrier = Barrier(8)
    target = tmp_path / 'pixels.json'
    def publish(i):
        value = {'worker': i, 'payload': [i]*4000}
        barrier.wait()
        _write_json(target, value)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(publish, range(8)))
    final = json.loads(target.read_text())
    assert final['payload'] == [final['worker']]*4000
    assert not list(tmp_path.glob('*.tmp'))
