"""rev11 item 6: owner->trace cross-frame matching rules.

Pins the DECLARED identity tests of build_owner_tracks:
- start within 6 px of the human attachment -> matched (distance
  alone; tube bases curve);
- 6-12 px -> matched only with initial-direction agreement
  (cos >= 0.7);
- > 12 px, or bad direction in the 6-12 px band -> UNMATCHED
  (stated, never guessed).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _write_csv(path: Path, rows: list[tuple]) -> None:
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['pollen_id', 'sample_index', 'source_frame',
                    'point_index', 'source_x_px', 'source_y_px',
                    'arc_length_px', 'causal_status'])
        for pid, f, pts in rows:
            for i, (x, y) in enumerate(pts):
                w.writerow([pid, 0, f, i, x, y,
                            float(i) * 4.0, 'measured'])


def _run(tmp_path: Path):
    import scripts.build_owner_tracks as bt
    owners = {'owners': [
        {'id': 'own-t-0001', 'movie': 'ld',
         'grain_native': [90.0, 100.0], 'attachment_native': [100.0, 100.0],
         'no_tube': False, 'identified_at_frame': 1000},
        {'id': 'own-t-0002', 'movie': 'ld',
         'grain_native': [190.0, 200.0], 'attachment_native': [200.0, 200.0],
         'no_tube': False, 'identified_at_frame': 1000},
        {'id': 'own-t-0003', 'movie': 'ld',
         'grain_native': [290.0, 300.0], 'attachment_native': [300.0, 300.0],
         'no_tube': False, 'identified_at_frame': 1000},
    ]}
    (tmp_path / 'owners.json').write_text(json.dumps(owners))
    _write_csv(tmp_path / 'centerlines.csv', [
        # near start (0.7 px) -> matched on distance alone
        (5, 1000, [(99.5, 100.5), (96.0, 103.0), (92.0, 106.0)]),
        # 8 px away, direction +x (agrees with grain->att) -> matched
        (6, 1000, [(208.0, 200.0), (220.0, 200.0), (232.0, 200.0)]),
        # 8 px away but direction -x (toward the grain) -> UNMATCHED
        (7, 1000, [(308.0, 300.0), (296.0, 300.0), (284.0, 300.0)]),
    ])
    bt.OWNERS = tmp_path / 'owners.json'
    bt.TRACES = tmp_path / 'centerlines.csv'
    bt.OUT = tmp_path / 'tracks.json'
    assert bt.main() == 0
    return json.loads((tmp_path / 'tracks.json').read_text())['owners']


def test_near_start_matches_on_distance_alone(tmp_path):
    out = _run(tmp_path)
    r = out['own-t-0001']
    assert r['matched'] is True
    assert r['pid'] == 5
    assert r['match_dist_px'] <= 6.0
    assert r['confidence'] == 'confirmed'


def test_mid_band_matches_with_direction_agreement(tmp_path):
    out = _run(tmp_path)
    r = out['own-t-0002']
    assert r['matched'] is True
    assert r['pid'] == 6
    assert r['match_cos'] is not None and r['match_cos'] >= 0.7


def test_mid_band_bad_direction_is_unmatched(tmp_path):
    out = _run(tmp_path)
    r = out['own-t-0003']
    assert r['matched'] is False
    assert 'direction' in r['reason']
    assert r['best_pid'] == 7


def test_output_declares_time_and_calibration(tmp_path):
    import scripts.build_owner_tracks as bt
    _run(tmp_path)
    doc = json.loads((tmp_path / 'tracks.json').read_text())
    assert doc['acquisition_cadence_s'] == 3.0
    assert 'pixels' in doc['calibration']
