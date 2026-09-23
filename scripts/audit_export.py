"""Audit exact current label sources, revisions and training eligibility."""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tubetracker.label_lineage import (audit_labels, label_key, load_project,
    read_csv, write_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-dir', required=True)
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--movie-tag', default='ld')
    ap.add_argument('--task-movie', default='')
    ap.add_argument('--validation-fraction', type=float, default=.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ds = Path(a.dataset_dir)
    manifest = json.loads((ds/'manifest.json').read_text())
    reviews = {r['image_name']: r for r in read_csv(ds/'frame_reviews.csv')}
    labels, lineage = read_csv(ds/'labels.csv'), read_csv(ds/'label_lineage.csv')
    evidence = load_project(a.project_dir)
    validated, errors, legacy = audit_labels(labels, lineage, reviews)
    all_ids = {r['uuid'] for rows in evidence['all'].values() for r in rows}
    for row in legacy:
        if row.get('task_uuid') in all_ids or row.get('obs_uuid') in all_ids:
            errors.append({'row': row, 'reason': 'legacy lineage cannot prove project identity or active revision'})
    known_images = {r['image_name'] for r in manifest['frames']}
    eligible = list(dict.fromkeys(r['image_name'] for r in manifest['frames']
        if any(str(reviews.get(r['image_name'], {}).get(c+'_reviewed')) in ('1', 'True', 'true')
               for c in ('grain', 'tip'))))
    random.Random(a.seed).shuffle(eligible)
    n_val = max(1, int(round(len(eligible)*a.validation_fraction)))
    split = {name: 'val' if i < n_val else 'train' for i, name in enumerate(eligible)}
    proved = {label_key(r) for r in validated}
    declared_legacy = {tuple(r) for r in manifest.get('legacy_unverified_label_keys', [])}
    for row in labels:
        if ('human' in row.get('provenance', '') and label_key(row) not in proved
                and label_key(row) not in declared_legacy):
            errors.append({'row': row, 'reason': 'human label has no verified active source'})
    scoped = [r for r in validated if r['source_project'] == evidence['project']
        and r['movie_tag'] == a.movie_tag and (not a.task_movie or r['movie_id'] == a.task_movie)]
    for row in scoped:
        if row['image_name'] not in known_images:
            errors.append({'row': row, 'reason': 'label missing from manifest'})
    rows = [dict(task=r['task_uuid'], source_project=r['source_project'],
        source_frame=int(r['image_name'].rsplit('_', 1)[-1].split('.')[0]),
        image=r['image_name'], label_type=r['label_type'], x=r['x'], y=r['y'],
        obs_uuid=r['obs_uuid'], obs_revision=r['obs_revision'], task_revision=r['task_revision'],
        split=split.get(r['image_name'], 'ineligible'), eligible=int(r['image_name'] in split)) for r in scoped]
    if a.out:
        write_csv(a.out, rows, ['task', 'source_project', 'source_frame', 'image', 'label_type',
            'x', 'y', 'obs_uuid', 'obs_revision', 'task_revision', 'split', 'eligible'])
    tip_keys = {label_key(r) for r in scoped if r['label_type'] == 'tip' and r['image_name'] in split}
    human = [r for r in labels if 'human' in r.get('provenance', '')]
    if human and not (proved & {label_key(r) for r in human}):
        errors.append({'reason': 'dataset claims human supervision but has zero verified eligible human labels'})
    report = {'eligible_frames': len({r['image'] for r in rows if r['eligible']}),
        'eligible_human_tips': len(tip_keys), 'verified_human_labels': len(proved & {label_key(r) for r in human}),
        'legacy_unverified_human_labels': len({label_key(r) for r in human} - proved),
        'legacy_unverified_lineage_rows': len(legacy), 'errors': errors}
    print(json.dumps(report, indent=2))
    print('AUDIT FAIL' if errors else 'AUDIT PASS (active sources verified; legacy coverage reported separately)')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
