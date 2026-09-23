"""Fold active human points with revisioned, retractable source lineage.

Point reviews license points only. Missing/partial/withdrawn observations never
license an apex or a whole-frame negative. The input dataset remains unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tubetracker.annotation_frames import FrameReader
from tubetracker.label_lineage import (LABEL_FIELDS, LINEAGE_FIELDS, label_key,
    licensed_points, load_project, read_csv, reconcile, write_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-dir', required=True)
    ap.add_argument('--movie', required=True)
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--movie-tag', default='ld')
    ap.add_argument('--task-movie', default='', help='Required when the project contains several movies')
    ap.add_argument('--actor', default='fold')
    a = ap.parse_args()
    src, out = Path(a.dataset_dir).resolve(), Path(a.out_dir).resolve()
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    if src == out or src in out.parents:
        raise SystemExit('output must be outside the source dataset')
    evidence = load_project(a.project_dir)
    movies = {str(r['data'].get('movie', '')) for r in evidence['all'].get('task', [])}
    if not a.task_movie and len(movies) > 1:
        raise SystemExit('mixed-movie project: specify --task-movie')
    movie_id = a.task_movie or next(iter(movies), '')
    movie = Path(a.movie)
    movie_stat = movie.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    digest = hashlib.sha256()
    with movie.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    movie_hash = digest.hexdigest()
    desired = licensed_points(evidence, a.movie_tag, movie_id, movie_hash, a.actor)
    old_labels = read_csv(src/'labels.csv')
    labels, lineage, audit = reconcile(old_labels, read_csv(src/'label_lineage.csv'),
                                      evidence, desired, scope=(a.movie_tag, movie_id, movie_hash))
    manifest = json.loads((src/'manifest.json').read_text())
    frame_map = {r['image_name']: r for r in manifest['frames']}
    reviews = {r['image_name']: r for r in read_csv(src/'frame_reviews.csv')}
    desired_frames = {r['image_name']: int(r['source_frame']) for r, _ in desired}
    image_hashes = manifest.setdefault('frame_movie_sha256', {})
    for image in desired_frames:
        if image_hashes.get(image) not in (None, movie_hash):
            raise ValueError('Frame filename already belongs to another movie: '+image)
    review_baseline = manifest.setdefault('label_review_baseline', {})
    for image, channel in audit['managed_channels']:
        review_baseline.setdefault(image, {}).setdefault(channel,
            str(reviews.get(image, {}).get(channel+'_reviewed', '0')))
    for image, frame in desired_frames.items():
        frame_map.setdefault(image, {'image_name': image, 'source_frame': frame,
            'movie_tag': a.movie_tag, 'provenance': 'human-v1'})
        reviews.setdefault(image, {'image_name': image, 'source_frame': str(frame),
                                   'grain_reviewed': '0', 'tip_reviewed': '0'})
    licensed_channels = {(r['image_name'], r['label_type']) for r in lineage
                         if r.get('lineage_schema') == '2'}
    for image, channel in audit['managed_channels']:
        if image in reviews:
            inherited = review_baseline[image][channel] in ('1', 'True', 'true')
            reviews[image][channel+'_reviewed'] = str(int(inherited or (image, channel) in licensed_channels))
    for label, _ in desired:
        reviews[label['image_name']][label['label_type']+'_reviewed'] = '1'
    manifest['frames'] = list(frame_map.values())
    manifest['label_lineage_schema'] = 2
    proved = {label_key(r) for r in lineage if r.get('lineage_schema') == '2'}
    manifest['legacy_unverified_label_keys'] = [list(label_key(r)) for r in labels
        if 'human' in r.get('provenance', '') and label_key(r) not in proved]
    audit.update(source_project=evidence['project'], movie_id=movie_id,
        movie_sha256=movie_hash, movie_tag=a.movie_tag,
        active_source_labels=len(desired), output_labels=len(labels),
        unique_output_human_labels=sum('human' in r.get('provenance', '') for r in labels))
    # Plan and provenance checks precede all output. A failed frame read cannot
    # leave a directory that looks like a finished dataset.
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fold-', dir=out.parent) as tmp:
        staged = Path(tmp)/'dataset'
        shutil.copytree(src, staged)
        (staged/'frames').mkdir(exist_ok=True)
        write_csv(staged/'labels.csv', labels, LABEL_FIELDS)
        write_csv(staged/'label_lineage.csv', lineage, LINEAGE_FIELDS)
        write_csv(staged/'frame_reviews.csv', list(reviews.values()),
                  ['image_name', 'source_frame', 'grain_reviewed', 'tip_reviewed'])
        (staged/'fold_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        missing = [(image, frame) for image, frame in desired_frames.items()
                   if not (staged/'frames'/image).exists()]
        verify = [(image, frame) for image, frame in desired_frames.items()
                  if image not in image_hashes or not (staged/'frames'/image).exists()]
        if verify:
            reader = FrameReader(a.movie)
            try:
                for image, frame in verify:
                    pixels = reader.read(frame).frame
                    target = staged/'frames'/image
                    if target.exists():
                        if not np.array_equal(cv2.imread(str(target), cv2.IMREAD_UNCHANGED), pixels):
                            raise ValueError('Inherited frame pixels do not match this movie: '+image)
                    elif not cv2.imwrite(str(target), pixels):
                        raise RuntimeError('Failed to write frame '+image)
                    image_hashes[image] = movie_hash
            finally:
                reader.close()
        if identity(movie.stat()) != identity(movie_stat):
            raise RuntimeError('Movie changed during fold; no dataset published')
        (staged/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        os.rename(staged, out)
    print(json.dumps({'frames_written': len(missing), 'labels': len(labels),
        'active_source_labels': len(desired), 'withdrawn_or_revised_licenses_removed': len(audit['removed_licenses']),
        'excluded_reviews': len(audit['excluded_reviews']),
        'legacy_unverified_lineage_rows': audit['legacy_unverified_lineage_rows']}))


if __name__ == '__main__':
    main()
