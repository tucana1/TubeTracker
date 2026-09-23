"""WO2 queue builder: propose germination-episode tasks, never publish them.

Inputs: a cf70-style rerun report (per-frame states + presence logits),
the movie grain registry / poses, and the snapshot's existing event and
presence answers. Outputs proposals JSON where EVERY task carries its
grain/movie/window, the specific unresolved error, why existing answers
cannot resolve it, the cheapest sufficient answer, the consuming loss or
frozen evaluation, its train/calibration/held-out role, and the
before/after comparison that will measure its contribution. A task with
no consumer or no actionable comparison raises instead of entering.

The live annotation DB is never touched: publishing is a separate,
explicit step after the existing-label comparison passes review.
"""
import json
import sys
from pathlib import Path

ROOT = Path('/Users/joshjiang/Documents/TubeTracker')
sys.path.insert(0, str(ROOT))
from tubetracker.annotation_tasks import germination_episode  # noqa: E402

MAX_BATCH = 12


def cf70_reference_event(movie_id, movie_hash):
    """Seed reference event from the existing reviewed bracket (300,6000]."""
    return {"uuid": "event-cf70-seed-bracket",
            "movie_uuid": movie_id, "movie_content_hash": movie_hash,
            "owner_uuid": "ld|review-53b55d8191864f77b5ed91beef56cf70",
            "task_uuid": "", "window_start": 0, "window_end": 9000,
            "verdict": "emerged_within",
            "last_absent_frame": 300, "first_visible_frame": 6000,
            "consulted_frames": [0, 300, 6000, 9000],
            "annotator": "reviewer", "revision": 1,
            "lineage": ["audit 2026-09-21: absent at 0/300, emerged at 6000/9000",
                        "bracket (300,6000] reused verbatim; no new judgement"]}


def propose_from_rerun(rerun_report, movie_id, movie_hash):
    """Candidate episodes from measured rerun errors (strata, not quotas)."""
    runs = rerun_report.get('runs', {})
    base = runs.get('disabled', {})
    props = []
    for frame, row in sorted(base.items(), key=lambda kv: int(kv[0])):
        f = int(frame)
        state, logit = row.get('state'), row.get('presence_logit')
        if state == 'present' and f <= 300:
            props.append(dict(
                grain_id='ld|review-53b55d8191864f77b5ed91beef56cf70',
                movie_id=movie_id, movie_content_hash=movie_hash,
                window_start=max(0, f - 300), window_end=f + 300,
                unresolved_error=(
                    f'model present at frame {f} with no reviewed emergence; '
                    f'presence_logit={logit}'),
                why_existing_insufficient=(
                    'existing absence answers are scalar verdicts without a '
                    'reviewed region; they cannot train a scoped pixel-rejection loss'),
                cheapest_answer='temporal verdict + bracket; scoped dense absence '
                                'only if a pixel-rejection loss will consume it',
                consuming_loss_or_eval='presence BCE (scalar) then frozen '
                                       'cf70 gate 0/300 absent',
                role='training',
                before_after_comparison='rerun states at 0/300 before vs after; '
                                       'false-presence count on reviewed bare rims',
                stratum='bare-rim-control'))
    return props


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    report_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    movie_id = sys.argv[3] if len(sys.argv) > 3 else 'ld'
    movie_hash = sys.argv[4] if len(sys.argv) > 4 else ''
    proposals = []
    if report_path and Path(report_path).exists():
        report = json.loads(Path(report_path).read_text())
        for p in propose_from_rerun(report, movie_id, movie_hash):
            proposals.append(germination_episode(**p))
    if len(proposals) > MAX_BATCH:
        raise ValueError(f'{len(proposals)} proposals exceed the {MAX_BATCH} cap; '
                         'narrow by measured error, not quota')
    seed = cf70_reference_event(movie_id, movie_hash)
    doc = {'reference_event_from_existing_answers': seed,
           'proposals': proposals,
           'published_to_db': False,
           'note': 'existing-label comparison first; publish only the smallest '
                   'set that tests a remaining failure'}
    if out:
        out.write_text(json.dumps(doc, indent=1) + '\n')
    print(json.dumps({'proposals': len(proposals),
                      'reference_verdict': seed['verdict'],
                      'reference_bracket': [seed['last_absent_frame'],
                                            seed['first_visible_frame']],
                      'published_to_db': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
