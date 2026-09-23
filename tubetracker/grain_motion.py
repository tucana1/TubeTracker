"""Offline grain motion from reviewed anchors and current-image evidence.

Outputs the shared model-pose contract. Detector/flow acceptance is a model
hypothesis, not human identity truth. Independent anchor disagreement and
ambiguous assignments remain explicit instead of being silently interpolated.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .analysis_contracts import stable_hash
from .frame_geometry import SCHEMA, grain_anchor_inputs
from .grain_identity import identity_review_hash


@dataclass(frozen=True)
class GrainMotionConfig:
    backend: str = 'local_flow_instances'
    maximum_frame_step: int = 300
    minimum_patch_correlation: float = .85
    maximum_fb_error_px: float = .5
    maximum_local_step_px: float = 5.
    maximum_stage_step_px: float = 32.
    minimum_flow_points: int = 12
    maximum_detection_correction_radii: float = 1.
    minimum_assignment_margin_px: float = 1.
    maximum_anchor_disagreement_px: float = 3.
    detector: dict = field(default_factory=dict)
    detector_device: str = 'cpu'
    detection_roi_xyxy: list | None = None

    @classmethod
    def from_dict(cls, value):
        cfg = cls(**(value or {}))
        if cfg.backend != 'local_flow_instances':
            raise ValueError('unknown grain motion provider')
        if (isinstance(cfg.maximum_frame_step, bool) or int(cfg.maximum_frame_step) != cfg.maximum_frame_step
                or cfg.maximum_frame_step < 1 or int(cfg.minimum_flow_points) != cfg.minimum_flow_points
                or not 1 <= cfg.minimum_flow_points <= 31):
            raise ValueError('grain motion needs positive source-frame steps and 1–31 flow points')
        numbers = [cfg.maximum_fb_error_px, cfg.maximum_local_step_px, cfg.maximum_stage_step_px,
                   cfg.maximum_detection_correction_radii, cfg.minimum_assignment_margin_px,
                   cfg.maximum_anchor_disagreement_px]
        if not np.isfinite(numbers).all() or min(numbers) <= 0 or not 0 <= cfg.minimum_patch_correlation <= 1:
            raise ValueError('invalid grain motion evidence thresholds')
        if cfg.detector_device not in ('cpu', 'mps', 'cuda'):
            raise ValueError('unknown grain detector device')
        if cfg.detector:
            from .grain_detection import GrainDetectionConfig
            cfg = cls(**{**asdict(cfg), 'detector': GrainDetectionConfig.from_dict(cfg.detector).to_dict()})
        if cfg.detection_roi_xyxy is not None:
            r = cfg.detection_roi_xyxy
            if len(r) != 4 or any(int(v) != v or v < 0 for v in r) or r[0] >= r[2] or r[1] >= r[3]:
                raise ValueError('grain detection ROI requires a native pixel rectangle')
        return cfg


def motion_dependencies(value, fp):
    """Can run in the Qt environment without importing inference runtimes."""
    cfg = GrainMotionConfig.from_dict(value)
    root = Path(__file__).resolve().parents[1]
    files = {name: fp.file(root/'tubetracker'/name) for name in
             ['grain_motion.py', 'grain_detection.py', 'frame_geometry.py', 'grain_identity.py',
              'annotation_frames.py', 'analysis_contracts.py', 'population.py']}
    if cfg.detector.get('checkpoint'):
        files['detector_weights'] = fp.file(cfg.detector['checkpoint'])
    return files


def assign_instances(predicted, owners, proposals, biases, cfg):
    """Global one-to-one assignment, with a second-best assignment margin."""
    from scipy.optimize import linear_sum_assignment
    if not len(owners):
        return predicted, []
    n, count = len(owners), len(proposals)
    points = np.asarray([d['xy'] for d in proposals], float).reshape(-1, 2)
    cost = np.full((n, count+n), 1e6)
    distances = np.zeros((n, count))
    for i, owner in enumerate(owners):
        radius = float(owner.get('grain_radius_px', 13))
        limit = radius * cfg.maximum_detection_correction_radii
        cost[i, count:] = limit + .1
        distances[i] = np.linalg.norm(points + biases[i] - predicted[i], axis=1)
        radius_cost = np.asarray([abs(d['radius']-radius)/radius for d in proposals])*2
        cost[i, :count] = np.where(distances[i] <= limit, distances[i]+radius_cost, 1e6)
    ri, ci = linear_sum_assignment(cost)
    best = float(cost[ri, ci].sum())
    evidence = [{} for _ in owners]
    corrected = predicted.copy()
    for i, col in zip(ri, ci):
        if col >= count:
            evidence[i] = {'assignment_accepted': False, 'reason': 'no_unique_current_instance'}
            continue
        alt = cost.copy(); alt[i, col] = 1e6
        ar, ac = linear_sum_assignment(alt)
        margin = float(alt[ar, ac].sum()-best)
        accepted = margin >= cfg.minimum_assignment_margin_px
        evidence[i] = {'assignment_accepted': accepted, 'assignment_margin_px': margin,
                       'detection_distance_px': float(distances[i, col]),
                       'detection_instance_id': str(proposals[col].get('instance_label', col)),
                       'reference_detection_bias': biases[i].tolist()}
        if accepted:
            corrected[i] = points[col] + biases[i]
    return corrected, evidence


def generate_grain_motion(request, owners, identity_reviews, movie_sha256, cache_dir,
                          *, progress=None, reader_factory=None, detector=None):
    import cv2
    from .analysis_dependencies import FileFingerprinter
    from .annotation_frames import FrameReader
    from .grain_detection import GrainDetector
    cfg = GrainMotionConfig.from_dict(request.grain_motion)
    fp = FileFingerprinter()
    files = motion_dependencies(request.grain_motion, fp)
    inputs = {'schema': SCHEMA, 'movie_sha256': movie_sha256, 'frames': request.frames,
        'owners': grain_anchor_inputs(owners), 'identity_reviews': identity_reviews,
        'roi_xyxy': request.roi_xyxy, 'configuration': asdict(cfg), 'files': files,
        'runtime': {'opencv': cv2.__version__, 'numpy': np.__version__}}
    key = stable_hash(inputs)
    directory = Path(cache_dir)/'grain-motion'/key
    target = directory/'poses.json'
    if target.exists():
        return str(target), {'cache_hit': True, 'input_identity': key}
    started = time.monotonic()
    progress = progress or (lambda event: None)
    reader = (reader_factory or FrameReader)(request.movie_path)
    images, hashes, phase = OrderedDict(), {}, {}
    detections = {}
    owner_map = {o['id']: o for o in owners}
    seeds = defaultdict(dict)
    for owner in owners:
        frame = owner.get('identified_at_frame')
        if frame is None or not owner.get('identity_verified'):
            reader.close()
            raise ValueError('grain motion requires a reviewed identity anchor for every selected grain')
        seeds[int(frame)][owner['id']] = {'xy': owner['grain_native'], 'basis': 'reviewed_inventory_anchor'}
    reviewed = defaultdict(list)
    for row in identity_reviews:
        reviewed[(row['owner_id'], row['source_frame'])].append(row)
        if row['identity_state'] == 'confirmed':
            seeds[row['source_frame']][row['owner_id']] = {'xy': row['grain_native'],
                'basis': 'reviewed_grain_identity', 'sources': [row]}
    schedule = sorted(set(request.frames) | set(seeds))
    # Add flow stepping frames independently of the output measurement schedule.
    for a, b in zip(schedule, schedule[1:]):
        schedule += list(range(a+cfg.maximum_frame_step, b, cfg.maximum_frame_step))
    schedule = sorted(set(schedule))
    if schedule[0] < 0 or schedule[-1] >= len(reader):
        reader.close()
        raise ValueError('grain motion frame or anchor lies outside the source movie')
    width, height = reader.native_size
    roi = list(request.roi_xyxy or [0, 0, width, height])
    for group in seeds.values():
        for seed in group.values():
            x, y = seed['xy']
            roi = [max(0, min(roi[0], int(x)-32)), max(0, min(roi[1], int(y)-32)),
                   min(width, max(roi[2], int(x)+33)), min(height, max(roi[3], int(y)+33))]
    origin = np.asarray(roi[:2], float)
    if cfg.detector and detector is None:
        detector = GrainDetector(cfg.detector, device=cfg.detector_device)
    detection_roi = cfg.detection_roi_xyxy or roi

    def image_at(frame):
        if frame not in images:
            read = reader.read(frame)
            if not read.exact:
                raise ValueError('grain motion refused an inexact source frame')
            full = cv2.cvtColor(read.frame, cv2.COLOR_BGR2GRAY) if read.frame.ndim == 3 else read.frame
            crop = full[roi[1]:roi[3], roi[0]:roi[2]].copy()
            images[frame] = crop
            hashes[str(frame)] = hashlib.sha256(crop.tobytes()).hexdigest()
            if detector is not None and frame in schedule and frame not in detections:
                proposals, audit = detector.detect(full, detection_roi)
                detections[frame] = [p for p in proposals if p.get('geometry_complete', True)]
            while len(images) > 16:
                images.popitem(last=False)
        images.move_to_end(frame)
        return images[frame]

    def flow(a_frame, b_frame, center, radius):
        a, b = image_at(a_frame), image_at(b_frame)
        pair = (a_frame, b_frame)
        if pair not in phase:
            window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
            phase[pair] = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32), window)
        shift, response = phase[pair]
        stage = np.asarray(shift) if response >= .2 else np.zeros(2)
        if not np.isfinite(stage).all() or np.linalg.norm(stage) > cfg.maximum_stage_step_px:
            stage = np.zeros(2)
        offsets = [[0., 0.]]
        for scale, count in [(.45, 6), (.75, 12), (.95, 12)]:
            angle = np.arange(count)*2*np.pi/count
            offsets.extend(np.column_stack([np.cos(angle), np.sin(angle)])*radius*scale)
        p = (center-origin+np.asarray(offsets)).astype(np.float32)[:, None]
        q, status, error = cv2.calcOpticalFlowPyrLK(a, b, p, (p+stage).astype(np.float32),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, winSize=(11,11), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.005))
        if q is None:
            return center, {'flow_accepted': False, 'reason': 'no_local_flow'}
        back, backward, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, winSize=(11,11), maxLevel=2)
        if back is None:
            return center, {'flow_accepted': False, 'reason': 'no_reverse_local_flow'}
        delta = (q-p)[:,0]
        fb = np.linalg.norm((back-p)[:,0],axis=1)
        good = ((status[:,0]>0) & (backward[:,0]>0) & np.isfinite(delta).all(1)
                & (fb<=cfg.maximum_fb_error_px) & (error[:,0]<=12))
        local = np.median(delta[good],axis=0) if good.any() else np.zeros(2)
        good &= np.linalg.norm(delta-local,axis=1)<=1.5
        local = np.median(delta[good],axis=0) if good.any() else np.zeros(2)
        predicted = center+local
        size = 2*int(round(radius))+1
        old = cv2.getRectSubPix(a,(size,size),tuple(map(float,center-origin))).astype(float)
        new = cv2.getRectSubPix(b,(size,size),tuple(map(float,predicted-origin))).astype(float)
        yy, xx = np.mgrid[:size,:size]
        circle = (xx-(size-1)/2)**2+(yy-(size-1)/2)**2 <= (radius*.95)**2
        x, y = old[circle], new[circle]; x-=x.mean(); y-=y.mean()
        den = np.linalg.norm(x)*np.linalg.norm(y)
        correlation = float(x@y/den) if den>0 else 0.
        accepted = bool(good.sum()>=cfg.minimum_flow_points and correlation>=cfg.minimum_patch_correlation
            and np.linalg.norm(local-stage)<=cfg.maximum_local_step_px
            and np.linalg.norm(local)<=cfg.maximum_stage_step_px)
        return predicted, {'flow_accepted': accepted, 'flow_points': int(good.sum()),
            'patch_correlation': correlation, 'stage_shift': stage.tolist(),
            'median_fb_error_px': float(np.median(fb[good])) if good.any() else None}

    def walk(a, b, center, radius):
        predicted, evidence = flow(a,b,center,radius)
        if evidence['flow_accepted'] or abs(b-a)<=1:
            return predicted, evidence
        middle = (a+b)//2
        first, e1 = walk(a,middle,center,radius)
        if not e1['flow_accepted']:
            return center, evidence
        second, e2 = walk(middle,b,first,radius)
        return (second, dict(e2, refined_interval=[a,b])) if e2['flow_accepted'] else (center,evidence)

    estimates = defaultdict(list)
    total = len(seeds)*len(schedule)
    completed = 0
    try:
        for anchor, group in sorted(seeds.items()):
            members = [owner_map[oid] for oid in sorted(group)]
            initial = np.asarray([group[o['id']]['xy'] for o in members],float)
            image_at(anchor)
            biases = np.zeros_like(initial)
            detector_members = [i for i,p in enumerate(initial) if detector is not None and
                detection_roi[0]<=p[0]<detection_roi[2] and detection_roi[1]<=p[1]<detection_roi[3]]
            if detector_members:
                pts, ev = assign_instances(initial[detector_members], [members[i] for i in detector_members],
                    detections[anchor], biases[detector_members],cfg)
                for k, i in enumerate(detector_members):
                    if ev[k].get('assignment_accepted'):
                        biases[i] = initial[i]-pts[k]
            index = schedule.index(anchor)
            for i, owner in enumerate(members):
                estimates[(owner['id'],anchor)].append({'anchor_frame':anchor,'xy':initial[i].tolist(),
                    'supported':True,'evidence':group[owner['id']]})
            for direction in [-1, 1]:
                current = initial.copy(); previous = anchor
                for frame in schedule[index+1:] if direction>0 else reversed(schedule[:index]):
                    predicted, evidence = [], []
                    for i, owner in enumerate(members):
                        point, ev = walk(previous,frame,current[i],owner.get('grain_radius_px',13))
                        predicted.append(point if ev['flow_accepted'] else current[i]); evidence.append(ev)
                    predicted = np.asarray(predicted)
                    if detector_members:
                        pts, ev = assign_instances(predicted[detector_members], [members[i] for i in detector_members],
                            detections[frame], biases[detector_members],cfg)
                        for k, i in enumerate(detector_members):
                            predicted[i]=pts[k]; evidence[i].update(ev[k])
                    for i, owner in enumerate(members):
                        supported = evidence[i]['flow_accepted'] and evidence[i].get('assignment_accepted',True)
                        estimates[(owner['id'],frame)].append({'anchor_frame':anchor,'xy':predicted[i].tolist(),
                            'supported':bool(supported),'evidence':evidence[i]})
                    current, previous = predicted, frame
                    completed += 1
                    progress({'stage':'grain motion','completed':completed,'total':total})
        poses = []
        for owner in owners:
            oid = owner['id']
            for frame in request.frames:
                choices = sorted(estimates[(oid,frame)],key=lambda e:abs(e['anchor_frame']-frame))
                lower = next((e for e in choices if e['anchor_frame']<=frame),None)
                upper = next((e for e in choices if e['anchor_frame']>=frame),None)
                chosen = choices[0]
                direct = chosen['anchor_frame']==frame
                disagreement = float(np.linalg.norm(np.asarray(lower['xy'])-upper['xy'])) if lower and upper else None
                supported = chosen['supported'] and (direct or disagreement is None or
                    (lower['supported'] and upper['supported'] and disagreement<=cfg.maximum_anchor_disagreement_px))
                point = chosen['xy']
                decisions = reviewed.get((oid,frame),[])
                state = 'current_image_supported' if supported else 'ambiguous'
                if any(r['identity_state']!='confirmed' for r in decisions):
                    supported = False
                    state = 'missing' if any(r['identity_state']=='out_of_field' for r in decisions) else 'ambiguous'
                    point = None
                evidence = dict(chosen['evidence'], reference_anchor_frame=chosen['anchor_frame'],
                    anchor_disagreement_px=disagreement, human_identity_reviews=decisions)
                poses.append({'owner_id':oid,'source_frame':frame,'grain_native':point,
                    'grain_radius_px':float(owner.get('grain_radius_px',13)), 'pose_status':state,
                    'identity_status':'tracked_provisional' if supported else 'unresolved',
                    'geometry_complete':bool(supported), 'review_origin':'model', 'evidence':evidence})
        duplicates = defaultdict(list)
        for pose in poses:
            token = pose['evidence'].get('detection_instance_id')
            if token and pose['geometry_complete']:
                duplicates[(pose['source_frame'],token)].append(pose)
        for group in duplicates.values():
            if len(group)>1:
                for pose in group:
                    pose.update(pose_status='ambiguous',identity_status='ambiguous',geometry_complete=False)
                    pose['evidence']['reason']='shared_current_instance'
        record = {'schema':SCHEMA,'movie_id':request.movie_id,'movie_sha256':movie_sha256,
            'coordinate_system':'native_pixels','image_size':[width,height], 'review_origin':'model',
            'provider':cfg.backend,'anchor_owners':grain_anchor_inputs(owners),
            'identity_review_hash':identity_review_hash(identity_reviews), 'identity_reviews':identity_reviews,
            'evidence_files':files, 'inputs':inputs, 'input_identity':key, 'motion_roi_xyxy':roi,
            'source_pixel_sha256':hashes,'poses':poses,'elapsed_s':time.monotonic()-started,
            'note':'Current-image tracking estimates, not independently validated human identities.'}
        if files != motion_dependencies(request.grain_motion,fp):
            raise RuntimeError('motion source or detector changed while running')
        directory.mkdir(parents=True,exist_ok=True)
        partial = target.with_suffix('.partial')
        partial.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        partial.replace(target)
        return str(target), {'cache_hit':False,'input_identity':key,'elapsed_s':record['elapsed_s']}
    finally:
        reader.close()
