"""Offline body proposals guided by current, provenance-checked owned masks.

The reviewed mask seeds remain human evidence at their own source frames.
Propagated masks are model proposals, including when they use later seeds.
SAM2 is optional and is imported only in the inference environment.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import sys
import threading

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter

from .analysis_contracts import stable_hash
from .analysis_dependencies import FileFingerprinter


@dataclass(frozen=True)
class SeededBodyConfig:
    backend: str = 'sam2_video'
    source_dir: str = ''
    checkpoint: str = ''
    model_config: str = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
    points_per_mask: int = 8
    seed_mask_ids: list[str] = field(default_factory=list)
    device: str = ''

    @classmethod
    def from_dict(cls, value):
        config = cls(**dict(value))
        if config.backend != 'sam2_video':
            raise ValueError('Unsupported reviewed-mask body backend')
        if config.device not in ('', 'cpu', 'mps', 'cuda'):
            raise ValueError('Reviewed-mask device must be cpu, mps, cuda or empty to inherit')
        source = Path(config.source_dir).resolve()
        checkpoint = Path(config.checkpoint).resolve()
        if not config.source_dir or not (source/'sam2/sam2_video_predictor.py').is_file():
            raise ValueError('Reviewed-mask tracking requires the configured official SAM2 source directory')
        if not config.checkpoint or not checkpoint.is_file():
            raise ValueError('Reviewed-mask tracking requires an installed SAM2 checkpoint')
        relative = Path(config.model_config)
        if relative.is_absolute() or '..' in relative.parts or not (source/'sam2'/relative).is_file():
            raise ValueError('SAM2 model configuration must exist inside its configured package')
        if (isinstance(config.points_per_mask, bool) or int(config.points_per_mask) != config.points_per_mask
                or not 1 <= config.points_per_mask <= 64):
            raise ValueError('Seed point count must be an integer from 1 to 64')
        if (not isinstance(config.seed_mask_ids, list)
                or any(not isinstance(s, str) or not s for s in config.seed_mask_ids)
                or len(set(config.seed_mask_ids)) != len(config.seed_mask_ids)):
            raise ValueError('Seed mask IDs must be distinct nonempty strings')
        return cls(**{**asdict(config), 'source_dir':str(source), 'checkpoint':str(checkpoint),
                      'points_per_mask':int(config.points_per_mask)})


def assistance_files(config, fingerprinter=None):
    cfg = SeededBodyConfig.from_dict(config)
    fp = fingerprinter or FileFingerprinter()
    package = Path(cfg.source_dir)/'sam2'
    files = {'checkpoint':fp.file(cfg.checkpoint)}
    for name in ('seeded_body.py', 'population.py', 'review_semantics.py', 'annotation_frames.py'):
        files['runtime/'+name] = fp.file(Path(__file__).parent/name)
    for path in sorted(package.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.yaml', '.yml'):
            files['package/'+str(path.relative_to(package))] = fp.file(path)
    return files


def mask_core_points(pixels, count):
    """Spread positive-only prompts along the actually reviewed interior."""
    mask = np.asarray(pixels, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError('A seed requires exact reviewed positive pixels')
    # Stored rasters may be tight bounding boxes. Explicit exterior zeros
    # make point selection independent of whether that box touches paint.
    distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    y, x = np.nonzero(mask & (distance >= maximum_filter(distance, size=3)))
    xy = np.column_stack((x, y)).astype(np.float32)
    chosen = [int(np.argmax(distance[y, x]))]
    remaining_distance = ((xy-xy[chosen[0]])**2).sum(1)
    for _ in range(1, min(count, len(xy))):
        index = int(np.argmax(remaining_distance))
        chosen.append(index)
        remaining_distance = np.minimum(remaining_distance, ((xy-xy[index])**2).sum(1))
    return xy[chosen]


def prepare_body_assistance(request, owners, entities, *, source='', tombstones=(),
                            fingerprinter=None, device='cpu'):
    if not request.body_assistance:
        return None
    from .annotation_frames import FrameReader
    from .population import resolve_owned_masks, reviewed_mask_pixels
    cfg = SeededBodyConfig.from_dict(request.body_assistance)
    if request.roi_xyxy is None:
        raise ValueError('Reviewed-mask tracking requires an explicit analysis field rectangle')
    reader = FrameReader(request.movie_path)
    try:
        image_size, frame_count = list(reader.native_size), len(reader)
    finally:
        reader.close()
    x0, y0, x1, y1 = request.roi_xyxy
    if x1 > image_size[0] or y1 > image_size[1]:
        raise ValueError('Reviewed-mask tracking field exceeds the source movie')
    masks, _ignored = resolve_owned_masks(entities, request, owners, source=source,
        tombstones=tombstones, image_size=image_size, all_movie_frames=True)
    if cfg.seed_mask_ids:
        counts = {uid:sum(m['evidence']['source_id']==uid for m in masks) for uid in cfg.seed_mask_ids}
        if any(n != 1 for n in counts.values()):
            raise ValueError('Requested seed mask is missing, excluded or ambiguous: '+
                             ', '.join(uid for uid,n in counts.items() if n != 1))
        masks = [m for m in masks if m['evidence']['source_id'] in cfg.seed_mask_ids]
    selected = {}
    for item in masks:
        evidence = item['evidence']
        if not 0 <= evidence['frame'] < frame_count:
            raise ValueError('Reviewed seed frame lies outside the source movie')
        positive, origin = reviewed_mask_pixels(item['mask'], image_size)
        y, x = np.mgrid[:positive.shape[0], :positive.shape[1]]
        inside = positive & (x+origin[0]>=x0) & (x+origin[0]<x1) & (y+origin[1]>=y0) & (y+origin[1]<y1)
        if not inside.any():
            if cfg.seed_mask_ids:
                raise ValueError('Requested seed has no reviewed tube pixels inside the selected field')
            continue
        seed = {**evidence, 'object_id':0,
                'points_native':(mask_core_points(inside,cfg.points_per_mask)+origin).tolist(),
                'positive_pixels_inside_roi':int(inside.sum())}
        seed['labels'] = [1]*len(seed['points_native'])
        previous = selected.get(seed['owner_id'])
        if previous is not None and cfg.seed_mask_ids:
            raise ValueError('Choose one reviewed seed mask per physical grain')
        # An explicit, deterministic one-mask policy; it consumes no held-out
        # tip/path targets and cannot silently prefer a model's best result.
        rank = lambda s:(s['frame'],s['revision'],s['source_project'],s['source_id'])
        if previous is None or rank(seed) > rank(previous):
            selected[seed['owner_id']] = seed
    if not selected:
        raise ValueError('No current eligible owned tube mask is available for reviewed-mask tracking')
    seeds = [selected[key] for key in sorted(selected)]
    for index, seed in enumerate(seeds, 1):
        seed['object_id'] = index
    movie_frames = sorted(set(request.frames) | {s['frame'] for s in seeds})
    plan = {'schema':'tubetracker.reviewed_mask_assistance.v1', 'config':asdict(cfg),
        'effective_device':cfg.device or device,
        'files':assistance_files(asdict(cfg),fingerprinter), 'movie':request.movie_id,
        'roi_xyxy':list(request.roi_xyxy),'analysis_frames':list(request.frames),
        'source_frames':movie_frames,'image_size':image_size,'seeds':seeds,
        'selection':'latest eligible reviewed mask per owner, or explicit unique mask IDs',
        'prediction_selection':'reverse at/before each seed; independent forward state afterward',
        'normalization':'lossless native grayscale; official PIL RGB resize; ImageNet mean/std',
        'memory_attention_dtype':'on MPS, stored bfloat16 memory is converted to float32 before attention',
        'classification':'human-seed-assisted model proposals; never propagated human truth',
        'postprocessing':False,'non_overlap_masks':False}
    plan['identity'] = stable_hash(plan)
    return plan


class LosslessFrameSequence:
    """SAM2-compatible lazy source frames with a bounded normalized-image cache."""
    def __init__(self, movie_path, frames, roi, image_size):
        from .annotation_frames import FrameReader
        self.reader = FrameReader(movie_path)
        self.frames, self.roi, self.image_size = list(frames), list(roi), image_size
        self.cache = OrderedDict()
        self.pixel_hashes = {}

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        import cv2
        import torch
        from PIL import Image
        index = int(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        frame = self.frames[index]
        read = self.reader.read(frame)
        if not read.exact:
            raise ValueError(f'Inexact source frame {frame} during reviewed-mask tracking')
        x0,y0,x1,y1 = self.roi
        gray = cv2.cvtColor(read.frame,cv2.COLOR_BGR2GRAY)[y0:y1,x0:x1]
        digest = hashlib.sha256(gray.tobytes()).hexdigest()
        if str(frame) in self.pixel_hashes and self.pixel_hashes[str(frame)] != digest:
            raise ValueError('Source pixels changed during offline propagation')
        self.pixel_hashes[str(frame)] = digest
        resized = np.array(Image.fromarray(gray).convert('RGB').resize((self.image_size,self.image_size)))
        tensor = torch.from_numpy(resized/255.0).permute(2,0,1).float()
        tensor.sub_(torch.tensor([.485,.456,.406])[:,None,None])
        tensor.div_(torch.tensor([.229,.224,.225])[:,None,None])
        self.cache[index] = tensor
        while len(self.cache)>2:
            self.cache.popitem(last=False)
        return tensor

    def close(self):
        self.cache.clear()
        self.reader.close()


_SAM_LOADER_LOCK = threading.Lock()


def run_sam2_body(request, plan, arrays, *, device, progress):
    """Write immutable native probability maps; return their per-owner bindings."""
    import torch
    from scipy.special import expit
    source = Path(plan['config']['source_dir'])
    if str(source) not in sys.path:
        sys.path.insert(0,str(source))
    import sam2
    if Path(sam2.__file__).resolve().parent != (source/'sam2').resolve():
        raise RuntimeError('This worker loaded a different SAM2 source; restart with the selected configuration')
    from sam2.build_sam import build_sam2_video_predictor
    import sam2.sam2_video_predictor as video_module
    predictor = build_sam2_video_predictor(plan['config']['model_config'],plan['config']['checkpoint'],
                                          device=device,apply_postprocessing=False)
    hook = None
    if str(device).startswith('mps'):
        def float32_memory(module,args,kwargs):
            def convert(value):
                if isinstance(value,torch.Tensor) and value.is_floating_point(): return value.float()
                if isinstance(value,list): return [convert(v) for v in value]
                if isinstance(value,tuple): return tuple(convert(v) for v in value)
                if isinstance(value,dict): return {k:convert(v) for k,v in value.items()}
                return value
            return convert(args),convert(kwargs)
        hook = predictor.memory_attention.register_forward_pre_hook(float32_memory,with_kwargs=True)
    frames,roi,seeds = plan['source_frames'],plan['roi_xyxy'],plan['seeds']
    sequence = LosslessFrameSequence(request.movie_path,frames,roi,predictor.image_size)
    by_id = {s['object_id']:s for s in seeds}
    requested = set(request.frames)
    bindings = {str(f):{} for f in request.frames}
    def loader(video_path,image_size,offload_video_to_cpu,**_kwargs):
        if image_size != predictor.image_size or not offload_video_to_cpu:
            raise ValueError('Unexpected SAM2 frame-loading contract')
        return sequence,roi[3]-roi[1],roi[2]-roi[0]
    try:
        with torch.inference_mode():
            for direction in ('forward','reverse'):
                # SAM2's directory loader accepts JPEG only. Substitute its
                # loader for initialization, restore it immediately, and retain
                # the original normalization with lossless source pixels.
                with _SAM_LOADER_LOCK:
                    original_loader = video_module.load_video_frames
                    try:
                        video_module.load_video_frames = loader
                        state = predictor.init_state(request.movie_path,
                            offload_video_to_cpu=True,offload_state_to_cpu=True)
                    finally:
                        video_module.load_video_frames = original_loader
                for seed in seeds:
                    predictor.add_new_points_or_box(state,frame_idx=frames.index(seed['frame']),
                        obj_id=seed['object_id'],points=np.asarray(seed['points_native'])-roi[:2],
                        labels=np.ones(len(seed['points_native']),dtype=np.int32))
                reverse = direction=='reverse'
                seed_indices = [frames.index(s['frame']) for s in seeds]
                start = max(seed_indices) if reverse else min(seed_indices)
                for index,ids,logits in predictor.propagate_in_video(state,start_frame_idx=start,reverse=reverse):
                    frame = frames[index]
                    if frame in requested:
                        values = logits.detach().float().cpu().numpy()[:,0]
                        for object_id,value in zip(ids,values):
                            seed = by_id[object_id]
                            if (frame<=seed['frame']) != reverse:
                                continue
                            if value.shape != (roi[3]-roi[1],roi[2]-roi[0]) or not np.isfinite(value).all():
                                raise ValueError('SAM2 returned invalid native body evidence')
                            key = 'seeded_'+plan['identity'][:16]+f'_f{frame}_o{object_id}'
                            arrays[key] = expit(value).astype(np.float32)
                            bindings[str(frame)][seed['owner_id']] = {
                                'body_key':key,'origin':roi[:2],
                                'source_frames':frames,'missing_frames':[],
                                'extent':{'kind':'analysis_roi','origin':roi[:2],'shape':list(value.shape)},
                                'attachment_source':None,
                                'prompt':{'schema':plan['schema'],'backend':'sam2_video',
                                    'assistance_identity':plan['identity'],'root_channel':False,
                                    'attachment_supplied':False,'direction':direction,
                                    'seed':{k:seed[k] for k in ('source_id','source_project','revision','frame','mask_sha256')},
                                    'classification':plan['classification']}}
                    progress({'stage':'reviewed-mask body '+direction,'completed':index+1,
                              'total':len(frames),'source_frame':frame})
                del state
                if str(device).startswith('mps'):
                    torch.mps.empty_cache()
        if any(set(row)!={s['owner_id'] for s in seeds} for row in bindings.values()):
            raise ValueError('Offline body propagation did not cover every requested owner/frame')
        return bindings,{'plan_identity':plan['identity'],'frame_pixel_sha256':sequence.pixel_hashes,
                         'classification':plan['classification'],'device':str(device),
                         'torch_version':torch.__version__}
    finally:
        sequence.close()
        if hook is not None:
            hook.remove()
        if 'state' in locals():
            del state
        del predictor
        if str(device).startswith('mps'):
            torch.mps.empty_cache()


def apply_body_assistance(request, pixels, arrays, plan, *, device, progress):
    if plan is None:
        return
    bindings,receipt = run_sam2_body(request,plan,arrays,
        device=plan.get('effective_device', device),progress=progress)
    expected_owners = {s['owner_id'] for s in plan['seeds']}
    if (set(bindings) != {str(f) for f in request.frames}
            or any(set(owners) != expected_owners for owners in bindings.values())):
        raise ValueError('Offline body propagation returned incomplete owner/frame bindings')
    roi = plan['roi_xyxy']
    for owners in bindings.values():
        for info in owners.values():
            value = arrays[info['body_key']]
            if (value.shape != (roi[3]-roi[1], roi[2]-roi[0])
                    or info['origin'] != roi[:2] or not np.isfinite(value).all()
                    or value.min() < 0 or value.max() > 1
                    or info.get('prompt', {}).get('assistance_identity') != plan['identity']):
                raise ValueError('Offline body propagation returned invalid native probability bindings')
    for frame,owners in bindings.items():
        for owner_id,info in owners.items():
            if owner_id not in pixels['frames'][frame]['owners']:
                raise ValueError('Assisted body belongs to an owner outside the analysis inventory')
            # Root edits cannot change SAM2's root-free pixel evidence. Replace
            # both native-mode bindings; unseeded owners retain native evidence.
            pixels['frames'][frame]['owners'][owner_id] = info
    pixels['body_assistance'] = receipt
