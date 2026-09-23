"""Native grain proposals, separate from reviewed inventory and tube evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GrainDetectionConfig:
    backend: str = 'radial'
    checkpoint: str = ''
    context_px: int = 32
    whole_frame_context: bool = True
    minimum_radius_px: float = 5.
    minimum_compactness: float = .42
    flow_threshold: float = .4
    cellprob_threshold: float = 0.
    minimum_area_px: int = 30

    @classmethod
    def from_dict(cls, value=None):
        values = dict(value or {})
        if values.get('backend') == 'cpdino' and not values.get('checkpoint'):
            values['checkpoint'] = str(Path.home() / '.cellpose/models/cpdino-vitb')
        cfg = cls(**values)
        cfg.validate()
        return cls(**{**asdict(cfg), 'context_px': int(cfg.context_px),
                      'minimum_area_px': int(cfg.minimum_area_px)})

    def validate(self):
        if self.backend not in ('radial', 'cpdino'):
            raise ValueError('Grain detector must be radial or cpdino')
        if not isinstance(self.whole_frame_context, bool):
            raise ValueError('whole_frame_context must be a boolean')
        if (int(self.context_px) != self.context_px or self.context_px < 0
                or int(self.minimum_area_px) != self.minimum_area_px or self.minimum_area_px < 1):
            raise ValueError('Grain context and minimum area require valid integer pixels')
        if not np.isfinite([self.minimum_radius_px, self.minimum_compactness,
                            self.flow_threshold, self.cellprob_threshold]).all():
            raise ValueError('Grain detector settings must be finite')
        if self.minimum_radius_px <= 0 or not 0 < self.minimum_compactness <= 1 or self.flow_threshold <= 0:
            raise ValueError('Invalid grain radius, compactness or flow threshold')
        if self.backend == 'cpdino' and not Path(self.checkpoint).is_file():
            # Cellpose otherwise silently falls back and may download a
            # different model. This analysis must use its declared weights.
            raise FileNotFoundError(f'Grain model is not installed: {self.checkpoint}')

    def to_dict(self):
        return asdict(self)


def grain_proposals_from_labels(labels, origin, roi_xyxy, config=None):
    """One center per compact instance; elongated or truncated masks are audited."""
    from scipy.ndimage import binary_fill_holes, distance_transform_edt, find_objects

    cfg = config or GrainDetectionConfig()
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError('Grain instance labels must be a 2-D integer image')
    x0, y0, x1, y1 = roi_xyxy
    ox, oy = origin
    proposals, instances = [], []
    for label, box in enumerate(find_objects(labels), 1):
        if box is None:
            continue
        mask = binary_fill_holes(labels[box] == label)
        # Padding prevents an image-edge mask from having an unbounded
        # inscribed distance. An incomplete instance stays unresolved.
        distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        py, px = np.unravel_index(np.argmax(distance), distance.shape)
        radius, area = float(distance[py, px]), int(mask.sum())
        compactness = min(1., float(np.pi * radius * radius / area))
        xy = [float(px + ox + box[1].start), float(py + oy + box[0].start)]
        border = bool((box[0].start == 0 and mask[0].any())
                      or (box[0].stop == labels.shape[0] and mask[-1].any())
                      or (box[1].start == 0 and mask[:, 0].any())
                      or (box[1].stop == labels.shape[1] and mask[:, -1].any()))
        item = {'instance_label': int(label), 'xy': xy, 'area_px': area,
                'inscribed_radius_px': radius, 'compactness': compactness,
                'touches_image_border': border}
        if not (x0 <= xy[0] < x1 and y0 <= xy[1] < y1):
            item['decision'] = 'outside_requested_roi'
        elif radius < cfg.minimum_radius_px or area < cfg.minimum_area_px:
            item['decision'] = 'below_grain_size'
        elif compactness < cfg.minimum_compactness:
            item['decision'] = 'elongated_or_merged_instance'
        else:
            item['decision'] = 'grain_proposal_truncated' if border else 'grain_proposal'
            proposals.append({'xy': xy, 'radius': radius, 'score': compactness,
                'instance_label': int(label), 'compactness': compactness,
                'geometry_complete': not border,
                'center_basis': 'visible_body_inscribed_disc',
                'source': 'automatic microscopy instance proposal',
                'score_kind': 'inscribed_disc_compactness_not_probability'})
        instances.append(item)
    return proposals, {'instances': instances,
        'unresolved_instances': [i for i in instances if i['decision'] in
                                ('grain_proposal_truncated', 'elongated_or_merged_instance')],
        'completeness_certified': False}


class GrainDetector:
    def __init__(self, config=None, *, device='cpu', model=None):
        self.config = GrainDetectionConfig.from_dict(config)
        self.device = device
        self.model = model

    def detect(self, image, roi_xyxy=None):
        gray = np.asarray(image)
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError('Grain discovery needs native uint8 grayscale pixels')
        height, width = gray.shape
        roi = list(roi_xyxy or [0, 0, width, height])
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('Grain discovery ROI leaves the image')
        cfg = self.config
        pad = cfg.context_px
        ox, oy = max(0, x0-pad), max(0, y0-pad)
        ex, ey = min(width, x1+pad), min(height, y1+pad)
        if cfg.backend == 'cpdino' and cfg.whole_frame_context:
            # The segmentation/normalization input must not change when the
            # user changes a reporting ROI around the very same grains.
            ox, oy, ex, ey = 0, 0, width, height
        crop = gray[oy:ey, ox:ex]
        report = {'config': cfg.to_dict(), 'input_origin': [ox, oy],
                  'input_shape': list(crop.shape), 'requested_roi_xyxy': roi,
                  'native_spatial_rescaling': False}
        if cfg.backend == 'radial':
            from prototypes.timesfm_tip_forecast.grain_detect import (
                detect_grains, peak_radii, grain_symmetry_maps)
            detections = detect_grains(crop)
            radii = peak_radii(grain_symmetry_maps(crop), [(d[0], d[1]) for d in detections])
            proposals = [{'xy': [float(x+ox), float(y+oy)], 'radius': float(r), 'score': float(score),
                          'source': 'automatic radial-symmetry proposal'}
                         for (x, y, score), r in zip(detections, radii)
                         if x0 <= x+ox < x1 and y0 <= y+oy < y1]
        else:
            if self.model is None:
                import torch
                from cellpose.models import CellposeModel
                self.model = CellposeModel(pretrained_model=cfg.checkpoint,
                    device=torch.device(self.device), use_bfloat16=False)
            labels, _, _ = self.model.eval(crop, batch_size=1, normalize=True, diameter=None,
                flow_threshold=cfg.flow_threshold, cellprob_threshold=cfg.cellprob_threshold,
                min_size=cfg.minimum_area_px)
            proposals, mask_report = grain_proposals_from_labels(labels, [ox, oy], roi, cfg)
            report.update(mask_report, package='cellpose', package_version=version('cellpose'))
        report.update(n_proposals=len(proposals), completeness_certified=False)
        return proposals, report
