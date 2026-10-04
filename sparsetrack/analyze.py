"""SparseTrack v1: per-grain germination onset and tube length from a bin cache.

For each grain, whole-movie and offline (bins before the cache's settled reference
start are treated as unobserved):

1. Crop the grain from every bin (global registration), then refine a per-bin
   residual shift on the grain itself.
2. Tube map: blurred ``|late - early|`` change, grain bodies masked (the rim band is
   kept so a tube wrapping round its own grain survives). A change region shared
   with other grains is split by geodesic ownership.
3. The change component attached to the grain's rim is the tube; its tip is the
   geodesically farthest point, and the centreline is the evidence-weighted
   shortest path from the rim to it, starting at the exit on the grain circle.
4. Kymograph of the (background-subtracted) change along that path for every bin,
   with the grain + tube allowed to rotate rigidly about the grain centre (smooth
   rotation track, judged on distal off-rim points, pinned at the end). The tube
   front is the globally best non-decreasing path through it (dynamic programming,
   capped growth rate); lengths stop where the path reaches another grain's rim.
5. Onset = first sustained rise of the excess change just outside the rim at the
   exit angle over its pre-emergence noise; no length is reported before onset.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import time
from collections import deque
from dataclasses import asdict, astuple, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import __version__, stack
from . import grains as census
from .evaluate import PRED_SCHEMA
from .render import Renderer


@dataclass
class Params:
    half: int = 150              # crop half-size around each grain (px)
    reg_pad: float = 12.0        # local registration window = grain radius + pad
    drift_check: bool = True     # an implausible grain track (plausible_drift) falls back to the field registration
    ref_bins: int = 3            # leading bins averaged as the "before" image
    late_bins: int = 3           # trailing full bins averaged as the "after" image
    map_sigma: float = 1.0
    map_k: float = 5.0           # tube-map threshold = max(map_floor, map_k * background sigma)
    map_floor: float = 5.0
    min_component_px: int = 12
    evid_k: float = 4.0          # kymograph threshold = max(evid_floor, base + evid_k * noise)
    evid_floor: float = 5.0
    lateral: float = 1.0         # sample +/- this many px across the path
    vmax_px: float = 4.0         # maximum front advance per bin (and the floor of the automatic cap)
    vmax_auto: bool = True       # raise it for movies whose tubes grow faster per bin (probe first)
    vmax_factor: float = 2.0     # cap = factor x the movie's fast growth per bin...
    vmax_cap: float = 16.0       # ...at most this
    vmax_probe: int = 10         # isolated grains sampled for the probe
    skip_px: float = 1.5         # evidence this close to the exit is ignored (rim band)
    onset_px: float = 2.0        # onset when the front passes this far beyond the exit
    step: float = 0.5            # path sampling (px)
    rotate: bool = True          # let the grain + tube rotate rigidly about the grain centre
    max_angle: float = 45.0      # rotation search range (degrees, either way)
    angle_step: float = 1.5
    angle_penalty: float = 0.4   # Viterbi cost per degree of change between consecutive bins
    max_turn: float = 6.0        # largest rotation change between consecutive bins (degrees)
    rot_min_s: float = 6.0       # rotation judged only on path points at least this far along...
    rot_rim_clear: float = 4.0   # ...and at least this far outside the grain rim
    rot_refine: bool = True      # second pass: rotation judged only on the tube the first front found
    # the far tube swings about its base far more than grains turn: human traces showed the base
    # angle changing by a median 3.6 deg while the rotation track (about the grain centre) wandered
    rot_pivot: str = "exit"      # rigid turn about the grain "centre", or swing about the tube's "exit"
    rot_refine_px: float = 10.0  # ...plus this far beyond it
    wedge_r: tuple = (1.0, 6.0)  # onset wedge: radii beyond the rim (px)
    wedge_halfwidth: float = 12.0  # degrees either side of the exit angle
    wedge_base_bins: int = 5     # bins defining the pre-emergence noise
    wedge_k: float = 5.0         # onset threshold = base + wedge_k * robust sigma (floor below)
    wedge_floor: float = 1.5
    wedge_hold: int = 8          # the rise must hold in >= 80% of the next wedge_hold bins
    persist_bins: int = 30       # ...and in >= 70% of the next persist_bins bins (0 = to the end)
    # onset detector: "matched" (stub matched filter, below; synthetic 42/71 in tolerance vs 17/71,
    # legacy 5/7), "wedge_fixed" (excess change at the end-state exit angle; legacy 5/7),
    # "wedge" (exit angle tracked back from the end; 2/7, drifts onto rim noise), "front" (1/7)
    onset_source: str = "matched"
    mf_len: float = 4.0          # matched-filter stub: first mf_len px of the path...
    mf_half: float = 3.0         # ...+/- mf_half px across it
    mf_search: float = 15.0      # exit angle search (degrees either way)
    mf_z: float = 3.0            # onset when the calibrated score exceeds this (sustained)
    mf_z_low: float | None = 2.0   # ...then reaching back while it stays above this (hysteresis)
    mf_back_bins: int = 6        # ...by at most this many bins (a slow drift is not a stub)
    mf_follow_rotation: bool | str = "both"  # True: stub placed along the rotation track; "both": max of the two
    bg_subtract: bool = True     # subtract each bin's background change before reading the path
    # front evidence: "matched" = signed change projected on the tube's own end-state cross-section
    # (rejects blobs, focus and uniform brightness changes; synthetic lengths 68% -> 84% in tolerance);
    # "abs" = |change| near the path (v1); "union" = per-point max of the two: as good on synthetic,
    # and it still reads real tubes whose look changes as they mature (g014, g022, g026 read 0 matched)
    evidence: str = "union"
    mk_half: float = 3.5         # matched evidence: template half-width across the path (px)
    mk_k: float = 4.0            # threshold = pre-onset base + mk_k * per-bin control sigma...
    mk_floor: float = 3.0        # ...but at least this
    mk_control_px: float = 9.0   # the noise control slides the template this far off the tube
    candidates: bool = True      # choose the centreline among branch/contact hypotheses by growth
    cand_tips: int = 4
    cand_nms_px: float = 10.0
    cand_branch_tips: int = 6    # + this many skeleton branch ends
    nest_px: float = 5.0         # a candidate within this of a longer one all along is the same tube
    ridge_px: float = 5.0        # tube-likeness: end-state change on the path vs this far beside it
    ridge_weight: float = 1.0    # score *= (1 - w) + w * fraction of path points on a ridge
    through_px: float = 12.0     # foreign-tube test: material this close to the exit...
    through_min_px: int = 10     # ...at least this many pixels of it, changed before the front left
    contact_px: float = 4.0      # lengths are censored where the path comes this close to another rim
    other_block_px: float = 2.0  # another grain's disc (radius + this) is never tube
    cover_px: float = 6.0        # review: region pixels farther than this from the path are unexplained
    min_tube_px: float = 8.0     # a front that never gets this long is not a tube (unless contact-censored)
    tip_offset_px: float = 2.5   # reported length = front - this (the signal's blurred end lies beyond the apex)
    front_lead_px: float = 12.0  # if the front is already this long at the stub's onset...
    run_min_px: float = 0.5      # ...onset = start of the growth run (> this per 3 bins) that led there
    # reader: "change" (the evidence above), "flood" (learned tube probabilities read by an arrival
    # flood, sparsetrack/learned.py) or "hybrid" (the flood only where the grain's change region
    # touches a neighbour: crowded grains, where change evidence picks up foreign tubes)
    reader: str = "hybrid"
    model: str | None = None     # tube-probability model (default learned.MODEL)
    flood_half: int = 270        # the flood's crop half-size (px): long tubes in crowded fields
    flood_recent: int = 12       # bins: a new piece joins if it touches pixels claimed this recently
    flood_bridge: int = 4        # px a new piece may be from them
    flood_start_band: float = 4.0  # px beyond the rim halo where a tube may start
    flood_tip_px: float = 0.0    # subtracted from the flood's reach
    flood_lookback: float = 0.25  # walk the onset back while P at the tube's exit stays above this (0 = off)
    hybrid_onset: str = "flood"   # 0.8.1 (was "change"; vs 0.8.0: m1 lengths +7, 95% CI +2 to +13, onsets +5; m2 +1;
                                  # ld unchanged; "change_unless_missed": m1 +4, others +0). The old maps marked rims
                                  # before a tube existed, the BatchNorm maps rarely do. Flooded grains on a noisy
                                  # background: whose germination call counts -
                                  # "change" (0.5.1-0.8.0: the change reader's, the flood's lengths from it; its "no
                                  # tube" zeroes the flood's), "change_unless_missed" (the flood's where the change
                                  # reader saw no tube but the flood read one of hybrid_min_px), or "flood"
    hybrid_min_px: float = 10.0
    # where the drawn tube lies across it (learned.centre_route): both readers' routes follow the strongest evidence,
    # a tube's darker wall, 3.5 px (median) from the middle annotators trace; moved onto the middle of the tube
    # network's band across the tube (ld 3.6 -> 1.8 px from the traces, m2 2.7 -> 1.8). Lengths are not changed.
    centre_route: bool = True
    centre_late: int = 3         # the network's map averaged over the last bins the grain was seen, where it was then
    centre_reach: float = 12.0   # px either side of the route searched for the band (a route on a wall is ~4.5 px
                                 # from the middle of a band ~9.5 px wide; young tubes lie further off the final route)
    centre_min_p: float = 0.35   # a band peaking lower than this is not used (the route stays there)
    centre_max_width: float = 14.0  # a band wider than this is two tubes or a clump (not used)
    centre_smooth: int = 7       # route points the shifts are median-filtered over
    centre_max_shift: float = 8.0  # 0.8.6 (was 5): drawn tube within 2 px of the traces ld 88 -> 93%, m2 76 -> 80%
    centre_per_bin: bool = True  # also how the route lay at each bin (tubes bend and are pushed as they grow)
    centre_knot_px: float = 5.0  # ...kept every this many px along it
    centre_lengths: bool = False  # lengths measured along the centred route (not the wall the reader followed)
    drawn_every: int = 10        # check the drawn tube against the network's map every this many bins (0 = off)
    drawn_min_on: float = 0.5    # ...flag drawn_off_tube where less of it than this lies on the map
    flood_from_exit: bool = True  # flood lengths along the tube from where it leaves the grain (not a rim detour)
    flood_routes_by_bin: bool = True  # draw each bin's own route where it leaves the final one (path_by_bin)
    flood_route_off_px: float = 2.5   # ...by more than this on average
    flood_speed_cap: bool = True   # a bin's tip at most vmax_px (the movie's growth cap) further than the last
                                   # one (0.8.7): a moved tube taken as growth jumped 27-121 px in a bin (m2
                                   # g069, g016; traced growth never exceeds 1.7 px a bin); no score changed
    # the flood's start and stop rules (learned.flood), settled on tubes_synth_v1's maps; options for other maps:
    flood_p: float = 0.5         # a pixel is tube where P >= this...
    flood_persist: int = 10      # ...and it arrives at the first bin from which it is tube in >= flood_frac of the
    flood_frac: float = 0.7      # next flood_persist bins
    flood_halo: float = 3.0      # nothing within this of the rim is claimed (the rim's own change; learned.HALO)
    flood_arc_deg: float = 60.0  # a start spanning more than this round the grain is an arc on the rim, not a stub
    flood_old_far_px: float = 10.0  # a start joined to material this far beyond the rim that arrived more than...
    flood_old_far_bins: int = 3     # ...this many bins earlier is the leading end of a structure already there
    flood_min_len: float = 8.0   # a tube that stops for flood_give_up bins before reaching this far beyond the rim...
    flood_give_up: int = 40      # ...was rim noise: forgotten, and the flood starts again
    flood_tip: str = "dist"      # the tip its length is read to: the pixel of greatest rim distance ("dist"), or also
                                 # the one farthest from the grain where that reads longer ("radial": a young blob
                                 # widening along the rim; with BatchNorm maps m2 +1, ld +1 lengths, old maps m2 -4)
    flood_exit_edge: bool = False  # flood lengths from the grain's visible edge along the exit (as exit_edge)
    exit_edge: bool = True       # change reader: lengths from the grain's visible edge along the exit, where an
                                 # annotator starts a trace, not from the census circle
    # tip continuation (change reader): a tube grows at its tip, so change that appears beyond the chosen path's
    # tip only after its front got there, and runs on from it, is the tube going on - typically back along its own
    # grain after a turn, where every point is nearer the rim than the tip and no path starting on the rim reaches
    # it the long way. A second front reads it from that bin on; the first reading is kept as it was.
    # ld 69 -> 73/104 lengths (95% CI +0 to +9; g022, g031, g002 gain, g031 loses one), m2 unchanged (flood-read);
    # synthetic v5 seeds 0-2, change reader: 331 -> 341/647 (+0 to +21). cont_back_px 3/6/10: ld -2/+4/+0,
    # synthetic +0/+10/+17.
    tip_continue: bool = True    # 0.7.0 (was off: the gain held on synthetic movies, where nothing was tuned)
    cont_min_px: float = 6.0       # ...when the second front gets at least this far
    cont_after_bins: int = 10      # ...and the first reached its tip at least this many bins before the end
    cont_margin_bins: int = 3      # material that changed from this many bins before then on counts
    cont_clear_px: float = 3.0     # ...away from the path itself (a swaying tube is not a continuation)
    cont_order: float = 0.6        # its change must arrive in order outwards (rank correlation with the distance)
    cont_back_px: float = 6.0      # it may leave the path this far before its end (a path overshoots a tight turn)
    cont_gate_bins: int = 2        # a continuation point is read as tube from this many bins before its arrival on
    settle: bool = True          # grains still arriving in the census bins are read from when they settle
    settle_bins: int = 24
    grain_min_rim: float = 1.5   # no rim at all in the early bins: not a grain (passing debris)
    # how each grain is followed through the movie (both readers and the labelling tool's views):
    # - "phase" (to 0.6.0): phase correlation bin to bin; the whole track is dropped (zero drift) by checked_drift
    #   when it jumps or wanders;
    # - "follow": sparsetrack/track.py follows the grain's own look bin by bin (through blobs, crossings, pushes and
    #   changes of look), refined by the phase correlation where the two agree (followed_drift); a grain it can no
    #   longer find is lost from that bin (lost_policy); every grain is read in its own frame;
    # - "auto": as "follow", but a grain is read in its own frame only once it has moved off its place (further than
    #   track_far_r radii), by the phase track as before nearer.
    grain_track: str = "auto"    # 0.7.0: followed, read in its own frame once off its place, held once lost
    track_far_r: float = 2.0     # auto: "off its place" = further than this many grain radii
    track_step_px: int = 8       # follow/auto: the tracker's search radius per bin (px)
    track_min_score: float = 0.5  # follow/auto: a match scoring below this is a missed bin...
    track_max_gap: int = 8       # ...and more missed bins in a row than this lose the grain
    track_tol_px: float = 4.0    # follow/auto: the phase track is kept where it is within this of the grain's track
    track_refind: bool = True    # follow/auto: a grain the bank misses is searched for by its last look, turned
                                 # (track.FollowConfig.refind: a knocked grain that jumps and turns is found again;
                                 # on from 0.8.6: m1 g027, g030, g056 followed to the end, no score changed on the
                                 # three movies)
    track_recentre_px: float = 25.0  # a grain read in its own frame that moves further than this is cropped where it
                                     # is (a fixed crop warped that far brings in that much replicated border)
    lost_policy: str = "hold"    # a lost grain: "hold" its readings from the loss on (flag grain_lost_after:<frame>),
                                 # or "read" on at its last place
    lost_min_bins: int = 15      # a grain followed for fewer bins than this is unobservable
    # a tube over its grain (over_grain(): movie 1's tubes from a pore facing the camera): "off"; "flag" (flagged
    # tube_over_grain, its onset over the grain kept apart in result["over_grain"]); "onset" (the onset is read over
    # the grain); "length" (and lengths from the pore, as the annotator traced such tubes, not from the rim)
    over_grain: str = "off"
    over_k: float = 5.0          # change over the grain: above this many times the disc's own noise...
    over_floor: float = 6.0      # ...and at least this
    over_min_px: float = 4.0     # the strip from the pore to the exit is at least this long...
    over_max_width: float = 7.0  # ...and on average at most this wide (px)...
    over_order: float = 0.4      # ...its change arrives in order towards the exit...
    over_lead_bins: int = 2      # ...at the pore at least this many bins before the rim onset
    over_join_px: float = 3.0    # the strip meets the exit within this
    # census check (sparsetrack/census_check.py): census "grains" that look like debris, clumps or passing things are
    # flagged likely_not_a_grain:<P> and left out of the population statistics (they are still read and reported).
    # Off: judged leave one movie out on the three labelled movies it finds dark debris well but does not bring every
    # movie's germinated share and T50 closer to the annotator's (prototypes/census_check/README.md)
    census_check: bool = False
    # learned tip detector (sparsetrack/tipdet.py; study prototypes/tip_track/README.md): a network trained on human
    # traces that puts a peak on growing tips. Off unless tipdet_model names a checkpoint (prototypes/tip_detector).
    tipdet_model: str | None = None
    tipdet_onset: str = "off"    # "later": the detector's onset where the reading's is later by > tipdet_late_bins;
                                 # "later_or_missing": also where the reading saw no tube; "off"
    tipdet_thr: float = 0.3      # detector onset: the rim response (best peak r - 2 .. r + 25 px) stays >= this...
    tipdet_hold: int = 3         # ...for this many bins
    tipdet_late_bins: int = 10
    tipdet_young: bool = False   # young tubes' lengths (reading < tipdet_young_px) from the detector's tip:
    tipdet_young_px: float = 20.0
    tipdet_young_k: float = 2.0  # length = tip's distance from the grain's visible edge + this
    tipdet_young_min: float = 0.2  # weakest peak taken as a tip
    tipdet_half: int = 64        # half-size of the detector's maps round each grain (px)
    # global tip-trajectory reader (sparsetrack/tiptraj.py, prototypes/tip_trajectory; on from 0.9.0): per grain the
    # tip at every bin chosen in one Viterbi over the movie, among the learned tip detector's peaks and the tube map's
    # piece ends, each with its body on the map, jumps onto other tubes ruled out. "flood" (it re-reads the grains the
    # hybrid floods, keeping their drift; leave one movie out against 0.8.8: lengths ld 75 -> 77/104, m2 27 -> 37/54,
    # m1 15 -> 27/50; ~20-30 s a grain); "all" (every grain); "off" (0.8.8's readings)
    tiptraj: str = "flood"
    tiptraj_model: str | None = None   # the tip detector checkpoint (default tiptraj.MODEL)
    tiptraj_det: str | None = None     # a directory of detector maps (tiptraj.det_cache) to use instead of building them
    tiptraj_weights: str | None = None  # the Viterbi's settings (JSON, or a tune_*.json file) over tiptraj.WEIGHTS
    tiptraj_half: int = 300            # crop half-size round each grain (px): the longest tubes to read
    tiptraj_guided: bool = False       # a second pass that follows the first reading's tube along the map (its tip,
                                       # points beyond it, a corridor along its body); twice the reading time
    tiptraj_mid: bool = True           # lengths along the middle of the tube's band, not the inside of its curves
                                       # (tiptraj.mid_correction; ld-chosen: ld +2, m2 +1, m1 +1 lengths)


def _highpass(img: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    img = img.astype(np.float64)
    return img - cv2.GaussianBlur(img, (0, 0), sigma)


def local_shifts(crops: np.ndarray, centre: float, radius: float, pad: float, ref_bins: int,
                 max_dev: float = 1.5, window: int = 3, follow: bool = True) -> np.ndarray:
    """Residual (dx, dy) per bin of the grain relative to its own early mean.

    With ``follow`` the correlation window moves with the grain (centred on the previous
    bin's estimate), so a grain can drift further than the window half-width.
    """
    from skimage.registration import phase_cross_correlation

    w = int(math.ceil(radius + pad))
    c = int(round(centre))
    size = crops.shape[1]
    ref = _highpass(crops[:ref_bins].mean(axis=0)[c - w:c + w, c - w:c + w])
    raw = np.zeros((len(crops), 2))
    prev = np.zeros(2)
    for b in range(len(crops)):
        ox, oy = (int(round(prev[0])), int(round(prev[1]))) if follow else (0, 0)
        ox, oy = int(np.clip(ox, w - c, size - w - c)), int(np.clip(oy, w - c, size - w - c))
        win = crops[b][c + oy - w:c + oy + w, c + ox - w:c + ox + w]
        shift, _, _ = phase_cross_correlation(ref, _highpass(win), upsample_factor=20, normalization=None)
        raw[b] = (ox - shift[1], oy - shift[0])
        if b >= ref_bins:  # the reference bins define the origin
            lo = max(ref_bins, b - window)
            recent = raw[lo:b + 1]
            prev = np.median(recent, axis=0) if len(recent) >= 3 else raw[b]
    med = np.array([np.median(raw[max(0, b - window):b + window + 1], axis=0) for b in range(len(raw))])
    bad = np.hypot(*(raw - med).T) > max_dev
    return np.where(bad[:, None], med, raw)


def exit_edge(img: np.ndarray, centre: float, r: float, theta: float, wedge_deg: float = 10.0, n_ang: int = 9,
              step: float = 0.25) -> float:
    """The grain's visible edge along a tube's exit direction ``theta`` (radians, image axes), as an offset from
    its census radius ``r``: where the before image ``img`` (grain-centred at (``centre``, ``centre``)) changes
    most steeply along the radius, 7 px inside to 5 px outside the census circle, over the median profile of a
    +/-10 degree wedge. Grains are not perfect discs, and an annotator starts a trace where the tube leaves this
    edge: over the benchmark grains it tracks their exit clicks (correlation 0.80 on ld, 0.62 on m2)."""
    rads = np.arange(0.0, r + 16.0, step)
    angs = theta + np.deg2rad(np.linspace(-wedge_deg, wedge_deg, n_ang))
    xs = (centre + rads[None] * np.cos(angs)[:, None]).astype(np.float32)
    ys = (centre + rads[None] * np.sin(angs)[:, None]).astype(np.float32)
    prof = np.median(cv2.remap(np.nan_to_num(img).astype(np.float32), xs, ys, cv2.INTER_LINEAR), axis=0)
    g = np.gradient(np.convolve(prof, np.ones(5) / 5, mode="same"), rads)
    win = (rads > r - 7.0) & (rads < r + 5.0)
    return float(rads[win][np.argmax(np.abs(g[win]))] - r)


def plausible_drift(ls: np.ndarray, max_step: float = 10.0) -> bool:
    """Is a grain's tracked drift physical? A real drift (even a sudden push) moves a few px per
    bin and mostly one way; locking onto a neighbour, a clump or the grain's own growing tube
    jumps far in one bin or wanders back and forth."""
    steps = np.hypot(*np.diff(ls, axis=0).T) if len(ls) > 1 else np.zeros(1)
    reach = float(np.hypot(*ls.T).max())
    return not (steps.max() > max_step or (steps > 2).sum() > 10 or steps.sum() > 4 * reach + 30)


def checked_drift(ls: np.ndarray, p: "Params") -> tuple[np.ndarray, str | None]:
    """A grain's tracked drift after the plausibility check: as tracked, else no drift at all (a track that
    jumped or wandered locked onto something else). Returns the drift and a flag ("drift_rejected" or None)."""
    if not p.drift_check or plausible_drift(ls):
        return ls, None
    return np.zeros_like(ls), "drift_rejected"


_FOLLOWED: dict = {}  # followed_drift's results in this process (both readers and the probe read the same grains)


def followed_drift(renderer: Renderer, meta: dict, grain: dict, others: list[dict], p: "Params") -> dict:
    """A grain's drift with ``p.grain_track = "follow"``: ``track.follow``'s track of the grain's own look, refined
    by ``local_shifts`` with its window centred on the track. Where the correlation agrees with the track within
    ``p.track_tol_px`` the drift is the correlation's (as with "phase": the same readings where the two agree);
    where it has jumped away (locked onto something else) it is the track, carried by the offset between the two
    nearby, so that the drift is continuous.

    Returns {"drift": (n, 2) (dx, dy) per bin from the reference start (NaN from the loss on), "lost_from": the
    first bin (absolute) the grain is lost at, or None, "lost_reason", "score": the track's match score per bin}."""
    from . import track
    rs, nb = int(meta.get("ref_start", 0)), int(meta["n_bins"])
    gx, gy, gr = float(grain["x"]), float(grain["y"]), float(grain["r"])
    near = tuple((round(float(o["x"]), 2), round(float(o["y"]), 2), round(float(o["r"]), 2)) for o in others
                 if math.hypot(o["x"] - gx, o["y"] - gy) < 120)
    cfg = track.FollowConfig(step_px=p.track_step_px, min_score=p.track_min_score, max_gap=p.track_max_gap,
                             refind=p.track_refind)
    key = (id(renderer.bins), round(gx, 2), round(gy, 2), round(gr, 2), rs, nb, near, astuple(cfg), p.reg_pad,
           p.ref_bins, p.track_tol_px)
    hit = _FOLLOWED.get(key)
    if hit is not None and hit[0] is renderer.bins:  # the same movie: an id alone can be reused once freed
        return hit[1]
    tr = track.follow(renderer, gx, gy, gr, rs, nb, near, cfg)
    guide = tr["xy"]
    n_fol = len(guide) if tr["lost_from"] is None else tr["lost_from"] - rs
    drift = np.full((nb - rs, 2), np.nan)
    if n_fol > p.ref_bins:
        g = guide[:n_fol]
        idx = np.arange(n_fol)

        def agreeing(est):
            """Bins where a correlation estimate stays with the track (within track_tol_px of it)."""
            diff = est - g
            return diff, np.hypot(*diff.T) <= p.track_tol_px

        # 1. the phase track as the readers made it before (the same readings where it stays with the grain)
        crops = np.stack([renderer.crop(b, gx, gy, p.half) for b in range(rs, rs + n_fol)])
        if np.isnan(crops).any():
            crops = np.nan_to_num(crops, nan=float(np.nanmedian(crops)) if np.isfinite(crops).any() else 0.0)
        diff0, good0 = agreeing(local_shifts(crops, p.half - 0.5, gr, p.reg_pad, p.ref_bins))
        del crops
        # 2. where it has jumped away (locked onto something else), the correlation with its window on the track
        off = np.round(g)
        off_abs = np.zeros((nb, 2))
        off_abs[rs:rs + n_fol] = off
        half = 2 * (int(math.ceil(gr + p.reg_pad)) // 2) + 6  # even: local_shifts' window is centred on half - 0.5
        crops = np.stack([renderer.crop(b, gx, gy, half, offsets=off_abs) for b in range(rs, rs + n_fol)])
        if np.isnan(crops).any():
            crops = np.nan_to_num(crops, nan=float(np.nanmedian(crops)) if np.isfinite(crops).any() else 0.0)
        diff1, good1 = agreeing(off + local_shifts(crops, half - 0.5, gr, p.reg_pad, p.ref_bins, follow=False))
        # 3. neither: the track, carried by the two's offset nearby, so that the drift never jumps
        diff = np.where(good0[:, None], diff0, diff1)
        good = good0 | good1
        corr = (np.stack([np.interp(idx, idx[good], diff[good, k]) for k in (0, 1)], axis=1) if good.any()
                else np.zeros_like(g))
        drift[:n_fol] = g + corr
    elif n_fol > 0:
        drift[:n_fol] = guide[:n_fol]
    out = {"drift": drift, "lost_from": tr["lost_from"], "lost_reason": tr["lost_reason"], "score": tr["score"],
           "angle": tr["angle"], "refound": tr["refound"]}
    if len(_FOLLOWED) > 512:
        _FOLLOWED.clear()
    _FOLLOWED[key] = (renderer.bins, out)  # keeps the (memory-mapped) movie alive, so its id is not reused
    return out


def _geodesic_far(mask: np.ndarray, seeds: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    """Breadth-first geodesic distance (8-connected) inside ``mask`` from ``seeds``; farthest pixel."""
    h, w = mask.shape
    dist = np.full((h, w), -1, np.int32)
    q = deque()
    for y, x in zip(*np.nonzero(seeds & mask)):
        dist[y, x] = 0
        q.append((y, x))
    far, fd = (int(q[0][0]), int(q[0][1])) if q else (0, 0), 0
    while q:
        y, x = q.popleft()
        d = dist[y, x]
        if d > fd:
            far, fd = (y, x), d
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and dist[yy, xx] < 0:
                    dist[yy, xx] = d + 1
                    q.append((yy, xx))
    return far, dist


def geodesic_owner(mask: np.ndarray, seeds: list[np.ndarray]) -> np.ndarray:
    """Label each ``mask`` pixel with the index of the seed set nearest *through the mask*.

    Multi-source breadth-first search (8-connected); -1 where no seed reaches.
    """
    h, w = mask.shape
    owner = np.full((h, w), -1, np.int32)
    q = deque()
    for k, s in enumerate(seeds):
        for y, x in zip(*np.nonzero(s & mask)):
            if owner[y, x] < 0:
                owner[y, x] = k
                q.append((y, x))
    while q:
        y, x = q.popleft()
        k = owner[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and owner[yy, xx] < 0:
                    owner[yy, xx] = k
                    q.append((yy, xx))
    return owner


def _cheapest_path(cost: np.ndarray, mask: np.ndarray, seeds: np.ndarray, target: tuple[int, int],
                   start_cost: np.ndarray | None = None) -> np.ndarray:
    """Dijkstra from any seed pixel (at ``start_cost`` there, default 0) to ``target`` through ``mask``; returns
    (n, 2) array of (y, x)."""
    h, w = mask.shape
    dist = np.full((h, w), np.inf)
    prev = np.full((h, w, 2), -1, np.int32)
    pq = []
    for y, x in zip(*np.nonzero(seeds & mask)):
        d0 = 0.0 if start_cost is None else float(start_cost[y, x])
        dist[y, x] = d0
        heapq.heappush(pq, (d0, int(y), int(x)))
    while pq:
        d, y, x = heapq.heappop(pq)
        if (y, x) == target:
            break
        if d > dist[y, x]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                    nd = d + cost[yy, xx] * math.hypot(dy, dx)
                    if nd < dist[yy, xx]:
                        dist[yy, xx] = nd
                        prev[yy, xx] = (y, x)
                        heapq.heappush(pq, (nd, yy, xx))
    path = [target]
    while prev[path[-1]][0] >= 0:
        path.append(tuple(prev[path[-1]]))
    return np.array(path[::-1], np.float64)


def _resample(path_xy: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    seg = np.hypot(*np.diff(path_xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ss = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(ss, s, path_xy[:, 0]), np.interp(ss, s, path_xy[:, 1])], axis=1), ss


def dp_front(evidence: np.ndarray, vmax: int) -> np.ndarray:
    """Non-decreasing front index per bin maximising summed evidence behind the front.

    ``evidence`` is (n_bins, n_points); the front at index f claims points [0, f).
    """
    n_bins, n = evidence.shape
    gain = np.concatenate([np.zeros((n_bins, 1)), np.cumsum(evidence, axis=1)], axis=1)
    score = gain[0].copy()
    back = np.zeros((n_bins, n + 1), np.int32)
    idx = np.arange(n + 1)
    for t in range(1, n_bins):
        pad = np.concatenate([np.full(vmax, -np.inf), score])
        windows = np.lib.stride_tricks.sliding_window_view(pad, vmax + 1)
        j = np.argmax(windows, axis=1)
        score = gain[t] + windows[idx, j]
        back[t] = idx - (vmax - j)
    front = np.zeros(n_bins, np.int32)
    front[-1] = int(np.argmax(score))
    for t in range(n_bins - 1, 0, -1):
        front[t - 1] = back[t, front[t]]
    return front


def rotation_track(score: np.ndarray, angles: np.ndarray, penalty: float, max_turn: float,
                   anchor_bins: int) -> np.ndarray:
    """Smooth per-bin rotation maximising summed ``score`` (n_bins, n_angles).

    Consecutive bins may differ by at most ``max_turn`` degrees at ``penalty`` per degree;
    the last ``anchor_bins`` bins are pinned to 0 degrees (the path was traced there).
    """
    n_bins, n_ang = score.shape
    zero = int(np.argmin(np.abs(angles)))
    diff = np.abs(angles[:, None] - angles[None, :])
    trans = np.where(diff <= max_turn + 1e-9, -penalty * diff, -np.inf)  # [prev, cur]
    pinned = np.full(n_ang, -np.inf)
    pinned[zero] = 0.0
    total = score[0] + (pinned if n_bins - anchor_bins <= 0 else 0.0)
    back = np.zeros((n_bins, n_ang), np.int32)
    for t in range(1, n_bins):
        cand = total[:, None] + trans
        back[t] = np.argmax(cand, axis=0)
        total = cand[back[t], np.arange(n_ang)] + score[t]
        if t >= n_bins - anchor_bins:
            total = total + pinned
    track = np.zeros(n_bins, np.int32)
    track[-1] = int(np.argmax(total))
    for t in range(n_bins - 1, 0, -1):
        track[t - 1] = back[t, track[t]]
    return angles[track]


def _kymograph(diffs: np.ndarray, pts: np.ndarray, normal: np.ndarray, centre: float, lateral: float,
               angles_deg: np.ndarray) -> np.ndarray:
    """|change| sampled along the path rotated by each angle about the grain centre.

    Returns (n_bins, n_angles, n_points), the max over -lateral, 0, +lateral across the path.
    """
    rad = np.deg2rad(angles_deg)
    cos, sin = np.cos(rad)[:, None], np.sin(rad)[:, None]
    cx, cy = (float(centre[0]), float(centre[1])) if np.ndim(centre) else (centre, centre)  # rotation pivot
    rel = pts - [cx, cy]
    maps = []
    for off in (-lateral, 0.0, lateral):
        q = rel + off * normal
        x = cx + cos * q[None, :, 0] - sin * q[None, :, 1]
        y = cy + sin * q[None, :, 0] + cos * q[None, :, 1]
        maps.append((x, y))
    mx = np.concatenate([m[0] for m in maps]).astype(np.float32)
    my = np.concatenate([m[1] for m in maps]).astype(np.float32)
    n_ang = len(angles_deg)
    out = np.stack([cv2.remap(d, mx, my, cv2.INTER_LINEAR, borderValue=0) for d in diffs])
    return out.reshape(len(diffs), 3, n_ang, -1).max(axis=1)


def cross_section_template(late_minus_early: np.ndarray, pts: np.ndarray, normal: np.ndarray, across: np.ndarray,
                           smooth: int = 2) -> np.ndarray:
    """The tube's own end-state cross-section at every path point: zero-mean, unit-norm rows.

    Rows are averaged over +/- ``smooth`` neighbouring points to cut noise.
    """
    q = pts[:, None, :] + across[None, :, None] * normal[:, None, :]
    prof = cv2.remap(late_minus_early.astype(np.float32), q[..., 0].astype(np.float32), q[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderValue=0)
    if smooth > 0 and len(prof) > 2 * smooth + 1:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        prof = np.stack([np.convolve(np.pad(prof[:, j], smooth, mode="edge"), k, "valid")
                         for j in range(prof.shape[1])], axis=1)
    prof = prof - prof.mean(axis=1, keepdims=True)
    return prof / np.maximum(np.linalg.norm(prof, axis=1, keepdims=True), 1e-3)


def matched_kymograph(signed: np.ndarray, template: np.ndarray, pts: np.ndarray, normal: np.ndarray, centre: float,
                      across: np.ndarray, angles_deg: np.ndarray, lateral_offset: float = 0.0) -> np.ndarray:
    """Signed change projected on the cross-section template, per (bin, angle, point).

    ``lateral_offset`` slides the whole placement sideways (off the tube: a noise control).
    """
    rad = np.deg2rad(angles_deg)
    cos, sin = np.cos(rad)[:, None, None], np.sin(rad)[:, None, None]
    cx, cy = (float(centre[0]), float(centre[1])) if np.ndim(centre) else (centre, centre)  # rotation pivot
    q = ((pts - [cx, cy])[None, :, None, :] + (across[None, None, :, None] + lateral_offset) * normal[None, :, None, :])
    n_ang, n_pts = len(angles_deg), len(pts)
    # one map row per angle: OpenCV's remap takes fewer than 32767 rows and columns, and one row
    # per (angle, point) passed that on paths longer than ~270 px
    x = (cx + cos * q[..., 0] - sin * q[..., 1]).reshape(n_ang, -1).astype(np.float32)
    y = (cy + sin * q[..., 0] + cos * q[..., 1]).reshape(n_ang, -1).astype(np.float32)
    out = np.empty((len(signed), n_ang, n_pts), np.float32)
    for t, d in enumerate(signed):
        smp = cv2.remap(d, x, y, cv2.INTER_LINEAR, borderValue=0).reshape(n_ang, n_pts, len(across))
        out[t] = np.einsum("apu,pu->ap", smp, template)
    return out


def wedge_signal(diffs: np.ndarray, centre: float, gr: float, exit_angles: np.ndarray, p: "Params") -> np.ndarray:
    """Per-bin excess change at the exit angle over the rim-wide median, just outside the rim."""
    phis = np.deg2rad(np.arange(0.0, 360.0, 3.0))
    radii = gr + np.arange(p.wedge_r[0], p.wedge_r[1] + 1e-9, 1.0)
    mx = (centre + radii[:, None] * np.cos(phis)[None]).astype(np.float32)
    my = (centre + radii[:, None] * np.sin(phis)[None]).astype(np.float32)
    prof = np.stack([cv2.remap(d, mx, my, cv2.INTER_LINEAR, borderValue=np.nan).mean(axis=0) for d in diffs])
    out = np.zeros(len(diffs))
    for t, a in enumerate(exit_angles):
        dphi = np.abs((np.rad2deg(phis) - a + 180.0) % 360.0 - 180.0)
        out[t] = np.nanmean(prof[t, dphi <= p.wedge_halfwidth]) - np.nanmedian(prof[t])
    return out


def exit_track_signal(diffs: np.ndarray, centre: float, gr: float, end_angle: float, p: "Params",
                      step_deg: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Track the exit angle backwards from the traced end-state exit and read its signal.

    The annulus just outside the rim is unwrapped to W[t, phi] = change at phi minus the
    rim-wide median (cancels whole-grain focus changes), smoothed over the tube's angular
    width. A smooth angle path pinned to ``end_angle`` in the final bins maximises
    summed W; the signal is W along that path. Returns (signal, angles_deg).
    """
    phis_deg = np.arange(0.0, 360.0, step_deg)
    phis = np.deg2rad(phis_deg)
    radii = gr + np.arange(p.wedge_r[0], p.wedge_r[1] + 1e-9, 1.0)
    mx = (centre + radii[:, None] * np.cos(phis)[None]).astype(np.float32)
    my = (centre + radii[:, None] * np.sin(phis)[None]).astype(np.float32)
    prof = np.stack([cv2.remap(d, mx, my, cv2.INTER_LINEAR, borderValue=0).mean(axis=0) for d in diffs])
    w = prof - np.median(prof, axis=1, keepdims=True)
    k = max(1, int(round(p.wedge_halfwidth / step_deg)))
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    w = np.stack([np.convolve(np.concatenate([row[-k:], row, row[:k]]), kernel, "valid") for row in w])
    n_bins, n_ang = w.shape
    dist = np.abs((phis_deg[:, None] - phis_deg[None, :] + 180.0) % 360.0 - 180.0)
    trans = np.where(dist <= p.max_turn + 1e-9, -p.angle_penalty * dist, -np.inf)
    end_i = int(np.argmin(np.abs((phis_deg - end_angle + 180.0) % 360.0 - 180.0)))
    pinned = np.full(n_ang, -np.inf)
    pinned[end_i] = 0.0
    anchor = p.late_bins + 1
    total = w[0].copy()
    back = np.zeros((n_bins, n_ang), np.int32)
    for t in range(1, n_bins):
        cand = total[:, None] + trans
        back[t] = np.argmax(cand, axis=0)
        total = cand[back[t], np.arange(n_ang)] + w[t]
        if t >= n_bins - anchor:
            total = total + pinned
    track = np.zeros(n_bins, np.int32)
    track[-1] = int(np.argmax(total))
    for t in range(n_bins - 1, 0, -1):
        track[t - 1] = back[t, track[t]]
    return w[np.arange(n_bins), track], phis_deg[track]


def matched_stub_signal(signed: np.ndarray, late_minus_early: np.ndarray, pts: np.ndarray, centre: float,
                        p: "Params", theta: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Calibrated matched-filter score of a short stub at the exit, per bin.

    The template is the grain's own end-state change over the first ``mf_len`` px of its
    path (signed, +/- ``mf_half`` px across), so dark and bright-cored tubes are both
    matched. Each bin's signed change is correlated with the template at the exit (best
    of +/- ``mf_search`` degrees) and at control angles round the rim (same search);
    the score is (exit - median control) / robust sigma of the controls over the movie.
    ``theta`` (degrees per bin) turns the whole placement with the grain's rotation track.
    """
    step = 0.5
    s_idx = np.nonzero(np.arange(len(pts)) * p.step <= p.mf_len)[0]
    stub = pts[s_idx]
    tang = np.gradient(stub, axis=0) if len(stub) > 1 else np.array([[1.0, 0.0]])
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    across = np.arange(-p.mf_half, p.mf_half + 1e-9, step)
    grid = (stub[:, None, :] + across[None, :, None] * normal[:, None, :]).reshape(-1, 2)  # template points
    tmpl = cv2.remap(late_minus_early.astype(np.float32), grid[None, :, 0].astype(np.float32),
                     grid[None, :, 1].astype(np.float32), cv2.INTER_LINEAR)[0]
    tmpl = tmpl - tmpl.mean()
    norm = float(np.linalg.norm(tmpl))
    info = {"template_norm": round(norm, 2)}
    if norm < 1e-3:
        return np.zeros(len(signed)), info
    tmpl /= norm
    rel = grid - centre
    exit_offsets = np.arange(-p.mf_search, p.mf_search + 1e-9, 3.0)
    ctrl_offsets = [a for a in range(45, 360, 45)]

    def placed(angle_deg):
        a = math.radians(angle_deg)
        x = centre + math.cos(a) * rel[:, 0] - math.sin(a) * rel[:, 1]
        y = centre + math.sin(a) * rel[:, 0] + math.cos(a) * rel[:, 1]
        return x.astype(np.float32), y.astype(np.float32)

    def maps(base):
        exit_maps = [placed(base + o) for o in exit_offsets]
        ctrl_maps = [[placed(base + c + o) for o in exit_offsets] for c in ctrl_offsets]
        return (np.concatenate([m[0] for m in exit_maps] + [m[0] for cm in ctrl_maps for m in cm])[None, :],
                np.concatenate([m[1] for m in exit_maps] + [m[1] for cm in ctrl_maps for m in cm])[None, :])

    n_pts, n_off = len(tmpl), len(exit_offsets)
    if theta is None:
        mx, my = maps(0.0)
        scores = np.stack([cv2.remap(d, mx, my, cv2.INTER_LINEAR, borderValue=0)[0] for d in signed])
    else:
        cache = {}
        rows = []
        for d, th in zip(signed, theta):
            key = round(float(th), 1)
            if key not in cache:
                cache[key] = maps(key)
            rows.append(cv2.remap(d, *cache[key], cv2.INTER_LINEAR, borderValue=0)[0])
        scores = np.stack(rows)
    scores = scores.reshape(len(signed), -1, n_pts) @ tmpl  # (bins, placements)
    exit_score = scores[:, :n_off].max(axis=1)
    ctrl = scores[:, n_off:].reshape(len(signed), len(ctrl_offsets), n_off).max(axis=2)
    base = np.median(ctrl, axis=1)
    resid = ctrl - base[:, None]
    sigma = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))), 1e-3)
    info["control_sigma"] = round(sigma, 3)
    return (exit_score - base) / sigma, info


def sustained_onset(signal: np.ndarray, p: "Params", threshold: float | None = None) -> tuple[int | None, float]:
    """First bin where ``signal`` rises above its pre-emergence noise and stays there.

    ``threshold`` (absolute) is used for already-calibrated scores; otherwise the
    threshold comes from the first ``wedge_base_bins`` bins.
    """
    if threshold is None:
        base = signal[:p.wedge_base_bins]
        mu = float(np.median(base))
        sigma = max(1.4826 * float(np.median(np.abs(base - mu))), 0.25)
        thr = mu + max(p.wedge_floor, p.wedge_k * sigma)
    else:
        thr = float(threshold)
    above = signal > thr
    n = len(signal)
    if above[-min(10, n):].mean() < 0.5:  # a tube never retracts: the exit stays changed to the end
        return None, thr
    for t in range(n):
        tail = above[t:t + p.persist_bins] if p.persist_bins > 0 else above[t:]
        if above[t] and above[t:t + p.wedge_hold].mean() >= 0.8 and tail.mean() >= 0.7:
            return t, thr
    return None, thr


def _local_maxima_tips(comp: np.ndarray, dist: np.ndarray, min_dist: int, nms_px: float, k: int) -> list:
    """Branch ends of a component: local maxima of the geodesic distance from the rim."""
    d = np.where(comp, dist, -1).astype(np.float32)
    mx = cv2.dilate(d, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(comp & (d >= mx) & (d >= min_dist))
    order = np.argsort(-d[ys, xs])
    tips = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if all(math.hypot(y - ty, x - tx) >= nms_px for ty, tx in tips):
            tips.append((y, x))
        if len(tips) >= k:
            break
    return tips


def candidate_paths(comp: np.ndarray, ring: np.ndarray, cost: np.ndarray, gr: float, centre: float,
                    p: "Params") -> list[np.ndarray]:
    """Centreline hypotheses: each branch end, reached from the nearest rim contact and, for a
    tube that wraps round and touches its grain again, from the other contacts too."""
    far, dist = _geodesic_far(comp, ring)
    tips = _local_maxima_tips(comp, dist, 4, p.cand_nms_px, p.cand_tips) or [far]
    # every branch of the region gets its end as a hypothesis, however far the others reach
    from skimage.morphology import skeletonize
    skel = skeletonize(comp)
    nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    ends = [(int(y), int(x)) for y, x in zip(*np.nonzero(skel & (nb == 2))) if dist[y, x] >= 6]
    for y, x in sorted(ends, key=lambda e: -dist[e]):
        if len(tips) >= p.cand_tips + p.cand_branch_tips:
            break
        if all(math.hypot(y - ty, x - tx) >= p.cand_nms_px for ty, tx in tips):
            tips.append((y, x))
    n_c, c_lab = cv2.connectedComponents((ring & comp).astype(np.uint8), connectivity=8)
    contacts = [c_lab == c for c in range(1, n_c) if (c_lab == c).sum() >= 2]
    out, seen = [], []
    for tip in tips:
        base = _cheapest_path(cost, comp, ring, tip)
        options = [base]
        if len(contacts) > 1:
            for c in contacts:
                if c[int(base[0][0]), int(base[0][1])]:
                    continue  # the default path already starts here
                alt = _cheapest_path(cost, comp, c, tip)
                if len(alt) >= 1.3 * len(base) and c[int(alt[0][0]), int(alt[0][1])]:
                    options.append(alt)
        for path in options:
            key = (int(path[0][0]) // 4, int(path[0][1]) // 4, tip)
            if key in seen or len(path) < 3:
                continue
            seen.append(key)
            out.append(path)
    return out


def read_path(ctx: dict, path_yx: np.ndarray, p: "Params") -> dict | None:
    """Kymograph, rotation track and growth front along one centreline hypothesis.

    Also scores the hypothesis: evidence explained by monotone growth from the exit, minus
    positive evidence the front leaves unexplained beyond it, and a penalty when the exit
    sits on material that was already there before the front left it (a foreign tube
    passing the rim, not one emerging from it).
    """
    centre, gr, reg, early, late = ctx["centre"], ctx["gr"], ctx["reg"], ctx["early"], ctx["late"]
    n_bins = ctx["n_bins"]
    path = path_yx[:, ::-1]  # (x, y) in crop coordinates
    if len(path) > 7:
        k = np.ones(5) / 5
        path = np.stack([np.convolve(np.pad(path[:, i], 2, mode="edge"), k, "valid") for i in range(2)], axis=1)
    v = path[0] - centre
    v = v / (np.linalg.norm(v) + 1e-9)
    path = np.vstack([centre + v * gr, path])
    pts, ss = _resample(path, p.step)
    if len(pts) < 4:
        return None
    flags, extra = [], {}
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    diffs, rg = ctx["diffs"], ctx["rg"]
    angles = (np.arange(-p.max_angle, p.max_angle + 1e-9, p.angle_step) if p.rotate else np.zeros(1))
    # the tube swings about its own exit (the base stays put) or turns with its grain
    pivot = pts[0].copy() if p.rot_pivot == "exit" else centre
    if p.evidence in ("matched", "union"):
        signed_all = ctx["signed"]
        across = np.arange(-p.mk_half, p.mk_half + 1e-9, 0.5)
        template = cross_section_template(late - early, pts, normal, across)
        kymo_all = matched_kymograph(signed_all, template, pts, normal, pivot, across, angles)
        zero = int(np.argmin(np.abs(angles)))
        ctrl = np.concatenate([matched_kymograph(signed_all, template, pts, normal, centre, across,
                                                 angles[zero:zero + 1], lateral_offset=off)[:, 0]
                               for off in (-p.mk_control_px, p.mk_control_px)], axis=1)  # (bins, 2 * points)
        ctrl_sigma = np.maximum(1.4826 * np.median(np.abs(ctrl - np.median(ctrl, axis=1, keepdims=True)), axis=1), 0.3)
        base = kymo_all[:p.ref_bins].mean(axis=0)
        tau_all = np.maximum(p.mk_floor, base[None] + p.mk_k * ctrl_sigma[:, None, None])  # (bins, angles, points)
        evid_all = np.clip((kymo_all - tau_all) / tau_all, -1.0, 1.0)
        extra["matched_control_sigma_median"] = round(float(np.median(ctrl_sigma)), 3)
        matched_evid = evid_all
    if p.evidence != "matched":
        kymo_all = _kymograph(diffs, pts, normal, pivot, p.lateral, angles)  # (bins, angles, points)
        if p.bg_subtract:
            # per-bin background change (focus / illumination drift) away from tube and grains; the same for
            # every candidate path of the grain, so worked out once
            if "bg_level" not in ctx:
                far = cv2.dilate(ctx["tube_mask"].astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
                bg_mask = (rg > gr + 8) & ~ctx["blocked"] & ~far
                ctx["bg_level"] = (np.median(diffs[:, bg_mask], axis=1).astype(np.float64) if bg_mask.any()
                                   else np.zeros(len(diffs)))
            bg_level = ctx["bg_level"]
            kymo_all = np.clip(kymo_all - bg_level[:, None, None], 0.0, None)
            extra["background_change_max"] = round(float(bg_level.max()), 2)
        base = kymo_all[:p.ref_bins].mean(axis=0)
        noise = np.maximum(kymo_all[:p.ref_bins].std(axis=0), 0.5)
        tau_all = np.maximum(p.evid_floor, base + p.evid_k * noise)[None]
        evid_all = np.clip((kymo_all - tau_all) / tau_all, -1.0, 1.0)
    if p.evidence == "union":  # either reading may carry the tube: shape-agnostic |change| or the template
        evid_all = np.maximum(evid_all, matched_evid)
    evid_all[:, :, ss < p.skip_px] = 0.0
    # Rotation is judged only on points that can reveal it: >= rot_min_s along the path and
    # clear of the rim (a rim-hugging segment slides along the rim under any rotation).
    radius_pts = np.hypot(*(pts - centre).T)
    informative = (ss >= p.rot_min_s) & (radius_pts >= gr + p.rot_rim_clear)
    if p.rotate and informative.sum() >= 4:
        rot_score = np.clip(evid_all[:, :, informative], 0.0, None).sum(axis=2)
        theta = rotation_track(rot_score, angles, p.angle_penalty, p.max_turn, p.late_bins + 1)
    else:
        theta = np.zeros(n_bins)
    ai = np.array([int(np.argmin(np.abs(angles - a))) for a in theta])
    evid = evid_all[np.arange(n_bins), ai]
    front = dp_front(evid, max(1, int(round(p.vmax_px / p.step))))
    if p.rotate and p.rot_refine and informative.sum() >= 4:
        # While the tube is short, the far path points read only neighbours and debris at every
        # angle, and the track follows those. Judge rotation again only on the part of the path
        # the first front says exists by then (plus a margin), and read the front once more.
        reach = front[:, None] * p.step + p.rot_refine_px
        mask = informative[None, :] & (ss[None, :] <= reach)
        rot_score = (np.clip(evid_all, 0.0, None) * mask[:, None, :]).sum(axis=2)
        theta = rotation_track(rot_score, angles, p.angle_penalty, p.max_turn, p.late_bins + 1)
        ai = np.array([int(np.argmin(np.abs(angles - a))) for a in theta])
        evid = evid_all[np.arange(n_bins), ai]
        front = dp_front(evid, max(1, int(round(p.vmax_px / p.step))))
    kymo = kymo_all[np.arange(n_bins), ai]
    tau = tau_all[-1, ai[-1]] if tau_all.ndim == 3 else tau_all[ai[-1]]
    behind = np.arange(len(pts))[None, :] < front[:, None]
    explained = float(np.sum(np.where(behind, evid, 0.0)))
    beyond = float(np.sum(np.where(~behind, np.clip(evid, 0.0, None), 0.0)))
    # an exit on a structure that predates the front's departure is a foreign tube passing by
    through = False
    moved = np.nonzero(front >= int(round(3.0 / p.step)))[0]
    if len(moved):
        # a drifting grain's own edge leaves change round the rim: keep clear of it
        margin = 3.0 + float(np.hypot(*ctx["ls"].T).max())
        yy, xx = np.nonzero(ctx["comp"] & (np.hypot(*(np.mgrid[0:rg.shape[0], 0:rg.shape[1]][::-1] -
                                                     pts[0][:, None, None])) <= p.through_px + margin)
                            & (rg > gr + margin))
        if len(yy):
            dpath = np.min(np.hypot(xx[:, None] - pts[None, :, 0], yy[:, None] - pts[None, :, 1]), axis=1)
            yy, xx = yy[dpath > 4.0], xx[dpath > 4.0]
        if len(yy) >= p.through_min_px:
            frac = (diffs[:, yy, xx] > ctx["thr"]).mean(axis=1)
            t0 = int(moved[0])
            early_bins = frac[max(0, t0 - 8):max(0, t0 - 2)]
            through = bool(len(early_bins) and np.median(early_bins) >= 0.5)
    score = explained
    # a tube is a ridge in the end-state change: higher on the path than just beside it
    chg = ctx["change"]
    lat = p.ridge_px
    samp = lambda q: cv2.remap(chg.astype(np.float32), q[:, 0].astype(np.float32)[None],
                               q[:, 1].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
    on_path = np.max([samp(pts + o * normal) for o in (-1.0, 0.0, 1.0)], axis=0)
    beside = np.maximum(samp(pts + lat * normal), samp(pts - lat * normal))
    far_pts = ss >= p.skip_px + 2.0
    ridge = float(np.mean((on_path - beside > 0.25 * on_path)[far_pts])) if far_pts.any() else 1.0
    if p.ridge_weight > 0 and score > 0:
        score *= (1.0 - p.ridge_weight) + p.ridge_weight * ridge
    if through:
        score = score * 0.25 if score > 0 else score - 10.0
    extra["rotation_deg"] = [round(float(t), 1) for t in theta]
    if p.rotate and np.max(np.abs(theta)) >= 10:
        flags.append(f"rotates:{np.max(np.abs(theta)):.0f}deg")
    return {"pts": pts, "ss": ss, "theta": theta, "front": front, "kymo": kymo, "tau": tau, "evid": evid,
            "pivot": pivot,
            "explained": explained, "beyond": beyond, "through": through, "ridge": ridge, "score": score,
            "flags": flags, "result": extra}


def arrival_map(diffs: np.ndarray, thr: float, sigma: float = 1.0, hold: float = 0.8) -> np.ndarray:
    """Per pixel, the first bin from which its (blurred) change stays above ``thr`` in at least ``hold`` of the
    remaining bins - a tube, once grown, stays; -1 where that never happens."""
    n = len(diffs)
    above = np.stack([cv2.GaussianBlur(np.asarray(d, np.float32), (0, 0), sigma) > thr for d in diffs])
    rest = np.cumsum(above[::-1], axis=0, dtype=np.int16 if n < 32000 else np.int32)[::-1]  # bins above, t to end
    ok = above & (rest >= hold * np.arange(n, 0, -1)[:, None, None])
    return np.where(ok.any(axis=0), ok.argmax(axis=0), -1)


def _point_arrivals(pts: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Per point (x, y): the earliest arrival (``arrival_map``) within 1 px; inf where nothing changed to stay."""
    a = np.where(arr < 0, np.inf, arr.astype(np.float64))
    h, w = a.shape
    out = np.full(len(pts), np.inf)
    for i, (x, y) in enumerate(np.round(pts).astype(int)):
        if 0 <= x < w and 0 <= y < h:
            out[i] = a[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].min()
    return out


def _gate_by_arrival(evid: np.ndarray, pts: np.ndarray, arr: np.ndarray, t0: int, gate_bins: int) -> np.ndarray:
    """``evid`` (bins from ``t0`` on, then points at ``pts``, (x, y); any axes between) set to -1 at each point
    before the bin its change arrived to stay, less ``gate_bins``; points that never changed to stay are kept."""
    for i, ai in enumerate(_point_arrivals(pts, arr)):
        if np.isfinite(ai):
            evid[:max(0, int(ai) - gate_bins - t0), ..., i] = -1.0
    return evid


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties); 0 when either side is constant."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def continue_read(ctx: dict, read: dict, p: "Params") -> dict | None:
    """The chosen read carried on beyond its tip (``p.tip_continue``), or None.

    Only when its front reached the path's end ``cont_after_bins`` or more before the last bin: the change region's
    material that changed from then on (``cont_margin_bins`` early), away from the path but connected to its last
    ``cont_back_px``, is the candidate continuation, followed to its geodesically farthest point; its change must
    arrive in order outwards (``cont_order``). Up to the bin the first front reached the tip the reading is kept
    as it was (capped where the continuation leaves the path); from then on a second front (the same dynamic
    programme, a point counting from the bin its change arrived) grows along the continuation, and it must get
    ``cont_min_px`` along it."""
    front, pts, path_yx = read["front"], read["pts"], read.get("path_yx")
    n_bins, step = ctx["n_bins"], p.step
    if path_yx is None or len(path_yx) < 2 or front[-1] < len(pts) - 1 - int(round(2.0 / step)):
        return None  # the tube never got to the end of this path: nothing grew beyond it
    t1 = int(np.argmax(front >= front[-1] - int(round(1.0 / step))))
    if t1 > n_bins - 1 - p.cont_after_bins:
        return None
    if ctx.get("arrival") is None:
        ctx["arrival"] = arrival_map(ctx["diffs"], ctx["thr"], p.map_sigma)
    comp, arr = ctx["comp"], ctx["arrival"]
    h, w = comp.shape
    tip = (int(round(path_yx[-1][0])), int(round(path_yx[-1][1])))
    if not (0 <= tip[0] < h and 0 <= tip[1] < w):
        return None
    on_path = np.zeros((h, w), np.uint8)
    cv2.polylines(on_path, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
    away = cv2.distanceTransform(1 - on_path, cv2.DIST_L2, 3) > p.cont_clear_px
    # the continuation may leave the path within its last cont_back_px (the path's end overshoots a tight turn):
    # wherever makes the whole route, along the path and then on, cheapest through the change map
    cost = 1.0 / (ctx["change"] + 1.0)
    yx = np.round(path_yx).astype(int)
    step_len = np.concatenate([[0.0], np.hypot(*np.diff(path_yx, axis=0).T)])
    raw_arc = np.cumsum(step_len)
    along = np.cumsum(cost[yx[:, 0], yx[:, 1]] * step_len)
    k0 = int(np.searchsorted(raw_arc, raw_arc[-1] - p.cont_back_px))
    seeds = np.zeros((h, w), bool)
    start_cost = np.full((h, w), np.inf)
    for i in range(k0, len(yx)):
        seeds[yx[i, 0], yx[i, 1]] = True
        start_cost[yx[i, 0], yx[i, 1]] = min(start_cost[yx[i, 0], yx[i, 1]], along[i] - along[k0])
    end = np.zeros((h, w), np.uint8)
    end[seeds] = 1
    near_end = comp & (cv2.distanceTransform(1 - end, cv2.DIST_L2, 3) <= p.cont_clear_px + 1.0)
    later = (comp & (arr >= t1 - p.cont_margin_bins) & away) | near_end | seeds
    _, lab = cv2.connectedComponents(later.astype(np.uint8), connectivity=8)
    part = lab == lab[tip]
    far, dist = _geodesic_far(part, seeds & part)
    if dist[far] < p.cont_min_px:
        return None
    cont = _cheapest_path(cost, part, seeds & part, far, start_cost=start_cost)
    # where it leaves the path: the path is cut there
    on = np.nonzero((yx[k0:, 0] == int(cont[0, 0])) & (yx[k0:, 1] == int(cont[0, 1])))[0]
    j = k0 + int(on[-1]) if len(on) else len(yx) - 1
    cut = path_yx[j, ::-1].astype(float)  # (x, y)
    path_yx = path_yx[:j + 1]
    # the tube grows at its tip: along a real continuation the change arrives later the farther out it is
    a = arr[cont[:, 0].astype(int), cont[:, 1].astype(int)].astype(float)
    ok = a >= 0
    if ok.sum() < 4 or _rank_corr(np.nonzero(ok)[0].astype(float), a[ok]) < p.cont_order:
        return None
    full = read_path(ctx, np.vstack([path_yx, cont[1:]]), p)
    if full is None:
        return None
    # the cut point on the new path (its first part is the old path's up to there)
    n1 = int(np.argmin(np.hypot(*(full["pts"][:len(pts)] - cut).T)))
    # read in the order it arrived: a continuation point is tube only from the bin its change came to stay
    ev2 = _gate_by_arrival(full["evid"][t1:, n1:].copy(), full["pts"][n1:], arr, t1, p.cont_gate_bins)
    # the second front starts at the tip (a first row nothing but the tip can explain)
    f2 = dp_front(np.vstack([np.full((1, ev2.shape[1]), -1e3), ev2]), max(1, int(round(p.vmax_px / step))))[1:]
    if f2[-1] * step < p.cont_min_px:
        return None
    merged = dict(full)
    merged["front"] = np.concatenate([np.minimum(front[:t1], n1), n1 + f2]).astype(np.int32)
    merged["theta"] = np.concatenate([read["theta"][:t1], full["theta"][t1:]])
    merged["kymo"] = full["kymo"]
    merged["flags"] = list(read["flags"])
    merged["result"] = {**read["result"], "rotation_deg": [round(float(t), 1) for t in merged["theta"]],
                        "continued_px": round(float(full["ss"][-1] - full["ss"][n1]), 1)}
    merged["path_yx"] = np.vstack([path_yx, cont[1:]])
    return merged


def over_grain(ctx: dict, exit_xy: np.ndarray, b_rim: int, p: "Params") -> dict | None:
    """Tube material that grew over the grain before the tube left it at ``exit_xy`` (crop (x, y)), or None.

    In movie 1 some tubes emerge from a pore facing the camera: they appear inside the grain's disc and grow over it
    to the rim (g015, g030: a dark curl inside the disc bins before anything shows outside). Both readers block the
    disc, so such a tube is seen only once it crosses the rim. Here, the pixels inside the disc whose change comes to
    stay (``arrival_map`` at ``over_k`` times the disc's own noise in the bins after the reference) by the rim onset
    ``b_rim`` and connect to the exit are the tube over the grain when they form a strip from a pore (the point
    farthest from the exit through them) to the exit, at least ``over_min_px`` long and at most ``over_max_width``
    wide on average, whose change arrived in order towards the exit (rank correlation ``over_order``) and, at the
    pore, ``over_lead_bins`` or more before the rim onset. A grain whose interior changes all at once (it turns or
    tumbles, its cytoplasm moves) fails the order or the width.

    Returns {"b_in": the bin the change came to stay at the pore, "inner_px": the strip's length from the pore to
    the exit, "grown": (n_bins,) its length grown by each bin, "route_yx": the strip's centreline from the pore,
    "order", "thr"}."""
    diffs, centre, gr, rg, n = ctx["diffs"], ctx["centre"], ctx["gr"], ctx["rg"], ctx["n_bins"]
    inside = rg < gr - 1.0
    k0 = min(p.ref_bins, n - 1)
    k1 = int(np.clip(b_rim - p.over_lead_bins, k0 + 1, k0 + 10))
    if k1 <= k0 or not inside.any():
        return None
    # each pixel's own noise over the quiet bins after the reference (the rim flickers more than the body), and the
    # disc's as a floor
    quiet = np.stack([cv2.GaussianBlur(np.asarray(d, np.float32), (0, 0), p.map_sigma) for d in diffs[k0:k1]])
    level = np.median(quiet, axis=0)
    spread = 1.4826 * np.median(np.abs(quiet - level), axis=0)
    sigma = 1.4826 * float(np.median(np.abs(quiet[:, inside] - np.median(quiet[:, inside]))))
    thr_px = np.maximum(p.over_floor, level + p.over_k * np.maximum(spread, sigma))
    thr = float(np.median(thr_px[inside]))
    arr = arrival_map(diffs, thr_px, p.map_sigma)
    cand = inside & (arr >= 0) & (arr <= b_rim)
    h, w = inside.shape
    yy, xx = np.mgrid[0:h, 0:w]
    near_exit = inside & (np.hypot(xx - exit_xy[0], yy - exit_xy[1]) <= p.over_join_px)
    if not near_exit.any():
        return None
    _, lab = cv2.connectedComponents((cand | near_exit).astype(np.uint8), connectivity=8)
    keep = np.unique(lab[near_exit])
    part = np.isin(lab, keep[keep > 0]) & (cand | near_exit)
    if not (part & cand & ~near_exit).any():
        return None
    far, dist = _geodesic_far(part, near_exit)
    route = _cheapest_path(1.0 / (ctx["change"] + 1.0), part, near_exit, far)[::-1]  # from the pore to the exit
    seg = np.hypot(*np.diff(route, axis=0).T) if len(route) > 1 else np.zeros(0)
    inner = float(seg.sum() + np.hypot(route[-1][1] - exit_xy[0], route[-1][0] - exit_xy[1]))
    area = float((part & cand).sum())
    if inner < p.over_min_px or area > p.over_max_width * max(inner, 1.0):
        return None
    a = arr[route[:, 0].astype(int), route[:, 1].astype(int)].astype(float)
    ok = a >= 0
    order = _rank_corr(np.nonzero(ok)[0].astype(float), a[ok]) if ok.sum() >= 3 else 0.0
    if order < p.over_order:
        return None
    head = a[ok][:max(2, int(ok.sum()) // 4)]
    b_in = int(np.median(head))
    if b_in > b_rim - p.over_lead_bins:
        return None
    # its length by each bin: as far along the strip as its change had come (every point up to there arrived)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    reached = np.maximum.accumulate(np.where(ok, a, np.inf))
    grown = np.array([float(s[reached <= t].max()) if (reached <= t).any() else 0.0 for t in range(n)])
    grown[b_rim:] = inner
    return {"b_in": b_in, "inner_px": inner, "grown": grown, "route_yx": route, "order": round(order, 2),
            "thr": round(thr, 2)}


def with_over_grain(flood_res: dict, change_res: dict, p: "Params") -> dict:
    """The flood's reading (under the change reader's onset) with the change reader's tube over the grain: its flag
    and, with ``over_grain="length"``, lengths from the pore (the strip over the grain as far as it had grown, then
    all of it and the flood's tube beyond the rim)."""
    og = change_res["over_grain"]
    out = dict(flood_res)
    out["over_grain"] = og
    out["flags"] = list(flood_res["flags"]) + [f for f in change_res["flags"] if f.startswith("tube_over_grain")]
    if p.over_grain == "length" and out.get("onset_frame") is not None:
        frames, px = out["length"]["frames"], out["length"]["px"]
        grown = og["grown_px"]
        px = [round(v + og["inner_px"], 2) if v > 0 else
              (grown[min(i, len(grown) - 1)] if f >= out["onset_frame"] else 0.0)
              for i, (f, v) in enumerate(zip(frames, px))]
        out["length"] = {"frames": frames, "px": px}
        out["final_length_px"] = round(float(px[-1]), 2) if px else 0.0
        out["flags"].append("length_from_pore")
    return out


def grain_settling(crops: np.ndarray, centre: float, gr: float, p: "Params") -> dict:
    """Is there a settled grain here, and from which bin?

    The census reads the field's first bins; a grain still arriving then (a blurred,
    moving blob) is found late or off-centre, and a passing piece of debris is found
    where no grain ever settles. The rim fit per early bin tells them apart.
    """
    n = min(len(crops), p.settle_bins)
    q = np.array([census.rim_fit(c, centre, centre, gr)[2] for c in crops[:n]])
    qmed = float(np.median(q))
    if qmed < p.grain_min_rim:
        return {"no_grain": True, "b0": 0, "rim_median": round(qmed, 2)}
    out = {"no_grain": False, "b0": 0, "rim_median": round(qmed, 2)}
    if float(np.min(q[:p.ref_bins])) >= 0.4 * qmed:
        return out
    for b in range(1, n - 2):
        if np.all(q[b:b + 3] >= 0.85 * qmed):  # the rim has come into focus and stays
            out["b0"] = b
            break
    return out


def _pad_front(res: dict, frames: list, b0: int) -> dict:
    """Re-express a result computed from bin b0 on the grain's full frame list."""
    res["length"] = {"frames": frames, "px": [0.0] * b0 + list(res["length"]["px"])}
    if res.get("tip"):
        res["tip"] = {"frames": frames, "xy": [res["tip"]["xy"][0]] * b0 + list(res["tip"]["xy"])}
    if res.get("rotation_deg"):
        res["rotation_deg"] = [res["rotation_deg"][0]] * b0 + list(res["rotation_deg"])
    if res.get("wedge"):
        res["wedge"]["signal"] = [0.0] * b0 + list(res["wedge"]["signal"])
    if res.get("drift"):
        res["drift"] = {"frames": frames, "xy": [res["drift"]["xy"][0]] * b0 + list(res["drift"]["xy"])}
    if res.get("over_grain"):
        res["over_grain"]["grown_px"] = [0.0] * b0 + list(res["over_grain"]["grown_px"])
    return res


def reads_in_grain_frame(drift: np.ndarray | None, gr: float, p: "Params") -> bool:
    """Whether a grain is read in its own frame (the crops registered by its followed drift) rather than by the
    phase track as before: always with grain_track "follow"; with "auto" once it has moved off its own place (further
    than ``track_far_r`` radii), where reading it at its old place is surely wrong. Nearer, a tube stuck to the
    substrate stays sharp in the field frame while its grain is pushed a few px (movie 2: g054 and g038, pushed 7 and
    20 px, each lost a length hit when read in their own frames; on synthetic movies, whose grains carry their tubes
    rigidly, following gains)."""
    if drift is None or p.grain_track == "phase":
        return False
    if p.grain_track == "follow":
        return True
    fin = drift[np.isfinite(drift).all(axis=1)]
    return bool(len(fin)) and float(np.hypot(*fin.T).max()) > p.track_far_r * gr


def hold_nan(a: np.ndarray) -> np.ndarray:
    """Rows of NaN after the last finite row repeat that row (a lost grain read on at its last place)."""
    a = np.array(a, float)
    fin = np.flatnonzero(np.isfinite(a).all(axis=1))
    if len(fin) and fin[-1] < len(a) - 1:
        a[fin[-1] + 1:] = a[fin[-1]]
    return np.nan_to_num(a)


def hold_after(res: dict, frames: list, n_fol: int, flag: str) -> dict:
    """Re-express a result read over the first ``n_fol`` bins of the grain's frame list ``frames`` on all of it:
    readings after the grain was lost are held at their last value (not extrapolated), and flagged."""
    n = len(frames)

    def held(seq, fill=None):
        seq = list(seq)
        last = seq[-1] if seq else fill
        return seq + [last] * (n - len(seq))

    res["length"] = {"frames": frames, "px": held(res["length"]["px"], 0.0)}
    for key in ("tip", "drift"):
        if res.get(key):
            res[key] = {"frames": frames, "xy": held(res[key]["xy"])}
    for key in ("rotation_deg", "exit_angle_deg"):
        if res.get(key):
            res[key] = held(res[key])
    if res.get("wedge"):
        res["wedge"]["signal"] = held(res["wedge"]["signal"], 0.0)
    res["flags"].append(flag)
    res["observed_until_frame"] = frames[n_fol - 1] if n_fol > 0 else frames[0]
    return res


def read_lost(renderer: Renderer, meta: dict, grain: dict, p: "Params", fd: dict, read, half: int,
              flags: tuple = ()) -> dict:
    """A followed grain lost from bin ``fd["lost_from"]``: ``read(truncated meta, drift)`` reads it over the bins it
    was followed and its readings are held from there (``hold_after``); followed for fewer than
    ``p.lost_min_bins`` bins, it is unobservable."""
    fpb, rs = int(meta["frames_per_bin"]), int(meta.get("ref_start", 0))
    n_bins = int(meta["n_bins"]) - rs
    frames = [b * fpb + fpb // 2 for b in range(rs, rs + n_bins)]
    n_fol = int(fd["lost_from"]) - rs
    last = frames[max(n_fol - 1, 0)]
    flag = f"grain_lost_after:{last}"
    if n_fol < p.lost_min_bins:
        c = np.nan_to_num(renderer.crop(rs, grain["x"], grain["y"], half))
        return {"id": grain["id"], "x": grain["x"], "y": grain["y"], "r": grain["r"], "flags": list(flags) + [flag],
                "map_threshold": 1.0, "status": "unobservable", "onset_frame": None, "onset_interval": None,
                "length": {"frames": frames, "px": [0.0] * n_bins}, "path": [],
                "observed_until_frame": last, "lost_reason": fd["lost_reason"],
                "_diag": (c, np.zeros_like(c), np.zeros(c.shape, bool), None, None, None, half - 0.5)}
    res = read({**meta, "n_bins": int(fd["lost_from"])}, fd["drift"][:n_fol])
    res["lost_reason"] = fd["lost_reason"]
    return hold_after(res, frames, n_fol, flag)


def analyze_grain(renderer: Renderer, meta: dict, grain: dict, others: list[dict], p: Params,
                  _settled: bool = False, route: list | None = None, _drift: np.ndarray | None = None) -> dict:
    fpb, rs = int(meta["frames_per_bin"]), int(meta.get("ref_start", 0))
    n_bins = int(meta["n_bins"]) - rs  # bins before the reference (settling) are not observed
    gx, gy, gr = grain["x"], grain["y"], grain["r"]
    half = p.half
    crops = np.stack([renderer.crop(b, gx, gy, half) for b in range(rs, rs + n_bins)])
    centre = half - 0.5  # crop pixel coordinate of the grain centre
    if p.settle and not _settled:
        st = grain_settling(crops, centre, gr, p)
        frames = [b * fpb + fpb // 2 for b in range(rs, rs + n_bins)]
        if st["no_grain"]:
            return {"id": grain["id"], "x": gx, "y": gy, "r": gr, "flags": ["no_grain"], "map_threshold": 1.0,
                    "status": "unobservable", "onset_frame": None, "onset_interval": None,
                    "length": {"frames": frames, "px": [0.0] * n_bins}, "path": [], "rim_median": st["rim_median"],
                    "_diag": (crops[-2], np.zeros_like(crops[0]), np.zeros(crops[0].shape, bool), None, None, None,
                              centre)}
        b0 = st["b0"]
        if b0 > 0:
            # re-find the settled grain (the census saw it arriving) and read it from bin b0 on
            ref = crops[b0:b0 + 3].mean(axis=0)
            w = int(gr + 30)
            c = int(round(centre))
            found = census.detect(ref[c - w:c + w, c - w:c + w], r_min=max(5, int(gr - 4)), r_max=int(gr + 4),
                                  ring_min=5.0, body_min=15.0)
            if found:
                best = min(found, key=lambda f: math.hypot(f["x"] - w, f["y"] - w))
                if math.hypot(best["x"] - w, best["y"] - w) <= 14:
                    gx, gy = gx + best["x"] - w + 0.5, gy + best["y"] - w + 0.5
            res = analyze_grain(renderer, {**meta, "ref_start": rs + b0}, {**grain, "x": gx, "y": gy}, others, p,
                                _settled=True)
            res["flags"].append(f"settled_from_bin:{b0}")
            return _pad_front(res, frames, b0)
    if p.grain_track in ("follow", "auto") and _drift is None:
        here = {**grain, "x": gx, "y": gy}
        fd = followed_drift(renderer, meta, here, others, p)
        if fd["lost_from"] is not None and p.lost_policy == "hold":
            return read_lost(renderer, meta, here, p, fd, half=half, read=lambda m, d: analyze_grain(
                renderer, m, here, others, p, _settled=True, route=route, _drift=d))
        _drift = hold_nan(fd["drift"])  # "read" on: at its last place
    followed = reads_in_grain_frame(_drift, gr, p)
    if followed:
        # a grain that has moved far is cropped at its whole-pixel place and registered by the rest of its drift;
        # nearer, its crop is warped by the whole drift, as the phase track's is (same readings where they agree)
        off = np.round(_drift) if np.abs(_drift).max() > p.track_recentre_px else np.zeros_like(_drift)
        if np.any(off != 0):
            off_abs = np.zeros((int(meta["n_bins"]), 2))
            off_abs[rs:rs + n_bins] = off
            crops = np.stack([renderer.crop(b, gx, gy, half, offsets=off_abs) for b in range(rs, rs + n_bins)])
        ls, resid, drift_flag = _drift, _drift - off, None
    else:
        ls = local_shifts(crops, centre, gr, p.reg_pad, p.ref_bins)
        ls, drift_flag = checked_drift(ls, p)  # a track that locked onto a neighbour or the grain's own tube is not used
        resid = ls
    reg = np.stack([cv2.warpAffine(c, np.float32([[1, 0, -dx], [0, 1, -dy]]), (2 * half, 2 * half),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    for c, (dx, dy) in zip(crops, resid)])
    late_idx = list(range(n_bins - 1 - p.late_bins, n_bins - 1))
    early = reg[:p.ref_bins].mean(axis=0)
    late = reg[late_idx].mean(axis=0)
    yy, xx = np.mgrid[0:2 * half, 0:2 * half].astype(np.float64)
    rg = np.hypot(xx - centre, yy - centre)
    # pixels whose source position leaves the frame in any bin are not evidence
    shifts = np.asarray(meta["shifts"])[rs:rs + n_bins] + ls
    ref_x, ref_y = gx - half + xx + 0.5, gy - half + yy + 0.5
    valid = ((ref_x + shifts[:, 0].min() >= 0) & (ref_x + shifts[:, 0].max() < renderer.width) &
             (ref_y + shifts[:, 1].min() >= 0) & (ref_y + shifts[:, 1].max() < renderer.height))
    change = cv2.GaussianBlur(np.abs(late - early), (0, 0), p.map_sigma)
    blocked = (rg < gr - 1.0) | ~valid
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        if -o["r"] - 5 < ox < 2 * half + o["r"] + 5 and -o["r"] - 5 < oy < 2 * half + o["r"] + 5:
            blocked |= np.hypot(xx - ox, yy - oy) < o["r"] + p.other_block_px
    bg = change[(rg > gr + 30) & ~blocked]
    bg = bg[bg < np.percentile(bg, 95)] if bg.size else bg
    sigma_bg = 1.4826 * float(np.median(np.abs(bg - np.median(bg)))) if bg.size else 1.0
    thr = max(p.map_floor, p.map_k * sigma_bg)
    tube_mask = (change > thr) & ~blocked
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(tube_mask.astype(np.uint8), connectivity=8)
    ring = (rg >= gr - 1.0) & (rg <= gr + 4.0)
    attached = [(stats[l, cv2.CC_STAT_AREA], l) for l in range(1, n_lab)
                if stats[l, cv2.CC_STAT_AREA] >= p.min_component_px and np.any(ring & (lab == l))]
    result = {"id": grain["id"], "x": gx, "y": gy, "r": gr, "flags": [drift_flag] if drift_flag else [],
              "map_threshold": round(thr, 2), "local_shift_max_px": round(float(np.hypot(*ls.T).max()), 2)}
    frames = [b * fpb + fpb // 2 for b in range(rs, rs + n_bins)]
    if followed or np.any(ls):
        # paths and tips are in the frame the grain was read in (its crops registered by its followed drift or its
        # phase track): its place in the field is census + drift. Until 0.8.2 only followed grains kept theirs, so
        # the app drew every other grain and tube where it started while the grain drifted (ld g017: 20 px)
        result["drift"] = {"frames": frames, "xy": np.round(ls, 2).tolist()}
    if not attached:
        result.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None,
                      length={"frames": frames, "px": [0.0] * n_bins}, path=[])
        result["_diag"] = (late, change, tube_mask, None, None, None, centre)
        return result
    attached.sort(reverse=True)
    # with growth-scored candidates every region touching the rim is a hypothesis; otherwise the largest
    comp = np.isin(lab, [l for _, l in attached]) if p.candidates else lab == attached[0][1]
    if len(attached) > 1:
        result["flags"].append("second_attached_component")
    # a change region shared with other grains is split by geodesic ownership: each pixel
    # belongs to the grain nearest to it through the change map itself
    rivals = []
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        o_ring = np.abs(np.hypot(xx - ox, yy - oy) - o["r"]) <= 4.0
        if np.any(comp & o_ring):
            result["flags"].append(f"touches:{o['id']}")
            rivals.append(o_ring)
    if rivals:
        owner = geodesic_owner(comp, [ring] + rivals)
        own = owner == 0
        n_own, own_lab = cv2.connectedComponents(own.astype(np.uint8), connectivity=8)
        keep = [l for l in range(1, n_own) if np.any(ring & (own_lab == l))]
        comp = np.isin(own_lab, keep) if keep else own
        result["flags"].append("shared_change_split")
    cost = 1.0 / (change + 1.0)
    diffs = np.abs(reg - early[None])
    ctx = {"reg": reg, "early": early, "late": late, "ls": ls, "late_idx": late_idx,
           "centre": centre, "gr": gr, "rg": rg, "blocked": blocked, "tube_mask": tube_mask, "diffs": diffs,
           "signed": (reg - early[None]).astype(np.float32), "n_bins": n_bins, "thr": thr, "comp": comp,
           "change": change}
    if route is not None:
        # a centreline chosen elsewhere (the flood's: which tube is the grain's), read here from the rim out;
        # reference (x, y) points -> crop (y, x), densified to ~1 px and cut where it leaves the crop
        q = np.array([[y - gy + centre, x - gx + centre] for x, y in route], float)
        seg = np.hypot(*np.diff(q, axis=0).T) if len(q) > 1 else np.zeros(0)
        dense = [q[0]] + [q[i] + (q[i + 1] - q[i]) * f for i in range(len(seg))
                          for f in np.linspace(0, 1, max(2, int(np.ceil(seg[i])) + 1))[1:]] if len(q) > 1 else list(q)
        dense = np.array(dense)
        out = np.nonzero((dense < 2).any(axis=1) | (dense > 2 * half - 3).any(axis=1))[0]
        dense = dense[:out[0]] if len(out) else dense
        read = read_path(ctx, dense, p) if len(dense) >= 3 else None
        result["flags"].append("route_given")
    elif p.candidates:
        cands = []
        for path_yx in candidate_paths(comp, ring, cost, gr, centre, p):
            read = read_path(ctx, path_yx, p)
            if read is not None:
                read["path_yx"] = path_yx
                cands.append(read)
        # a candidate that is only a shorter stretch of another along the same tube is dropped:
        # the growth front, not the path, decides how far the tube got (and tips are faint)
        live = []
        for i, a in enumerate(cands):
            nested = False
            for j, b in enumerate(cands):
                # only a longer path from the same exit counts as the same tube reaching further
                if (j != i and b["ss"][-1] > a["ss"][-1] + 1.0
                        and math.hypot(*(a["pts"][0] - b["pts"][0])) <= p.nest_px + 1.0):
                    d = np.min(np.hypot(a["pts"][:, None, 0] - b["pts"][None, :, 0],
                                        a["pts"][:, None, 1] - b["pts"][None, :, 1]), axis=1)
                    if np.all(d <= p.nest_px):
                        nested = True
                        break
            if not nested:
                live.append(a)
        read = max(live or cands, key=lambda c: c["score"]) if cands else None
        result["path_candidates"] = [{"length_px": round(float(c["ss"][-1]), 1), "score": round(c["score"], 1),
                                      "explained": round(c["explained"], 1), "beyond": round(c["beyond"], 1),
                                      "ridge": round(c["ridge"], 2), "through": c["through"]} for c in cands]
        if len(cands) > 1 and read is not cands[0]:
            result["flags"].append("path_by_growth")
        if p.tip_continue and read is not None:
            cont = continue_read(ctx, read, p)
            if cont is not None:
                read = cont
                result["flags"].append(f"tip_continued:{cont['result']['continued_px']:.0f}px")
    else:
        far, _ = _geodesic_far(comp, ring)
        read = read_path(ctx, _cheapest_path(cost, comp, ring, far), p)
    if read is None:  # a one- or two-pixel path is not a tube
        result["flags"].append("degenerate_path")
        result.update(status="no_emergence_by_end", onset_frame=None, onset_interval=None,
                      length={"frames": frames, "px": [0.0] * n_bins}, path=[])
        result["_diag"] = (late, change, tube_mask, None, None, None, centre)
        return result
    pts, ss, theta, front, kymo, tau = (read[k] for k in ("pts", "ss", "theta", "front", "kymo", "tau"))
    result["flags"] += read["flags"]
    # how much of its own change region the path accounts for: a tube that curls, turns back
    # or wraps round its grain leaves part of that region far from the path (a review signal)
    on_path = np.zeros(comp.shape, np.uint8)
    cv2.polylines(on_path, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
    by_path = cv2.distanceTransform(1 - on_path, cv2.DIST_L2, 3) <= p.cover_px
    _, comp_lab = cv2.connectedComponents(comp.astype(np.uint8), connectivity=8)
    own = np.isin(comp_lab, [l for l in np.unique(comp_lab[on_path.astype(bool) & comp]) if l > 0])
    result["uncovered_px"] = int((own & ~by_path).sum())
    result["path_coverage"] = round(float((own & by_path).sum()) / max(int(own.sum()), 1), 3)
    result.update({k: v for k, v in read["result"].items()})
    # contact censoring: stop measuring where the path first reaches another grain's rim
    contact_idx = None
    for o in others:
        ox, oy = o["x"] - gx + centre, o["y"] - gy + centre
        near = np.nonzero(np.hypot(pts[:, 0] - ox, pts[:, 1] - oy) <= o["r"] + p.contact_px)[0]
        if len(near) and (contact_idx is None or near[0] < contact_idx):
            contact_idx = int(near[0])
    if contact_idx is not None:
        censor = np.nonzero(front > contact_idx)[0]
        if len(censor):
            result["length_censored_from_frame"] = frames[int(censor[0])]
            result["flags"].append("contact_censored")
        front = np.minimum(front, contact_idx)
    length = np.array([ss[f - 1] if f > 0 else 0.0 for f in front])

    piv = np.broadcast_to(np.asarray(read.get("pivot", centre), float), (2,))

    def rotated(pt, deg):
        a = math.radians(deg)
        r = pt - piv
        return np.array([piv[0] + math.cos(a) * r[0] - math.sin(a) * r[1],
                         piv[1] + math.sin(a) * r[0] + math.cos(a) * r[1]])

    tips = np.array([rotated(pts[max(f - 1, 0)], th) for f, th in zip(front, theta)])
    above = np.nonzero(length >= p.onset_px)[0]
    front_onset = int(above[0]) if len(above) else None
    end_angle = math.degrees(math.atan2(pts[0][1] - centre, pts[0][0] - centre))
    u_exit = (pts[0] - centre) / max(float(np.hypot(*(pts[0] - centre))), 1e-6)
    e_exit = (float(np.clip(exit_edge(early, centre, gr, math.atan2(u_exit[1], u_exit[0])), -(gr - 1.0), ss[-1] - 1.0))
              if p.exit_edge else 0.0)
    stub_pts = pts
    if p.onset_source == "matched":
        signed = (reg - early[None]).astype(np.float32)
        z, mf_info = matched_stub_signal(signed, (late - early), stub_pts, centre, p,
                                         theta=theta if p.mf_follow_rotation is True and p.rotate else None)
        if p.mf_follow_rotation == "both" and p.rotate and np.max(np.abs(theta)) >= 5:
            # a rotating grain's tube emerged elsewhere on the rim: also look along the rotation track
            z_rot, _ = matched_stub_signal(signed, (late - early), stub_pts, centre, p, theta=theta)
            z = np.maximum(z, z_rot)
        result["matched_filter"] = mf_info
        wedge, exit_track = z, np.full(n_bins, end_angle)
    elif p.onset_source == "wedge_fixed":
        wedge = wedge_signal(diffs, centre, gr, np.full(n_bins, end_angle), p)
        exit_track = np.full(n_bins, end_angle)
    else:
        wedge, exit_track = exit_track_signal(diffs, centre, gr, end_angle, p)
    result["exit_angle_deg"] = [round(float(a), 1) for a in exit_track]
    wedge_onset, wedge_thr = sustained_onset(wedge, p, threshold=p.mf_z if p.onset_source == "matched" else None)
    if wedge_onset is not None and p.onset_source == "matched" and p.mf_z_low is not None:
        # hysteresis: a confirmed stub reaches back while its score stays above the lower threshold
        confirmed = wedge_onset
        while wedge_onset > 0 and wedge[wedge_onset - 1] > p.mf_z_low and confirmed - wedge_onset < p.mf_back_bins:
            wedge_onset -= 1
    result["onset_bins"] = {"front": front_onset, "wedge": wedge_onset}
    result["wedge"] = {"signal": [round(float(v), 2) for v in wedge], "threshold": round(wedge_thr, 2)}
    b = front_onset if p.onset_source == "front" else wedge_onset
    if b is None and front_onset is not None and p.onset_source != "front":
        b = front_onset
        result["flags"].append("onset_from_front")
    elif b is not None and p.front_lead_px > 0 and length[b] >= p.front_lead_px:
        # The stub was confirmed only once the front was already long: the tube was there before.
        # Go back to the start of the growth run that led there, stopping at a plateau (a pore
        # bulge or rim change can hold the front a few px out long before the tube emerges).
        t = b
        while t > 2 and length[t - 1] > p.onset_px and length[t] - length[t - 3] > p.run_min_px:
            t -= 1
        if t < b:
            b = t
            result["flags"].append("onset_moved_to_front")
    og = over_grain(ctx, pts[0], b, p) if p.over_grain != "off" and b is not None and b > p.ref_bins else None
    if og is not None:
        result["flags"].append(f"tube_over_grain:{og['inner_px']:.0f}px")
        pore = og["route_yx"][0]
        result["over_grain"] = {"onset_frame": frames[og["b_in"]], "rim_onset_frame": frames[b],
                                "inner_px": round(og["inner_px"], 1), "order": og["order"],
                                "pore_xy": [round(float(pore[1] - centre + gx), 2), round(float(pore[0] - centre + gy), 2)],
                                "grown_px": [round(float(v), 2) for v in og["grown"]]}
        if p.over_grain in ("onset", "length"):
            b = og["b_in"]  # the tube was there, over its grain, before it reached the rim
    if b is None:
        status, onset, interval = "no_emergence_by_end", None, None
        result["flags"].append("tube_map_without_onset")
    elif b == 0:
        status, onset, interval = "emerged_at_start", frames[0], None
    else:
        status, onset, interval = "emerged_within", frames[b], [frames[b - 1], frames[b]]
    if b is None:
        length[:] = 0.0
    elif b > 0:
        length[:b] = 0.0  # no tube before its own onset
        tips[:b] = rotated(pts[0], 0.0)
    if b is not None and length[-1] < p.min_tube_px and "contact_censored" not in result["flags"]:
        result["flags"].append("front_too_short")
        status, onset, interval = "no_emergence_by_end", None, None
        length[:] = 0.0
    if p.tip_offset_px > 0:
        # The change signal outlasts the tube's visible end by the optical blur: on the real
        # benchmark the front ran a median 2.9 px beyond the human-clicked apex. Report the apex.
        grown = length > 0
        length = np.where(grown, np.maximum(length - p.tip_offset_px, 0.0), 0.0)
        back = int(round(p.tip_offset_px / p.step))
        for t in np.nonzero(grown)[0]:
            tips[t] = rotated(pts[max(int(front[t]) - 1 - back, 0)], theta[t])
    to_ref = lambda xy: [round(float(xy[0] - centre + gx), 2), round(float(xy[1] - centre + gy), 2)]
    drawn = list(pts[:: max(1, int(2 / p.step))]) + [pts[-1]]
    path_len = float(ss[-1])
    if og is not None and p.over_grain == "length" and b is not None:
        # lengths from the pore: over the grain as far as it had grown, then that strip and the tube beyond the exit
        grown = np.where(np.arange(n_bins) >= b, og["grown"], 0.0)
        length = np.where(length > 0, length + og["inner_px"], grown)
        drawn = [q[::-1] for q in og["route_yx"][::2]] + drawn
        path_len += og["inner_px"]
        result["flags"].append("length_from_pore")
    elif p.exit_edge and (length > 0).any():
        # measure from where the tube leaves the grain's visible edge; the path starts there too
        u, e = u_exit, e_exit
        length = np.where(length > 0, np.maximum(length - e, 0.0), 0.0)
        start = np.asarray(pts[0], float) + e * u if e < 0 else pts[min(int(np.searchsorted(ss, e)), len(pts) - 1)]
        drawn = [start] + [q for q, sq in zip(drawn, ss[:: max(1, int(2 / p.step))].tolist() + [ss[-1]]) if sq > e]
        path_len -= e
        result["exit_edge_px"] = round(e, 2)
    result.update(status=status, onset_frame=onset, onset_interval=interval,
                  length={"frames": frames, "px": [round(float(v), 2) for v in length]},
                  tip={"frames": frames, "xy": [to_ref(t) for t in tips]},
                  path=[to_ref(q) for q in drawn],
                  exit_xy=to_ref(drawn[0]), final_length_px=round(float(length[-1]), 2),
                  path_length_px=round(path_len, 2))
    result["_diag"] = (late, change, tube_mask, pts, kymo, (front, tau), centre)
    return result


def _diagnostic(res: dict, fpb: int) -> np.ndarray:
    late, change, mask, pts, kymo, front_tau, centre = res["_diag"]
    lo, hi = np.percentile(late, [0.5, 99.5])
    a = cv2.cvtColor(np.clip((late - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    c = cv2.cvtColor(np.clip(change / max(res["map_threshold"] * 3, 1e-6) * 255, 0, 255).astype(np.uint8),
                     cv2.COLOR_GRAY2BGR)
    c[mask] = (0.5 * c[mask] + (0, 90, 0)).astype(np.uint8)
    if pts is not None:
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(a, [poly], False, (0, 200, 255), 1)
        cv2.circle(a, tuple(np.round(pts[0]).astype(int)), 3, (0, 0, 255), -1)
    panels = [a, c]
    if kymo is not None:
        front, tau = front_tau
        k = np.clip(kymo / max(float(np.percentile(kymo, 99)), 1e-6) * 255, 0, 255).astype(np.uint8)
        k = cv2.cvtColor(cv2.resize(k, (late.shape[1], late.shape[0]), interpolation=cv2.INTER_NEAREST),
                         cv2.COLOR_GRAY2BGR)
        sy, sx = late.shape[0] / kymo.shape[0], late.shape[1] / kymo.shape[1]
        line = np.array([[f * sx, (t + 0.5) * sy] for t, f in enumerate(front)], np.int32).reshape(-1, 1, 2)
        cv2.polylines(k, [line], False, (0, 0, 255), 1)
        panels.append(k)
    sheet = np.concatenate(panels, axis=1)
    label = (f"{res['id']} {res['status']} onset {res.get('onset_frame')} final {res.get('final_length_px', 0)} px "
             f"{' '.join(res['flags'])}")
    band = np.full((18, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(band, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([band, sheet], axis=0)


def growth_scale(renderer: Renderer, meta: dict, grains: list[dict], p: "Params", log=print,
                 physical: list[dict] | None = None) -> float | None:
    """Fast growth per bin in this movie: the median over a sample of isolated grains of the
    90th percentile of their 3-bin front advance, read with a generous speed cap. None when
    too few tubes to tell."""
    from dataclasses import replace
    sample = [g for g in grains if g.get("isolated", True) and not g.get("border")]
    sample = sample[:: max(1, len(sample) // p.vmax_probe)][:p.vmax_probe]
    probe = replace(p, vmax_px=p.vmax_cap, vmax_auto=False)
    rates = []
    for g in sample:
        res = analyze_grain(renderer, meta, g, [o for o in (physical or grains) if o["id"] != g["id"]], probe)
        L = np.asarray(res.get("length", {}).get("px") or [], float)
        if res.get("status") != "emerged_within" or len(L) < 8 or L[-1] < 2 * p.min_tube_px:
            continue
        adv = (L[3:] - L[:-3]) / 3.0
        adv = adv[adv > 0.05]
        if len(adv) >= 5:
            rates.append(float(np.percentile(adv, 90)))
    if len(rates) < 3:
        return None
    return float(np.median(rates))


def focus_changes(bins: np.ndarray, meta: dict, window: int = 10, factor: float = 2.0) -> list[dict]:
    """Bins where the movie's focus changes: the median sharpness (variance of the Laplacian of the frame's
    centre) over the next ``window`` bins differs from the median over the previous ``window`` by more than
    ``factor`` either way. Movie 2 refocused at bin 63 (6.3x sharper after): tubes that emerged while it was out of
    focus became visible only then, so onsets there say "visible by", not "emerged at". Returns [{"bin", "frame",
    "ratio"}], the strongest bin of each change."""
    nb, fpb = int(meta["n_bins"]), int(meta["frames_per_bin"])
    h, w = bins.shape[1:]
    y0, y1, x0, x1 = h // 5, h - h // 5, w // 6, w - w // 6
    sharp = np.array([float(cv2.Laplacian(np.asarray(bins[b][y0:y1, x0:x1], np.float32), cv2.CV_32F).var())
                      for b in range(nb)])
    lr = np.zeros(nb)
    for b in range(window, nb - window + 1):
        lr[b] = math.log(max(np.median(sharp[b:b + window]), 1e-6) / max(np.median(sharp[b - window:b]), 1e-6))
    out, b = [], 0
    while b < nb:
        if abs(lr[b]) > math.log(factor):
            run = b
            while run + 1 < nb and abs(lr[run + 1]) > math.log(factor) and np.sign(lr[run + 1]) == np.sign(lr[b]):
                run += 1
            k = b + int(np.argmax(np.abs(lr[b:run + 1])))
            out.append({"bin": int(k), "frame": int(k * fpb + fpb // 2), "ratio": round(float(math.exp(lr[k])), 2)})
            b = run + 1
        else:
            b += 1
    return out


def analyze(cache_dir: str | Path, out_dir: str | Path, grains_path: str | Path | None = None,
            params: Params | None = None, only: list[str] | None = None, video: bool = False, log=print,
            units: tuple[float, float] | None = None) -> dict:
    p = params or Params()
    bins, meta = stack.load(cache_dir)
    renderer = Renderer(bins, meta)
    out_dir = Path(out_dir)
    (out_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    src = Path(grains_path) if grains_path else Path(cache_dir) / "grains.json"
    doc = json.loads(src.read_text())
    census = list(doc["grains"].values()) if isinstance(doc["grains"], dict) else doc["grains"]
    grains = [g for g in census if not g.get("excluded")]
    # grains excluded from scoring (clump members, edge grains) are still grains: they stay
    # obstacles and ownership rivals for their neighbours; only "not a grain" is dropped
    physical = [g for g in census if g.get("exclude_reason") != "not_a_grain"]
    started = time.time()
    if p.vmax_auto:
        from dataclasses import replace
        scale = growth_scale(renderer, meta, grains, p, log, physical)
        if scale is not None:
            vmax = float(np.clip(p.vmax_factor * scale, p.vmax_px, p.vmax_cap))
            log(f"growth scale {scale:.2f} px/bin (90th pct, median of isolated grains): front speed cap "
                f"{vmax:.1f} px/bin")
            p = replace(p, vmax_px=vmax)
    if p.reader not in ("change", "flood", "hybrid"):
        raise ValueError(f"unknown reader {p.reader!r}: change, flood or hybrid")
    prob = None
    if p.reader != "change":
        from dataclasses import replace
        from . import learned
        try:
            prob = Renderer(*stack.load(learned.prob_cache(cache_dir, p.model or learned.MODEL, log)))
        except ImportError:  # building the probability movie needs torch (pip install .[cnn])
            log("torch is not installed: reading every grain from change evidence (reader=change)")
            p = replace(p, reader="change")
    det = det_scale = None
    if p.tiptraj not in ("off", "flood", "all"):
        raise ValueError(f"unknown tiptraj {p.tiptraj!r}: off, flood or all")
    if p.tiptraj == "all" or (p.tiptraj == "flood" and p.reader != "change"):  # reader=change floods no grain
        from . import learned, tiptraj
        if prob is None:
            prob = Renderer(*stack.load(learned.prob_cache(cache_dir, p.model or learned.MODEL, log)))
        det = tiptraj.DetMaps(p.tiptraj_det or tiptraj.det_cache(cache_dir, p.tiptraj_model or tiptraj.MODEL, log))
        det_scale = det.scale()
    results = []
    for g in grains:
        if only and g["id"] not in only:
            continue
        others = [o for o in physical if o["id"] != g["id"]]
        res = analyze_grain(renderer, meta, g, others, p) if p.reader != "flood" else None
        crowded = res is not None and any(f.startswith(("touches:", "shared_change_split")) for f in res["flags"])
        noisy = res is not None and res.get("map_threshold", 0.0) > p.map_floor  # background change above the floor
        if p.reader == "flood" or (p.reader == "hybrid" and (crowded or noisy)):
            from . import learned
            fl = learned.read_grain(renderer, prob, meta, g, others, p)
            missed = (res is not None and res["status"] == "no_emergence_by_end"
                      and fl["status"] != "no_emergence_by_end" and fl.get("final_length_px", 0.0) >= p.hybrid_min_px)
            if res is not None and not crowded and p.hybrid_onset != "flood" and not (
                    p.hybrid_onset == "change_unless_missed" and missed):
                # a clean rim still gives the better onset: keep the change reader's germination call,
                # and the flood's lengths from that onset on
                fl = learned.with_onset(fl, res)
                if res.get("over_grain"):
                    fl = with_over_grain(fl, res, p)
            elif missed:
                fl["flags"].append("onset:flood_over_change")
            res = fl
        if det is not None and res.get("status") != "unobservable" and (p.tiptraj == "all" or "reader:flood" in res["flags"]):
            from . import tiptraj
            res = tiptraj.read(res, renderer, prob, meta, g, p, det, det_scale)
        if p.centre_route and prob is not None and len(res.get("path") or []) >= 2:
            from . import learned
            learned.centre_route(res, prob, meta, p)
        if p.tipdet_model and (p.tipdet_onset != "off" or p.tipdet_young):
            from . import tipdet  # after centring: its tips are peaks on the tube, not route points to move
            tipdet.read(res, renderer, meta, p)
        if prob is not None:
            from . import learned
            learned.drawn_check(res, prob, meta, p)
        cv2.imwrite(str(out_dir / "diagnostics" / f"{g['id']}.png"), _diagnostic(res, meta["frames_per_bin"]))
        res.pop("_diag", None)
        results.append(res)
        log(f"{g['id']}: {res['status']:<20} onset {str(res.get('onset_frame')):>6}  "
            f"final {res.get('final_length_px', 0):6.1f} px  {' '.join(res['flags'])}")
    focus = focus_changes(bins, meta)
    fpb = int(meta["frames_per_bin"])
    for res in results:  # an onset at a refocus says "visible by", not "emerged at"
        of = res.get("onset_frame")
        if of is not None and any(abs(of // fpb - f["bin"]) <= 3 for f in focus):
            res["flags"].append("onset_at_focus_change")
    not_grains: set[str] = set()
    if p.census_check:  # likely not grains: flagged, and left out of the population statistics below
        from . import census_check
        suspects = census_check.flagged(cache_dir, grains=[g for g in grains if any(r["id"] == g["id"] for r in results)])
        for res in results:
            if res["id"] in suspects:
                res["flags"].append(f"{census_check.FLAG}:{suspects[res['id']]:.2f}")
        not_grains = set(suspects)
    pred = {"schema": PRED_SCHEMA, "method": f"sparsetrack-v1 {__version__}", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cache": str(cache_dir), "grains_source": str(src), "params": asdict(p),
            "frames_per_bin": meta["frames_per_bin"], "movie": meta["movie"], "focus_changes": focus, "grains": results}
    (out_dir / "predictions.json").write_text(json.dumps(pred))
    from . import report
    um, spf = units or (None, None)  # (um per px, s per frame): physical units in the tables when both are known
    cal = bool(um and spf)
    with open(out_dir / "grains.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grain", "x", "y", "status", "onset_frame", "onset_after", "onset_by", "final_length_px",
                    "growth_px_per_bin", "model_confidence"]
                   + (["onset_min", "final_length_um", "growth_um_per_min"] if cal else []) + ["flags"])
        for r in results:
            iv = r.get("onset_interval") or [None, None]
            rate = report.growth_rate(r["length"]["frames"], r["length"]["px"]) if r["status"].startswith("emerged") else None
            conf = report.grain_confidence(r, meta["frames_per_bin"])
            row = [r["id"], r["x"], r["y"], r["status"], r.get("onset_frame"), iv[0], iv[1], r.get("final_length_px", 0),
                   "" if rate is None else round(rate * meta["frames_per_bin"], 3), "" if conf is None else round(conf, 3)]
            if cal:
                row += ["" if r.get("onset_frame") is None else round(r["onset_frame"] * spf / 60.0, 2),
                        round((r.get("final_length_px") or 0) * um, 2),
                        "" if rate is None else round(rate * um * 60.0 / spf, 4)]
            w.writerow(row + [";".join(r["flags"])])
    with open(out_dir / "growth.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grain", "frame", "length_px", "tip_x", "tip_y"] + (["minutes", "length_um"] if cal else []))
        for r in results:
            tips = (r.get("tip") or {}).get("xy") or [[None, None]] * len(r["length"]["frames"])
            for f, L, (tx, ty) in zip(r["length"]["frames"], r["length"]["px"], tips):
                w.writerow([r["id"], f, L, tx, ty] + ([round(f * spf / 60.0, 2), round(L * um, 2)] if cal else []))
    isolated = [r["id"] for r in results if next((g for g in grains if g["id"] == r["id"]), {}).get("isolated", True)
                and r["id"] not in not_grains]
    pop = report.write_population(pred, out_dir, set(isolated))
    report.write_growth_curves(pred, out_dir, isolated)
    report.write_gallery(pred, out_dir, set(isolated), population=pop, units=units)
    for w in report.movie_warnings(pred, set(isolated)):
        log(f"WARNING: {w}")
    if pop and pop.get("t50_interval"):
        log(f"population ({pop['n']} isolated grains): half germinated by frame {pop['t50_interval'][1]:.0f}")
    if video:
        report.write_video(renderer, meta, pred, out_dir / "field_overlay.mp4", ids=set(isolated))
        log(f"video -> {out_dir / 'field_overlay.mp4'}")
    log(f"{len(results)} grains in {time.time() - started:.0f} s -> {out_dir}")
    return pred
