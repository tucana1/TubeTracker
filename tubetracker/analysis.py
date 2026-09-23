"""Image processing, detection linking, event scoring, and result export."""

import csv
from heapq import heappop, heappush
import math
from random import randint
import statistics as stat

import cv2 as cv
from laptrack import LapTrack
import numpy as np
import pandas as pd

from tubetracker_resources import load_tip_templates

from .models import Point, ROI, Track

# Fixed hit-or-miss definitions for skeleton endpoints; these are not learned weights.
TIP_ENDPOINT_KERNELS = (
	np.array(([0, 0, 0], [-1, 1, -1], [-1, -1, -1]), dtype=np.int8),
	np.array(([0, -1, -1], [0, 1, -1], [0, -1, -1]), dtype=np.int8),
	np.array(([-1, -1, 0], [-1, 1, 0], [-1, -1, 0]), dtype=np.int8),
	np.array(([-1, -1, -1], [-1, 1, -1], [0, 0, 0]), dtype=np.int8),
)

def bbox_iou_distance(left, right):
	"""Return one minus intersection-over-union for two bounding boxes."""
	x_left = max(left[0], right[0])
	y_top = max(left[1], right[1])
	x_right = min(left[2], right[2])
	y_bottom = min(left[3], right[3])
	intersection = max(0.0, x_right-x_left)*max(0.0, y_bottom-y_top)
	left_area = max(0.0, left[2]-left[0])*max(0.0, left[3]-left[1])
	right_area = max(0.0, right[2]-right[0])*max(0.0, right[3]-right[1])
	union = left_area + right_area - intersection
	return 1.0 - intersection/union if union > 0 else 1.0

def bbox_center_squared_distance(left, right):
	"""Return squared centroid distance for two bounding boxes."""
	dx = (left[0]+left[2]-right[0]-right[2])/2
	dy = (left[1]+left[3]-right[1]-right[3])/2
	return dx*dx + dy*dy

class Detections:
	"""Preprocess microscopy frames and retain foreground component detections."""

	def __init__(self, img_list = None, bg_threshold = 10, blur_radius = 1):
		"""Initialize frame storage and segment the supplied image sequence."""
		if img_list != None:
			self.img_list_input = img_list
			self.img_list_gray = []
			self.img_list_gray_noiseless = []
			self.rois = []
			self.img_list_length = len(img_list)
			self.h, self.w, l = img_list[0].shape
			self.bg_threshold = bg_threshold
			self.blur_radius = blur_radius
			self.segment_inputs()
			self.remove_bg_and_locate_rois()

	@staticmethod
	def color_palette():
		"""Build the grayscale-to-BGR lookup table used by heatmap displays."""
		v = 127.5
		a = -230/(v*v)
		rows = []
		for x in range(256):
			b = int(0.35*a*x*x + 230)
			g = int(a*(x-127.5)*(x-127.5) + 255)
			r = int(0.35*a*(x-300)*(x-300) + 230)
			if r < 1:
				r = 1
			if r > 255:
				r = 255
			if b < 1:
				b = 1
			if b > 255:
				b = 255
			if g < 1:
				g = 1
			if g > 255:
				g = 255
			rows.append([b, g, r])
		return np.asarray(rows, dtype=np.uint8)

	def false_color(self, gray, min_pxl = 1):
		"""Map grayscale intensities to the application's false-color palette."""
		mask = self.color_palette()[gray]
		mask[gray < min_pxl] = 0
		return mask

	def get_noiseless_frame(self, frame, bnr = False, gray = False, fill_holes = False):
		"""Return a processed frame in binary, grayscale, or BGR form."""
		if self.img_list_length > frame:
			if bnr == True:
				x = self.img_list_gray_noiseless[frame].copy()
				if fill_holes:
					im = np.zeros_like(cv.cvtColor(x, cv.COLOR_GRAY2BGR))
					cnts, _ = cv.findContours(x, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
					for c in cnts:
						cv.drawContours(im, [c], 0, (255, 255, 255), -1)
					x = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
				else:
					x[x>0] = 255

				if gray == True:
					return x
				else:
					return cv.cvtColor(x, cv.COLOR_GRAY2BGR)
			else:
				if gray == True:
					return self.img_list_gray_noiseless[frame]
				else:
					return cv.cvtColor(self.img_list_gray_noiseless[frame], cv.COLOR_GRAY2BGR)

	def get_raw_colored_frame(self, frame):
		"""Return a false-colored processed frame for display."""
		if self.img_list_length > frame:
			return self.false_color(gray = self.img_list_gray_noiseless[frame])

	def remove_bg_and_locate_rois(self, bg_threshold = None, frame = None, filter_radius = None, sigma = None, blur_radius = None):
		"""Threshold foreground and record component bounding boxes by frame."""
		if self.img_list_length > 0:
			if bg_threshold == None:
				bg_threshold = self.bg_threshold
			ev_knl = False
			if blur_radius == None:
				if self.blur_radius % 2 == 0:
					ev_knl = True
					knl = (self.blur_radius-1, self.blur_radius-1)
					knl2 = (self.blur_radius+1, self.blur_radius+1)
				else:
					knl = (self.blur_radius, self.blur_radius)
			else:
				if blur_radius % 2 == 0:
					knl = (blur_radius-1, blur_radius-1)
					knl2 = (blur_radius+1, blur_radius+1)
					ev_knl = True
				else:
					knl = (blur_radius, blur_radius)
			if frame == None:
				self.img_list_gray_noiseless = []
				self.rois = []
				for i in range(self.img_list_length):
					if ev_knl == True:
						gray = cv.addWeighted(cv.GaussianBlur(self.img_list_gray[i].copy(), knl, 0), 0.5, cv.GaussianBlur(self.img_list_gray[i].copy(), knl2, 0), 0.5, 0.0)
					else:
						gray = cv.GaussianBlur(self.img_list_gray[i].copy(), knl, 0)
					gray[gray < bg_threshold] = 0
					cnts, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
					rois_in_i = []
					for c in cnts:
						(x, y, w, h) = cv.boundingRect(c)
						rois_in_i.append(ROI(x_l = x, y_t = y, x_r = x+w, y_b = y+h, frame = i))
					self.rois.append(rois_in_i)
					self.img_list_gray_noiseless.append(gray)
			else:
				if ev_knl == True:
					gray = cv.addWeighted(cv.GaussianBlur(self.img_list_gray[frame].copy(), knl, 0), 0.5, cv.GaussianBlur(self.img_list_gray[frame].copy(), knl2, 0), 0.5, 0.0)
				else:
					gray = cv.GaussianBlur(self.img_list_gray[frame].copy(), knl, 0)
				gray[gray < bg_threshold] = 0
				cnts, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
				rois_in_i = []
				for c in cnts:
					(x, y, w, h) = cv.boundingRect(c)
					rois_in_i.append(ROI(x_l = x, y_t = y, x_r = x+w, y_b = y+h, frame = frame))
				self.rois[frame] = rois_in_i
				self.img_list_gray_noiseless[frame] = gray

	def segment_inputs(self):
		"""Convert source images into morphology-enhanced edge frames."""
		self.img_list_gray = []
		self.kernel = np.ones((3, 3), np.uint8)
		img_list = self.img_list_input.copy()
		k = 1
		for img in img_list:
			k+=1
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			gray = cv.morphologyEx(cv.morphologyEx(cv.add(cv.morphologyEx(cv.morphologyEx(cv.add(cv.convertScaleAbs(cv.Scharr(gray,cv.CV_16S,1,0)), cv.convertScaleAbs(cv.Scharr(gray,cv.CV_16S,0,1))), cv.MORPH_CLOSE, self.kernel), cv.MORPH_CLOSE, self.kernel), cv.morphologyEx(cv.morphologyEx(cv.add(cv.convertScaleAbs(cv.Sobel(gray, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)), cv.convertScaleAbs(cv.Sobel(gray, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT))), cv.MORPH_CLOSE, self.kernel), cv.MORPH_CLOSE, self.kernel)) , cv.MORPH_OPEN, self.kernel), cv.MORPH_CLOSE, self.kernel)
			self.img_list_gray.append(gray)

class Tracker:
	"""Coordinate pollen-grain detection, tip tracking, events, and exports."""

	def __init__(self, screen_size = (1000,800)):
		"""Initialize analysis parameters, counters, and empty result collections."""
		self.pxl_dis = 1.00
		self.dis_unit = "um"
		self.gv3 = []
		self.tip_gap_closing = 10
		self.gv5 = 0.05
		self.grain_det_start = 0
		self.grain_det_stop = 9
		self.gv8 = 1
		self.img_rp = Point(x = 1, y = 1)
		self.img_ratio = 1
		self.max_grain_radius = 20*self.img_ratio
		self.max_grain_area = int(math.pi*self.max_grain_radius*self.max_grain_radius)
		self.min_grain_radius = 10*self.img_ratio
		self.min_grain_area = int(math.pi*self.min_grain_radius*self.min_grain_radius)
		self.screen_size = screen_size
		self.gv11 = []
		self.gv12 = None
		self.gv13 = None
		self.file_names = None
		self.tip_max_step = 0.1
		self.gv16 = 110
		self.min_tip_per_trk = 10
		self.gv18 = 50
		self.all_detections = []
		self.gv20 = (10, 10)
		self.gv21 = 10000000
		self.gv22 = 4
		self.time_p_frame = 60
		self.time_unit = "sec"
		self.tip_box_h = 25
		self.tip_box_w = 25
		self.tip_area = int((self.tip_box_h + self.tip_box_w)*150*self.img_ratio/1250)
		self.assist_start = 10
		self.assist_lookback = 9
		self.gv28 = []
		self.gv29 = []
		self.gv30 = []
		self.gv31 = 1
		self.gv32 = []
		self.gv31p = 1
		self.gv32p = []
		self.valid_tracks = []
		self.other_tracks = []
		self.valid_grains = []
		self.valid_tips = []
		self.input_imgages = []
		self.grain_tresh = 20
		self.pot_grains = []
		self.bg_threshold = 55
		self.tip_template_det_percent = 0.03
		self.tip_det_threshold_percent = 0.80
		self.tip_template_use_fr = 1.0
		self.min_tip_side = 18
		self.filter_radius = 10
		self.filled_in_tips = []
		self.aceptance_ratio = 0.5
		self.ger_confirm_frames = 8
		self.burst_candidates = []
		self.enable_burst_candidates = False
		self.burst_candidates_evaluated = False
		self.burst_score_threshold = 0.5
		self.burst_end_margin = 3
		self.tube_background_sigma = 15
		self.tube_min_component_area = 12
		self.tube_length_search_radius = 180
		self.tube_connection_gap = 15
		self._tube_skeleton_cache = {}
		self._tube_length_cache = {}
		self.burst_area_spike_ratio = 1.6
		self.tracking_engine = "laptrack"

	def link_detections(self, detections_by_frame, max_distance = None, min_overlap = None, gap_frames = 0, gap_distance = None):
		"""Link frame-indexed ROIs with LapTrack and return observed ROI groups."""
		records = []
		sources = []
		for frame, detections in enumerate(detections_by_frame):
			for detection in detections:
				source_index = len(sources)
				sources.append(detection)
				records.append({
					"source_index": source_index,
					"frame": frame,
					"x": detection.gv3.x,
					"y": detection.gv3.y,
					"x_l": detection.gv1.x,
					"y_t": detection.gv1.y,
					"x_r": detection.gv2.x,
					"y_b": detection.gv2.y,
				})
		if not records:
			return []
		if max_distance is None and min_overlap is None:
			raise ValueError("max_distance or min_overlap must be provided")
		if max_distance is not None and max_distance <= 0:
			raise ValueError("max_distance must be greater than zero")
		if min_overlap is not None and not 0 < min_overlap < 1:
			raise ValueError("min_overlap must be in (0, 1)")
		if gap_distance is None:
			gap_distance = max_distance
		if gap_frames > 0 and (gap_distance is None or gap_distance <= 0):
			raise ValueError("gap_distance must be positive when closing gaps")

		if min_overlap is None:
			metric = "sqeuclidean"
			cutoff = float(max_distance)**2
			coordinate_cols = ["x", "y"]
			gap_metric = "sqeuclidean"
		else:
			metric = bbox_iou_distance
			cutoff = 1.0-float(min_overlap)
			coordinate_cols = ["x_l", "y_t", "x_r", "y_b"]
			gap_metric = bbox_center_squared_distance

		linker = LapTrack(
			metric=metric,
			cutoff=cutoff,
			gap_closing_metric=gap_metric,
			gap_closing_cutoff=(
				float(gap_distance)**2 if gap_frames > 0 else False
			),
			gap_closing_max_frame_count=max(1, int(gap_frames)),
			splitting_cutoff=False,
			merging_cutoff=False,
		)
		linked, _, _ = linker.predict_dataframe(
			pd.DataFrame.from_records(records),
			coordinate_cols=coordinate_cols,
			frame_col="frame",
			only_coordinate_cols=False,
		)

		groups = []
		for _, rows in linked.sort_values(["track_id", "frame"]).groupby(
			"track_id", sort=True
		):
			group = []
			for source_index in rows["source_index"]:
				source = sources[int(source_index)]
				group.append(ROI(
					x_l=source.gv1.x,
					y_t=source.gv1.y,
					x_r=source.gv2.x,
					y_b=source.gv2.y,
					frame=source.gv6,
					detection_method=source.detection_method,
					is_tip=source.is_tip,
				))
			groups.append(group)
		return groups

	def stitch_track_fragments(self, groups, max_gap_frames, prediction_tolerance):
		"""Join fragments when recent velocity predicts a compatible future start."""
		fragments = [sorted(group, key=lambda roi: roi.gv6) for group in groups if group]
		while True:
			candidates = []
			for source_index, source in enumerate(fragments):
				if len(source) < 2:
					continue
				previous, current = source[-2], source[-1]
				frame_delta = current.gv6 - previous.gv6
				if frame_delta <= 0:
					continue
				velocity_x = (current.gv3.x - previous.gv3.x)/frame_delta
				velocity_y = (current.gv3.y - previous.gv3.y)/frame_delta
				for target_index, target in enumerate(fragments):
					if source_index == target_index:
						continue
					gap = target[0].gv6 - current.gv6
					if gap <= 1 or gap - 1 > max_gap_frames:
						continue
					predicted_x = current.gv3.x + velocity_x*gap
					predicted_y = current.gv3.y + velocity_y*gap
					prediction_error = math.hypot(
						target[0].gv3.x - predicted_x,
						target[0].gv3.y - predicted_y,
					)
					if prediction_error > prediction_tolerance:
						continue
					connector_x = target[0].gv3.x - current.gv3.x
					connector_y = target[0].gv3.y - current.gv3.y
					velocity_norm = math.hypot(velocity_x, velocity_y)
					connector_norm = math.hypot(connector_x, connector_y)
					if velocity_norm > 0 and connector_norm > 0:
						direction_cosine = (
							velocity_x*connector_x + velocity_y*connector_y
						)/(velocity_norm*connector_norm)
						if direction_cosine < 0.5:
							continue
					candidates.append((prediction_error, source_index, target_index))
			if not candidates:
				break
			_, source_index, target_index = min(candidates)
			target = fragments[target_index]
			target[0].detection_method = (
				target[0].detection_method + "+trajectory_assisted"
			)
			fragments[source_index] = fragments[source_index] + target
			fragments.pop(target_index)
		return fragments

	def draw_results(self , tracks = True):
		"""Render track paths or grain states onto copies of source frames."""
		img_list = []
		lwd = 1
		for img in self.all_detections.img_list_input:
			img_list.append(img.copy())
		if tracks == True:
			if len(self.valid_tracks) > 0:
				for t in self.valid_tracks:
					for i in range(t.gv1[0].gv6, len(img_list)):
						cv.putText(img_list[i], str(t.id), t.gv1[0].gv1.coor, cv.FONT_HERSHEY_SIMPLEX, 0.65, t.color, lwd, cv.LINE_AA)
						k = 0
						while(True):
							k += 1
							if k >= t.gv2 or t.gv1[k].gv6 > i:
								break
							cv.line(img_list[i], t.gv1[k-1].set_pos.coor, t.gv1[k].set_pos.coor, t.color, lwd)
		else:
			if len(self.valid_grains) > 0:
				k=-1
				for img in img_list:
					k+=1
					for g in self.valid_grains:
						for r in g.gv1:
							if r.gv6 == k:
								cv.putText(img, str(g.id), r.gv1.coor, cv.FONT_HERSHEY_SIMPLEX, 0.65, r.grain_color(paint = False), lwd, cv.LINE_AA)
								cv.rectangle(img, r.gv1.coor, r.gv2.coor, r.grain_color(paint = False), lwd)
		return img_list

	def exclude_grains(self):
		"""Remove grains whose identifiers are listed in the exclusion state."""
		ids = self.gv3
		if len(self.valid_grains) > 0 and len(self.gv3) > 0:
			for grain in self.valid_grains:
				if grain.id in ids:
					self.release_id(grain.id, what = "g")
					self.valid_grains.remove(grain)

	def random_color(self, genre = None):
		"""Generate a random BGR display color, optionally restricted to light tones."""
		if genre == None:
			return (randint(0, 255),randint(0, 255),randint(0, 255))
		else:
			return (randint(100, 255),randint(100, 255),randint(100, 255))

	def allocate_id(self, what = "t"):
		"""Allocate or recycle an identifier for a grain, track, or tip."""
		ID = ''
		if what == "g":
			if len(self.gv11) > 0:
				ID = self.gv11[0]
				if len(self.gv11) == 1:
					self.gv11 = []
				else:
					self.gv11.remove(ID)
			else:
				ID = self.gv8
				self.gv8 += 1
		if what == "t":
			if len(self.gv32) > 0:
				ID = self.gv32[0]
				if len(self.gv32) == 1:
					self.gv32 = []
				else:
					self.gv32.remove(ID)
			else:
				ID = self.gv31
				self.gv31 += 1
		if what == "tp":
			if len(self.gv32p) > 0:
				ID = self.gv32p[0]
				if len(self.gv32p) == 1:
					self.gv32p = []
				else:
					self.gv32p.remove(ID)
			else:
				ID = self.gv31p
				self.gv31p += 1
		return str(ID)

	def release_id(self, ID, what = "t"):
		"""Return an identifier to the appropriate reusable-ID pool."""
		ID = int(ID)
		if what == "g" and ID not in self.gv11:
			self.gv11.append(ID)
			self.gv11.sort()
		if what == "t" and ID not in self.gv32:
			self.gv32.append(ID)
			self.gv32.sort()
		if what == "tp" and ID not in self.gv32p:
			self.gv32p.append(ID)
			self.gv32p.sort()

	def write_video(self, filename, img_list, img_per_sec = 25):
		"""Encode an image sequence as an MJPG AVI video."""
		try:
			h, w, l = img_list[0].shape
		except:
			h, w = img_list[0].shape
		writer = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*'MJPG'), img_per_sec, (w, h))
		for img in img_list:
			writer.write(img)
		writer.release()
		writer = None

	def find_grains(self):
		"""Detect circular pollen grains and link observations across frames."""
		self.clear_tube_length_cache()
		size_cutoff = 1.3
		self.valid_grains = []
		self.gv3 = []
		self.gv11 = []
		self.gv8 = 1
		overlap = 0.3
		if self.grain_det_start >= self.all_detections.img_list_length:
			self.grain_det_start = 0
		if self.grain_det_stop <= self.grain_det_start:
			self.grain_det_stop = self.grain_det_start + 5
		k = self.grain_det_start - 1
		grain_detections = [
			[] for _ in range(self.all_detections.img_list_length)
		]
		all_grains = []
		voys = 0
		while(True):
			k += 1
			if k >= self.all_detections.img_list_length:
				break
			gr_tr = int(self.grain_tresh)
			circles = cv.HoughCircles(self.all_detections.img_list_gray_noiseless[k], cv.HOUGH_GRADIENT, 1, int(0.75*(self.max_grain_radius+self.min_grain_radius)), param1=210, param2 = gr_tr, minRadius = int(self.min_grain_radius), maxRadius = int(self.max_grain_radius))
			if circles is None:
				continue
			grains = np.around(circles).astype(np.int32)
			for i in grains[0, :]:
				x, y, radius = (int(value) for value in i)
				grain = ROI(x_l = x - radius, y_t = y - radius, x_r = x + radius, y_b = y + radius, frame = k, group = str(voys))
				voys += 1
				if grain.overlaps_any(self.all_detections.rois[k], overlap = overlap):
					if k <= self.grain_det_stop:
						if k > self.grain_det_start:
							if grain.overlaps_any(all_grains, overlap = overlap):
								p_g = grain.best_overlap(all_grains)
								kk = -1
								for g in all_grains:
									kk+=1
									if g.gv6 == p_g.gv6:
										if g.group == p_g.group:
											all_grains.pop(kk)
											break
						grain_detections[k].append(grain)
						all_grains.append(grain)
					else:
						if grain.overlaps_any(all_grains, overlap = overlap):
							grain_detections[k].append(grain)
							all_grains.append(grain)
							p_g = grain.best_overlap(all_grains)
							kk = -1
							for g in all_grains:
								kk+=1
								if g.gv6 == p_g.gv6:
									if g.group == p_g.group:
										break
		grain_tracks = self.link_detections(
			grain_detections,
			min_overlap=0.2,
			gap_frames=5,
			gap_distance=max(1.0, float(self.max_grain_radius)),
		)
		self.gv8 = 1
		for cur_g in grain_tracks:
			frame1 = self.grain_det_stop + 1
			for roi in cur_g:
				if roi.gv6 < frame1:
					frame1 = roi.gv6
			observed_span = cur_g[-1].gv6 - cur_g[0].gv6 if cur_g else 0
			if frame1 <= self.grain_det_stop and observed_span >= self.grain_det_stop - self.grain_det_start:
				grain = Track(boxes = cur_g, color = self.random_color(), ID = self.allocate_id("g"))
				grain.fill_missing_frames(up_to_frame = self.all_detections.img_list_length)
				self.valid_grains.append(grain)

	def find_tips_se(self):
		"""Detect candidate tube tips as endpoints of foreground skeletons."""
		overlap = 0.5
		frame = -1
		valid_tips = []
		knl_r = 2
		while(True):
			t1 = []
			t2 = []
			frame += 1
			if frame >= self.all_detections.img_list_length:
				break
			img = self.all_detections.get_noiseless_frame(frame = frame, gray = True, bnr = True, fill_holes = True)
			rois = []
			cnts, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			for c in cnts:
				(x, y, w, h) = cv.boundingRect(c)
				rois.append(ROI(x_l = x, y_t = y, x_r = x+w, y_b = y+h, frame = frame))
			sk = cv.ximgproc.thinning(img, None, 1)
			H,W = img.shape
			out = cv.morphologyEx(sk, cv.MORPH_HITMISS, TIP_ENDPOINT_KERNELS[0])
			for kernel in TIP_ENDPOINT_KERNELS[1:]:
				out = out + cv.morphologyEx(sk, cv.MORPH_HITMISS, kernel)
			tips = np.argwhere(out == 255)
			try:
				tips_in_k = self.valid_tips[frame]
			except:
				tips_in_k = []
			for pt in tips:
				xl = int(pt[1]-self.min_tip_side/2)
				xr = int(pt[1]+self.min_tip_side/2)
				yt = int(pt[0]-self.min_tip_side/2)
				yb = int(pt[0]+self.min_tip_side/2)
				if xl >=0 and xr <= W and yt >= 0 and yb <= H:
					im = img[yt:yb, xl:xr]
					if len(np.argwhere(im > 0)) <= 0.95*self.min_tip_side*self.min_tip_side:
						tip = ROI(x_l = xl, y_t = yt, x_r = xr, y_b = yb, frame = frame, is_tip = True)
						tip_obj = tip.best_overlap(rois)
						if tip_obj != None:
							if tip_obj.gv4 > tip.gv4:
								M1, M2, m1, m2 = tip.get_axes_int_points(img = img.copy())
								if M1 != None and M2 != None:
									if M1.y < H and M1.x < W and M2.y < H and M2.x < W:
										if img[M1.y, M1.x] > 0 and img[M2.y, M2.x] > 0:
											pass
										elif img[M1.y, M1.x] < 200 and img[M2.y, M2.x] < 200:
											pass
										else:
											if not tip.overlaps_any(tips_in_k, overlap = overlap):
												tips_in_k.append(tip)
			print("Frame: " + str(frame) + "; tips: " + str(len(tips_in_k)))
			valid_tips.append(tips_in_k)
		self.valid_tips = valid_tips

	def find_tips_tm(self):
		"""Detect tube tips with the versioned grayscale template library."""
		overlap = 0.1
		templates = self.tip_templates()
		area = []
		if len(self.valid_tips) > 0:
			for tips_in_k in self.valid_tips:
				for tip in tips_in_k:
					area.append(tip.gv4)
		frame = -1
		valid_tips = []
		while(True):
			frame += 1
			if frame >= self.all_detections.img_list_length:
				break
			if len(self.valid_tips) > frame:
				c_tips = self.valid_tips[frame].copy()
			else:
				c_tips = []
			tips_in_k = []
			detections = self.match_templates(img = self.all_detections.get_noiseless_frame(frame, bnr = True), templates = templates, frame = frame, threshold1 = self.tip_det_threshold_percent)
			k=0
			for tip in detections:
				if tip.w >= self.min_tip_side and tip.h >= self.min_tip_side :
					tip.is_tip = True
					if len(c_tips) >0:
						if not tip.overlaps_any(c_tips, overlap = overlap):
							tips_in_k.append(tip)
							area.append(tip.gv4)
							k+=1
					else:
						tips_in_k.append(tip)
						area.append(tip.gv4)
						k+=1
			valid_tips.append(tips_in_k)
			print("Frame: " + str(frame) + "; tips: " + str(k))
		if len(area) > 0:
			area = stat.median(area)
			side_ratio = 0.5
			area_ratio = 2
			k=-1
			for tips_in_k in valid_tips:
				k+=1
				if len(self.valid_tips) > k:
					if len(tips_in_k) > 0:
						for tip in tips_in_k:
							if tip.side_ratio <= side_ratio and tip.gv4/area < area_ratio:
								self.valid_tips[k].append(tip)
				else:
					self.valid_tips.append(tips_in_k)

	def match_templates(self, img, templates, frame = -1, threshold1 = 0.75):
		"""Match tip templates in one frame and filter geometric false positives."""
		tmp = []
		knl_r = 2
		flt1 = 3.5/5
		flt2 = 3/5
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		H, W = gray.shape
		for template in templates:
			h, w = template.shape
			mask = np.zeros(img.shape, dtype = np.uint8)
			rois = cv.matchTemplate(cv.cvtColor(img, cv.COLOR_BGR2GRAY), template, cv.TM_CCOEFF_NORMED)
			for pt in zip(*np.where(rois >= threshold1)[::-1]):
				tip = ROI(x_l = pt[0], y_t = pt[1], x_r = pt[0]+w, y_b = pt[1]+h, frame = frame, img = gray.copy())
				int_pts = [[[tip.ma_ax_int_p1, tip.ma_ax_int_p2], 0], [[tip.mi_ax_int_p1, tip.mi_ax_int_p2], 0]]
				ax = []
				for axis in int_pts:
					for p in axis[0]:
						ki =0
						kt = 0
						for y in range(int(p.y - knl_r), int(p.y + knl_r + 1)):
							if y - knl_r > 0:
								ytt = y - knl_r
							else:
								ytt = 0
							if y + knl_r < H:
								ybb = y + knl_r
							else:
								ybb = H-1
							dy = abs(ytt - ybb)
							for x in range(int(p.x - knl_r),int(p.x + knl_r + 1)):
								kt+=1
								if x - knl_r > 0:
									xll = x - knl_r
								else:
									xll = 0
								if x + knl_r < W:
									xrr = x + knl_r
								else:
									xrr = W-1
								dx = abs(xll - xrr)
								g = np.resize(gray[ytt:ybb, xll:xrr].copy(), (1, int(dy*dx)))[0]
								if len(g)>0:
									if len(g[g<=int(self.bg_threshold)])/len(g) >= flt1:
										ki+=1
						if ki/kt >= flt2:
							axis[1] +=1
					ax.append(axis[1])
				if 1 in ax:
					mask[pt[1]:(pt[1] + h), pt[0]:(pt[0] + w)] = img[pt[1]:(pt[1] + h), pt[0]:(pt[0] + w)]
			tmp.append(mask)
		detections = []
		cnts, _ = cv.findContours(cv.inRange(cv.cvtColor(np.mean(tmp, axis = 0).astype(np.uint8), cv.COLOR_BGR2GRAY), np.array([1]), np.array([255])), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		if len(cnts) > 0:
			for c in cnts:
				(x, y, w, h) = cv.boundingRect(c)
				xr = x+w
				if xr > W:
					xr = W - 1
				yb = y+h
				if yb > H:
					yb = H - 1
				detections.append(ROI(x_l = x, y_t = y, x_r = xr, y_b = yb, frame = frame))
		return detections

	def save_results(self, save_dir = ""):
		"""Write annotated videos, trajectories, events, and summary CSV files."""
		if len(self.valid_tracks) > 0:
			for track in self.valid_tracks:
				track.calculate_metrics(coef = self.img_rp, disp = self.pxl_dis, disp_u = self.dis_unit, num_frames = len(self.file_names), time_p_frame = self.time_p_frame)
			name = save_dir + "tip.tracks.avi"
			self.write_video(filename = name, img_list = self.draw_results())
			tracks = []
			k = 0
			while(True):
				k += 1
				if len(tracks) >= len(self.valid_tracks):
					break
				for track in self.valid_tracks:
					if int(track.id) == k:
						tracks.append(track)
						break
			num_rows = []
			for track in self.valid_tracks:
				num_rows.append(track.gv2 + 2)
			self.gv38 = [["track_id", "frame", "centroid_x", "centroid_y", "time (" + self.time_unit + ")", "cumulative tip movement (pixel)", "cumulative tip movement (" + self.dis_unit + ")", "detection_method"]]
			for track in tracks:
				for row in track.gv3:
					self.gv38.append(row)
			num_rows = len(self.file_names) + 1
			self.gv39 = []
			self.gv40 = []
			for row in range(num_rows):
				if row > 0:
					sync_row = [int((row - 1)*self.time_p_frame)]
					unsync_row = [int((row - 1)*self.time_p_frame)]
				else:
					sync_row = ["time (" + self.time_unit + ")"]
					unsync_row = ["time (" + self.time_unit + ")"]
				for track in tracks:
					sync_row.append(track.gv4[row])
					unsync_row.append(track.gv5[row])
				self.gv39.append(sync_row)
				self.gv40.append(unsync_row)
			all_data = [[save_dir + "tracks.raw.data.csv", self.gv38], [save_dir + "tracks.synchronized.csv", self.gv39], [save_dir + "tracks.unsynchronized.csv", self.gv40]]
			for data in all_data:
				with open(data[0], 'w', newline='') as f:
					csv.writer(f).writerows(data[1])
		if len(self.valid_grains) > 0:
			name = save_dir + "survival.avi"
			self.write_video(filename = name, img_list = self.draw_results(tracks = False))
			rows = [["grain_id", "detection_time ("+self.time_unit+")", "germination_time ("+self.time_unit+")", "burst_time ("+self.time_unit+")", "detection_frame", "germination_frame", "burst_frame", "germination_p_value", "centroid_x", "centroid_y", "area_at_detection", "detection_method", "germination_method", "burst_method"]]
			ger_frames = []
			burst_frames = []
			for grain in self.valid_grains:
				rows.append(grain.survival_values(tbf = self.time_p_frame))
				if grain.ger_frame < 0:
					ger_frames.append(self.all_detections.img_list_length+10)
				else:
					ger_frames.append(grain.ger_frame)
				if grain.burst_frame < 0:
					burst_frames.append(self.all_detections.img_list_length+10)
				else:
					burst_frames.append(grain.burst_frame)
			name = save_dir + "survival.raw.data.csv"
			with open(name, 'w', newline='') as f:
				csv.writer(f).writerows(rows)
			name = save_dir + "survival.curves.csv"

			rows = [["frame", "time ("+self.time_unit+")", "num_germinated", "fraction_germinated", "num_bursted", "fraction_bursted"]]
			frame = -1
			while(True):
				frame+=1
				if frame >= self.all_detections.img_list_length:
					break
				g = 0
				for i in ger_frames:
					if i <= frame:
						g+=1
				b = 0
				for i in burst_frames:
					if i <= frame:
						b+=1
				rows.append([frame, frame*self.time_p_frame, g, g/len(ger_frames), b, b/len(burst_frames)])
			with open(name, 'w', newline='') as f:
				csv.writer(f).writerows(rows)
			if self.enable_burst_candidates and not self.burst_candidates_evaluated:
				self.find_burst_candidates()
			if self.enable_burst_candidates and len(self.burst_candidates) > 0:
				name = save_dir + "burst.candidates.csv"
				rows = [["grain_id", "frame", "time ("+self.time_unit+")", "confidence", "method", "associated_track_id", "evidence"]]
				for c in self.burst_candidates:
					rows.append([c["grain_id"], c["frame"], c["time"], c["confidence"], c["method"], c["track_id"], c["reason"]])
				with open(name, 'w', newline='') as f:
					csv.writer(f).writerows(rows)

	def coordinate_rows(self):
		"""Return a tidy coordinate/time table for all accepted tip tracks."""
		header = [
			"grain_id",
			"track_id",
			"frame",
			"time_" + self.time_unit,
			"centroid_x_original_px",
			"centroid_y_original_px",
			"cumulative_tip_movement_px",
			"cumulative_tip_movement_" + self.dis_unit,
			"tip_growth_rate_" + self.dis_unit + "_per_" + self.time_unit,
			"grain_boundary_to_tracked_tip_straight_distance_px",
			"grain_boundary_to_tracked_tip_straight_distance_" + self.dis_unit,
			"trajectory_based_tube_length_px",
			"trajectory_based_tube_length_" + self.dis_unit,
			"trajectory_length_qc_status",
			"image_centerline_length_px",
			"image_centerline_length_" + self.dis_unit,
			"image_centerline_qc_status",
			"tip_position_status",
			"detection_method",
		]
		if not self.file_names:
			return [header]
		track_to_grain = {}
		for grain in self.valid_grains:
			track = self.associate_grain_to_track(grain)
			if track is not None and track.id not in track_to_grain:
				track_to_grain[track.id] = grain.id
		rows = [header]
		for track in self.valid_tracks:
			grain_id = track_to_grain.get(track.id, "")
			grain = next(
				(grain for grain in self.valid_grains if grain.id == grain_id),
				None,
			)
			track.calculate_metrics(
				coef=self.img_rp,
				disp=self.pxl_dis,
				disp_u=self.dis_unit,
				num_frames=len(self.file_names),
				time_p_frame=self.time_p_frame,
			)
			initial_length_px = 0.0
			if grain is not None:
				first_tip = track.gv1[0]
				first_grain = grain.roi_closest_to(first_tip.gv6)
				center_distance = grain.centroid_distance(
					first_grain, first_tip, img_ratio=self.img_rp
				)
				grain_radius = max(
					first_grain.w/self.img_rp.x,
					first_grain.h/self.img_rp.y,
				)/2
				initial_length_px = max(0.0, center_distance - grain_radius)
			previous_time = None
			previous_length = None
			for point in track.gv3:
				tip_roi = track.roi_closest_to(point[1])
				growth_rate = ""
				if previous_time is not None and point[4] > previous_time:
					growth_rate = (point[6] - previous_length)/(point[4] - previous_time)
				length_result = self.estimate_tube_length(grain, point[1]) if grain else None
				straight_px = ""
				trajectory_px = ""
				trajectory_status = "unassociated_track"
				if grain is not None:
					grain_roi = grain.roi_closest_to(point[1])
					center_distance = grain.centroid_distance(
						grain_roi, tip_roi, img_ratio=self.img_rp
					)
					grain_radius = max(
						grain_roi.w/self.img_rp.x,
						grain_roi.h/self.img_rp.y,
					)/2
					straight_px = max(0.0, center_distance - grain_radius)
					trajectory_px = initial_length_px + point[5]
					trajectory_status = (
						"interpolated_tip_position"
						if tip_roi.is_filled_in
						else "ok"
					)
				rows.append([
					grain_id,
					point[0],
					point[1],
					point[4],
					point[2],
					point[3],
					point[5],
					point[6],
					growth_rate,
					straight_px,
					straight_px*self.pxl_dis if straight_px != "" else "",
					trajectory_px,
					trajectory_px*self.pxl_dis if trajectory_px != "" else "",
					trajectory_status,
					length_result["centerline_px"] if length_result else "",
					length_result["centerline_calibrated"] if length_result else "",
					length_result["status"] if length_result else "unassociated_track",
					"interpolated" if tip_roi.is_filled_in else "detected",
					point[7],
				])
				previous_time = point[4]
				previous_length = point[6]
		return rows

	def clear_tube_length_cache(self):
		"""Discard derived tube skeletons and length estimates after image changes."""
		self._tube_skeleton_cache = {}
		self._tube_length_cache = {}

	def tube_skeleton(self, frame):
		"""Return a cached skeleton of dark tube-like structures for one frame."""
		if frame in self._tube_skeleton_cache:
			return self._tube_skeleton_cache[frame]
		if not hasattr(self.all_detections, "img_list_input"):
			return None
		if frame < 0 or frame >= self.all_detections.img_list_length:
			return None
		image = self.all_detections.img_list_input[frame]
		gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		background = cv.GaussianBlur(gray, (0, 0), self.tube_background_sigma)
		enhanced = cv.subtract(background, gray)
		enhanced = cv.normalize(enhanced, None, 0, 255, cv.NORM_MINMAX)
		_, binary = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
		binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
		count, labels, stats, _ = cv.connectedComponentsWithStats(binary)
		cleaned = np.zeros_like(binary)
		for label in range(1, count):
			if stats[label, cv.CC_STAT_AREA] >= self.tube_min_component_area:
				cleaned[labels == label] = 255
		skeleton = cv.ximgproc.thinning(cleaned)
		self._tube_skeleton_cache[frame] = skeleton
		return skeleton

	def estimate_tube_length(self, grain, frame):
		"""Estimate a grain-to-tip centerline and straight distance with QC state."""
		if grain is None:
			return None
		cache_key = (grain.id, int(frame))
		if cache_key in self._tube_length_cache:
			return self._tube_length_cache[cache_key]
		skeleton = self.tube_skeleton(frame)
		if skeleton is None:
			return None
		grain_roi = grain.roi_closest_to(frame)
		center_x, center_y = grain_roi.gv3.x, grain_roi.gv3.y
		half = self.tube_length_search_radius
		x1 = max(0, center_x - half)
		y1 = max(0, center_y - half)
		x2 = min(skeleton.shape[1], center_x + half)
		y2 = min(skeleton.shape[0], center_y + half)
		crop = skeleton[y1:y2, x1:x2].copy()
		local_center = (center_x - x1, center_y - y1)

		other_grains = []
		for candidate in self.valid_grains:
			roi = candidate.roi_closest_to(frame)
			left = max(0, roi.gv1.x - x1 - 1)
			top = max(0, roi.gv1.y - y1 - 1)
			right = min(crop.shape[1], roi.gv2.x - x1 + 1)
			bottom = min(crop.shape[0], roi.gv2.y - y1 + 1)
			if left < right and top < bottom:
				crop[top:bottom, left:right] = 0
			if candidate.id != grain.id:
				other_grains.append(roi)

		component_count, component_labels = cv.connectedComponents(crop)
		y_grid, x_grid = np.indices(crop.shape)
		distance_from_grain = np.hypot(
			x_grid - local_center[0], y_grid - local_center[1]
		)
		grain_radius = max(grain_roi.w, grain_roi.h)/2
		candidate_labels = []
		for label in range(1, component_count):
			distances = distance_from_grain[component_labels == label]
			if len(distances) and distances.min() <= grain_radius + self.tube_connection_gap:
				candidate_labels.append(label)

		best = None
		for label in candidate_labels:
			pixels = np.column_stack(np.where(component_labels == label))
			start = tuple(
				pixels[np.argmin(np.hypot(
					pixels[:, 1] - local_center[0],
					pixels[:, 0] - local_center[1],
				))]
			)
			pixel_set = {tuple(pixel) for pixel in pixels}
			distances = {start: 0.0}
			parents = {}
			queue = [(0.0, start)]
			while queue:
				cost, node = heappop(queue)
				if cost != distances[node]:
					continue
				y, x = node
				for dy in (-1, 0, 1):
					for dx in (-1, 0, 1):
						if not (dx or dy):
							continue
						next_node = (y + dy, x + dx)
						if next_node not in pixel_set:
							continue
						step = math.sqrt(2) if dx and dy else 1.0
						next_cost = cost + step
						if next_cost < distances.get(next_node, float("inf")):
							distances[next_node] = next_cost
							parents[next_node] = node
							heappush(queue, (next_cost, next_node))
			end = max(distances, key=distances.get)
			if best is None or distances[end] > best[0]:
				best = (distances[end], start, end, parents, pixel_set)

		if best is None:
			result = {
				"straight_px": "",
				"straight_calibrated": "",
				"centerline_px": "",
				"centerline_calibrated": "",
				"status": "no_connected_centerline",
			}
			self._tube_length_cache[cache_key] = result
			return result

		_, start, end, parents, pixel_set = best
		path = [end]
		while path[-1] != start:
			path.append(parents[path[-1]])
		path.reverse()
		centerline_px = 0.0
		for first, second in zip(path, path[1:]):
			dx = (second[1] - first[1])/self.img_rp.x
			dy = (second[0] - first[0])/self.img_rp.y
			centerline_px += math.hypot(dx, dy)
		straight_px = math.hypot(
			(end[1] - start[1])/self.img_rp.x,
			(end[0] - start[0])/self.img_rp.y,
		)
		endpoint_count = 0
		for y, x in pixel_set:
			degree = sum(
				(y + dy, x + dx) in pixel_set
				for dy in (-1, 0, 1)
				for dx in (-1, 0, 1)
				if dx or dy
			)
			endpoint_count += degree == 1
		status = "ok"
		if endpoint_count > 2:
			status = "ambiguous_branch"
		for other in other_grains:
			other_center = (other.gv3.x - x1, other.gv3.y - y1)
			other_radius = max(other.w, other.h)/2
			if any(
				math.hypot(x - other_center[0], y - other_center[1])
				<= other_radius + self.tube_connection_gap
				for y, x in pixel_set
			):
				status = "touches_other_grain"
				break
		if centerline_px < 5:
			status = "too_short"
		result = {
			"straight_px": straight_px,
			"straight_calibrated": straight_px*self.pxl_dis,
			"centerline_px": centerline_px,
			"centerline_calibrated": centerline_px*self.pxl_dis,
			"status": status,
		}
		self._tube_length_cache[cache_key] = result
		return result

	def track_summary_rows(self):
		"""Return one final cumulative-movement summary row per accepted track."""
		header = [
			"grain_id",
			"track_id",
			"start_frame",
			"end_frame",
			"start_time_" + self.time_unit,
			"end_time_" + self.time_unit,
			"duration_" + self.time_unit,
			"observation_count",
			"final_cumulative_tip_movement_px",
			"final_cumulative_tip_movement_" + self.dis_unit,
			"average_tip_growth_rate_" + self.dis_unit + "_per_" + self.time_unit,
			"final_grain_boundary_to_tracked_tip_straight_distance_px",
			"final_grain_boundary_to_tracked_tip_straight_distance_" + self.dis_unit,
			"final_trajectory_based_tube_length_px",
			"final_trajectory_based_tube_length_" + self.dis_unit,
			"trajectory_length_qc_status",
			"final_image_centerline_length_px",
			"final_image_centerline_length_" + self.dis_unit,
			"final_image_centerline_qc_status",
			"interpolated_point_count",
			"detection_methods",
		]
		details = self.coordinate_rows()
		if len(details) == 1:
			return [header]
		detail_header = details[0]
		by_track = {}
		for values in details[1:]:
			row = dict(zip(detail_header, values))
			by_track.setdefault(row["track_id"], []).append(row)
		rows = [header]
		for track in self.valid_tracks:
			track_rows = by_track.get(track.id, [])
			if not track_rows:
				continue
			first = track_rows[0]
			last = track_rows[-1]
			duration = last["time_" + self.time_unit] - first["time_" + self.time_unit]
			final_movement = last["cumulative_tip_movement_" + self.dis_unit]
			average_rate = final_movement/duration if duration > 0 else ""
			methods = ",".join(dict.fromkeys(row["detection_method"] for row in track_rows))
			rows.append([
				first["grain_id"],
				track.id,
				first["frame"],
				last["frame"],
				first["time_" + self.time_unit],
				last["time_" + self.time_unit],
				duration,
				len(track_rows),
				last["cumulative_tip_movement_px"],
				final_movement,
				average_rate,
				last["grain_boundary_to_tracked_tip_straight_distance_px"],
				last["grain_boundary_to_tracked_tip_straight_distance_" + self.dis_unit],
				last["trajectory_based_tube_length_px"],
				last["trajectory_based_tube_length_" + self.dis_unit],
				(
					"contains_interpolated_tip_positions"
					if any(row["tip_position_status"] == "interpolated" for row in track_rows)
					else last["trajectory_length_qc_status"]
				),
				last["image_centerline_length_px"],
				last["image_centerline_length_" + self.dis_unit],
				last["image_centerline_qc_status"],
				sum(row["tip_position_status"] == "interpolated" for row in track_rows),
				methods,
			])
		return rows

	def segment_inputs(self, for_ger = True, false_color = True):
		"""Load extracted frame files and build the processed detection sequence."""
		if self.file_names is None:
			pass
		else:
			self.all_detections = []
			img_list = []
			self.img_ratio = (self.img_rp.x+self.img_rp.y)/2
			img = cv.resize(cv.imread(self.file_names[0]), self.screen_size)
			h, w, l = img.shape
			self.gv13 = Point(x = w, y = h)
			self.gv12 = Point(x = int(self.gv13.x*self.img_rp.x), y = int(self.gv13.y**self.img_rp.y))
			for img_path in self.file_names:
				img_list.append(cv.resize(cv.imread(img_path), self.screen_size))
			self.all_detections = Detections(img_list = img_list, bg_threshold = self.bg_threshold, blur_radius = self.filter_radius)
			self.clear_tube_length_cache()

	def tip_templates(self):
		"""Return validated image templates for the legacy tip detector."""
		return load_tip_templates()

	def track_elongation(self):
		"""Link detected tips into filtered tube-growth trajectories."""
		disp_ratio = 0.25
		self.valid_tracks  = []
		self.filled_in_tips = []
		self.gv31 = 1
		self.gv32 = []
		if len(self.valid_tips) > 0:
			tmp = []
			for tips_in_k in self.valid_tips:
				for tip in tips_in_k:
					if tip.w >= self.min_tip_side and tip.h >= self.min_tip_side:
						tmp.append(tip.w)
						tmp.append(tip.h)
			if len(tmp) == 0:
				return
			mean_side = stat.mean(tmp)
			overlap = min(0.99, max(0.0, float(self.tip_max_step)))
			max_distance = mean_side*(1-overlap)/(1+overlap)
			max_distance = max(1.0, max_distance)
			linked_tracks = self.link_detections(
				self.valid_tips,
				min_overlap=overlap,
				gap_frames=max(0, int(self.tip_gap_closing)),
				gap_distance=2*max_distance,
			)
			linked_tracks = self.stitch_track_fragments(
				linked_tracks,
				max_gap_frames=max(0, int(self.tip_gap_closing)),
				prediction_tolerance=max(mean_side, 2*max_distance),
			)
			for cur_trk in linked_tracks:
				if len(cur_trk)>1:
					trk = Track(boxes = cur_trk, color = self.random_color(), ID = self.allocate_id())
					xtr_tips = trk.fill_missing_frames(ret = True)
					if trk.gv2 >= self.min_tip_per_trk or trk.last_frame() - trk.first_frame() >= self.min_tip_per_trk:
						if trk.straightness_ratio() >= disp_ratio:
							self.valid_tracks.append(trk)
							if len(xtr_tips) > 0:
								for tip in xtr_tips:
									self.filled_in_tips.append(tip)
						else:
							self.release_id(trk.id)
					else:
						self.release_id(trk.id)

	def track_germination_via_tips(self):
		"""Infer germination from sustained tip movement overlapping a grain."""
		disp_ratio = 0.3
		cutoff = self.ger_confirm_frames
		confirm_at = self.aceptance_ratio
		tips = []
		grains = []
		k = -1
		for tips_in_k in self.valid_tips:
			for tip in tips_in_k:
				tip.is_used = False
			tips.append(tips_in_k.copy())
		while(True):
			k+=1
			if k >= self.all_detections.img_list_length:
				break
			grains_in_k = []
			for grain in self.valid_grains:
				if grain.first_frame() <= k:
					grains_in_k.append(grain.roi_closest_to(frame = k))
			grains.append(grains_in_k)
		n_unger = len(self.valid_grains)
		k = -1
		while(True):
			k+=1
			if k >= self.all_detections.img_list_length:
				break
			if n_unger == 0:
				break
			print("Frame: " + str(k) + ", ungerminated: " + str(n_unger))
			tips_in_k = tips[k]
			grains_in_k = grains[k]
			if len(tips_in_k) > 0 and len(grains_in_k) > 0:
				for tip in tips_in_k:
					if not tip.is_used:
						if tip.overlaps_any(grains_in_k):
							g = tip.best_overlap(grains_in_k)
							for grain in self.valid_grains:
								if grain.id == g.group:
									if not grain.is_germinated:
										c=1
										ct=1
										t = tip
										disp = 0
										for i in range(k+1, k+cutoff+1):
											if i >= self.all_detections.img_list_length:
												break
											ct+=1
											if t.overlaps_any(tips[i]):
												c+=1
												disp += t.distance_btw(t.gv3, t.best_overlap(tips[i]).gv3)
												t = t.best_overlap(tips[i])
										if c/ct >= confirm_at:
											if disp <= 0:
												disp = 1
											if tip.distance_btw(tip.gv3, t.gv3)/disp > disp_ratio:
												grain.update_germination(frame = tip.gv6, p_value = 0.9*self.gv5, method = "tip_overlap")
												n_unger-=1
												tip.is_used = True
												t = tip
												for i in range(k+1, k+cutoff+1):
													if i >= self.all_detections.img_list_length:
														break
													if t.overlaps_any(tips[i]):
														t.is_used = True
														t = t.best_overlap(tips[i])

	def track_germination_via_area(self):
		"""Infer germination from statistically unusual local area changes."""
		cutoff = self.ger_confirm_frames
		confirm_at = self.aceptance_ratio
		area = []
		for grain in self.valid_grains:
			for roi in grain.gv1:
				if not roi.is_filled_in:
					area.append(roi.gv4)
		if len(area) > 1:
			sd = stat.stdev(area)/stat.mean(area)
		else:
			sd = 0.08270356
		null_dist = np.random.normal(1, sd, self.gv21)
		n_unger = 0
		for grain in self.valid_grains:
			if not grain.is_germinated:
				n_unger+=1
		sz_co = 1 + 2*sd
		fr = -1
		for rois_in_fr in self.all_detections.rois:
			if n_unger < 1:
				break
			fr += 1
			print("Frame: " + str(fr) + ", ungerminated: " + str(n_unger))
			for grain in self.valid_grains:
				if grain.is_germinated:
					pass
				else:
					if grain.first_frame() <= fr:
						r1 = grain.roi_closest_to(grain.first_frame()).best_overlap(self.all_detections.rois[grain.first_frame()])
						r2 = grain.roi_closest_to(fr).best_overlap(rois_in_fr)
						if r1 != None and r2 != None:
							if r2.gv4/r1.gv4 <= sz_co:
								p_val = len(null_dist[null_dist >= r2.gv4/r1.gv4])/self.gv21
								if p_val <= self.gv5:
									c = 0
									for i in range(fr + 1, fr + cutoff + 1):
										if i >= self.all_detections.img_list_length:
											break
										r3 = grain.roi_closest_to(i).best_overlap(self.all_detections.rois[i])
										if r3 is not None:
											if r3.gv4/r1.gv4 <= 1.5*sz_co:
												p = len(null_dist[null_dist >= r3.gv4/r1.gv4])/self.gv21
												if p <= self.gv5:
													c += 1
									if c >= confirm_at:
										grain.update_germination(frame = r2.gv6, p_value = p_val, method = "area_change")
										n_unger -= 1
								elif p_val < grain.ger_p_value:
									grain.ger_p_value = p_val

	def associate_grain_to_track(self, grain):
		"""Return the most plausible tip trajectory associated with a grain."""
		if len(self.valid_tracks) == 0:
			return None
		ref_frame = grain.ger_frame if grain.ger_frame >= 0 else grain.first_frame()
		grain_roi = grain.roi_closest_to(ref_frame)
		best = None
		best_score = None
		max_dist = 4*(self.max_grain_radius + self.min_grain_radius)
		for track in self.valid_tracks:
			t0 = track.gv1[0]
			overlaps = False
			for i in range(min(3, track.gv2)):
				if grain_roi.overlap_fraction(track.gv1[i]) > 0 or track.gv1[i].overlap_fraction(grain_roi) > 0:
					overlaps = True
					break
			d = grain.centroid_distance(grain_roi, t0)
			score = d - 1000000 if overlaps else d
			if best_score is None or score < best_score:
				best_score = score
				best = track
		if best is not None:
			if grain.centroid_distance(grain_roi, best.gv1[0]) > max_dist:
				return None
		return best

	def local_foreground(self, center, frame, half):
		"""Measure foreground area and component count near a point in one frame."""
		if frame < 0 or frame >= self.all_detections.img_list_length:
			return (0, 0)
		img = self.all_detections.get_noiseless_frame(frame = frame, bnr = True, gray = True)
		if img is None:
			return (0, 0)
		h, w = img.shape
		x0 = max(0, int(center.x - half))
		x1 = min(w, int(center.x + half))
		y0 = max(0, int(center.y - half))
		y1 = min(h, int(center.y + half))
		if x1 <= x0 or y1 <= y0:
			return (0, 0)
		crop = img[y0:y1, x0:x1]
		area = int(np.count_nonzero(crop))
		cnts, _ = cv.findContours(crop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		return (area, len(cnts))

	def score_burst(self, grain, track, last_valid_frame):
		"""Score a possible rupture from track termination and local morphology."""
		if track is None or track.gv2 < 2:
			return None
		end_frame = track.last_frame()
		tip_end = track.roi_closest_to(end_frame)
		center = tip_end.gv3
		half = int(2*max(tip_end.w, tip_end.h))
		half = max(half, self.min_tip_side*2)
		score = 0.0
		reasons = []
		terminates = end_frame < (last_valid_frame - self.burst_end_margin)
		if terminates:
			score += 0.3
			reasons.append("track_ends_early")
		before_frame = max(track.first_frame(), end_frame - self.burst_end_margin)
		after_frame = min(last_valid_frame, end_frame + self.burst_end_margin)
		before_area, before_cnts = self.local_foreground(center, before_frame, half)
		after_area, after_cnts = self.local_foreground(center, after_frame, half)
		if before_area > 0 and after_area/before_area >= self.burst_area_spike_ratio:
			score += 0.4
			reasons.append("local_area_spike")
		early = track.gv1[0]
		if early.side_ratio - tip_end.side_ratio > 0.25 and tip_end.side_ratio < 0.35:
			score += 0.2
			reasons.append("tip_rounding")
		if after_cnts > before_cnts and after_cnts >= 2:
			score += 0.2
			reasons.append("fragmentation")
		if score > 1.0:
			score = 1.0
		if len(reasons) == 0:
			return None
		return {"grain_id": grain.id, "frame": end_frame, "time": end_frame*self.time_p_frame, "confidence": round(score, 3), "method": "auto_tip_burst", "track_id": track.id, "reason": "+".join(reasons)}

	def find_burst_candidates(self):
		"""Generate reviewer-only rupture suggestions for germinated grains."""
		self.burst_candidates = []
		self.burst_candidates_evaluated = True
		if len(self.valid_grains) == 0 or self.all_detections == []:
			return self.burst_candidates
		last_valid_frame = self.all_detections.img_list_length - 1
		for grain in self.valid_grains:
			if not grain.is_bursted:
				grain.clear_burst_candidate()
		for grain in self.valid_grains:
			if grain.is_bursted or not grain.is_germinated:
				continue
			track = self.associate_grain_to_track(grain)
			cand = self.score_burst(grain, track, last_valid_frame)
			if cand is not None and cand["confidence"] >= self.burst_score_threshold:
				cf = min(cand["frame"], grain.last_frame())
				cand["frame"] = cf
				cand["time"] = cf*self.time_p_frame
				grain.set_burst_candidate(frame = cf, confidence = cand["confidence"], reason = cand["reason"], track_id = cand["track_id"])
				self.burst_candidates.append(cand)
		return self.burst_candidates
