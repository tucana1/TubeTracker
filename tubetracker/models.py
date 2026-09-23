"""Domain objects shared by TubeTracker analysis and review workflows."""

import math
import statistics as stat

import cv2 as cv
import numpy as np


class Point:
	"""Represent a two-dimensional coordinate with optional frame metadata."""

	def __init__(self, x=0, y=0, frame = None, detection_method = "auto"):
		"""Store coordinate, frame, and detection provenance values."""
		self.x = x
		self.y = y
		self.coor = (x, y)
		self.is_a = "point"
		self.detection_method = detection_method
		if frame != None:
			self.frame = int(frame)
		else:
			self.frame = None

class ROI:
	"""Represent a rectangular detection and its biological display state."""

	def __init__(self, x_l, y_t, x_r, y_b, color = (255, 0, 0), ID = '', frame = 0, img = None, detection_method = "auto", group = '', is_germinated = False, is_bursted = False, is_tip = False, filled_in = False, is_used = False, is_burst_candidate = False):
		"""Initialize bounds, geometry, identity, and tracking state."""
		self.gv1 = Point(x = x_l, y = y_t)
		self.gv2 = Point(x = x_r, y = y_b)
		self.gv3 = Point(x = int((x_r+x_l)/2), y = int((y_b+y_t)/2))
		self.w = int(abs(x_r - x_l))
		self.h = int(abs(y_b - y_t))
		self.gv4 = self.w*self.h
		self.side_ratio = 1 - min([self.w, self.h])/max([self.w, self.h])
		self.set_pos = Point(x = int((x_r+x_l)/2), y = int((y_b+y_t)/2))
		self.is_a = "roi"
		self.color = color
		self.set_id(ID)
		self.gv6 = int(frame)
		self.gv7 = None
		self.gv8 = 1
		self.detection_method = detection_method
		self.gv12 = None
		self.gv13 = (255,255,0)
		self.group = str(group)
		self.is_germinated = is_germinated
		self.is_bursted = is_bursted
		self.is_burst_candidate = is_burst_candidate
		self.is_tip = is_tip
		self.is_filled_in = filled_in
		self.ma_ax_int_p1 = None
		self.ma_ax_int_p2 = None
		self.mi_ax_int_p1 = None
		self.mi_ax_int_p2 = None
		self.is_used = is_used
		if img is not None:
			self.get_axes_int_points(img = img, ret = False)

	def distance_btw(self, p1, p2):
		"""Return Euclidean distance between two point-like objects."""
		return math.sqrt(math.pow(p1.x - p2.x, 2) + math.pow(p1.y - p2.y, 2))

	def overlap_fraction(self, other):
		"""Return overlap area with another ROI as a fraction of this ROI."""
		if (min(self.gv2.x, other.gv2.x) < max(self.gv1.x, other.gv1.x)) or (min(self.gv2.y, other.gv2.y) < max(self.gv1.y, other.gv1.y)):
			return 0
		elif self.gv1.x < other.gv1.x and self.gv2.x > other.gv2.x and self.gv1.y < other.gv1.y and self.gv2.y > other.gv2.y:
			return 1
		elif self.gv1.x > other.gv1.x and self.gv2.x < other.gv2.x and self.gv1.y > other.gv1.y and self.gv2.y < other.gv2.y:
			return 1
		else:
			return (abs(min(self.gv2.x, other.gv2.x) - max(self.gv1.x, other.gv1.x))*abs(min(self.gv2.y, other.gv2.y) - max(self.gv1.y, other.gv1.y)))/self.gv4

	def overlaps_any(self, others, overlap = 0.10, check_frame = False):
		"""Report whether this ROI sufficiently overlaps any ROI in a collection."""
		ret = 0
		for other in others:
			if self.overlap_fraction(other) >= overlap:
				if check_frame == True:
					if self.gv6 == other.gv6:
						ret = 1
					else:
						ret = 0
				else:
					ret = 1
				break
		if ret == 0:
			return False
		else:
			return True

	def best_overlap(self, others, idx = False, check_frame = False):
		"""Return the best-overlapping ROI or its index from a collection."""
		ret = None
		if self.overlaps_any(others, check_frame = check_frame):
			overlap = 0
			index = None
			i = 0
			for other in others:
				if self.overlap_fraction(other) > overlap:
					if check_frame == True:
						if self.gv6 == other.gv6:
							overlap = self.overlap_fraction(other)
							index = i
					else:
						overlap = self.overlap_fraction(other)
						index = i
				i += 1
			if index != None:
				if idx == True:
					ret = index
				else:
					ret = others[index]
		return ret

	def grain_color(self, paint = True):
		"""Choose a display color for tip, germination, and rupture state."""
		if self.is_tip:
			if paint:
				return (255,0,255)
			else:
				return (255,0,255)
		else:
			if self.is_burst_candidate and not self.is_bursted:
				return (0,165,255)
			if self.is_germinated:
				if self.is_bursted:
					if paint:
						return (0,255,255)
					else:
						return (255,255,0)
				else:
					if paint:
						return (0,255,0)
					else:
						return (0,255,0)
			else:
				if self.is_bursted:
					if paint:
						return (0,0,255)
					else:
						return (255,0,0)
				else:
					if paint:
						return (255,0,0)
					else:
						return (0,0,255)

	def get_axes_int_points(self, img, ret = True):
		"""Estimate major and minor axis intersections using contour PCA."""
		dxx = 1
		area = -1
		cnts, _ = cv.findContours(img[self.gv1.y:self.gv2.y, self.gv1.x:self.gv2.x], cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
		for c in cnts:
			if cv.contourArea(c) > area:
				pts = c
				area = cv.contourArea(c)
		sz = len(pts)
		data_pts = np.empty((sz, 2), dtype=np.float64)
		for i in range(data_pts.shape[0]):
			data_pts[i,0] = pts[i,0,0]
			data_pts[i,1] = pts[i,0,1]
		mean = np.empty((0))
		mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
		c = Point(x = self.gv1.x + int(mean[0,0]), y = self.gv1.y + int(mean[0,1]), frame = self.gv6)
		self.tip_center = c
		p = Point(x = self.gv1.x + int(mean[0,0]) + eigenvectors[0,0]*eigenvalues[0,0], y = self.gv1.y + int(mean[0,1]) + eigenvectors[0,1]*eigenvalues[0,0], frame = self.gv6)
		if abs(c.x - p.x) < dxx:
			self.ma_ax_int_p1 = Point(x = int(c.x), y = int(self.gv1.y), frame = self.gv6)
			self.ma_ax_int_p2 = Point(x = int(c.x), y = int(self.gv2.y), frame = self.gv6)
		elif abs(c.y - p.y) < dxx:
			self.ma_ax_int_p1 = Point(x = int(self.gv1.x), y = int(c.y), frame = self.gv6)
			self.ma_ax_int_p2 = Point(x = int(self.gv2.x), y = int(c.y), frame = self.gv6)
		else:
			m = (c.y - p.y)/(c.x - p.x)
			b = c.y - m*c.x
			int_pts = [[Point(x = int(self.gv1.x), y = int(b+m*self.gv1.x), frame = self.gv6), 0, 0], [Point(x = int((self.gv1.y - b)/m), y = int(self.gv1.y), frame = self.gv6), 0, 1], [Point(x = int(self.gv2.x), y = int(b+m*self.gv2.x), frame = self.gv6), 0, 2], [Point(x = int((self.gv2.y - b)/m), y = int(self.gv2.y), frame = self.gv6), 0, 3]]
			dx = None
			i = 0
			for p in int_pts:
				p[1] = self.distance_btw(p1=c, p2=p[0])
				if dx == None:
					dx = p[1]
				else:
					if p[1] < dx:
						dx = p[1]
						i = int(p[2])
			self.ma_ax_int_p1 = int_pts[i][0]
			dx = 100000*self.distance_btw(p1 = self.gv1, p2 = self.gv2)
			for p in int_pts:
				if p[2] != i:
					if(p[0].x - c.x)*(self.ma_ax_int_p1.x - c.x) < 0:
						if (p[0].y - c.y)*(self.ma_ax_int_p1.y - c.y) < 0:
							if self.distance_btw(p1 = c, p2 = p[0]) <= dx:
								dx = self.distance_btw(p1 = c, p2 = p[0])
								i = int(p[2])
			self.ma_ax_int_p2 = int_pts[i][0]
		p = Point(x = self.gv1.x + int(mean[0,0]) - eigenvectors[1,0]*eigenvalues[1,0], y = self.gv1.y + int(mean[0,1]) - eigenvectors[1,1]*eigenvalues[1,0], frame = self.gv6)
		if abs(c.x - p.x) < dxx:
			self.mi_ax_int_p1 = Point(x = int(c.x), y = int(self.gv1.y), frame = self.gv6)
			self.mi_ax_int_p2 = Point(x = int(c.x), y = int(self.gv2.y), frame = self.gv6)
		elif abs(c.y - p.y) < dxx:
			self.mi_ax_int_p1 = Point(x = int(self.gv1.x), y = int(c.y), frame = self.gv6)
			self.mi_ax_int_p2 = Point(x = int(self.gv2.x), y = int(c.y), frame = self.gv6)
		else:
			m = (c.y - p.y)/(c.x - p.x)
			b = c.y - m*c.x
			int_pts = [[Point(x = int(self.gv1.x), y = int(b+m*self.gv1.x), frame = self.gv6), 0, 0], [Point(x = int((self.gv1.y - b)/m), y = int(self.gv1.y), frame = self.gv6), 0, 1], [Point(x = int(self.gv2.x), y = int(b+m*self.gv2.x), frame = self.gv6), 0, 2], [Point(x = int((self.gv2.y - b)/m), y = int(self.gv2.y), frame = self.gv6), 0, 3]]
			dx = None
			i = 0
			for p in int_pts:
				p[1] = self.distance_btw(p1=c, p2=p[0])
				if dx == None:
					dx = p[1]
				else:
					if p[1] < dx:
						dx = p[1]
						i = int(p[2])
			self.mi_ax_int_p1 = int_pts[i][0]
			dx = 100000*self.distance_btw(p1 = self.gv1, p2 = self.gv2)
			for p in int_pts:
				if p[2] != i:
					if(p[0].x - c.x)*(self.mi_ax_int_p1.x - c.x) < 0:
						if (p[0].y - c.y)*(self.mi_ax_int_p1.y - c.y) < 0:
							if self.distance_btw(p1 = c, p2 = p[0]) <= dx:
								dx = self.distance_btw(p1 = c, p2 = p[0])
								i = int(p[2])
			self.mi_ax_int_p2 = int_pts[i][0]
		if ret:
			return self.ma_ax_int_p1, self.ma_ax_int_p2, self.mi_ax_int_p1, self.mi_ax_int_p2

	def set_id(self, ID):
		"""Assign a string identifier to the detection."""
		self.id = str(ID)

class Track:
	"""Represent a time-ordered trajectory or grain observation sequence."""

	def __init__(self, boxes, color = (0,0,0), ID = '', fill_holes = False):
		"""Initialize trajectory state from a collection of frame-specific ROIs."""
		self.burst_frame = -1
		self.burst_method = "manual"
		self.is_burst_candidate = False
		self.burst_candidate_frame = -1
		self.burst_candidate_confidence = 0.0
		self.burst_candidate_reason = ""
		self.burst_candidate_track_id = ""
		self.color = color
		self.detection_method = "auto"
		self.germination_method = "auto"
		self.ger_frame = -1
		self.ger_p_value = 1
		self.gv1 = []
		self.gv2 = 0
		self.gv3 = []
		self.is_germinated = False
		self.is_bursted = False
		self.set_id(ID)
		self.add_rois(boxes)
		if fill_holes == True:
			self.fill_missing_frames()
		self.gv4 = ["track." + self.id + " cumulative tip movement ()"]
		self.gv5 = ["track." + self.id + " cumulative tip movement ()"]

	def add_roi(self, box):
		"""Insert an ROI into the trajectory while preserving frame order."""
		box.group = self.id
		if self.gv1 == []:
			self.gv1.append(box)
			self.gv2 += 1
		elif box.gv6 < self.gv1[0].gv6:
			self.gv1.insert(0,box)
			self.gv2 += 1
		elif box.gv6 > self.gv1[-1].gv6:
			self.gv1.append(box)
			self.gv2 += 1
		else:
			for i in range(1, self.gv2):
				if self.gv1[i-1].gv6 < box.gv6 < self.gv1[i].gv6:
					self.gv1.insert(i, box)
					self.gv2 += 1
					break

	def add_rois(self, boxes):
		"""Insert each supplied ROI into the trajectory."""
		for box in boxes:
			self.add_roi(box)

	def calculate_metrics(self, coef = Point(x=1,y=1), disp = 2.00, disp_u = "um", num_frames = 300, time_p_frame = 60):
		"""Calculate calibrated coordinates and cumulative movement over time."""
		if self.gv2 > 1:
			self.gv3 = [[self.id, self.gv1[0].gv6, int(self.gv1[0].gv3.x/coef.x), int(self.gv1[0].gv3.y/coef.y), self.gv1[0].gv6*time_p_frame, 0, 0, self.gv1[0].detection_method]]
			length = 0
			for i in range(1,self.gv2):
				box = self.gv1[i]
				box1 = self.gv1[i-1]
				length += self.centroid_distance(box1, box, img_ratio = coef)
				self.gv3.append([self.id, box.gv6, int(box.gv3.x/coef.x), int(box.gv3.y/coef.y), box.gv6*time_p_frame, length, length*disp, box.detection_method])
			length = 0
			times = 0
			gv4 = [[times, length]]
			gv5 = [[self.gv1[0].gv6, length]]
			for i in range(1,self.gv2):
				box1 = self.gv1[i-1]
				box2 = self.gv1[i]
				length += self.centroid_distance(box1, box2, img_ratio = coef)*disp
				times = int(box2.gv6 - self.gv1[0].gv6)
				gv4.append([times, length])
				gv5.append([box2.gv6, length])
			self.gv4 = ["track." + self.id + " cumulative tip movement (" + disp_u + ")"]
			self.gv5 = ["track." + self.id + " cumulative tip movement (" + disp_u + ")"]
			for i in range(num_frames):
				k = ""
				for pt in gv4:
					if pt[0] == i:
						k = pt[1]
						break
				self.gv4.append(k)
				k = ""
				for pt in gv5:
					if pt[0] == i:
						k = pt[1]
						break
				self.gv5.append(k)

	def truncate_near(self, box, keep_leftover = False):
		"""Truncate the trajectory near an ROI and optionally return its tail."""
		distance = 1000000
		index = None
		min_length = 2
		if len(self.gv1) > 2*min_length:
			i = 0
			for bx in self.gv1:
				if self.centroid_distance(box, bx) < distance:
					distance = self.centroid_distance(box, bx)
					index = i
				i+=1
		if index != None:
			if keep_leftover and (index >= self.gv2 - min_length or index < min_length):
				return None
			else:
				if keep_leftover:
					r = self.gv2 - 1
					temp = self.gv1[index:r]
				r = index - 1
				self.gv1 = self.gv1[0:r]
				self.gv2 = len(self.gv1)
				if keep_leftover:
					return temp
		elif keep_leftover:
				return None

	def straightness_ratio(self):
		"""Return endpoint displacement divided by total trajectory distance."""
		abs_len = 0
		for i in range(1, self.gv2):
			abs_len += self.centroid_distance(self.gv1[i-1], self.gv1[i])
		if abs_len == 0:
			return 0
		else:
			return self.centroid_distance(self.gv1[0], self.gv1[self.gv2-1])/abs_len

	def centroid_distance(self, box1, box2, img_ratio = None):
		"""Measure centroid distance with optional image-scale correction."""
		if img_ratio ==None:
			return math.sqrt(math.pow(float(box1.gv3.x - box2.gv3.x), 2) + math.pow(float(box1.gv3.y - box2.gv3.y), 2))
		else:
			return math.sqrt(math.pow(float(box1.gv3.x - box2.gv3.x)/img_ratio.x, 2) + math.pow(float(box1.gv3.y - box2.gv3.y)/img_ratio.y, 2))

	def fill_missing_frames(self, ret = False, up_to_frame = -1):
		"""Linearly interpolate missing trajectory frames and optional trailing frames."""
		k = 0
		h = []
		w = []
		for roi in self.gv1:
			h.append(roi.h)
			w.append(roi.w)
		h = int(stat.mean(h)/2)
		w = int(stat.mean(w)/2)
		boxes = []
		while(True):
			k+=1
			if k >= self.gv2:
				break
			if self.gv1[k].gv6 - self.gv1[k-1].gv6 > 1:
				box1 = self.gv1[k-1]
				box2 = self.gv1[k]
				nbox = box2.gv6 - box1.gv6
				mxl = (box2.gv1.x - box1.gv1.x)/nbox
				mxr = (box2.gv2.x - box1.gv2.x)/nbox
				myt = (box2.gv1.y - box1.gv1.y)/nbox
				myb = (box2.gv2.y - box1.gv2.y)/nbox
				nbox-=1
				for i in range(nbox):
					df = i+1
					frame = box1.gv6 + df
					boxes.append(ROI(x_l = int(box1.gv1.x + df*mxl), y_t = int(box1.gv1.y + df*myt), x_r = int(box1.gv2.x + df*mxr), y_b = int(box1.gv2.y + df*myb), frame = frame, group = box1.group, filled_in = True, detection_method = "interpolated"))
		self.add_rois(boxes)
		if up_to_frame > self.last_frame():
			b = []
			for i in range(self.last_frame(), up_to_frame, 1):
				b.append(ROI(x_l = self.gv1[-1].gv1.x, y_t = self.gv1[-1].gv1.y, x_r = self.gv1[-1].gv2.x, y_b = self.gv1[-1].gv2.y, frame = i, group = self.gv1[-1].group, filled_in = True, detection_method = "interpolated"))
			if len(b) > 0:
				self.add_rois(b)
		if ret:
			return boxes

	def first_frame(self):
		"""Return the earliest frame represented by the trajectory."""
		f = 100000000
		for roi in self.gv1:
			if roi.gv6 < f:
				f = roi.gv6
		return f

	def last_frame(self):
		"""Return the latest frame represented by the trajectory."""
		f = 0
		for roi in self.gv1:
			if roi.gv6 > f:
				f = roi.gv6
		return f

	def remove_survival(self, what = "g"):
		"""Clear a germination or rupture annotation from this grain track."""
		if what == "g":
			self.ger_p_value = 1
			self.ger_frame = -1
			self.is_germinated = False
			for r in self.gv1:
				r.is_germinated = False
			self.germination_method = "manual"
		elif what == "b":
			self.burst_frame = -1
			self.is_bursted = False
			for r in self.gv1:
				r.is_bursted = False
			self.burst_method = "manual"

	def roi_closest_to(self, frame):
		"""Return the observation nearest to a requested frame."""
		dt = 100000000
		idx = 0
		k = -1
		for roi in self.gv1:
			k+=1
			if abs(roi.gv6 - frame) < dt:
				idx = k
				dt = abs(roi.gv6 - frame)
				if dt == 0:
					break
		return self.gv1[idx]

	def set_id(self, ID):
		"""Assign a string identifier to the trajectory."""
		self.id = str(ID)

	def survival_values(self, tbf = 30):
		"""Serialize grain detection, germination, rupture, and provenance values."""
		if self.is_germinated:
			gt = self.ger_frame*tbf
		else:
			gt = -1
		if self.is_bursted:
			bt = self.burst_frame*tbf
		else:
			bt = -1
		return [self.id, self.gv1[0].gv6*tbf, gt, bt, self.gv1[0].gv6, self.ger_frame, self.burst_frame, self.ger_p_value, self.gv1[0].gv3.x, self.gv1[0].gv3.y, self.gv1[0].gv4, self.detection_method, self.germination_method, self.burst_method]

	def update_germination(self, frame, p_value, method = "auto"):
		"""Mark germination at a frame when it does not conflict with rupture."""
		if frame <= self.last_frame():
			add_ger = True
			if self.is_bursted:
				if self.burst_frame <= frame:
					add_ger = False
			if add_ger:
				self.ger_p_value = p_value
				self.ger_frame = frame
				self.is_germinated = True
				self.germination_method = method
				for r in self.gv1:
					if r.gv6 >= frame:
						r.is_germinated = True
					else:
						r.is_germinated = False

	def update_burst(self, frame, method = "manual"):
		"""Mark a reviewed rupture event and update per-frame grain state."""
		if frame <= self.last_frame():
			self.burst_frame = frame
			self.is_bursted = True
			self.burst_method = method
			self.clear_burst_candidate()
			for r in self.gv1:
				if r.gv6 >= frame:
					r.is_bursted = True
				else:
					r.is_bursted = False
			if self.is_germinated:
				if self.burst_frame <= self.ger_frame:
					self.ger_frame = -1
					self.ger_p_value = 1
					self.is_germinated = False
					for r in self.gv1:
						r.is_germinated = False

	def set_burst_candidate(self, frame, confidence, reason, track_id = ""):
		"""Attach an unconfirmed rupture suggestion and supporting evidence."""
		if frame <= self.last_frame() and not self.is_bursted:
			self.is_burst_candidate = True
			self.burst_candidate_frame = frame
			self.burst_candidate_confidence = confidence
			self.burst_candidate_reason = reason
			self.burst_candidate_track_id = str(track_id)
			for r in self.gv1:
				if r.gv6 >= frame:
					r.is_burst_candidate = True
				else:
					r.is_burst_candidate = False

	def clear_burst_candidate(self):
		"""Remove any unconfirmed rupture suggestion from this grain."""
		self.is_burst_candidate = False
		self.burst_candidate_frame = -1
		self.burst_candidate_confidence = 0.0
		self.burst_candidate_reason = ""
		self.burst_candidate_track_id = ""
		for r in self.gv1:
			r.is_burst_candidate = False

	def accept_burst_candidate(self):
		"""Convert the current rupture suggestion into a reviewed event."""
		if self.is_burst_candidate:
			frame = self.burst_candidate_frame
			self.update_burst(frame = frame, method = "reviewed_burst_candidate")
