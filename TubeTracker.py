#!/opt/miniconda3/envs/tbtkr/bin/python3
#-*- coding: utf-8 -*-
"""Desktop and analysis engine for pollen germination and tube tracking."""

import os
import wx
import cv2 as cv
import csv
import math
import tempfile
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import statistics as stat
from random import randint
from motpy import Detection, MultiObjectTracker

from tubetracker_resources import load_logo, load_tip_templates

tempdir = tempfile.TemporaryDirectory()
temp_dir = tempdir.name
title = 'TubeTracker'

UI_BACKGROUND = wx.Colour(231, 236, 241)
UI_PANEL = wx.Colour(247, 249, 251)
UI_CANVAS = wx.Colour(255, 255, 255)
UI_FIELD = wx.Colour(255, 255, 255)
UI_TEXT = wx.Colour(36, 45, 54)
UI_MUTED_TEXT = wx.Colour(83, 96, 109)
UI_ACCENT = wx.Colour(35, 94, 145)

# Fixed hit-or-miss definitions for skeleton endpoints; these are not learned weights.
TIP_ENDPOINT_KERNELS = (
	np.array(([0, 0, 0], [-1, 1, -1], [-1, -1, -1]), dtype=np.int8),
	np.array(([0, -1, -1], [0, 1, -1], [0, -1, -1]), dtype=np.int8),
	np.array(([-1, -1, 0], [-1, 1, 0], [-1, -1, 0]), dtype=np.int8),
	np.array(([-1, -1, -1], [-1, 1, -1], [0, 0, 0]), dtype=np.int8),
)

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

	def color_pallette(self):
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
			rows.append([x, b, g, r])
		data_dir = temp_dir + "/pallette.csv"
		with open(data_dir, 'w', newline = '') as f:
			csv.writer(f).writerows(rows)
		return pd.read_csv(os.path.expanduser(data_dir), names=['gray', 'b', 'g', 'r'])

	def false_color(self, gray, min_pxl = 1):
		"""Map grayscale intensities to the application's false-color palette."""
		col_p = self.color_pallette()
		h, w = gray.shape
		mask = np.zeros((h,w,3), dtype = np.uint8)
		for pxl in range(min_pxl, 256, 1):
			pxl_coor = np.column_stack(np.where(gray == pxl))
			if len(pxl_coor) > 0:
				bgr = col_p[col_p.gray == pxl]
				mask[gray == pxl] = (int(bgr.b.iloc[0]), int(bgr.g.iloc[0]), int(bgr.r.iloc[0]))
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

	def get_raw_frame(self, frame):
		"""Return the stored grayscale edge frame converted to BGR."""
		if self.img_list_length > frame:
			return cv.cvtColor(self.img_list_gray[frame], cv.COLOR_GRAY2BGR)

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

	def f1(self, other):
		"""Return overlap area with another ROI as a fraction of this ROI."""
		if (min(self.gv2.x, other.gv2.x) < max(self.gv1.x, other.gv1.x)) or (min(self.gv2.y, other.gv2.y) < max(self.gv1.y, other.gv1.y)):
			return 0
		elif self.gv1.x < other.gv1.x and self.gv2.x > other.gv2.x and self.gv1.y < other.gv1.y and self.gv2.y > other.gv2.y:
			return 1
		elif self.gv1.x > other.gv1.x and self.gv2.x < other.gv2.x and self.gv1.y > other.gv1.y and self.gv2.y < other.gv2.y:
			return 1
		else:
			return (abs(min(self.gv2.x, other.gv2.x) - max(self.gv1.x, other.gv1.x))*abs(min(self.gv2.y, other.gv2.y) - max(self.gv1.y, other.gv1.y)))/self.gv4

	def f3(self, others, overlap = 0.10, check_frame = False):
		"""Report whether this ROI sufficiently overlaps any ROI in a collection."""
		ret = 0
		for other in others:
			if self.f1(other) >= overlap:
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

	def f4(self, others, idx = False, check_frame = False):
		"""Return the best-overlapping ROI or its index from a collection."""
		ret = None
		if self.f3(others, check_frame = check_frame):
			overlap = 0
			index = None
			i = 0
			for other in others:
				if self.f1(other) > overlap:
					if check_frame == True:
						if self.gv6 == other.gv6:
							overlap = self.f1(other)
							index = i
					else:
						overlap = self.f1(other)
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

class Screen(wx.Panel):
	"""Display the active microscopy frame inside the desktop application."""

	def __init__(self, parent, path, size = (1000, 800), pos = (222, 20)):
		"""Create the image panel and initialize it with the application logo."""
		self.size = Point(x = size[0], y = size[1])
		wx.Panel.__init__(self, parent, size = self.size.coor, pos = pos)
		self.parent = parent
		self.path = path
		self.ratios = (1, 1)
		cv.imwrite(self.path + "/logo.png", cv.resize(load_logo(), self.size.coor))
		self.imageCtrl = wx.StaticBitmap(self, wx.ID_ANY, wx.Bitmap(wx.Image(self.path + "/logo.png", wx.BITMAP_TYPE_ANY)))
		self.Layout()

	def display(self, img):
		"""Resize, cache, and display an OpenCV image in the panel."""
		img_path = self.path + "/img.png"
		cv.imwrite(img_path, cv.resize(img, self.size.coor))
		self.imageCtrl.SetBitmap(wx.Bitmap(wx.Image(img_path, wx.BITMAP_TYPE_ANY)))
		self.Refresh()

class Screen_Control(wx.Panel):
	"""Handle frame-canvas keyboard, pointer, and overlay interactions."""

	def __init__(self, panel, parent, path, size = (1000, 800), pos = (222, 20)):
		"""Create the transparent interaction layer over the image display."""
		wx.Panel.__init__(self, parent, size = size, pos = pos)
		self.parent = parent
		self.Bind(wx.EVT_MOTION, self.on_mouse_move)
		self.Bind(wx.EVT_LEFT_DOWN, self.on_mouse_click)
		self.Bind(wx.EVT_RIGHT_DOWN, self.on_mouse_click)
		self.Bind(wx.EVT_LEFT_UP, self.on_mouse_up)
		self.Bind(wx.EVT_PAINT, self.on_paint)
		self.Bind(wx.EVT_KEY_DOWN, self.on_keyboard)
		self.c2 = Point(x=0,y=0)
		self.SetCursor(wx.Cursor(wx.CURSOR_CROSS))
		self.frame = 0
		self.screen_size = size
		self.user_clicks = []
		self.Layout()

	def on_keyboard(self, e):
		"""Navigate frames with arrow keys and preserve canvas focus."""
		if e.GetKeyCode() == wx.WXK_UP:
			self.parent.change_frame(self.parent.move_xl)
		elif e.GetKeyCode() == wx.WXK_DOWN:
			self.parent.change_frame(-self.parent.move_xl)
		elif e.GetKeyCode() == wx.WXK_RIGHT:
			self.parent.change_frame(1)
		elif e.GetKeyCode() == wx.WXK_LEFT:
			self.parent.change_frame(-1)
		else:
			e.Skip()
		self.SetFocus()

	def on_mouse_click(self, e):
		"""Record manual edit points according to the selected QC operation."""
		self.SetFocus()
		pos = self.ScreenToClient(e.GetPosition())
		screen_pos = self.GetScreenPosition()
		if self.parent.chb3.GetValue() == True or self.parent.chb4.GetValue() == True or self.parent.chb5.GetValue() == True or self.parent.chb6.GetValue() == True or self.parent.chb8a.GetValue() == True:
			self.user_clicks.append(Point(x = int((pos[0] + screen_pos[0])), y = int((pos[1] + screen_pos[1])), frame = self.frame))
		if self.parent.chb8.GetValue() == True:
			pt = Point(x = int((pos[0] + screen_pos[0])), y = int((pos[1] + screen_pos[1])), frame = self.frame)
			pt.is_a = "grain"
			self.user_clicks.append(pt)
		if self.parent.chb7.GetValue() == True or self.parent.chb9.GetValue() == True:
			pt = Point(x = int((pos[0] + screen_pos[0])), y = int((pos[1] + screen_pos[1])), frame = self.frame)
			pt.is_a = "g"
			self.user_clicks.append(pt)

	def on_mouse_move(self, e):
		"""Update the crosshair position while the pointer moves."""
		self.SetFocus()
		self.c2 = e.GetPosition()
		self.Refresh()

	def on_mouse_up(self, e):
		"""Restore the normal cursor after a pointer interaction."""
		self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
		self.SetFocus()

	def on_paint(self, e):
		"""Draw crosshairs, tracks, tips, grains, and manual annotations."""
		dc = wx.PaintDC(self)
		dc.SetPen(wx.Pen('gray', 1))
		dc.SetBrush(wx.Brush("BLACK", wx.TRANSPARENT))
		dc.DrawLine(0, self.c2.y,18000, self.c2.y)
		dc.DrawLine(self.c2.x, 0, self.c2.x, 18000)
		if self.parent.chb14.GetValue() == True:
			if len(self.parent.tracker.valid_tracks) > 0:
				for track in self.parent.tracker.valid_tracks:
					if track.first_frame() <= self.frame:
						dc.SetPen(wx.Pen(track.color, 2))
						dc.DrawText(track.id, track.gv1[0].gv3.x, track.gv1[0].gv3.y)
						for i in range(1, track.gv2):
							if track.gv1[i].gv6 <= self.frame:
								dc.DrawLine(track.gv1[i-1].set_pos.x, track.gv1[i-1].set_pos.y, track.gv1[i].set_pos.x, track.gv1[i].set_pos.y)
							else:
								break
		if self.parent.chb13.GetValue() == True:
			if len(self.parent.tracker.valid_tips) > 0:
				for tip in self.parent.tracker.valid_tips[self.frame]:
					if tip.gv6 == self.frame:
						dc.SetPen(wx.Pen(tip.grain_color(), 2))
						dc.DrawRectangle(tip.gv1.x, tip.gv1.y, int(tip.w), int(tip.h))
			if len(self.user_clicks) > 0:
				dc.SetPen(wx.Pen('gray', 2))
				for click in self.user_clicks:
					if click.is_a == "point" and click.frame == self.frame:
						dc.DrawRectangle(int(click.x - self.parent.tracker.min_tip_side/2), int(click.y - self.parent.tracker.min_tip_side/2), self.parent.tracker.min_tip_side, self.parent.tracker.min_tip_side)
		if self.parent.chb15.GetValue() == True:
			if len(self.parent.tracker.valid_grains) > 0:
				for grain in self.parent.tracker.valid_grains:
					for roi in grain.gv1:
						if roi.gv6 == self.frame:
							dc.SetPen(wx.Pen(roi.grain_color(), 2))
							dc.DrawRectangle(roi.gv1.x, roi.gv1.y, int(roi.w), int(roi.h))
							dc.DrawText(grain.id, roi.gv1.x, roi.gv1.y)
					if grain.last_frame() < self.frame:
						dc.SetPen(wx.Pen(grain.gv1[-1].grain_color(), 2))
						dc.DrawRectangle(grain.gv1[-1].gv1.x, grain.gv1[-1].gv1.y, int(grain.gv1[-1].w), int(grain.gv1[-1].h))
						dc.DrawText(grain.id, grain.gv1[-1].gv1.x, grain.gv1[-1].gv1.y)
			if len(self.user_clicks) > 0:
				dc.SetPen(wx.Pen('gray', 2))
				for click in self.user_clicks:
					if click.is_a == "grain" and click.frame <= self.frame:
						dc.DrawRectangle(int(click.x - (self.parent.tracker.max_grain_radius + self.parent.tracker.min_grain_radius)/2), int(click.y - (self.parent.tracker.max_grain_radius + self.parent.tracker.min_grain_radius)/2), int((self.parent.tracker.max_grain_radius + self.parent.tracker.min_grain_radius)), int((self.parent.tracker.max_grain_radius + self.parent.tracker.min_grain_radius)))
					if click.is_a == "g" and click.frame == self.frame:
						dc.DrawRectangle(int(click.x - (self.parent.tracker.max_grain_radius + self.parent.tracker.min_grain_radius)/4), int(click.y - (self.parent.tracker.max_grain_radius + self.parent.tracker.min_grain_radius)/4), int((self.parent.tracker.max_grain_radius + self.parent.tracker.min_grain_radius)/2), int((self.parent.tracker.max_grain_radius + self.parent.tracker.min_grain_radius)/2))
		self.SetFocus()

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
		self.f2(boxes)
		if fill_holes == True:
			self.fill_missing_frames()
		self.gv4 = ["track." + self.id + " length ()"]
		self.gv5 = ["track." + self.id + " length ()"]

	def f1(self, box):
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

	def f2(self, boxes):
		"""Insert each supplied ROI into the trajectory."""
		for box in boxes:
			self.f1(box)

	def f3(self, coef = Point(x=1,y=1), disp = 2.00, disp_u = "um", num_frames = 300, time_p_frame = 60):
		"""Calculate calibrated coordinates and cumulative movement over time."""
		if self.gv2 > 1:
			self.gv3 = [[self.id, self.gv1[0].gv6, int(self.gv1[0].gv3.x/coef.x), int(self.gv1[0].gv3.y/coef.y), self.gv1[0].gv6*time_p_frame, 0, 0, self.gv1[0].detection_method]]
			length = 0
			for i in range(1,self.gv2):
				box = self.gv1[i]
				box1 = self.gv1[i-1]
				length += self.f6(box1, box, img_ratio = coef)
				self.gv3.append([self.id, box.gv6, int(box.gv3.x/coef.x), int(box.gv3.y/coef.y), box.gv6*time_p_frame, length, length*disp, box.detection_method])
			length = 0
			times = 0
			gv4 = [[times, length]]
			gv5 = [[self.gv1[0].gv6, length]]
			for i in range(1,self.gv2):
				box1 = self.gv1[i-1]
				box2 = self.gv1[i]
				length += self.f6(box1, box2, img_ratio = coef)*disp
				times = int(box2.gv6 - self.gv1[0].gv6)
				gv4.append([times, length])
				gv5.append([box2.gv6, length])
			self.gv4 = ["track." + self.id + " length (" + disp_u + ")"]
			self.gv5 = ["track." + self.id + " length (" + disp_u + ")"]
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

	def f4(self, box, keep_leftover = False):
		"""Truncate the trajectory near an ROI and optionally return its tail."""
		distance = 1000000
		index = None
		min_length = 2
		if len(self.gv1) > 2*min_length:
			i = 0
			for bx in self.gv1:
				if self.f6(box, bx) < distance:
					distance = self.f6(box, bx)
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

	def f5(self):
		"""Return endpoint displacement divided by total trajectory distance."""
		abs_len = 0
		for i in range(1, self.gv2):
			abs_len += self.f6(self.gv1[i-1], self.gv1[i])
		if abs_len == 0:
			return 0
		else:
			return self.f6(self.gv1[0], self.gv1[self.gv2-1])/abs_len

	def f6(self, box1, box2, img_ratio = None):
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
					boxes.append(ROI(x_l = int(box1.gv1.x + df*mxl), y_t = int(box1.gv1.y + df*myt), x_r = int(box1.gv2.x + df*mxr), y_b = int(box1.gv2.y + df*myb), frame = frame, group = box1.group, filled_in = True))
		self.f2(boxes)
		if up_to_frame > self.last_frame():
			b = []
			for i in range(self.last_frame(), up_to_frame, 1):
				b.append(ROI(x_l = self.gv1[-1].gv1.x, y_t = self.gv1[-1].gv1.y, x_r = self.gv1[-1].gv2.x, y_b = self.gv1[-1].gv2.y, frame = i, group = self.gv1[-1].group, filled_in = True))
			if len(b) > 0:
				self.f2(b)
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
		self.flaten_gap = 4
		self.aceptance_ratio = 0.5
		self.ger_confirm_frames = 8
		self.burst_candidates = []
		self.enable_burst_candidates = False
		self.burst_candidates_evaluated = False
		self.burst_score_threshold = 0.5
		self.burst_end_margin = 3
		self.burst_area_spike_ratio = 1.6

	def detections_generator(self, data_dir = ""):
		"""Yield Motpy detections grouped by frame from a bounding-box CSV."""
		if not os.path.isfile(os.path.expanduser(data_dir)):
			raise ValueError('file '+ data_dir + ' does not exist')
		df = pd.read_csv(os.path.expanduser(data_dir), names=['frame', 'x_l', 'y_t', 'x_r', 'y_b'])
		if df.empty:
			return
		for frame in range(df.frame.max() + 1):
			detections = []
			for _, row in df[df.frame == frame].iterrows():
				detections.append(Detection(box = [row.x_l, row.y_t, row.x_r, row.y_b]))
			yield frame, detections

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
					self.f12(grain.id, what = "g")
					self.valid_grains.remove(grain)

	def f9(self, genre = None):
		"""Generate a random BGR display color, optionally restricted to light tones."""
		if genre == None:
			return (randint(0, 255),randint(0, 255),randint(0, 255))
		else:
			return (randint(100, 255),randint(100, 255),randint(100, 255))

	def f10(self, what = "t"):
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

	def f12(self, ID, what = "t"):
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

	def f17(self, filename, img_list, img_per_sec = 25):
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
		grain_dir = temp_dir + '/pot_grains.csv'
		all_grains = []
		voys = 0
		with open(grain_dir, 'w', newline = '') as f:
			writer = csv.writer(f)
			while(True):
				k += 1
				if k >= self.all_detections.img_list_length:
					break
				gr_tr = int(self.grain_tresh)
				circles = cv.HoughCircles(self.all_detections.img_list_gray_noiseless[k], cv.HOUGH_GRADIENT, 1, int(0.75*(self.max_grain_radius+self.min_grain_radius)), param1=210, param2 = gr_tr, minRadius = int(self.min_grain_radius), maxRadius = int(self.max_grain_radius))
				if circles is None:
					continue
				grains = np.uint16(np.around(circles))
				for i in grains[0, :]:
					grain = ROI(x_l = int(i[0] - i[2]), y_t = int(i[1] - i[2]), x_r = int(i[0] + i[2]), y_b = int(i[1] + i[2]), frame = k, group = str(voys))
					voys += 1
					if grain.f3(self.all_detections.rois[k], overlap = overlap):
						if k <= self.grain_det_stop:
							if k > self.grain_det_start:
								if grain.f3(all_grains, overlap = overlap):
									p_g = grain.f4(all_grains)
									kk = -1
									for g in all_grains:
										kk+=1
										if g.gv6 == p_g.gv6:
											if g.group == p_g.group:
												all_grains.pop(kk) 
												break
							writer.writerow([grain.gv6, grain.gv1.x, grain.gv1.y, grain.gv2.x, grain.gv2.y])
							all_grains.append(grain)
						else:
							if grain.f3(all_grains, overlap = overlap):
								writer.writerow([grain.gv6, grain.gv1.x, grain.gv1.y, grain.gv2.x, grain.gv2.y])
								all_grains.append(grain)
								p_g = grain.f4(all_grains)
								kk = -1
								for g in all_grains:
									kk+=1
									if g.gv6 == p_g.gv6:
										if g.group == p_g.group:
											break
		dets_gen = self.detections_generator(data_dir = grain_dir)
		tracker = MultiObjectTracker(dt = 0.1, tracker_kwargs = {'max_staleness': 5}, model_spec = 'constant_velocity_and_static_box_size_2d', matching_fn_kwargs = {'min_iou': 0.2}, active_tracks_kwargs = {'min_steps_alive': 1, 'max_staleness': 5})
		grain_tracks = []
		trk_ids = []
		self.gv8 = 1
		while(True):
			try:
				frame, detections = next(dets_gen)
			except Exception as e:
				break
			active_tracks = tracker.step(detections)
			if len(active_tracks) > 0:
				for track in active_tracks:
					if track.id not in trk_ids:
						trk_ids.append(track.id)
					grain_tracks.append(ROI(x_l = int(track.box[0]), y_t = int(track.box[1]), x_r = int(track.box[2]), y_b = int(track.box[3]), frame = frame, group = track.id))
		for trk_id in trk_ids:
			cur_g = []
			frame1 = self.grain_det_stop + 1
			for roi in grain_tracks:
				if roi.group == trk_id:
					cur_g.append(roi)
					if roi.gv6 < frame1:
						frame1 = roi.gv6
			if frame1 <= self.grain_det_stop and len(cur_g) > self.grain_det_stop - self.grain_det_start:
				grain = Track(boxes = cur_g, color = self.f9(), ID = self.f10("g"))
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
						tip_obj = tip.f4(rois)
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
											if not tip.f3(tips_in_k, overlap = overlap):
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
						if not tip.f3(c_tips, overlap = overlap):
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
				track.f3(coef = self.img_rp, disp = self.pxl_dis, disp_u = self.dis_unit, num_frames = len(self.file_names), time_p_frame = self.time_p_frame)
			name = save_dir + "tip.tracks.avi"
			self.f17(filename = name, img_list = self.draw_results())
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
			self.gv38 = [["track_id", "frame", "centroid_x", "centroid_y", "time (" + self.time_unit + ")",  "length (pixel)", "length (" + self.dis_unit + ")", "detection_method"]]
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
			self.f17(filename = name, img_list = self.draw_results(tracks = False))
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
			"cumulative_movement_px",
			"cumulative_movement_" + self.dis_unit,
			"growth_rate_" + self.dis_unit + "_per_" + self.time_unit,
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
			track.f3(
				coef=self.img_rp,
				disp=self.pxl_dis,
				disp_u=self.dis_unit,
				num_frames=len(self.file_names),
				time_p_frame=self.time_p_frame,
			)
			previous_time = None
			previous_length = None
			for point in track.gv3:
				growth_rate = ""
				if previous_time is not None and point[4] > previous_time:
					growth_rate = (point[6] - previous_length)/(point[4] - previous_time)
				rows.append([
					track_to_grain.get(track.id, ""),
					point[0],
					point[1],
					point[4],
					point[2],
					point[3],
					point[5],
					point[6],
					growth_rate,
					point[7],
				])
				previous_time = point[4]
				previous_length = point[6]
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
		track_threshold = 1
		if self.flaten_gap >= self.min_tip_per_trk:
			flaten_gap = self.min_tip_per_trk-1
		else:
			flaten_gap = self.flaten_gap
		if len(self.valid_tips) > 0:
			tmp = []
			tmp_tips = []
			for tips_in_k in self.valid_tips:
				for tip in tips_in_k:
					if tip.w >= self.min_tip_side and tip.h >= self.min_tip_side:
						tmp.append(tip.w)
						tmp.append(tip.h)
			if len(tmp) == 0:
				return
			rd = stat.mean(tmp)/2
			tip_path = temp_dir + "/tips.csv"
			with open(tip_path, 'w', newline = '') as f:
				writer = csv.writer(f)
				for tips_in_k in self.valid_tips:
					for tip in tips_in_k:
						writer.writerow([tip.gv6, tip.gv1.x, tip.gv1.y, tip.gv2.x, tip.gv2.y])
			dets_gen = self.detections_generator(data_dir = tip_path)	
			mot = MultiObjectTracker(dt = 1, tracker_kwargs = {'max_staleness': int(2*self.tip_gap_closing)}, model_spec = 'constant_velocity_and_static_box_size_2d', matching_fn_kwargs = {'min_iou': self.tip_max_step}, active_tracks_kwargs = {'min_steps_alive': track_threshold, 'max_staleness': self.tip_gap_closing})
			trk_ids = []
			csv_path = temp_dir + '/trks.csv'
			with open(csv_path, 'w', newline = '') as f:
				writer = csv.writer(f)
				while(True):
					try:
						frame, detections = next(dets_gen)
					except Exception as e:
						break
					active_tracks = mot.step(detections)
					if len(active_tracks) > 0:
						for track in active_tracks:
							if track.id not in trk_ids:
								trk_ids.append(track.id)
							writer.writerow([track.id, int(track.box[0]), int(track.box[1]), int(track.box[2]), int(track.box[3]), frame])
			df = pd.read_csv(csv_path, names=['id', "x_l", "y_t", "x_r", "y_b", "frame"])
			for track_id in trk_ids:
				cur_trk = []
				for _, row in df[df.id == track_id].iterrows():
					tip = ROI(x_l = int(row.x_l), y_t = int(row.y_t), x_r = int(row.x_r), y_b = int(row.y_b), frame = row.frame)
					if tip.f3(self.valid_tips[tip.gv6], overlap = 0.50):
						cur_trk.append(tip)
				if len(cur_trk)>1:
					trk = Track(boxes = cur_trk, color = self.f9(), ID = self.f10())
					xtr_tips = trk.fill_missing_frames(ret = True)
					if trk.gv2 >= self.min_tip_per_trk or trk.last_frame() - trk.first_frame() >= self.min_tip_per_trk:
						if trk.f5() >= disp_ratio:
							if flaten_gap > 1:
								k=0
								cr = []
								while(True):
									if k >= trk.gv2:
										if k - flaten_gap < trk.gv2 -1:
											cr.append(trk.gv1[trk.gv2 -1])
										break
									cr.append(trk.gv1[k])
									k+= flaten_gap
								self.valid_tracks.append(Track(boxes = cr, color = self.f9(), ID = trk.id, fill_holes = True))
							else:
								self.valid_tracks.append(trk)
							if len(xtr_tips) > 0:
								for tip in xtr_tips:
									self.filled_in_tips.append(tip)
						else:
							self.f12(trk.id)
					else:
						self.f12(trk.id)

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
						if tip.f3(grains_in_k):
							g = tip.f4(grains_in_k)
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
											if t.f3(tips[i]):
												c+=1
												disp += t.distance_btw(t.gv3, t.f4(tips[i]).gv3)
												t = t.f4(tips[i])
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
													if t.f3(tips[i]):
														t.is_used = True
														t = t.f4(tips[i])

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
						r1 = grain.roi_closest_to(grain.first_frame()).f4(self.all_detections.rois[grain.first_frame()])
						r2 = grain.roi_closest_to(fr).f4(rois_in_fr)
						if r1 != None and r2 != None:
							if r2.gv4/r1.gv4 <= sz_co:
								p_val = len(null_dist[null_dist >= r2.gv4/r1.gv4])/self.gv21
								if p_val <= self.gv5:
									c = 0
									for i in range(fr + 1, fr + cutoff + 1):
										if i >= self.all_detections.img_list_length:
											break
										r3 = grain.roi_closest_to(i).f4(self.all_detections.rois[i])
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
				if grain_roi.f1(track.gv1[i]) > 0 or track.gv1[i].f1(grain_roi) > 0:
					overlaps = True
					break
			d = grain.f6(grain_roi, t0)
			score = d - 1000000 if overlaps else d
			if best_score is None or score < best_score:
				best_score = score
				best = track
		if best is not None:
			if grain.f6(grain_roi, best.gv1[0]) > max_dist:
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

class Tracker_GUI(wx.Frame):
	"""Provide the desktop workflow for loading, reviewing, and exporting analyses."""

	def __init__(self,title = title):
		"""Construct controls, analysis state, menus, canvas, and review panels."""
		wx.Frame.__init__(self, None, title=title)
		self.first_use = True
		self.gv1 = wx.GetClientDisplayRect().GetSize()
		self.tracker = Tracker(screen_size = self.f47("tt4"))
		self.input_ext = ".avi"
		self.ids_to_process = ''
		self.img_to_display = 0
		self.move_xl = 5
		self.number_of_uses = 0
		self.screen_size = self.f47("tt4")
		self.save_name = 'Enter output name'
		self.output_directory = None
		self.panel_lt = wx.Panel(self, pos=self.f45("pnl1"), size=self.f47("pnl1"))
		wx.StaticBox(self.panel_lt, label='Image Parameters', pos=self.f45("ly1"), size=self.f47("ly1"))
		self.rotate_input = wx.CheckBox(self.panel_lt, label = "Rotate images", pos = self.f45("cb_b1"))
		b1 = wx.Button(self.panel_lt, label = "Choose Folder", size = self.f47("b1"), pos = self.f45("b1"))
		b1.Bind(wx.EVT_BUTTON, self.on_data_directory)
		b2 = wx.Button(self.panel_lt, label = "Save Results", size = self.f47("b2"), pos = self.f45("b2"))
		b2.Bind(wx.EVT_BUTTON, self.on_save)
		self.suported_file_ext = ['.avi', ".mp4",'.png', '.tiff', '.tif', '.jpeg', '.jpg']
		self.cb1 = wx.ComboBox(self.panel_lt, choices=self.suported_file_ext, size = self.f47("cb1"), pos = self.f45("cb1"))
		self.cb1.Bind(wx.EVT_COMBOBOX, self.on_cb1)
		self.cb2 = wx.ComboBox(self.panel_lt, choices=['sec', 'min', 'hour', "day", "week", "month", "year"], size = self.f47("cb2"), pos = self.f45("cb2"))
		self.cb2.Bind(wx.EVT_COMBOBOX, self.on_time_unit)
		self.cb3 = wx.ComboBox(self.panel_lt, choices=['um', 'nm','mm', 'cm','m','in','ft','pxl'], size = self.f47("cb3"), pos = self.f45("cb3"))
		self.cb3.Bind(wx.EVT_COMBOBOX, self.on_dis_unit)
		self.tc3 = wx.TextCtrl(self.panel_lt, value = str(self.tracker.pxl_dis), size = self.f47("tc"), pos = self.f45("tc3"))
		self.tc3.Bind(wx.EVT_TEXT, self.on_pxl_dis)
		self.tc2 = wx.TextCtrl(self.panel_lt, value = str(self.tracker.time_p_frame), size = self.f47("tc"), pos = self.f45("tc2"))
		self.tc2.Bind(wx.EVT_TEXT, self.on_time_p_frame)
		self.tc4 = wx.TextCtrl(self.panel_lt, value = self.save_name, size = self.f47("tc4"), pos = self.f45("tc4"))
		self.tc4.Bind(wx.EVT_TEXT, self.on_save_name)
		wx.StaticText(self.panel_lt, label="Blur radius:", style=wx.ALIGN_LEFT, pos = self.f45("st1sa"))
		self.smouthing_radius = wx.Slider(self.panel_lt, value=int(self.tracker.filter_radius), minValue=1, maxValue=20, style=wx.SL_HORIZONTAL, pos=self.f45("lt_sl2"), size=self.f47("br_sl1"))
		self.smouthing_radius.Bind(wx.EVT_SLIDER, self.on_filter_radius)
		self.smouthing_radius_lab = wx.StaticText(self.panel_lt, label=str(self.tracker.filter_radius), style=wx.ALIGN_LEFT, pos = self.f45("st1sal"))
		wx.StaticText(self.panel_lt, label="File type:", style=wx.ALIGN_LEFT, pos = self.f45("st1"))
		wx.StaticText(self.panel_lt, label='''Time per frame:''', style=wx.ALIGN_LEFT, pos = self.f45("st9"))
		wx.StaticText(self.panel_lt, label="1 pixel:", style=wx.ALIGN_LEFT, pos = self.f45("st10"))
		cv.imwrite(temp_dir + "/heatmap.png", cv.resize(self.get_heatmap(), self.f47("heatmap")))
		self.heatmap = wx.StaticBitmap(self.panel_lt, wx.ID_ANY, wx.Bitmap(wx.Image(temp_dir + "/heatmap.png", wx.BITMAP_TYPE_ANY)), size = self.f47("heatmap"), pos = self.f45("heatmap"))
		self.min_tresh_bar = wx.Slider(self.panel_lt, value=self.tracker.bg_threshold, minValue=1, maxValue=255, style=wx.SL_HORIZONTAL, pos=self.f45("sl_heatmap"), size=self.f47("heatmap"))
		self.min_tresh_bar.Bind(wx.EVT_SLIDER, self.on_min_threshold)#
		wx.StaticText(self.panel_lt, label="Background cutoff:", style=wx.ALIGN_LEFT, pos = self.f45("st1_heatmap"))
		self.st_bg_cutoff = wx.StaticText(self.panel_lt, label= str(self.tracker.bg_threshold), style=wx.ALIGN_LEFT, pos = self.f45("st2_heatmap"))
		wx.StaticText(self.panel_lt, label="Apply to: ", style=wx.ALIGN_LEFT, pos = self.f45("st_apply_bg"))
		b4a = wx.Button(self.panel_lt, label = "All", size = self.f47("b4"), pos = self.f45("b4a"))
		b4a.Bind(wx.EVT_BUTTON, self.on_update_all_bg)
		b4b = wx.Button(self.panel_lt, label = "Frame", size = self.f47("b4"), pos = self.f45("b4b"))
		b4b.Bind(wx.EVT_BUTTON, self.on_update_frame_bg)
		self.panel_lb = wx.Panel(self, pos=self.f45("pnl2"), size=self.f47("pnl2"))
		wx.StaticBox(self.panel_lb, label='QC and Manual Tracking', pos=self.f45("ly2"), size=self.f47("ly2"))
		b11 = wx.Button(self.panel_lb, label = "Apply Edit", size = self.f47("b11"), pos = self.f45("b11"))
		b11.Bind(wx.EVT_BUTTON, self.on_manual_update)
		self.st11 = wx.StaticText(self.panel_lb, label=self.f29(16), pos = self.f45("st11"))
		self.cb_id = 0
		labels = [["Remove track", self.f45("chb1")], ["Link tracks", self.f45("chb2")], ["Extend track", self.f45("chb3")], ["Add track", self.f45("chb4")], ["Split track", self.f45("chb5")], ["Truncate track", self.f45("chb6")], ["Add tip", self.f45("chb7")], ["Add grain", self.f45("chb8")], ["Add germ frame",self.f45("chb9")], ["Add burst frame",self.f45("chb9a")]]
		for label in labels:
			self.make_checkbox(self.panel_lb, label[0], label[1], self.on_manual_checkboxes, self.cb_id)
			self.cb_id+=1
		self.chb1 = self.FindWindowByLabel("Remove track")
		self.chb2 = self.FindWindowByLabel("Link tracks")
		self.chb3 = self.FindWindowByLabel("Extend track")
		self.chb4 = self.FindWindowByLabel("Add track")
		self.chb5 = self.FindWindowByLabel("Split track")
		self.chb6 = self.FindWindowByLabel("Truncate track")
		self.chb7 = self.FindWindowByLabel("Add germ frame")
		self.chb8 = self.FindWindowByLabel("Add grain")
		self.chb9 = self.FindWindowByLabel("Add burst frame")
		self.chb8a = self.FindWindowByLabel("Add tip")
		self.tc10 = wx.TextCtrl(self.panel_lb, value=self.ids_to_process, size = self.f47("tc10"), pos = self.f45("tc10"))
		self.tc10.Bind(wx.EVT_TEXT, self.on_ids_to_process)
		self.panel_rt = wx.Panel(self, pos=self.f45("pnl5"), size=self.f47("pnl5"))
		wx.StaticBox(self.panel_rt, label='Automated Pollen Detection', pos=self.f45("ly4"), size=self.f47("ly4"))
		wx.StaticText(self.panel_rt, label="Grain Detection", style=wx.ALIGN_LEFT, pos = self.f45("st13"))
		wx.StaticText(self.panel_rt, label="Detection frames:", style=wx.ALIGN_LEFT, pos = self.f45("st14"))
		wx.StaticText(self.panel_rt, label="-", style=wx.ALIGN_LEFT, pos = self.f45("st15"))
		self.tc5 = wx.TextCtrl(self.panel_rt, value = str(self.tracker.grain_det_start + 1), size = self.f47("tc"), pos = self.f45("tc5"))
		self.tc5.Bind(wx.EVT_TEXT, self.on_grain_det_start)
		self.tc6 = wx.TextCtrl(self.panel_rt, value=str(self.tracker.grain_det_stop + 1), size = self.f47("tc"), pos = self.f45("tc6"))
		self.tc6.Bind(wx.EVT_TEXT, self.on_grain_det_stop)
		wx.StaticText(self.panel_rt, label="≤ radius ≤", style=wx.ALIGN_LEFT, pos = self.f45("st17"))
		self.tc7 = wx.TextCtrl(self.panel_rt, value=str(int(self.tracker.min_grain_radius/self.tracker.img_ratio)), size = self.f47("tc"), pos = self.f45("tc7"))
		self.tc7.Bind(wx.EVT_TEXT, self.on_min_grain_radius)
		self.tc8 = wx.TextCtrl(self.panel_rt, value=str(int(self.tracker.max_grain_radius/self.tracker.img_ratio)), size = self.f47("tc"), pos = self.f45("tc8"))
		self.tc8.Bind(wx.EVT_TEXT, self.on_max_grain_radius)
		wx.StaticText(self.panel_rt, label="Threshold:", style=wx.ALIGN_LEFT, pos = self.f45("rt_st6"))
		self.grain_det_tresh_lab = wx.StaticText(self.panel_rt, label= str(self.tracker.grain_tresh), style=wx.ALIGN_LEFT, pos = self.f45("rt_st6l"))
		self.grain_det_tresh = wx.Slider(self.panel_rt, value=int(self.tracker.grain_tresh), minValue=10, maxValue=40, style=wx.SL_HORIZONTAL, pos=self.f45("rt_sl2"), size=self.f47("br_sl1"))
		self.grain_det_tresh.Bind(wx.EVT_SLIDER, self.on_grain_det_threshold)
		b3 = wx.Button(self.panel_rt, label = "Find Grains", size = self.f47("b3"), pos = self.f45("b3"))
		b3.Bind(wx.EVT_BUTTON, self.on_find_grains)
		wx.StaticText(self.panel_rt, label="Tip Detection", style=wx.ALIGN_LEFT, pos = self.f45("st20"))
		wx.StaticText(self.panel_rt, label="Match:", style=wx.ALIGN_LEFT, pos = self.f45("st23"))
		wx.StaticText(self.panel_rt, label="Segment edge", style=wx.ALIGN_LEFT, pos = self.f45("st23a"))
		wx.StaticText(self.panel_rt, label="Template", style=wx.ALIGN_LEFT, pos = self.f45("st23b"))
		self.tip_det_mthd = wx.Slider(self.panel_rt, value=1, minValue=1, maxValue=2, style=wx.SL_HORIZONTAL, pos=self.f45("st23c"), size=self.f47("rt_sl1a"))
		self.tip_tresh_det_lab = wx.StaticText(self.panel_rt, label= str(int(self.tracker.tip_det_threshold_percent*100)), style=wx.ALIGN_LEFT, pos = self.f45("br_st2"))
		self.tip_tresh_det = wx.Slider(self.panel_rt, value=int(self.tracker.tip_det_threshold_percent*100), minValue=1, maxValue=100, style=wx.SL_HORIZONTAL, pos=self.f45("br_sl2"), size=self.f47("br_sl1"))
		self.tip_tresh_det.Bind(wx.EVT_SLIDER, self.on_tip_det_thresh)
		wx.StaticText(self.panel_rt, label="Minimum tip size:", style=wx.ALIGN_LEFT, pos = self.f45("st22"))
		self.tc15a = wx.TextCtrl(self.panel_rt, value=str(self.tracker.min_tip_side), size = self.f47("tc"), pos = self.f45("tc15a"))
		self.tc15a.Bind(wx.EVT_TEXT, self.on_min_tip_sidelength)
		b5a = wx.Button(self.panel_rt, label = "Find Tips", size = self.f47("b3"), pos = self.f45("b5a"))
		b5a.Bind(wx.EVT_BUTTON, self.on_find_tips)
		self.panel_rb = wx.Panel(self, pos=self.f45("pnl6"), size=self.f47("pnl6"))
		wx.StaticBox(self.panel_rb, label='Automated Tip Tracking', pos=self.f45("ly5a"), size=self.f47("ly5a"))
		wx.StaticBox(self.panel_rb, label='Automated Germination Tracking', pos=self.f45("ly5b"), size=self.f47("ly5b"))
		wx.StaticText(self.panel_rb, label="Germination p-value:", style=wx.ALIGN_LEFT, pos = self.f45("st19"))
		self.tc11 = wx.TextCtrl(self.panel_rb, value=str(self.tracker.gv5), size = self.f47("tc11"), pos = self.f45("tc11"))
		self.tc11.Bind(wx.EVT_TEXT, self.f20)
		wx.StaticText(self.panel_rb, label="Tip overlap", style=wx.ALIGN_LEFT, pos = self.f45("st23d"))
		wx.StaticText(self.panel_rb, label="Area change", style=wx.ALIGN_LEFT, pos = self.f45("st23e"))
		self.ger_track_mthd = wx.Slider(self.panel_rb, value=1, minValue=1, maxValue=2, style=wx.SL_HORIZONTAL, pos=self.f45("st23f"), size=self.f47("rt_sl1a"))
		wx.StaticText(self.panel_rb, label="Confirmation frames:", style=wx.ALIGN_LEFT, pos = self.f45("st19x"))
		self.tc11x = wx.TextCtrl(self.panel_rb, value=str(self.tracker.ger_confirm_frames), size = self.f47("tc16"), pos = self.f45("tc11x"))
		self.tc11x.Bind(wx.EVT_TEXT, self.on_ger_confirm_frames)
		wx.StaticText(self.panel_rb, label="Acceptance ratio:", style=wx.ALIGN_LEFT, pos = self.f45("st19y"))
		self.tc11y_lab = wx.StaticText(self.panel_rb, label=str(self.tracker.aceptance_ratio), style=wx.ALIGN_LEFT, pos = self.f45("tc11yl"))
		self.tc11y = wx.Slider(self.panel_rb, value=int(100*self.tracker.aceptance_ratio), minValue=10, maxValue=100, style=wx.SL_HORIZONTAL, pos=self.f45("tc11y"), size=self.f47("tc11y"))
		self.tc11y.Bind(wx.EVT_SLIDER, self.on_aceptance_ratio)
		b4 = wx.Button(self.panel_rb, label = "Track Germination", size = self.f47("b5"), pos = self.f45("b4"))
		b4.Bind(wx.EVT_BUTTON, self.on_track_germination)
		wx.StaticText(self.panel_rb, label="Gap closing:", style=wx.ALIGN_LEFT, pos = self.f45("st25"))
		self.tc15 = wx.TextCtrl(self.panel_rb, value=str(self.tracker.tip_gap_closing), size = self.f47("tc15"), pos = self.f45("tc15"))
		self.tc15.Bind(wx.EVT_TEXT, self.on_gap_closing)
		wx.StaticText(self.panel_rb, label="Min. points/track:", style=wx.ALIGN_LEFT, pos = self.f45("st26"))
		self.tc16 = wx.TextCtrl(self.panel_rb, value=str(self.tracker.min_tip_per_trk), size = self.f47("tc16"), pos = self.f45("tc16"))
		self.tc16.Bind(wx.EVT_TEXT, self.on_min_tip_per_trk)
		wx.StaticText(self.panel_rb, label="Min. overlap:", style=wx.ALIGN_LEFT, pos = self.f45("st27"))
		self.tc17_lab = wx.StaticText(self.panel_rb, label=str(self.tracker.tip_max_step), style=wx.ALIGN_LEFT, pos = self.f45("tc17l"))
		self.tc17 = wx.Slider(self.panel_rb, value=int(100*self.tracker.tip_max_step), minValue=1, maxValue=100, style=wx.SL_HORIZONTAL, pos=self.f45("tc17"), size=self.f47("tc11y"))
		self.tc17.Bind(wx.EVT_SLIDER, self.on_tip_max_step)
		wx.StaticText(self.panel_rb, label="Smoothing gap:", style=wx.ALIGN_LEFT, pos = self.f45("st27x"))
		self.tc17x = wx.TextCtrl(self.panel_rb, value=str(self.tracker.flaten_gap), size = self.f47("tc17"), pos = self.f45("tc17x"))
		self.tc17x.Bind(wx.EVT_TEXT, self.on_flaten_gap)
		b5 = wx.Button(self.panel_rb, label = "Track Tips", size = self.f47("b5"), pos = self.f45("b5"))
		b5.Bind(wx.EVT_BUTTON, self.on_track_elongation)
		self.panel_mb = wx.Panel(self, pos=self.f45("pnl4"), size=self.f47("pnl4"))
		b6 = wx.Button(self.panel_mb, label = "Go to Frame", size = self.f47("b6"), pos = self.f45("b6"))
		b6.Bind(wx.EVT_BUTTON, self.on_bt_show)
		b7 = wx.Button(self.panel_mb, label = "", size = self.f47("b7"), pos = self.f45("b7"))
		b7.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_GO_BACK, wx.ART_BUTTON, (16, 16)))
		b7.SetToolTip("Previous frame")
		b7.Bind(wx.EVT_BUTTON, self.on_reverse)
		b8 = wx.Button(self.panel_mb, label = "-5", size = self.f47("b8"), pos = self.f45("b8"))
		b8.SetToolTip("Back 5 frames")
		b8.Bind(wx.EVT_BUTTON, self.on_reverse_xl)
		b9 = wx.Button(self.panel_mb, label = "", size = self.f47("b9"), pos = self.f45("b9"))
		b9.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_GO_FORWARD, wx.ART_BUTTON, (16, 16)))
		b9.SetToolTip("Next frame")
		b9.Bind(wx.EVT_BUTTON, self.on_forward)
		b10 = wx.Button(self.panel_mb, label = "+5", size = self.f47("b10"), pos = self.f45("b10"))
		b10.SetToolTip("Forward 5 frames")
		b10.Bind(wx.EVT_BUTTON, self.on_forward_xl)
		self.tc1 = wx.TextCtrl(self.panel_mb, value = "1", size = self.f47("tc1"), pos = self.f45("tc1"))
		self.st12 = wx.StaticText(self.panel_mb, label="1/1", style=wx.ALIGN_LEFT, pos = self.f45("mb_st2"))
		wx.StaticText(self.panel_mb, label="Overlays:", style=wx.ALIGN_LEFT, pos = self.f45("mb_st3"))
		labels = [["Heatmap", self.f45("chb11")], ["Binary", self.f45("chb10")], ["Tips", self.f45("chb13")], ["Tracks", self.f45("chb14")], ["Grain status", self.f45("chb15")]]
		for label in labels:
			self.make_checkbox(self.panel_mb, label[0], label[1], self.on_what_to_display, self.cb_id)
			self.cb_id+=1
		self.chb10 = self.FindWindowByLabel("Binary")
		self.chb11 = self.FindWindowByLabel("Heatmap")
		self.chb13 = self.FindWindowByLabel("Tips")
		self.chb14 = self.FindWindowByLabel("Tracks")
		self.chb15 = self.FindWindowByLabel("Grain status")
		self.panel_mt = wx.Panel(self, pos=self.f45("pnl3"), size=self.f47("pnl3"))
		wx.StaticBox(self.panel_mt, label='Display', pos=self.f45("ly3"), size=self.f47("ly3"))
		self.screen = Screen(self.panel_mt,temp_dir, size = self.f47("tt4"), pos = self.f45("tt4"))
		self.screen_control = Screen_Control(self.panel_mt, parent = self, path = temp_dir, size = self.f47("tt4"), pos = (self.f45("tt4")[0] + self.f45("pnl3")[0], self.f45("tt4")[1] + self.f45("pnl3")[1]))
		self.gv2 = self.tracker.gv5
		self.panel_rbb = wx.Panel(self, pos=self.f45("pnl7"), size=self.f47("pnl7"))
		wx.StaticBox(self.panel_rbb, label='Burst Review', pos=self.f45("ly6"), size=self.f47("ly6"))
		self.burst_candidate_records = []
		burst_find = wx.Button(self.panel_rbb, label="Find Candidates", pos=self.f45("burst_find"), size=self.f47("burst_find"))
		burst_find.SetToolTip("Suggest possible ruptures after tracking germination")
		burst_find.Bind(wx.EVT_BUTTON, self.on_find_burst_candidates)
		self.burst_choice = wx.ComboBox(self.panel_rbb, choices=[], style=wx.CB_READONLY, pos=self.f45("burst_choice"), size=self.f47("burst_choice"))
		self.burst_choice.SetToolTip("Select a candidate to jump to its proposed rupture frame")
		self.burst_choice.Bind(wx.EVT_COMBOBOX, self.on_select_burst_candidate)
		self.burst_confirm = wx.Button(self.panel_rbb, label="Confirm", pos=self.f45("burst_confirm"), size=self.f47("burst_action"))
		self.burst_confirm.SetToolTip("Confirm the selected candidate as a reviewed rupture event")
		self.burst_confirm.Bind(wx.EVT_BUTTON, self.on_confirm_burst_candidate)
		self.burst_dismiss = wx.Button(self.panel_rbb, label="Dismiss", pos=self.f45("burst_dismiss"), size=self.f47("burst_action"))
		self.burst_dismiss.SetToolTip("Dismiss the selected candidate")
		self.burst_dismiss.Bind(wx.EVT_BUTTON, self.on_dismiss_burst_candidate)
		self.refresh_burst_candidates()
		self.build_menu_bar()
		self.CreateStatusBar(2)
		self.SetStatusWidths([-2, -3])
		self.update_output_status()
		self.apply_ui_style()
		self.panel_mb.Raise()
		self.Maximize(True)

	def build_menu_bar(self):
		"""Create file-export and session-recovery application menus."""
		menu_bar = wx.MenuBar()
		file_menu = wx.Menu()
		open_item = file_menu.Append(wx.ID_ANY, "Open Video or Images...")
		output_item = file_menu.Append(wx.ID_ANY, "Choose Output Folder...")
		file_menu.AppendSeparator()
		export_csv_item = file_menu.Append(wx.ID_ANY, "Export Coordinates CSV")
		save_item = file_menu.Append(wx.ID_ANY, "Save All Results")
		menu_bar.Append(file_menu, "File")

		session_menu = wx.Menu()
		reset_tracking_item = session_menu.Append(wx.ID_ANY, "Restart Tracking")
		restart_analysis_item = session_menu.Append(wx.ID_ANY, "Restart Analysis")
		start_over_item = session_menu.Append(wx.ID_ANY, "Start Over")
		menu_bar.Append(session_menu, "Session")
		self.SetMenuBar(menu_bar)

		self.Bind(wx.EVT_MENU, self.on_data_directory, open_item)
		self.Bind(wx.EVT_MENU, self.on_choose_output_directory, output_item)
		self.Bind(wx.EVT_MENU, self.on_export_coordinates, export_csv_item)
		self.Bind(wx.EVT_MENU, self.on_save, save_item)
		self.Bind(wx.EVT_MENU, self.on_reset_tracking, reset_tracking_item)
		self.Bind(wx.EVT_MENU, self.on_restart_analysis, restart_analysis_item)
		self.Bind(wx.EVT_MENU, self.on_start_over, start_over_item)

	def apply_ui_style(self):
		"""Apply a restrained visual hierarchy without changing the legacy layout."""
		self.SetBackgroundColour(UI_BACKGROUND)
		base_font = wx.Font(wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT))
		base_font.SetPointSize(10)
		heading_font = wx.Font(base_font)
		heading_font.SetWeight(wx.FONTWEIGHT_BOLD)
		section_labels = {"Grain Detection", "Tip Detection"}
		primary_actions = {
			"Apply Edit",
			"Find Candidates",
			"Find Grains",
			"Find Tips",
			"Save Results",
			"Track Germination",
			"Track Tips",
		}
		panels = (
			self.panel_lt,
			self.panel_lb,
			self.panel_mt,
			self.panel_mb,
			self.panel_rt,
			self.panel_rb,
			self.panel_rbb,
		)
		for panel in panels:
			panel.SetBackgroundColour(UI_PANEL)
			for child in panel.GetChildren():
				child.SetFont(base_font)
				if isinstance(child, wx.StaticBox):
					child.SetFont(heading_font)
					child.SetForegroundColour(UI_MUTED_TEXT)
				elif isinstance(child, wx.StaticText):
					child.SetForegroundColour(UI_TEXT)
					if child.GetLabel().strip() in section_labels:
						child.SetFont(heading_font)
						child.SetForegroundColour(UI_ACCENT)
				elif isinstance(child, (wx.TextCtrl, wx.ComboBox)):
					child.SetBackgroundColour(UI_FIELD)
					child.SetForegroundColour(UI_TEXT)
				elif isinstance(child, wx.Button):
					child.SetCursor(wx.Cursor(wx.CURSOR_HAND))
					if child.GetLabel() in primary_actions:
						child.SetFont(heading_font)
				else:
					child.SetForegroundColour(UI_TEXT)
		self.panel_mt.SetBackgroundColour(UI_CANVAS)
		self.Layout()

	def update_output_status(self, message = None):
		"""Show the persistent output folder and latest operation status."""
		if self.output_directory is None:
			self.SetStatusText("Output folder: not selected", 0)
		else:
			self.SetStatusText("Output folder: " + str(self.output_directory), 0)
		self.SetStatusText(message or "Ready", 1)

	def normalized_output_name(self):
		"""Return a safe nonempty base name for exported files."""
		name = self.save_name.strip()
		if name == "" or name == "Enter output name":
			name = "result"
		name = "".join(
			character if character.isalnum() or character in "-_." else "_"
			for character in name
		)
		return name.strip(".") or "result"

	def on_choose_output_directory(self, e):
		"""Prompt for and remember the session's export directory."""
		with wx.DirDialog(
			self,
			"Choose an output folder",
			style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
		) as dialog:
			if dialog.ShowModal() != wx.ID_OK:
				return None
			self.output_directory = Path(dialog.GetPath()).resolve()
		self.update_output_status()
		return self.output_directory

	def ensure_output_directory(self):
		"""Return the selected output directory, prompting when absent."""
		if self.output_directory is None:
			return self.on_choose_output_directory(None)
		return self.output_directory

	def clear_burst_review(self):
		"""Reset all pending rupture suggestions and review state."""
		self.tracker.burst_candidates = []
		self.tracker.enable_burst_candidates = False
		self.tracker.burst_candidates_evaluated = False
		self.burst_candidate_records = []
		self.refresh_burst_candidates()

	def reset_tracking_state(self):
		"""Clear tracks and downstream events while retaining grains and tips."""
		self.tracker.valid_tracks = []
		self.tracker.other_tracks = []
		self.tracker.filled_in_tips = []
		self.tracker.gv31 = 1
		self.tracker.gv32 = []
		for tips in self.tracker.valid_tips:
			for tip in tips:
				tip.is_used = False
		for grain in self.tracker.valid_grains:
			grain.remove_survival(what="g")
			grain.remove_survival(what="b")
			grain.clear_burst_candidate()
		self.clear_burst_review()
		self.screen_control.user_clicks = []
		self.chb14.SetValue(False)
		self.chb15.SetValue(False)

	def on_reset_tracking(self, e):
		"""Confirm and perform a tracking-only restart."""
		if not self.tracker.file_names:
			return
		choice = wx.MessageBox(
			"Clear tip tracks, germination calls, and burst review while keeping detected grains and tips?",
			"Restart Tracking",
			wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION,
			self,
		)
		if choice != wx.YES:
			return
		self.reset_tracking_state()
		self.update_screen()
		self.update_output_status("Tracking reset. Detected grains and tips were kept.")

	def on_restart_analysis(self, e):
		"""Confirm and clear detections while retaining video and parameters."""
		if not self.tracker.file_names:
			return
		choice = wx.MessageBox(
			"Clear all detections and tracking results while keeping the loaded video and current parameters?",
			"Restart Analysis",
			wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION,
			self,
		)
		if choice != wx.YES:
			return
		self.reset_tracking_state()
		self.tracker.valid_grains = []
		self.tracker.valid_tips = []
		self.tracker.pot_grains = []
		self.tracker.gv8 = 1
		self.update_screen()
		self.update_output_status("Analysis restarted. Video and parameters were kept.")

	def on_start_over(self, e):
		"""Confirm and clear the entire loaded-video analysis session."""
		if not self.tracker.file_names:
			return
		choice = wx.MessageBox(
			"Close the current video and clear all unsaved analysis results?",
			"Start Over",
			wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
			self,
		)
		if choice != wx.YES:
			return
		self.reset_tracking_state()
		self.tracker.valid_grains = []
		self.tracker.valid_tips = []
		self.tracker.file_names = None
		self.tracker.all_detections = []
		self.first_use = True
		self.img_to_display = 0
		self.screen_control.frame = 0
		self.screen_control.user_clicks = []
		for checkbox in (self.chb10, self.chb11, self.chb13, self.chb14, self.chb15):
			checkbox.SetValue(False)
		self.screen.display(load_logo())
		self.st12.SetLabel("1/1")
		self.update_output_status("Session cleared. Ready for another video.")

	def refresh_burst_candidates(self):
		"""Rebuild rupture-review choices and action availability."""
		self.burst_choice.Clear()
		for candidate in self.burst_candidate_records:
			self.burst_choice.Append(
				"Grain {} | frame {} | score {:.2f}".format(
					candidate["grain_id"],
					candidate["frame"] + 1,
					candidate["confidence"],
				)
			)
		has_candidates = len(self.burst_candidate_records) > 0
		self.burst_confirm.Enable(has_candidates)
		self.burst_dismiss.Enable(has_candidates)
		if has_candidates:
			self.burst_choice.SetSelection(0)
		else:
			self.burst_choice.Append("No candidates yet")
			self.burst_choice.SetSelection(0)

	def on_find_burst_candidates(self, e):
		"""Validate prerequisites and populate experimental rupture suggestions."""
		if not self.tracker.file_names:
			wx.MessageBox("Load a video before reviewing ruptures.", "Burst Review", wx.OK | wx.ICON_INFORMATION, self)
			return
		if len(self.tracker.valid_grains) == 0:
			wx.MessageBox("Run Find Grains first.", "Burst Review", wx.OK | wx.ICON_INFORMATION, self)
			return
		if len(self.tracker.valid_tracks) == 0:
			wx.MessageBox("Run Find Tips and Track Tips first.", "Burst Review", wx.OK | wx.ICON_INFORMATION, self)
			return
		if not any(grain.is_germinated for grain in self.tracker.valid_grains):
			wx.MessageBox("Run Track Germination first.", "Burst Review", wx.OK | wx.ICON_INFORMATION, self)
			return
		self.tracker.enable_burst_candidates = True
		self.burst_candidate_records = list(self.tracker.find_burst_candidates())
		self.refresh_burst_candidates()
		if len(self.burst_candidate_records) == 0:
			wx.MessageBox("No rupture candidates were found.", "Burst Review", wx.OK | wx.ICON_INFORMATION, self)
		else:
			self.on_select_burst_candidate(None)

	def on_select_burst_candidate(self, e):
		"""Jump to a selected suggestion and enable grain-state overlays."""
		index = self.burst_choice.GetSelection()
		if index < 0 or index >= len(self.burst_candidate_records):
			return
		candidate = self.burst_candidate_records[index]
		frame = int(candidate["frame"])
		self.img_to_display = frame
		self.screen_control.frame = frame
		self.chb15.SetValue(True)
		self.burst_choice.SetToolTip("Evidence: " + candidate["reason"])
		self.update_screen()

	def selected_burst_candidate(self):
		"""Return the selected candidate record and corresponding grain."""
		index = self.burst_choice.GetSelection()
		if index < 0 or index >= len(self.burst_candidate_records):
			return (None, None, None)
		candidate = self.burst_candidate_records[index]
		grain = next(
			(
				grain
				for grain in self.tracker.valid_grains
				if str(grain.id) == str(candidate["grain_id"])
			),
			None,
		)
		return (index, candidate, grain)

	def remove_selected_burst_candidate(self, index, candidate):
		"""Remove a reviewed suggestion and advance to the next candidate."""
		self.burst_candidate_records.pop(index)
		if candidate in self.tracker.burst_candidates:
			self.tracker.burst_candidates.remove(candidate)
		self.refresh_burst_candidates()
		if len(self.burst_candidate_records) > 0:
			self.on_select_burst_candidate(None)
		else:
			self.update_screen()

	def on_confirm_burst_candidate(self, e):
		"""Record the selected rupture suggestion as manually reviewed."""
		index, candidate, grain = self.selected_burst_candidate()
		if grain is None:
			return
		grain.accept_burst_candidate()
		self.remove_selected_burst_candidate(index, candidate)

	def on_dismiss_burst_candidate(self, e):
		"""Dismiss the selected rupture suggestion without recording an event."""
		index, candidate, grain = self.selected_burst_candidate()
		if grain is None:
			return
		grain.clear_burst_candidate()
		self.remove_selected_burst_candidate(index, candidate)

	def add_grain_frame(self, for_ger = True):
		"""Apply manual germination or rupture frames from canvas clicks."""
		if len(self.screen_control.user_clicks) > 0:
			dt = int(0.5*(self.tracker.max_grain_radius + self.tracker.min_grain_radius))
			for center in self.screen_control.user_clicks:
				rois = []
				for grain in self.tracker.valid_grains:
					rois.append(grain.roi_closest_to(center.frame))
				roi = ROI(x_l = int(center.x - dt), y_t = int(center.y - dt), x_r = int(center.x + dt), y_b = int(center.y + dt), frame = center.frame)
				idx = roi.f4(others = rois, idx = True)
				if idx != None:
					if for_ger:
						if center.frame >= self.tracker.all_detections.img_list_length - 1:
							self.tracker.valid_grains[idx].remove_survival(what = "g")
						else:
							self.tracker.valid_grains[idx].update_germination(frame = center.frame, p_value = 0.01*self.tracker.gv5, method = "manual")
					else:
						if center.frame >= self.tracker.all_detections.img_list_length - 1:
							self.tracker.valid_grains[idx].remove_survival(what = "b")
						else:
							self.tracker.valid_grains[idx].update_burst(frame = center.frame, method = "manual")

	def add_grains(self):
		"""Create manually positioned grain tracks from canvas selections."""
		dt = int(0.5*(self.tracker.max_grain_radius + self.tracker.min_grain_radius))
		if self.tracker.file_names != None:
			if len(self.screen_control.user_clicks) > 0:
				for center in self.screen_control.user_clicks:
					if center.is_a == "grain":
						g = [ROI(x_l = int(center.x - dt), y_t = int(center.y - dt), x_r = int(center.x + dt), y_b = int(center.y + dt), frame = center.frame, detection_method = "manual"), ROI(x_l = int(center.x - dt), y_t = int(center.y - dt), x_r = int(center.x + dt), y_b = int(center.y + dt), frame = len(self.tracker.file_names)-1, detection_method = "manual")]
						g = Track(boxes = g, color = self.tracker.f9(), ID = self.tracker.f10("g"))
						g.detection_method = "manual"
						g.fill_missing_frames()
						self.tracker.valid_grains.append(g)

	def add_tips(self):
		"""Add or remove manual tip detections at selected frame coordinates."""
		dt = int(0.6*self.tracker.min_tip_side)
		if len(self.screen_control.user_clicks) > 0:
			for center in self.screen_control.user_clicks:
				tip = ROI(x_l = int(center.x - dt), y_t = int(center.y - dt), x_r = int(center.x + dt), y_b = int(center.y + dt), frame = center.frame, detection_method = "manual", is_tip = True)
				if tip.f3(self.tracker.valid_tips[center.frame]):
					if self.ids_to_process == "a" or self.ids_to_process == "A":
						t = tip.f4(self.tracker.valid_tips[center.frame], idx = False)
						for i in range(center.frame, len(self.tracker.valid_tips)):
							while(True):
								idx = t.f4(self.tracker.valid_tips[i], idx = True)
								if idx == None:
									break
								self.tracker.valid_tips[i].remove(self.tracker.valid_tips[i][idx])
					else:
						self.tracker.valid_tips[center.frame].remove(self.tracker.valid_tips[center.frame][tip.f4(self.tracker.valid_tips[center.frame], idx = True)])
				else:
					tip.set_id(self.tracker.f10(what = "tp"))
					self.tracker.valid_tips[center.frame].append(tip)

	def add_track(self):
		"""Create a manual tip trajectory from selected frame coordinates."""
		dt = int(0.6*self.tracker.min_tip_side)
		if len(self.screen_control.user_clicks) > 0:
			track = []
			for center in self.screen_control.user_clicks:
				track.append(ROI(x_l = int(center.x - dt), y_t = int(center.y - dt), x_r = int(center.x + dt), y_b = int(center.y + dt), frame = center.frame, detection_method = "manual", is_tip = True, ID = self.tracker.f10(what = "tp")))
			track = Track(boxes = track, color = self.tracker.f9(), ID = self.tracker.f10())
			xtr_roi = track.fill_missing_frames(ret = True)
			self.tracker.valid_tracks.append(track)
			for roi in xtr_roi:
				self.screen_control.user_clicks.append(Point(x = roi.gv3.x, y = roi.gv3.y, frame = roi.gv6))
				if not roi.f3(self.tracker.valid_tips[roi.gv6], overlap = 0.7):
					roi.is_tip = True
					roi.set_id(self.tracker.f10(what = "tp"))
					self.tracker.valid_tips.append(roi)

	def change_frame(self, by):
		"""Move the display by a wrapped frame offset."""
		if self.tracker.file_names != None:
			self.img_to_display = self.img_to_display + by
			if self.img_to_display >= len(self.tracker.file_names):
				self.img_to_display = self.img_to_display - len(self.tracker.file_names)
			if self.img_to_display < 0:
				self.img_to_display = len(self.tracker.file_names) + self.img_to_display
			self.screen_control.frame = self.img_to_display
			self.update_screen()
			self.screen_control.Refresh()

	def cut_track(self, keep_leftover = False):
		"""Truncate selected tracks and optionally retain their trailing segments."""
		ids = self.f2(self.ids_to_process, separation = ',')
		if len(ids) > 0 and len(self.screen_control.user_clicks) > 0:
			if len(ids) == len(self.screen_control.user_clicks):
				temp = []
				dp = 5
				for i in range(len(ids)):
					pt = self.screen_control.user_clicks[i]
					for track in self.tracker.valid_tracks:
						if track.id == ids[i]:
							split = track.f4(ROI(x_l = pt.x - dp, y_t = pt.y - dp, x_r = pt.x + dp, y_b = pt.y + dp, frame = pt.frame), keep_leftover = keep_leftover)
							if split != None:
								temp.append(Track(split, color = self.tracker.f9(), ID = self.tracker.f10()))
				if temp != []:
					for track in temp:
						self.tracker.valid_tracks.append(track)
			else:
				pass

	def extend_track(self):
		"""Extend selected tracks with manually clicked tip positions."""
		dt = int(0.6*self.tracker.min_tip_side)
		trk_id = self.f2(self.ids_to_process, separation = ',')[0]
		for track in self.tracker.valid_tracks:
			if track.id == trk_id:
				tmp = []
				for center in self.screen_control.user_clicks:
					roi = ROI(x_l = int(center.x - dt), y_t = int(center.y - dt), x_r = int(center.x + dt), y_b = int(center.y + self.tracker.min_tip_side), frame = center.frame, detection_method = "manual", is_tip = True, ID = self.tracker.f10(what = "tp"))
					tmp.append(roi)
					self.tracker.valid_tips.append(roi)
				track.f2(tmp)
				xtr_roi = track.fill_missing_frames(ret = True)
				for roi in xtr_roi:
					self.screen_control.user_clicks.append(Point(x = roi.gv3.x, y = roi.gv3.y, frame = roi.gv6))

	def f2(self, string, separation = ",", track_undo = False):
		"""Parse a separated identifier string into validated IDs."""
		if separation is None:
			num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
			x = ''
			for s in string:
				if s in num:
					x = x + s
			return x
		else:
			separation = str(separation)
			x = ''
			ids = []
			undo = False
			for i in range(len(string)):
				if string[i] == separation:
					if x != '':
						ids.append(x)
						x=''
				else:
					if string[i] == 'u' or string[i] == 'U':
						undo = True
						pass
					else:
						x = x + string[i]
				if i == len(string) - 1:
					if x != '':
						ids.append(x)
			if track_undo:
				return ids, undo
			else:
				return ids

	def f3(self, string, track_undo = False, sep1 = "/", sep2 = ","):
		"""Parse grouped track identifiers for multi-track editing operations."""
		entries = self.f2(string, separation = sep1)
		res = []
		undo = []
		if len(entries) > 0:
			for entry in entries:
				if track_undo:
					x, u = self.f2(entry, separation = sep2, track_undo = track_undo)
					res.append(x)
					undo.append(u)
				else:
					res.append(self.f2(entry, separation = sep2))
		if track_undo:
			return res, undo
		else:
			return res

	def f20(self, e):
		"""Update the germination significance threshold from its text field."""
		self.tracker.gv5 = float(self.tc11.GetValue())
		if self.tracker.gv5 <= 0:
			self.tracker.gv5 = self.gv2

	def f29(self, item):
		"""Return contextual manual-edit instructions for a selected operation."""
		if item == 1:
			cmt = '''To manually track or update burst, select (click inside) the BURSTED grain on the frame the event first happens. To remove burst, select the grain in the final frame. Click Update when done. '''
		elif item == 2:
			cmt = '''To truncate tracks, please enter the list of tracks separated by commas (,). Next, click on the screen at the position to truncate in the same order as the list of tracks. '''
		elif item == 3:
			cmt = '''Recomended: Please before tracking germination, Click "remove grains" from QC panel and enter the id of grains that should not  be tracked. '''
		elif item == 4:
			cmt = '''grains detected. Please enter the IDs of grains to "Exclude" from tracking then "track". '''
		elif item == 5:
			cmt = '''To manually track or update germination, select (click inside) the GERMINATED grain on the frame the event first happens. To remove germination, select the grain in the final frame. Click Update when done. '''
		elif item == 6:
			cmt = '''To manually track a tip, Click on it location in all frames to be included in the track then press Update. '''
		elif item == 7:
			cmt = '''To add to a track, please enter the track's id bellow then on each frame to include, click on the position of the tip. Press Update when done. '''
		elif item == 8:
			cmt = '''To link tracks, please enter the track ids bellow separated by commas (,). Separate multiples entries with "/". (i.e.: ../1,2/..). Press Update when done. '''
		elif item == 9:
			cmt = '''To manually add grains, select the center of each grain on the frame it first appears then press Update. To remove, enter the list of grains ids separated by commas (,). '''
		elif item == 10:
			cmt = '''To remove tracks, please enter the list of tracks separated by commas (,) then press Update. '''
		elif item == 11:
			cmt = '''To split tracks, please enter the list of tracks separated by commas (,). next, click on the screen at the position to split in the same order as the list of tracks. '''
		elif item == 12:
			cmt = '''Loading Completed. Please do not modify the input directory untill completed. '''
		elif item == 13:
			cmt = '''Error: Please select a directory containg a series of pictures with selected extention. '''
		elif item == 14:
			cmt = '''No Frame Found. Uploading the previous frame. Please make sure to enter values from 0 to number of frames. '''
		elif item == 15:
			cmt = '''No image to show. '''
		elif item == 16:
			cmt = '''Select a box to display instructions. '''
		elif item == 17:
			cmt = '''Error, Could not identify tips with specified arguments. Please, increase Start and lookback or change the tip average area. '''
		elif item == 18:
			cmt = '''The number of IDs entered and the number of frames selected must be equal. Please Try Again. '''
		elif item == 19:
			cmt = '''No Grain Available. '''
		elif item == 20:
			cmt = '''Germination Tracked. '''
		elif item == 21:
			cmt = '''To manually add tips, select the center of each tip on the frame it appears then press Update. "add track" will add the same tip over multiple frames. To remove a tip, click inside an already existing tip. Enter "a" or "A" below if you want to remove overlaping tips in later frames too. '''
		else:
			cmt = None
		if cmt == None:
			return("")
		else:
			by = int(self.gv1[0]*30/1445)
			word = []
			tmp = ""
			for i in cmt:
				if i == " ":
					word.append(tmp)
					tmp = ""
				else:
					tmp = tmp + i
			res = "* "
			lne = ""
			k = 0
			kk = 0
			for wrd in word:
				if len(wrd) + len(lne) < by:
					lne = lne + wrd + " "
					kk+=1
				else:
					res = res + lne + "\n  "
					lne = wrd + " "
					k+=kk
					kk = 0
			if k < len(word):
				res = res + lne
			return res

	def f45(self, item):
		"""Scale a named legacy control position to the usable display size."""
		xy = None
		poss = 5
		if item == "pnl1":
			xy = (0, 0)
		elif item == "ly1":
			xy = (5, 0)
		elif item == "cb1":
			xy = (130, 29)
		elif item == "st1":
			xy = (15, 30)
		elif item == "cb_b1":
			xy = (45, 60)
		elif item == "b1":
			xy = (45, 90)
		elif item == "heatmap":
			xy = (20, 125)
		elif item == "sl_heatmap":
			xy = (20, 125)
		elif item == "st1_heatmap":
			xy = (30, 140)
		elif item == "st2_heatmap":
			xy = (165, 140)
		elif item == "lt_sl2":
			xy = (130, 170)
		elif item == "st1sa":
			xy = (20, 170)
		elif item == "st1sal":
			xy = (100, 170)
		elif item == "lt_sl3":
			xy = (130, 170)
		elif item == "st1sb":
			xy = (20, 170)
		elif item == "st1sbl":
			xy = (100, 170)
		elif item == "st_apply_bg":
			xy = (20, 200)
		elif item == "b4a":
			xy = (150, 200)
		elif item == "b4b":
			xy = (90, 200)
		elif item == "st9":
			xy = (20, 230)
		elif item == "tc2":
			xy = (100, 230)
		elif item == "cb2":
			xy = (140, 230)
		elif item == "st10":
			xy = (20, 260)
		elif item == "tc3":
			xy = (100, 260)
		elif item == "cb3":
			xy = (140, 260)
		elif item == "tc4":
			xy = (15, 295)
		elif item == "b2":
			xy = (55, 325)
		elif item == "pnl2":
			xy = (5, 360)
		elif item == "ly2":
			xy = (0, 0)
		elif item == "chb1":
			xy = (10, 25)
		elif item == "chb2":
			xy = (105, 25)
		elif item == "chb3":
			xy = (10,50)
		elif item == "chb4":
			xy = (105, 50)
		elif item == "chb5":
			xy = (10, 75)
		elif item == "chb6":
			xy = (105, 75)
		elif item == "chb7":
			xy = (10, 100)
		elif item == "chb8":
			xy = (105, 100)
		elif item == "chb9":
			xy = (10, 125)
		elif item == "chb9a":
			xy = (105, 125)
		elif item == "st11":
			xy = (10, 150)
		elif item == "tc10":
			xy = (10, 350)
		elif item == "b11":
			xy = (55, 385)
		elif item == "pnl3":
			xy = (215, 0)
		elif item == "ly3":
			xy = (0, 0)
		elif item == "tt4":
			xy = (8,22)
		elif item == "pnl4":
			xy = (215, 750)
		elif item == "b8":
			xy = (45, poss)
		elif item == "b7":
			xy = (105, poss)
		elif item == "b6":
			xy = (325, poss)
		elif item == "b9":
			xy = (175, poss)
		elif item == "b10":
			xy = (235, poss)
		elif item == "tc1":
			xy = (420, poss)
		elif item == "mb_st2":
			xy = (495, poss)
		elif item == "mb_st3":
			xy = (570, poss)
		elif item == "chb10":
			xy = (640, poss)
		elif item == "chb11":
			xy = (710, poss)
		elif item == "chb13":
			xy = (795, poss)
		elif item == "chb14":
			xy = (850, poss)
		elif item == "chb15":
			xy = (915, poss)
		elif item == "pnl5":
			xy = (1228, 0)
		elif item == "ly4":
			xy = (5, 0)
		elif item == "st13":
			xy = (50, 20)
		elif item == "st14":
			xy = (10, 45)
		elif item == "tc5":
			xy = (125, 45)
		elif item == "st15":
			xy = (155, 45)
		elif item == "tc6":
			xy = (165, 45)
		elif item == "tc7":
			xy = (15, 75)
		elif item == "st17":
			xy = (55, 75)
		elif item == "tc8":
			xy = (165, 75)
		elif item == "rt_st6":
			xy = (10, 105)
		elif item == "rt_st6l":
			xy = (105, 105)
		elif item == "rt_sl2":
			xy = (130, 105)
		elif item == "rt_tc5":
			xy = (150, 105)
		elif item == "b3":
			xy = (60,130)
		elif item == "st20":
			xy = (60, 160)
		elif item == "st23a":
			xy = (10, 185)
		elif item == "st23b":
			xy = (140, 185)
		elif item == "st23c":
			xy = (75, 185)
		elif item == "st23":
			xy = (15, 210)
		elif item == "br_st2":
			xy = (90, 210)
		elif item == "br_sl2":
			xy = (130, 210)
		elif item == "st22":
			xy = (15, 235)
		elif item == "tc15a":
			xy = (165, 235)
		elif item == "b5a":
			xy = (60, 265)
		elif item == "st18":
			xy = (15, 190)
		elif item == "tc9":
			xy = (75, 190)
		elif item == "st19a":
			xy = (40, 20)
		elif item == "st19":
			xy = (15, 295)
		elif item == "tc11":
			xy = (150, 295)
		elif item == "st19x":
			xy = (15, 235)
		elif item == "tc11x":
			xy = (160, 235)
		elif item == "st19y":
			xy = (15, 265)
		elif item == "tc11y":
			xy = (160, 265)
		elif item == "tc11yl":
			xy = (125, 265)
		elif item == "st23d":
			xy = (10, 205)
		elif item == "st23e":
			xy = (140, 205)
		elif item == "st23f":
			xy = (75, 205)
		elif item == "b4":
			xy = (35, 330)
		elif item == "pnl6":
			xy = (1228, 300)
		elif item == "ly5a":
			xy = (5, 0)
		elif item == "ly5b":
			xy = (5, 180)
		elif item == "st21":
			xy = (15, 40)
		elif item == "br_sl1":
			xy = (120, 40)
		elif item == "br_st1":
			xy = (80, 40)
		elif item == "tc12":
			xy = (55, 50)
		elif item == "tc13":
			xy = (160, 50)
		elif item == "br_sl3":
			xy = (120, 80)
		elif item == "br_st3":
			xy = (80, 80)
		elif item == "tc14":
			xy = (100, 80)
		elif item == "tc14a":
			xy = (160, 80)
		elif item == "st24":
			xy = (45, 125)
		elif item == "st25":
			xy = (20, 25)
		elif item == "tc15":
			xy = (150, 25)
		elif item == "st26":
			xy = (20, 55)
		elif item == "tc16":
			xy = (150, 55)
		elif item == "st27":
			xy = (20, 85)
		elif item == "tc17l":
			xy = (105, 85)
		elif item == "tc17":
			xy = (150, 85)
		elif item == "st27x":
			xy = (20, 115)
		elif item == "tc17x":
			xy = (150, 115)
		elif item == "b5":
			xy = (35, 150)
		elif item == "pnl7":
			xy = (1228, 660)
		elif item == "ly6":
			xy = (5, 0)
		elif item == "burst_find":
			xy = (20, 22)
		elif item == "burst_choice":
			xy = (10, 51)
		elif item == "burst_confirm":
			xy = (10, 82)
		elif item == "burst_dismiss":
			xy = (110, 82)
		elif item == "st28":
			xy = (1235, 650)
		if xy == None:
			return None
		elif xy[0] < 0:
			return (-1, int(self.gv1[1]*xy[1]/865))
		elif xy[1] < 0:
			return (int(self.gv1[0]*xy[0]/1445), -1)
		else:
			return (int(self.gv1[0]*xy[0]/1445), int(self.gv1[1]*xy[1]/865))

	def f47(self, item):
		"""Scale a named legacy control size to the usable display size."""
		xy = None
		if item == "pnl1":
			xy = (210, 370)
		elif item == "ly1":
			xy = (205, 360)
		elif item == "b1":
			xy = (120, -1)
		elif item == "cb1":
			xy = (60, -1)
		elif item == "tc2":
			xy = (50, -1)
		elif item == "cb2":
			xy = (60, -1)
		elif item == "tc3":
			xy = (50, -1)
		elif item == "cb3":
			xy = (60, -1)
		elif item == "tc4":
			xy = (190, -1)
		elif item == "b2":
			xy = (100, -1)
		elif item == "pnl2":
			xy = (210, 420)
		elif item == "ly2":
			xy = (205, 420)
		elif item == "tc10":
			xy = (190,-1)
		elif item == "b11":
			xy = (100, -1)
		elif item == "pnl3":
			xy = (1013, 750)
		elif item == "ly3":
			xy = (1013, 750)
		elif item == "tt4":
			xy = (1000,725)
		elif item == "pnl4":
			xy = (1013, 50)
		elif item == "st12":
			xy = (100,10)
		elif item == "tc1":
			xy = (50,-1)
		elif item == "b8":
			xy = (50,-1)
		elif item == "b7":
			xy = (50,-1)
		elif item == "b6":
			xy = (85,-1)
		elif item == "b9":
			xy = (50,-1)
		elif item == "b10":
			xy = (50,-1)
		elif item == "pnl5":
			xy = (210, 300)
		elif item == "ly4":
			xy = (205, 300)
		elif item == "tc":
			xy = (30, -1)
		elif item == "rt_tc5":
			xy = (40, -1)
		elif item == "b3":
			xy = (100, -1)
		elif item == "tc9":
			xy = (120, -1)
		elif item == "tc11":
			xy = (50, -1)
		elif item == "b4":
			xy = (50, -1)
		elif item == "heatmap":
			xy = (180, 10)
		elif item == "pnl6":
			xy = (210, 370)
		elif item == "ly5a":
			xy = (205, 180)
		elif item == "ly5b":
			xy = (205, 180)
		elif item == "tc12":
			xy = (30,-1)
		elif item == "tc13":
			xy = (30,-1)
		elif item == "tc14":
			xy = (40,-1)
		elif item == "tc15":
			xy = (40,-1)
		elif item == "tc16":
			xy = (40,-1)
		elif item == "tc17":
			xy = (40,-1)
		elif item == "b5":
			xy = (150, -1)
		elif item == "br_sl1":
			xy = (70, -1)
		elif item == "rt_sl1a":
			xy = (50, -1)
		elif item == "tc11y":
			xy = (40, -1)
		elif item == "pnl7":
			xy = (210, 120)
		elif item == "ly6":
			xy = (205, 120)
		elif item == "burst_find":
			xy = (170, -1)
		elif item == "burst_choice":
			xy = (190, -1)
		elif item == "burst_action":
			xy = (90, -1)
		if xy == None:
			return None
		elif xy[0] < 0:
			return (-1, int(self.gv1[1]*xy[1]/865))
		elif xy[1] < 0:
			return (int(self.gv1[0]*xy[0]/1445), -1)
		else:
			return (int(self.gv1[0]*xy[0]/1445), int(self.gv1[1]*xy[1]/865))

	def get_file_names(self, heading = '', ending = '.png'):
		"""Yield collision-resistant sequential temporary frame filenames."""
		num = list(range(10))
		for a in num:
			for b in num:
				for c in num:
					for d in num:
						for e in num:
							for f in num:
								for g in num:
									for h in num:
										for i in num:
											for j in num:
												yield str(heading) + str(a) + str(b) + str(c) + str(d) + str(e) + str(f) + str(g) + str(h) + str(i) + str(j) + str(ending)

	def get_heatmap(self, h=50, w = 256):
		"""Render the grayscale false-color palette as a horizontal legend."""
		mask = np.zeros((h,w,3), dtype = np.uint8)
		col_p = Detections.color_pallette(None)
		for pxl in range(256):
			bgr = col_p[col_p.gray == pxl]
			for j in range(h):
				mask[j,pxl] = (int(bgr.b.iloc[0]), int(bgr.g.iloc[0]), int(bgr.r.iloc[0]))
		return mask

	def link_tracks(self):
		"""Merge manually grouped trajectories and restore interpolated tips."""
		track_groups = self.f3(self.ids_to_process)
		for tracks_ids in track_groups:
			if len(self.tracker.valid_tracks) > 0:
				temp = []
				idx = []
				k = -1
				for track in self.tracker.valid_tracks:
					k+=1
					if track.id in tracks_ids:
						idx.append(k)
						for box in track.gv1:
							if not box.is_filled_in:
								temp.append(box)
				if len(temp) > 0:
					self.remove_track(ids = tracks_ids)
					track = Track(boxes = temp, color = self.tracker.f9(), ID = self.tracker.f10())
					xtr_roi = track.fill_missing_frames(ret = True)
					self.tracker.valid_tracks.append(track)
					for roi in xtr_roi:
						self.screen_control.user_clicks.append(Point(x = roi.gv3.x, y = roi.gv3.y, frame = roi.gv6))

	def make_checkbox(self, panel, label, pos, bind, id):
		"""Create and bind a consistently styled checkbox control."""
		cb = wx.CheckBox(panel, id, label = label, pos = pos)
		self.Bind(wx.EVT_CHECKBOX, bind, cb)
		cb.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.NORMAL))

	def on_aceptance_ratio(self, e):
		"""Update the required fraction of frames confirming germination."""
		val = int(self.tc11y.GetValue())/100
		self.tracker.aceptance_ratio = val
		self.tc11y_lab.SetLabel('{}'.format(val))
		self.reset_focus()

	def on_bt_show(self, e):
		"""Jump to the frame entered in the frame-number field."""
		if self.tracker.file_names != None:
			if self.tc1.GetValue() != '':
				self.img_to_display = int(int(self.tc1.GetValue()) - 1)
				if self.img_to_display >= len(self.tracker.file_names):
					self.img_to_display = int(len(self.tracker.file_names) - 1)
				if self.img_to_display < 0:
					self.img_to_display = 0
				self.screen_control.frame = self.img_to_display
				self.update_screen()
		self.reset_focus()

	def on_cb1(self, e):
		"""Update the selected input file extension."""
		self.input_ext = self.cb1.GetValue()

	def on_data_directory(self, e):
		"""Load a video or image sequence and initialize frame segmentation."""
		if self.input_ext in self.suported_file_ext:
			if self.first_use:
				move_on = True
			else:
				if wx.MessageBox("Are you sure you want to continue? All unsaved data will be deleted...", "Please confirm",wx.ICON_QUESTION | wx.YES_NO, self) == wx.NO:
					move_on = False
				else:
					move_on = True
			if move_on:
				self.first_use = False
				self.number_of_uses+=1
				self.tracker.file_names = []
				f_n = self.get_file_names(heading = temp_dir + '/input_' + str(self.number_of_uses))
				if self.input_ext in ['.avi', '.mp4']:
					with wx.FileDialog(self, "Choose a video with specified extention:", style = wx.FD_DEFAULT_STYLE|wx.FD_FILE_MUST_EXIST) as dlg:
						if dlg.ShowModal() == wx.ID_CANCEL:
							return
						cap = cv.VideoCapture(dlg.GetPath())
						k=-1
						temp = []
						while(True):
							k+=1
							ret, frame = cap.read()
							if ret == False:
								break
							if k == 0:
								h, w, l = frame.shape
								if self.rotate_input.GetValue() == True:
									self.tracker.img_rp = Point(x = self.screen_size[0]/h, y = self.screen_size[1]/w)
								else:
									self.tracker.img_rp = Point(x = self.screen_size[0]/w, y = self.screen_size[1]/h)
							xx = next(f_n)
							self.tracker.file_names.append(xx)
							if self.rotate_input.GetValue() == True:
								cv.imwrite(xx, cv.resize(cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE), self.screen_size))
							else:
								cv.imwrite(xx, cv.resize(frame, self.screen_size))
				else:
					with wx.DirDialog(self, "Choose Images Directory:", style = wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST|wx.DD_CHANGE_DIR) as dlg:
						if dlg.ShowModal() == wx.ID_CANCEL:
							return
						temp_path = sorted(glob(dlg.GetPath() + '/*' + self.input_ext))
						h, w, l = cv.imread(temp_path[0]).shape
						if self.rotate_input.GetValue() == True:
							self.tracker.img_rp = Point(x = self.screen_size[0]/h, y = self.screen_size[1]/w)
						else:
							self.tracker.img_rp = Point(x = self.screen_size[0]/w, y = self.screen_size[1]/h)
						for pth in temp_path:
							xx = next(f_n)
							self.tracker.file_names.append(xx)
							if self.rotate_input.GetValue() == True:
								cv.imwrite(xx, cv.resize(cv.rotate(cv.imread(pth), cv.ROTATE_90_CLOCKWISE), self.screen_size))
							else:
								cv.imwrite(xx, cv.resize(cv.imread(pth), self.screen_size))
				self.tracker.gv3 = []
				self.tracker.gv11 = []
				self.tracker.all_detections = []
				self.tracker.gv29 = []
				self.tracker.gv30 = []
				self.tracker.gv32 = []
				self.tracker.valid_tracks = []
				self.tracker.valid_grains = []
				self.tracker.burst_candidates = []
				self.tracker.enable_burst_candidates = False
				self.tracker.burst_candidates_evaluated = False
				self.burst_candidate_records = []
				self.refresh_burst_candidates()
				self.tracker.gv31 = 1
				self.tracker.gv8 = 1
				self.tracker.segment_inputs()
				self.update_screen()
		self.reset_focus()

	def on_dis_unit(self, e):
		"""Update the physical distance unit used in exports."""
		self.tracker.dis_unit = self.cb3.GetValue()

	def on_filter_radius(self, e):
		"""Update the foreground smoothing radius from its slider."""
		val = self.smouthing_radius.GetValue()
		self.tracker.filter_radius = int(val)
		self.smouthing_radius_lab.SetLabel('{}'.format(val))
		self.reset_focus()

	def on_find_grains(self, e):
		"""Run automated grain detection for the loaded sequence."""
		if self.tracker.file_names != None:
			self.tracker.find_grains()
		self.reset_focus()

	def on_find_tips(self, e):
		"""Run the selected tip detector and optionally replace prior tips."""
		mthd = int(self.tip_det_mthd.GetValue())
		if self.tracker.file_names != None:
			if len(self.tracker.valid_tips) > 0:
				if wx.MessageBox( "Would you like to erase the tips already detected?", "Please confirm",wx.ICON_QUESTION | wx.YES_NO, self) == wx.YES:
					self.tracker.valid_tips = []
			else:
				self.tracker.valid_tips = []
			if mthd == 1:
				print("finding tips via Segment Edge")
				self.tracker.find_tips_se()
			else:
				print("finding tips via Template Match")
				self.tracker.find_tips_tm()
		self.reset_focus()

	def on_forward(self, e):
		"""Advance the display by one frame."""
		self.change_frame(1)
		self.reset_focus()

	def on_forward_xl(self, e):
		"""Advance the display by the configured multi-frame step."""
		self.change_frame(self.move_xl)
		self.reset_focus()

	def on_gap_closing(self, e):
		"""Update the maximum gap used when linking tip detections."""
		self.tracker.tip_gap_closing = int(self.tc15.GetValue())

	def on_ger_confirm_frames(self, e):
		"""Update the number of frames used to confirm germination."""
		self.tracker.ger_confirm_frames = int(self.tc11x.GetValue())

	def on_grain_det_start(self, e):
		"""Update the first frame considered for new-grain detection."""
		self.tracker.grain_det_start = int(self.tc5.GetValue()) - 1
		if self.tracker.grain_det_start < 0:
			self.tracker.grain_det_start = 0

	def on_grain_det_stop(self, e):
		"""Update the last frame considered for new-grain detection."""
		self.tracker.grain_det_stop = int(self.tc6.GetValue()) - 1
		if self.tracker.grain_det_stop < 0:
			self.tracker.grain_det_stop = 0

	def on_grain_det_threshold(self, e):
		"""Update the Hough-circle grain detection threshold."""
		val = self.grain_det_tresh.GetValue()
		self.tracker.grain_tresh = int(val)
		self.grain_det_tresh_lab.SetLabel('{}'.format(val))
		self.reset_focus()

	def on_ids_to_process(self, e):
		"""Store identifiers entered for a manual QC operation."""
		self.ids_to_process = self.tc10.GetValue()

	def on_manual_checkboxes(self, e):
		"""Enforce one manual edit mode and show its instructions."""
		if self.chb1.GetValue()==self.chb2.GetValue()==self.chb3.GetValue()==self.chb4.GetValue()==self.chb7.GetValue()==self.chb5.GetValue()==self.chb6.GetValue()==self.chb8.GetValue()==self.chb9.GetValue()==self.chb8a.GetValue()==False:
			self.chb2.Enable(True)
			self.chb3.Enable(True)
			self.chb4.Enable(True)
			self.chb7.Enable(True)
			self.chb1.Enable(True)
			self.chb5.Enable(True)
			self.chb6.Enable(True)
			self.chb8.Enable(True)
			self.chb8a.Enable(True)
			self.chb9.Enable(True)
			self.st11.SetLabel(self.f29(16))
		elif self.chb1.GetValue() == True:
			self.st11.SetLabel(self.f29(10))
			self.chb2.Enable(False)
			self.chb3.Enable(False)
			self.chb4.Enable(False)
			self.chb7.Enable(False)
			self.chb5.Enable(False)
			self.chb6.Enable(False)
			self.chb8.Enable(False)
			self.chb8a.Enable(False)
			self.chb9.Enable(False)
		elif self.chb2.GetValue() == True:
			self.st11.SetLabel(self.f29(8))
			self.chb1.Enable(False)
			self.chb3.Enable(False)
			self.chb4.Enable(False)
			self.chb7.Enable(False)
			self.chb5.Enable(False)
			self.chb6.Enable(False)
			self.chb8.Enable(False)
			self.chb8a.Enable(False)
			self.chb9.Enable(False)
		elif self.chb3.GetValue() == True:
			self.st11.SetLabel(self.f29(7))
			self.chb1.Enable(False)
			self.chb2.Enable(False)
			self.chb4.Enable(False)
			self.chb7.Enable(False)
			self.chb5.Enable(False)
			self.chb6.Enable(False)
			self.chb8.Enable(False)
			self.chb9.Enable(False)
			self.chb8a.Enable(False)
		elif self.chb4.GetValue() == True:
			self.st11.SetLabel(self.f29(6))
			self.chb1.Enable(False)
			self.chb2.Enable(False)
			self.chb3.Enable(False)
			self.chb7.Enable(False)
			self.chb5.Enable(False)
			self.chb6.Enable(False)
			self.chb8.Enable(False)
			self.chb8a.Enable(False)
			self.chb9.Enable(False)
		elif self.chb7.GetValue() == True:
			self.st11.SetLabel(self.f29(5))
			self.chb1.Enable(False)
			self.chb2.Enable(False)
			self.chb3.Enable(False)
			self.chb4.Enable(False)
			self.chb5.Enable(False)
			self.chb6.Enable(False)
			self.chb8.Enable(False)
			self.chb8a.Enable(False)
			self.chb9.Enable(False)
		elif self.chb5.GetValue() == True:
			self.st11.SetLabel(self.f29(11))
			self.chb1.Enable(False)
			self.chb2.Enable(False)
			self.chb3.Enable(False)
			self.chb4.Enable(False)
			self.chb7.Enable(False)
			self.chb6.Enable(False)
			self.chb8.Enable(False)
			self.chb8a.Enable(False)
			self.chb9.Enable(False)
		elif self.chb6.GetValue() == True:
			self.st11.SetLabel(self.f29(2))
			self.chb1.Enable(False)
			self.chb2.Enable(False)
			self.chb3.Enable(False)
			self.chb4.Enable(False)
			self.chb7.Enable(False)
			self.chb5.Enable(False)
			self.chb8.Enable(False)
			self.chb8a.Enable(False)
			self.chb9.Enable(False)
		elif self.chb8.GetValue() == True:
			self.st11.SetLabel(self.f29(9))
			self.chb1.Enable(False)
			self.chb2.Enable(False)
			self.chb3.Enable(False)
			self.chb4.Enable(False)
			self.chb7.Enable(False)
			self.chb5.Enable(False)
			self.chb6.Enable(False)
			self.chb8a.Enable(False)
			self.chb9.Enable(False)
		elif self.chb9.GetValue() == True:
			self.st11.SetLabel(self.f29(1))
			self.chb1.Enable(False)
			self.chb2.Enable(False)
			self.chb3.Enable(False)
			self.chb4.Enable(False)
			self.chb7.Enable(False)
			self.chb5.Enable(False)
			self.chb6.Enable(False)
			self.chb8a.Enable(False)
			self.chb8.Enable(False)
		elif self.chb8a.GetValue() == True:
			self.st11.SetLabel(self.f29(21))
			self.chb1.Enable(False)
			self.chb2.Enable(False)
			self.chb3.Enable(False)
			self.chb4.Enable(False)
			self.chb7.Enable(False)
			self.chb5.Enable(False)
			self.chb6.Enable(False)
			self.chb8.Enable(False)
			self.chb9.Enable(False)
		self.reset_focus()

	def on_manual_update(self, e):
		"""Apply the currently selected manual QC operation."""
		if len(self.tracker.valid_tracks) > 0:
			if self.chb1.GetValue() == True:
				self.remove_track()
			if self.chb2.GetValue() == True:
				self.link_tracks()
				self.add_tips()
			if self.chb3.GetValue() == True:
				self.extend_track()
				self.add_tips()
			if self.chb5.GetValue() == True:
				self.cut_track(keep_leftover = True)
			if self.chb6.GetValue() == True:
				self.cut_track()
		if self.chb4.GetValue() == True:
			self.add_track()
			self.add_tips()
		if self.chb8a.GetValue() == True:
			self.add_tips()
		if self.chb8.GetValue() == True:
			self.tracker.gv3 = self.f2(self.ids_to_process, separation = ',')
			self.tracker.exclude_grains()
			self.tracker.gv3 = []
			self.add_grains()
		if len(self.tracker.valid_grains) > 0:
			if self.chb7.GetValue() == True:
				self.add_grain_frame()
			if self.chb9.GetValue() == True:
				self.add_grain_frame(for_ger = False)
		self.screen_control.user_clicks = []
		self.screen_control.Refresh()
		self.reset_focus()

	def on_max_grain_radius(self, e):
		"""Update the maximum expected grain radius and area."""
		self.tracker.max_grain_radius = int(int(self.tc8.GetValue())*self.tracker.img_ratio)
		self.tracker.max_grain_area = int(math.pi*self.tracker.max_grain_radius*self.tracker.max_grain_radius)

	def on_min_grain_radius(self, e):
		"""Update the minimum expected grain radius and area."""
		self.tracker.min_grain_radius = int(int(self.tc7.GetValue())*self.tracker.img_ratio)
		self.tracker.min_grain_area = int(math.pi*self.tracker.min_grain_radius*self.tracker.min_grain_radius)

	def on_min_threshold(self, e):
		"""Update the foreground background-cutoff threshold."""
		val = self.min_tresh_bar.GetValue()
		self.tracker.bg_threshold = int(val)
		self.st_bg_cutoff.SetLabel('{}'.format(val))
		self.reset_focus()

	def on_min_tip_per_trk(self, e):
		"""Update the minimum observations required for a tip track."""
		self.tracker.min_tip_per_trk = int(self.tc16.GetValue())

	def on_min_tip_sidelength(self, e):
		"""Update the minimum accepted tip bounding-box side length."""
		self.tracker.min_tip_side = int(self.tc15a.GetValue())

	def on_pxl_dis(self, e):
		"""Update the physical distance represented by one source pixel."""
		self.tracker.pxl_dis = float(self.tc3.GetValue())

	def on_reverse(self, e):
		"""Move the display backward by one frame."""
		self.change_frame(-1)
		self.reset_focus()

	def on_reverse_xl(self, e):
		"""Move the display backward by the configured multi-frame step."""
		self.change_frame(-self.move_xl)
		self.reset_focus()

	def on_save(self, e):
		"""Save all available analysis outputs to the selected directory."""
		if len(self.tracker.valid_tracks) == 0 and len(self.tracker.valid_grains) == 0:
			wx.MessageBox("There are no analysis results to save yet.", "Save Results", wx.OK | wx.ICON_INFORMATION, self)
			return
		output_directory = self.ensure_output_directory()
		if output_directory is None:
			return
		name = self.normalized_output_name()
		save_prefix = str(output_directory / (name + "."))
		self.tracker.save_results(save_prefix)
		self.update_output_status("Saved all results to " + str(output_directory))
		self.reset_focus()

	def on_export_coordinates(self, e):
		"""Write the tidy track coordinate and biological-time CSV."""
		if len(self.tracker.valid_tracks) == 0:
			wx.MessageBox("Track tips before exporting coordinates.", "Export Coordinates", wx.OK | wx.ICON_INFORMATION, self)
			return
		output_directory = self.ensure_output_directory()
		if output_directory is None:
			return
		path = output_directory / (self.normalized_output_name() + ".coordinates.csv")
		with path.open("w", newline="") as handle:
			csv.writer(handle).writerows(self.tracker.coordinate_rows())
		self.update_output_status("Exported coordinates to " + str(path))

	def on_save_name(self, e):
		"""Update the base name used for exported files."""
		self.save_name = self.tc4.GetValue()

	def on_time_p_frame(self, e):
		"""Update the biological time represented by one frame."""
		self.tracker.time_p_frame = int(self.tc2.GetValue())

	def on_time_unit(self, e):
		"""Update the biological time unit used in displays and exports."""
		self.tracker.time_unit = self.cb2.GetValue()

	def on_tip_det_thresh(self, e):
		"""Update the template-matching identity threshold."""
		val = self.tip_tresh_det.GetValue()
		self.tracker.tip_det_threshold_percent = int(val)/100
		self.tip_tresh_det_lab.SetLabel('{}'.format(val))
		self.reset_focus()

	def on_tip_max_step(self, e):
		"""Update the minimum overlap used to link tip detections."""
		val = int(self.tc17.GetValue())/100
		self.tracker.tip_max_step = val
		self.tc17_lab.SetLabel('{}'.format(val))
		self.reset_focus()

	def on_flaten_gap(self, e):
		"""Update the trajectory smoothing interval."""
		self.tracker.flaten_gap = int(self.tc17x.GetValue())

	def on_track_elongation(self, e):
		"""Link tip detections into trajectories and review interpolated tips."""
		if self.tracker.file_names == None:
			pass
		else:
			if len(self.tracker.valid_tips) <= 0:
				self.on_find_tips(None)
				if wx.MessageBox("Tips detected. Would you like to manually add more tips before tracking? if yes, use the 'add tip' option of manual tracking to add additional tips, then click track again.", "Please confirm",wx.ICON_QUESTION | wx.YES_NO, self) == wx.NO:
					c = True
				else:
					c = False
			else:
				c = True
			if c == True:
				self.tracker.track_elongation()
				if len(self.tracker.filled_in_tips) > 0:
					if wx.MessageBox("Would you like to save tip positions predicted to close gaps?", "Please confirm",wx.ICON_QUESTION | wx.YES_NO, self) == wx.YES:
						for tip in self.tracker.filled_in_tips:
							tip.is_tip = True
							self.tracker.valid_tips[tip.gv6].append(tip)
		self.reset_focus()

	def on_track_germination(self, e):
		"""Run the selected germination inference method and refresh overlays."""
		if self.tracker.file_names is None:
			pass
		else:
			mthd = int(self.ger_track_mthd.GetValue())
			if self.tracker.all_detections == []:
				self.tracker.segment_inputs()
			if self.tracker.valid_grains == []:
				self.tracker.find_grains()
			else:
				for grain in self.tracker.valid_grains:
					if grain.is_germinated:
						grain.is_germinated = False
						grain.ger_p_value = 1
						grain.ger_frame = -1
			if len(self.tracker.valid_grains) >0:
				if mthd == 1:
					if self.tracker.valid_tips == []:
						self.on_find_tips(None)
					print("Tracking Germination via Tip Overlap")
					self.tracker.track_germination_via_tips()
				else:
					print("Tracking Germination via Area Change")
					self.tracker.track_germination_via_area()
			self.update_screen()
		self.reset_focus()

	def on_update_all_bg(self, e):
		"""Recompute foreground segmentation for every loaded frame."""
		if self.tracker.all_detections != []:
			self.tracker.all_detections.remove_bg_and_locate_rois(bg_threshold = self.tracker.bg_threshold, blur_radius = self.tracker.filter_radius)
			self.update_screen()
		self.reset_focus()

	def on_update_frame_bg(self, e):
		"""Recompute foreground segmentation for only the displayed frame."""
		if self.tracker.all_detections != []:
			self.tracker.all_detections.remove_bg_and_locate_rois(bg_threshold = self.tracker.bg_threshold, frame = self.screen_control.frame, blur_radius = self.tracker.filter_radius)
			self.update_screen()
		self.reset_focus()

	def on_what_to_display(self, e):
		"""Keep display modes consistent and redraw selected overlays."""
		if self.chb10.GetValue() == self.chb11.GetValue() == False:
			self.chb10.Enable(True)
			self.chb11.Enable(True)
		elif self.chb10.GetValue() == True:
			self.chb11.Enable(False)
		elif self.chb11.GetValue() == True:
			self.chb10.Enable(False)
		if self.tracker.file_names != None:
			self.update_screen()
		self.reset_focus()

	def remove_track(self, ids = None):
		"""Remove selected trajectories and recycle their identifiers."""
		if ids == None:
			ids = self.f2(self.ids_to_process, separation = ',')
		temp = []
		for track in self.tracker.valid_tracks:
			if track.id in ids:
				temp.append(track)
		if len(temp) > 0:
			for track in temp:
				self.tracker.f12(track.id)
				self.tracker.valid_tracks.remove(track)

	def reset_focus(self):
		"""Return keyboard focus to the interactive image canvas."""
		self.screen_control.SetFocus()

	def update_screen(self):
		"""Render the selected frame mode and synchronize frame indicators."""
		if self.tracker.file_names != None:
			if self.chb11.GetValue() == True:
				self.screen.display(self.tracker.all_detections.get_raw_colored_frame(self.img_to_display))
			elif self.chb10.GetValue() == True:
				self.screen.display(self.tracker.all_detections.get_noiseless_frame(self.img_to_display))
			else:
				self.screen.display(self.tracker.all_detections.img_list_input[self.img_to_display])
			self.st12.SetLabel(str(self.screen_control.frame + 1) + "/" + str(self.tracker.all_detections.img_list_length))
			self.screen_control.Refresh()

def main():
	"""Launch the wxPython TubeTracker desktop application."""
	app = wx.App()
	ex = Tracker_GUI()
	ex.Show()
	app.MainLoop()

if __name__ == '__main__':
	main()
