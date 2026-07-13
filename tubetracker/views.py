"""Reusable wxPython image canvas controls."""

import cv2 as cv
import wx

from tubetracker_resources import load_logo

from .models import Point


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
