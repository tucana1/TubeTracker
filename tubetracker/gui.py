"""TubeTracker's wxPython desktop application and event handlers."""

import csv
from glob import glob
import math
from pathlib import Path
import tempfile

import cv2 as cv
import numpy as np
import wx

from tubetracker_resources import load_logo

from .analysis import Detections, Tracker
from .models import Point, ROI, Track
from .views import Screen, Screen_Control

tempdir = tempfile.TemporaryDirectory()
temp_dir = tempdir.name
title = "TubeTracker"

UI_BACKGROUND = wx.Colour(231, 236, 241)
UI_PANEL = wx.Colour(247, 249, 251)
UI_CANVAS = wx.Colour(255, 255, 255)
UI_FIELD = wx.Colour(255, 255, 255)
UI_TEXT = wx.Colour(36, 45, 54)
UI_MUTED_TEXT = wx.Colour(83, 96, 109)
UI_ACCENT = wx.Colour(35, 94, 145)


class Tracker_GUI(wx.Frame):
	"""Provide the desktop workflow for loading, reviewing, and exporting analyses."""

	def __init__(self,title = title):
		"""Construct controls, analysis state, menus, canvas, and review panels."""
		wx.Frame.__init__(self, None, title=title)
		self.first_use = True
		self.gv1 = wx.GetClientDisplayRect().GetSize()
		self.tracker = Tracker(screen_size = self.size_for("tt4"))
		self.input_ext = ".avi"
		self.ids_to_process = ''
		self.img_to_display = 0
		self.move_xl = 5
		self.number_of_uses = 0
		self.screen_size = self.size_for("tt4")
		self.save_name = 'Enter output name'
		self.output_directory = None
		self.panel_lt = wx.Panel(self, pos=self.position_for("pnl1"), size=self.size_for("pnl1"))
		wx.StaticBox(self.panel_lt, label='Image Parameters', pos=self.position_for("ly1"), size=self.size_for("ly1"))
		self.rotate_input = wx.CheckBox(self.panel_lt, label = "Rotate images", pos = self.position_for("cb_b1"))
		b1 = wx.Button(self.panel_lt, label = "Choose Folder", size = self.size_for("b1"), pos = self.position_for("b1"))
		b1.Bind(wx.EVT_BUTTON, self.on_data_directory)
		b2 = wx.Button(self.panel_lt, label = "Save Results", size = self.size_for("b2"), pos = self.position_for("b2"))
		b2.Bind(wx.EVT_BUTTON, self.on_save)
		self.suported_file_ext = ['.avi', ".mp4",'.png', '.tiff', '.tif', '.jpeg', '.jpg']
		self.cb1 = wx.ComboBox(self.panel_lt, choices=self.suported_file_ext, size = self.size_for("cb1"), pos = self.position_for("cb1"))
		self.cb1.Bind(wx.EVT_COMBOBOX, self.on_cb1)
		self.cb2 = wx.ComboBox(self.panel_lt, choices=['sec', 'min', 'hour', "day", "week", "month", "year"], size = self.size_for("cb2"), pos = self.position_for("cb2"))
		self.cb2.Bind(wx.EVT_COMBOBOX, self.on_time_unit)
		self.cb3 = wx.ComboBox(self.panel_lt, choices=['um', 'nm','mm', 'cm','m','in','ft','pxl'], size = self.size_for("cb3"), pos = self.position_for("cb3"))
		self.cb3.Bind(wx.EVT_COMBOBOX, self.on_dis_unit)
		self.tc3 = wx.TextCtrl(self.panel_lt, value = str(self.tracker.pxl_dis), size = self.size_for("tc"), pos = self.position_for("tc3"))
		self.tc3.Bind(wx.EVT_TEXT, self.on_pxl_dis)
		self.tc2 = wx.TextCtrl(self.panel_lt, value = str(self.tracker.time_p_frame), size = self.size_for("tc"), pos = self.position_for("tc2"))
		self.tc2.Bind(wx.EVT_TEXT, self.on_time_p_frame)
		self.tc4 = wx.TextCtrl(self.panel_lt, value = self.save_name, size = self.size_for("tc4"), pos = self.position_for("tc4"))
		self.tc4.Bind(wx.EVT_TEXT, self.on_save_name)
		wx.StaticText(self.panel_lt, label="Blur radius:", style=wx.ALIGN_LEFT, pos = self.position_for("st1sa"))
		self.smouthing_radius = wx.Slider(self.panel_lt, value=int(self.tracker.filter_radius), minValue=1, maxValue=20, style=wx.SL_HORIZONTAL, pos=self.position_for("lt_sl2"), size=self.size_for("br_sl1"))
		self.smouthing_radius.Bind(wx.EVT_SLIDER, self.on_filter_radius)
		self.smouthing_radius_lab = wx.StaticText(self.panel_lt, label=str(self.tracker.filter_radius), style=wx.ALIGN_LEFT, pos = self.position_for("st1sal"))
		wx.StaticText(self.panel_lt, label="File type:", style=wx.ALIGN_LEFT, pos = self.position_for("st1"))
		wx.StaticText(self.panel_lt, label='''Time per frame:''', style=wx.ALIGN_LEFT, pos = self.position_for("st9"))
		wx.StaticText(self.panel_lt, label="1 pixel:", style=wx.ALIGN_LEFT, pos = self.position_for("st10"))
		cv.imwrite(temp_dir + "/heatmap.png", cv.resize(self.get_heatmap(), self.size_for("heatmap")))
		self.heatmap = wx.StaticBitmap(self.panel_lt, wx.ID_ANY, wx.Bitmap(wx.Image(temp_dir + "/heatmap.png", wx.BITMAP_TYPE_ANY)), size = self.size_for("heatmap"), pos = self.position_for("heatmap"))
		self.min_tresh_bar = wx.Slider(self.panel_lt, value=self.tracker.bg_threshold, minValue=1, maxValue=255, style=wx.SL_HORIZONTAL, pos=self.position_for("sl_heatmap"), size=self.size_for("heatmap"))
		self.min_tresh_bar.Bind(wx.EVT_SLIDER, self.on_min_threshold)#
		wx.StaticText(self.panel_lt, label="Background cutoff:", style=wx.ALIGN_LEFT, pos = self.position_for("st1_heatmap"))
		self.st_bg_cutoff = wx.StaticText(self.panel_lt, label= str(self.tracker.bg_threshold), style=wx.ALIGN_LEFT, pos = self.position_for("st2_heatmap"))
		wx.StaticText(self.panel_lt, label="Apply to: ", style=wx.ALIGN_LEFT, pos = self.position_for("st_apply_bg"))
		b4a = wx.Button(self.panel_lt, label = "All", size = self.size_for("b4"), pos = self.position_for("b4a"))
		b4a.Bind(wx.EVT_BUTTON, self.on_update_all_bg)
		b4b = wx.Button(self.panel_lt, label = "Frame", size = self.size_for("b4"), pos = self.position_for("b4b"))
		b4b.Bind(wx.EVT_BUTTON, self.on_update_frame_bg)
		self.panel_lb = wx.Panel(self, pos=self.position_for("pnl2"), size=self.size_for("pnl2"))
		wx.StaticBox(self.panel_lb, label='QC and Manual Tracking', pos=self.position_for("ly2"), size=self.size_for("ly2"))
		b11 = wx.Button(self.panel_lb, label = "Apply Edit", size = self.size_for("b11"), pos = self.position_for("b11"))
		b11.Bind(wx.EVT_BUTTON, self.on_manual_update)
		self.st11 = wx.StaticText(self.panel_lb, label=self.instructions_for(16), pos = self.position_for("st11"))
		self.cb_id = 0
		labels = [["Remove track", self.position_for("chb1")], ["Link tracks", self.position_for("chb2")], ["Extend track", self.position_for("chb3")], ["Add track", self.position_for("chb4")], ["Split track", self.position_for("chb5")], ["Truncate track", self.position_for("chb6")], ["Add tip", self.position_for("chb7")], ["Add grain", self.position_for("chb8")], ["Add germ frame",self.position_for("chb9")], ["Add burst frame",self.position_for("chb9a")]]
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
		self.tc10 = wx.TextCtrl(self.panel_lb, value=self.ids_to_process, size = self.size_for("tc10"), pos = self.position_for("tc10"))
		self.tc10.Bind(wx.EVT_TEXT, self.on_ids_to_process)
		self.panel_rt = wx.Panel(self, pos=self.position_for("pnl5"), size=self.size_for("pnl5"))
		wx.StaticBox(self.panel_rt, label='Automated Pollen Detection', pos=self.position_for("ly4"), size=self.size_for("ly4"))
		wx.StaticText(self.panel_rt, label="Grain Detection", style=wx.ALIGN_LEFT, pos = self.position_for("st13"))
		wx.StaticText(self.panel_rt, label="Detection frames:", style=wx.ALIGN_LEFT, pos = self.position_for("st14"))
		wx.StaticText(self.panel_rt, label="-", style=wx.ALIGN_LEFT, pos = self.position_for("st15"))
		self.tc5 = wx.TextCtrl(self.panel_rt, value = str(self.tracker.grain_det_start + 1), size = self.size_for("tc"), pos = self.position_for("tc5"))
		self.tc5.Bind(wx.EVT_TEXT, self.on_grain_det_start)
		self.tc6 = wx.TextCtrl(self.panel_rt, value=str(self.tracker.grain_det_stop + 1), size = self.size_for("tc"), pos = self.position_for("tc6"))
		self.tc6.Bind(wx.EVT_TEXT, self.on_grain_det_stop)
		wx.StaticText(self.panel_rt, label="≤ radius ≤", style=wx.ALIGN_LEFT, pos = self.position_for("st17"))
		self.tc7 = wx.TextCtrl(self.panel_rt, value=str(int(self.tracker.min_grain_radius/self.tracker.img_ratio)), size = self.size_for("tc"), pos = self.position_for("tc7"))
		self.tc7.Bind(wx.EVT_TEXT, self.on_min_grain_radius)
		self.tc8 = wx.TextCtrl(self.panel_rt, value=str(int(self.tracker.max_grain_radius/self.tracker.img_ratio)), size = self.size_for("tc"), pos = self.position_for("tc8"))
		self.tc8.Bind(wx.EVT_TEXT, self.on_max_grain_radius)
		wx.StaticText(self.panel_rt, label="Threshold:", style=wx.ALIGN_LEFT, pos = self.position_for("rt_st6"))
		self.grain_det_tresh_lab = wx.StaticText(self.panel_rt, label= str(self.tracker.grain_tresh), style=wx.ALIGN_LEFT, pos = self.position_for("rt_st6l"))
		self.grain_det_tresh = wx.Slider(self.panel_rt, value=int(self.tracker.grain_tresh), minValue=10, maxValue=40, style=wx.SL_HORIZONTAL, pos=self.position_for("rt_sl2"), size=self.size_for("br_sl1"))
		self.grain_det_tresh.Bind(wx.EVT_SLIDER, self.on_grain_det_threshold)
		b3 = wx.Button(self.panel_rt, label = "Find Grains", size = self.size_for("b3"), pos = self.position_for("b3"))
		b3.Bind(wx.EVT_BUTTON, self.on_find_grains)
		wx.StaticText(self.panel_rt, label="Tip Detection", style=wx.ALIGN_LEFT, pos = self.position_for("st20"))
		wx.StaticText(self.panel_rt, label="Match:", style=wx.ALIGN_LEFT, pos = self.position_for("st23"))
		wx.StaticText(self.panel_rt, label="Segment edge", style=wx.ALIGN_LEFT, pos = self.position_for("st23a"))
		wx.StaticText(self.panel_rt, label="Template", style=wx.ALIGN_LEFT, pos = self.position_for("st23b"))
		self.tip_det_mthd = wx.Slider(self.panel_rt, value=1, minValue=1, maxValue=2, style=wx.SL_HORIZONTAL, pos=self.position_for("st23c"), size=self.size_for("rt_sl1a"))
		self.tip_tresh_det_lab = wx.StaticText(self.panel_rt, label= str(int(self.tracker.tip_det_threshold_percent*100)), style=wx.ALIGN_LEFT, pos = self.position_for("br_st2"))
		self.tip_tresh_det = wx.Slider(self.panel_rt, value=int(self.tracker.tip_det_threshold_percent*100), minValue=1, maxValue=100, style=wx.SL_HORIZONTAL, pos=self.position_for("br_sl2"), size=self.size_for("br_sl1"))
		self.tip_tresh_det.Bind(wx.EVT_SLIDER, self.on_tip_det_thresh)
		wx.StaticText(self.panel_rt, label="Minimum tip size:", style=wx.ALIGN_LEFT, pos = self.position_for("st22"))
		self.tc15a = wx.TextCtrl(self.panel_rt, value=str(self.tracker.min_tip_side), size = self.size_for("tc"), pos = self.position_for("tc15a"))
		self.tc15a.Bind(wx.EVT_TEXT, self.on_min_tip_sidelength)
		b5a = wx.Button(self.panel_rt, label = "Find Tips", size = self.size_for("b3"), pos = self.position_for("b5a"))
		b5a.Bind(wx.EVT_BUTTON, self.on_find_tips)
		self.panel_rb = wx.Panel(self, pos=self.position_for("pnl6"), size=self.size_for("pnl6"))
		wx.StaticBox(self.panel_rb, label='Automated Tip Tracking', pos=self.position_for("ly5a"), size=self.size_for("ly5a"))
		wx.StaticBox(self.panel_rb, label='Automated Germination Tracking', pos=self.position_for("ly5b"), size=self.size_for("ly5b"))
		wx.StaticText(self.panel_rb, label="Germination p-value:", style=wx.ALIGN_LEFT, pos = self.position_for("st19"))
		self.tc11 = wx.TextCtrl(self.panel_rb, value=str(self.tracker.gv5), size = self.size_for("tc11"), pos = self.position_for("tc11"))
		self.tc11.Bind(wx.EVT_TEXT, self.on_germination_significance)
		wx.StaticText(self.panel_rb, label="Tip overlap", style=wx.ALIGN_LEFT, pos = self.position_for("st23d"))
		wx.StaticText(self.panel_rb, label="Area change", style=wx.ALIGN_LEFT, pos = self.position_for("st23e"))
		self.ger_track_mthd = wx.Slider(self.panel_rb, value=1, minValue=1, maxValue=2, style=wx.SL_HORIZONTAL, pos=self.position_for("st23f"), size=self.size_for("rt_sl1a"))
		wx.StaticText(self.panel_rb, label="Confirmation frames:", style=wx.ALIGN_LEFT, pos = self.position_for("st19x"))
		self.tc11x = wx.TextCtrl(self.panel_rb, value=str(self.tracker.ger_confirm_frames), size = self.size_for("tc16"), pos = self.position_for("tc11x"))
		self.tc11x.Bind(wx.EVT_TEXT, self.on_ger_confirm_frames)
		wx.StaticText(self.panel_rb, label="Acceptance ratio:", style=wx.ALIGN_LEFT, pos = self.position_for("st19y"))
		self.tc11y_lab = wx.StaticText(self.panel_rb, label=str(self.tracker.aceptance_ratio), style=wx.ALIGN_LEFT, pos = self.position_for("tc11yl"))
		self.tc11y = wx.Slider(self.panel_rb, value=int(100*self.tracker.aceptance_ratio), minValue=10, maxValue=100, style=wx.SL_HORIZONTAL, pos=self.position_for("tc11y"), size=self.size_for("tc11y"))
		self.tc11y.Bind(wx.EVT_SLIDER, self.on_aceptance_ratio)
		b4 = wx.Button(self.panel_rb, label = "Track Germination", size = self.size_for("b5"), pos = self.position_for("b4"))
		b4.Bind(wx.EVT_BUTTON, self.on_track_germination)
		wx.StaticText(self.panel_rb, label="Gap closing:", style=wx.ALIGN_LEFT, pos = self.position_for("st25"))
		self.tc15 = wx.TextCtrl(self.panel_rb, value=str(self.tracker.tip_gap_closing), size = self.size_for("tc15"), pos = self.position_for("tc15"))
		self.tc15.Bind(wx.EVT_TEXT, self.on_gap_closing)
		wx.StaticText(self.panel_rb, label="Min. points/track:", style=wx.ALIGN_LEFT, pos = self.position_for("st26"))
		self.tc16 = wx.TextCtrl(self.panel_rb, value=str(self.tracker.min_tip_per_trk), size = self.size_for("tc16"), pos = self.position_for("tc16"))
		self.tc16.Bind(wx.EVT_TEXT, self.on_min_tip_per_trk)
		wx.StaticText(self.panel_rb, label="Min. overlap:", style=wx.ALIGN_LEFT, pos = self.position_for("st27"))
		self.tc17_lab = wx.StaticText(self.panel_rb, label=str(self.tracker.tip_max_step), style=wx.ALIGN_LEFT, pos = self.position_for("tc17l"))
		self.tc17 = wx.Slider(self.panel_rb, value=int(100*self.tracker.tip_max_step), minValue=1, maxValue=100, style=wx.SL_HORIZONTAL, pos=self.position_for("tc17"), size=self.size_for("tc11y"))
		self.tc17.Bind(wx.EVT_SLIDER, self.on_tip_max_step)
		wx.StaticText(self.panel_rb, label="Smoothing gap:", style=wx.ALIGN_LEFT, pos = self.position_for("st27x"))
		self.tc17x = wx.TextCtrl(self.panel_rb, value=str(self.tracker.flaten_gap), size = self.size_for("tc17"), pos = self.position_for("tc17x"))
		self.tc17x.Bind(wx.EVT_TEXT, self.on_flaten_gap)
		b5 = wx.Button(self.panel_rb, label = "Track Tips", size = self.size_for("b5"), pos = self.position_for("b5"))
		b5.Bind(wx.EVT_BUTTON, self.on_track_elongation)
		self.panel_mb = wx.Panel(self, pos=self.position_for("pnl4"), size=self.size_for("pnl4"))
		b6 = wx.Button(self.panel_mb, label = "Go to Frame", size = self.size_for("b6"), pos = self.position_for("b6"))
		b6.Bind(wx.EVT_BUTTON, self.on_bt_show)
		b7 = wx.Button(self.panel_mb, label = "", size = self.size_for("b7"), pos = self.position_for("b7"))
		b7.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_GO_BACK, wx.ART_BUTTON, (16, 16)))
		b7.SetToolTip("Previous frame")
		b7.Bind(wx.EVT_BUTTON, self.on_reverse)
		b8 = wx.Button(self.panel_mb, label = "-5", size = self.size_for("b8"), pos = self.position_for("b8"))
		b8.SetToolTip("Back 5 frames")
		b8.Bind(wx.EVT_BUTTON, self.on_reverse_xl)
		b9 = wx.Button(self.panel_mb, label = "", size = self.size_for("b9"), pos = self.position_for("b9"))
		b9.SetBitmap(wx.ArtProvider.GetBitmap(wx.ART_GO_FORWARD, wx.ART_BUTTON, (16, 16)))
		b9.SetToolTip("Next frame")
		b9.Bind(wx.EVT_BUTTON, self.on_forward)
		b10 = wx.Button(self.panel_mb, label = "+5", size = self.size_for("b10"), pos = self.position_for("b10"))
		b10.SetToolTip("Forward 5 frames")
		b10.Bind(wx.EVT_BUTTON, self.on_forward_xl)
		self.tc1 = wx.TextCtrl(self.panel_mb, value = "1", size = self.size_for("tc1"), pos = self.position_for("tc1"))
		self.st12 = wx.StaticText(self.panel_mb, label="1/1", style=wx.ALIGN_LEFT, pos = self.position_for("mb_st2"))
		wx.StaticText(self.panel_mb, label="Overlays:", style=wx.ALIGN_LEFT, pos = self.position_for("mb_st3"))
		labels = [["Heatmap", self.position_for("chb11")], ["Binary", self.position_for("chb10")], ["Tips", self.position_for("chb13")], ["Tracks", self.position_for("chb14")], ["Grain status", self.position_for("chb15")]]
		for label in labels:
			self.make_checkbox(self.panel_mb, label[0], label[1], self.on_what_to_display, self.cb_id)
			self.cb_id+=1
		self.chb10 = self.FindWindowByLabel("Binary")
		self.chb11 = self.FindWindowByLabel("Heatmap")
		self.chb13 = self.FindWindowByLabel("Tips")
		self.chb14 = self.FindWindowByLabel("Tracks")
		self.chb15 = self.FindWindowByLabel("Grain status")
		self.panel_mt = wx.Panel(self, pos=self.position_for("pnl3"), size=self.size_for("pnl3"))
		wx.StaticBox(self.panel_mt, label='Display', pos=self.position_for("ly3"), size=self.size_for("ly3"))
		self.screen = Screen(self.panel_mt,temp_dir, size = self.size_for("tt4"), pos = self.position_for("tt4"))
		self.screen_control = Screen_Control(self.panel_mt, parent = self, path = temp_dir, size = self.size_for("tt4"), pos = (self.position_for("tt4")[0] + self.position_for("pnl3")[0], self.position_for("tt4")[1] + self.position_for("pnl3")[1]))
		self.gv2 = self.tracker.gv5
		self.panel_rbb = wx.Panel(self, pos=self.position_for("pnl7"), size=self.size_for("pnl7"))
		wx.StaticBox(self.panel_rbb, label='Burst Review', pos=self.position_for("ly6"), size=self.size_for("ly6"))
		self.burst_candidate_records = []
		burst_find = wx.Button(self.panel_rbb, label="Find Candidates", pos=self.position_for("burst_find"), size=self.size_for("burst_find"))
		burst_find.SetToolTip("Suggest possible ruptures after tracking germination")
		burst_find.Bind(wx.EVT_BUTTON, self.on_find_burst_candidates)
		self.burst_choice = wx.ComboBox(self.panel_rbb, choices=[], style=wx.CB_READONLY, pos=self.position_for("burst_choice"), size=self.size_for("burst_choice"))
		self.burst_choice.SetToolTip("Select a candidate to jump to its proposed rupture frame")
		self.burst_choice.Bind(wx.EVT_COMBOBOX, self.on_select_burst_candidate)
		self.burst_confirm = wx.Button(self.panel_rbb, label="Confirm", pos=self.position_for("burst_confirm"), size=self.size_for("burst_action"))
		self.burst_confirm.SetToolTip("Confirm the selected candidate as a reviewed rupture event")
		self.burst_confirm.Bind(wx.EVT_BUTTON, self.on_confirm_burst_candidate)
		self.burst_dismiss = wx.Button(self.panel_rbb, label="Dismiss", pos=self.position_for("burst_dismiss"), size=self.size_for("burst_action"))
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
				idx = roi.best_overlap(others = rois, idx = True)
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
						g = Track(boxes = g, color = self.tracker.random_color(), ID = self.tracker.allocate_id("g"))
						g.detection_method = "manual"
						g.fill_missing_frames()
						self.tracker.valid_grains.append(g)

	def add_tips(self):
		"""Add or remove manual tip detections at selected frame coordinates."""
		dt = int(0.6*self.tracker.min_tip_side)
		if len(self.screen_control.user_clicks) > 0:
			for center in self.screen_control.user_clicks:
				tip = ROI(x_l = int(center.x - dt), y_t = int(center.y - dt), x_r = int(center.x + dt), y_b = int(center.y + dt), frame = center.frame, detection_method = "manual", is_tip = True)
				if tip.overlaps_any(self.tracker.valid_tips[center.frame]):
					if self.ids_to_process == "a" or self.ids_to_process == "A":
						t = tip.best_overlap(self.tracker.valid_tips[center.frame], idx = False)
						for i in range(center.frame, len(self.tracker.valid_tips)):
							while(True):
								idx = t.best_overlap(self.tracker.valid_tips[i], idx = True)
								if idx == None:
									break
								self.tracker.valid_tips[i].remove(self.tracker.valid_tips[i][idx])
					else:
						self.tracker.valid_tips[center.frame].remove(self.tracker.valid_tips[center.frame][tip.best_overlap(self.tracker.valid_tips[center.frame], idx = True)])
				else:
					tip.set_id(self.tracker.allocate_id(what = "tp"))
					self.tracker.valid_tips[center.frame].append(tip)

	def add_track(self):
		"""Create a manual tip trajectory from selected frame coordinates."""
		dt = int(0.6*self.tracker.min_tip_side)
		if len(self.screen_control.user_clicks) > 0:
			track = []
			for center in self.screen_control.user_clicks:
				track.append(ROI(x_l = int(center.x - dt), y_t = int(center.y - dt), x_r = int(center.x + dt), y_b = int(center.y + dt), frame = center.frame, detection_method = "manual", is_tip = True, ID = self.tracker.allocate_id(what = "tp")))
			track = Track(boxes = track, color = self.tracker.random_color(), ID = self.tracker.allocate_id())
			xtr_roi = track.fill_missing_frames(ret = True)
			self.tracker.valid_tracks.append(track)
			for roi in xtr_roi:
				self.screen_control.user_clicks.append(Point(x = roi.gv3.x, y = roi.gv3.y, frame = roi.gv6))
				if not roi.overlaps_any(self.tracker.valid_tips[roi.gv6], overlap = 0.7):
					roi.is_tip = True
					roi.set_id(self.tracker.allocate_id(what = "tp"))
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
		ids = self.parse_ids(self.ids_to_process, separation = ',')
		if len(ids) > 0 and len(self.screen_control.user_clicks) > 0:
			if len(ids) == len(self.screen_control.user_clicks):
				temp = []
				dp = 5
				for i in range(len(ids)):
					pt = self.screen_control.user_clicks[i]
					for track in self.tracker.valid_tracks:
						if track.id == ids[i]:
							split = track.truncate_near(ROI(x_l = pt.x - dp, y_t = pt.y - dp, x_r = pt.x + dp, y_b = pt.y + dp, frame = pt.frame), keep_leftover = keep_leftover)
							if split != None:
								temp.append(Track(split, color = self.tracker.random_color(), ID = self.tracker.allocate_id()))
				if temp != []:
					for track in temp:
						self.tracker.valid_tracks.append(track)
			else:
				pass

	def extend_track(self):
		"""Extend selected tracks with manually clicked tip positions."""
		dt = int(0.6*self.tracker.min_tip_side)
		trk_id = self.parse_ids(self.ids_to_process, separation = ',')[0]
		for track in self.tracker.valid_tracks:
			if track.id == trk_id:
				tmp = []
				for center in self.screen_control.user_clicks:
					roi = ROI(x_l = int(center.x - dt), y_t = int(center.y - dt), x_r = int(center.x + dt), y_b = int(center.y + self.tracker.min_tip_side), frame = center.frame, detection_method = "manual", is_tip = True, ID = self.tracker.allocate_id(what = "tp"))
					tmp.append(roi)
					self.tracker.valid_tips.append(roi)
				track.add_rois(tmp)
				xtr_roi = track.fill_missing_frames(ret = True)
				for roi in xtr_roi:
					self.screen_control.user_clicks.append(Point(x = roi.gv3.x, y = roi.gv3.y, frame = roi.gv6))

	def parse_ids(self, string, separation = ",", track_undo = False):
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

	def parse_id_groups(self, string, track_undo = False, sep1 = "/", sep2 = ","):
		"""Parse grouped track identifiers for multi-track editing operations."""
		entries = self.parse_ids(string, separation = sep1)
		res = []
		undo = []
		if len(entries) > 0:
			for entry in entries:
				if track_undo:
					x, u = self.parse_ids(entry, separation = sep2, track_undo = track_undo)
					res.append(x)
					undo.append(u)
				else:
					res.append(self.parse_ids(entry, separation = sep2))
		if track_undo:
			return res, undo
		else:
			return res

	def on_germination_significance(self, e):
		"""Update the germination significance threshold from its text field."""
		self.tracker.gv5 = float(self.tc11.GetValue())
		if self.tracker.gv5 <= 0:
			self.tracker.gv5 = self.gv2

	def instructions_for(self, item):
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

	def position_for(self, item):
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

	def size_for(self, item):
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
		gray = np.linspace(0, 255, num=w, dtype=np.uint8)
		row = Detections.color_palette()[gray]
		return np.repeat(row[np.newaxis, :, :], h, axis=0)

	def link_tracks(self):
		"""Merge manually grouped trajectories and restore interpolated tips."""
		track_groups = self.parse_id_groups(self.ids_to_process)
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
					track = Track(boxes = temp, color = self.tracker.random_color(), ID = self.tracker.allocate_id())
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
			self.st11.SetLabel(self.instructions_for(16))
		elif self.chb1.GetValue() == True:
			self.st11.SetLabel(self.instructions_for(10))
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
			self.st11.SetLabel(self.instructions_for(8))
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
			self.st11.SetLabel(self.instructions_for(7))
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
			self.st11.SetLabel(self.instructions_for(6))
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
			self.st11.SetLabel(self.instructions_for(5))
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
			self.st11.SetLabel(self.instructions_for(11))
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
			self.st11.SetLabel(self.instructions_for(2))
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
			self.st11.SetLabel(self.instructions_for(9))
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
			self.st11.SetLabel(self.instructions_for(1))
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
			self.st11.SetLabel(self.instructions_for(21))
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
			self.tracker.gv3 = self.parse_ids(self.ids_to_process, separation = ',')
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
			ids = self.parse_ids(self.ids_to_process, separation = ',')
		temp = []
		for track in self.tracker.valid_tracks:
			if track.id in ids:
				temp.append(track)
		if len(temp) > 0:
			for track in temp:
				self.tracker.release_id(track.id)
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
	frame = Tracker_GUI()
	frame.Show()
	app.MainLoop()
