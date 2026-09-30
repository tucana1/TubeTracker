"""TubeTracker's main window and ``main``.

One window: the start screen (open a movie, or one analysed before), a movie's analysis in progress, or the movie
itself (the field with every grain and tube drawn on it, the timeline of events, and the side panel for checking
and correcting grains). The analysis runs in a child process the window starts and follows (``jobs``); nothing
else has to be opened.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import wx
import wx.adv
import wx.svg

from . import theme
from .canvas import FieldCanvas
from .charts import Drawn, Timeline
from .corrections import Reviewer, ReviewError
from .dialogs import CompareFrame, ResultsFrame, SetupDialog, ShortcutsDialog
from .jobs import JobManager
from .overlay import EMERGED, needs_check, pos_at, state_at
from .panels import AnalysisPanel, SidePanel, StartPanel, bg
from .runfolder import DEFAULT_RUNS_ROOT, REPO, MOVIE_SUFFIXES, RunFolder, find_runs, folder_for, is_movie, \
    load_prefs, looks_like_run, movie_frame_count, save_prefs

ICON = Path(__file__).with_name("icon.svg")
MODES = (("n", "Movie"), ("h", "Contrast"), ("g", "Growth"))


class Legend(Drawn):
    ITEMS = (("germinated", "germinated"), ("notyet", "not yet"), ("never", "never"), ("lost", "lost"),
             ("excluded", "excluded"))

    def __init__(self, parent):
        super().__init__(parent, size=(-1, 22))
        self.SetMinSize((520, 22))
        self.SetBackgroundColour(theme.panel_bg())

    def _paint(self, e):
        dc = wx.PaintDC(self)
        dc.SetBackground(wx.Brush(theme.panel_bg()))
        dc.Clear()
        gc = wx.GraphicsContext.Create(dc)
        x, y = 4.0, self.GetClientSize()[1] / 2
        fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        gc.SetFont(theme.font(10), fg)
        for key, name in self.ITEMS:
            style = wx.PENSTYLE_SHORT_DASH if key in theme.DASHED else wx.PENSTYLE_SOLID
            gc.SetBrush(wx.TRANSPARENT_BRUSH)
            gc.SetPen(theme.pen((60, 60, 60), 4))
            gc.DrawEllipse(x, y - 5, 10, 10)
            gc.SetPen(theme.pen(theme.STATE[key], 2, style))
            gc.DrawEllipse(x, y - 5, 10, 10)
            gc.DrawText(name, x + 14, y - gc.GetTextExtent(name)[1] / 2)
            x += 20 + gc.GetTextExtent(name)[0] + 8
        gc.SetPen(theme.pen(theme.TUBE, 3))
        gc.StrokeLine(x, y, x + 14, y)
        gc.DrawText("tube", x + 18, y - gc.GetTextExtent("tube")[1] / 2)
        x += 18 + gc.GetTextExtent("tube")[0] + 10
        for rgb, name in ((theme.TIP, "tip"), (theme.CHECK, "to check")):
            gc.SetPen(wx.TRANSPARENT_PEN)
            gc.SetBrush(wx.Brush(theme.colour(rgb)))
            gc.DrawEllipse(x, y - 4, 8, 8)
            gc.DrawText(name, x + 12, y - gc.GetTextExtent(name)[1] / 2)
            x += 12 + gc.GetTextExtent(name)[0] + 10


class MainFrame(wx.Frame):
    def __init__(self, runs_root: str | Path = DEFAULT_RUNS_ROOT):
        super().__init__(None, title="TubeTracker", size=(1440, 900))
        self.runs_root = Path(runs_root).expanduser().resolve()
        self.jobs = JobManager(on_finish=lambda job: wx.CallAfter(self._job_finished, job))
        self.folder = self.data = self.frames = self.reviewer = None
        self.grains, self.events, self.checks = [], [], []
        self.population = self.summary = None
        self.b, self.mode, self.sel = 0, "n", None
        self.overlays = {"grains": True, "tubes": True, "names": True}
        self.tool, self.path_pts = None, []
        self.speed = 8
        self.results_win = None
        self.pool = ThreadPoolExecutor(max_workers=2)
        self._pending_prefetch = set()
        self._icons = None
        self.SetMinSize((1100, 700))
        self.SetIcons(self.icons())
        self._menus()
        self.root = bg(wx.Panel(self))
        self.start = StartPanel(self.root, self)
        self.analysis = AnalysisPanel(self.root, self)
        self.movie = self._movie_panel(self.root)
        s = wx.BoxSizer(wx.VERTICAL)
        for p in (self.start, self.analysis, self.movie):
            s.Add(p, 1, wx.EXPAND)
        self.root.SetSizer(s)
        self.CreateStatusBar(2)
        self.SetStatusWidths([-3, -2])
        self.poll = wx.Timer(self)
        self.player = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, lambda e: self._poll(), self.poll)
        self.Bind(wx.EVT_TIMER, lambda e: self._tick(), self.player)
        self.Bind(wx.EVT_CHAR_HOOK, self._on_key)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.show_start()
        self.poll.Start(600)

    # ================================================================ building
    def icons(self) -> wx.IconBundle:
        if self._icons is None:
            self._icons = wx.IconBundle()
            for size in (16, 32, 64, 128, 256):
                bmp = self.icon_bitmap(size)
                if bmp:
                    icon = wx.Icon()
                    icon.CopyFromBitmap(bmp)
                    self._icons.AddIcon(icon)
        return self._icons

    @staticmethod
    def icon_bitmap(size: int):
        try:
            return wx.svg.SVGimage.CreateFromFile(str(ICON)).ConvertToScaledBitmap((size, size))
        except Exception:  # noqa: BLE001 - no icon is better than no window
            return None

    def _menus(self):
        mb = wx.MenuBar()
        self._items = {}

        def add(menu, key, label, handler, kind=wx.ITEM_NORMAL):
            item = menu.Append(wx.ID_ANY, label, kind=kind)
            self.Bind(wx.EVT_MENU, lambda e: handler(), item)
            self._items[key] = item
            return item

        f = wx.Menu()
        add(f, "open", "Open Movie...\tCtrl+O", self.on_open_movie)
        add(f, "open_folder", "Open Analysis Folder...\tCtrl+Shift+O", self.on_open_folder)
        add(f, "close", "Close Movie\tCtrl+W", self.close_movie)
        f.AppendSeparator()
        add(f, "results", "Results...\tCtrl+R", self.on_results)
        add(f, "export", "Export Results\tCtrl+E", self.on_export)
        add(f, "compare", "Compare Movies...", self.on_compare)
        f.AppendSeparator()
        quit_ = f.Append(wx.ID_EXIT, "Quit TubeTracker\tCtrl+Q")
        self.Bind(wx.EVT_MENU, lambda e: self.Close(), quit_)
        mb.Append(f, "File")
        m = wx.Menu()
        add(m, "settings", "Settings...\tCtrl+,", self.on_settings)
        add(m, "analyse", "Analyse Again...", self.on_analyse_again)
        add(m, "cancel", "Cancel Analysis", self.on_cancel_analysis)
        mb.Append(m, "Movie")
        v = wx.Menu()
        for key, name in MODES:
            add(v, f"mode_{key}", name, lambda k=key: self.set_mode(k), wx.ITEM_RADIO)
        v.AppendSeparator()
        add(v, "cycle", "Next View (C)", lambda: self.set_mode({"n": "h", "h": "g", "g": "n"}[self.mode]))
        v.AppendSeparator()
        for key, name in (("grains", "Grains"), ("tubes", "Tubes"), ("names", "Names (I)")):
            it = add(v, f"ov_{key}", name, lambda k=key: self.toggle_overlay(k), wx.ITEM_CHECK)
            it.Check(True)
        v.AppendSeparator()
        add(v, "fit", "Fit Field (F)", self.fit)
        add(v, "zoomgrain", "Zoom to Grain (Z)", self.zoom_to_selected)
        add(v, "zin", "Zoom In\tCtrl+=", lambda: self.canvas.zoom_at(1.4))
        add(v, "zout", "Zoom Out\tCtrl+-", lambda: self.canvas.zoom_at(1 / 1.4))
        mb.Append(v, "View")
        g = wx.Menu()
        add(g, "play", "Play / Pause (Space)", self.toggle_play)
        add(g, "nextcheck", "Next Check (N)", lambda: self.goto_check(1))
        add(g, "prevcheck", "Previous Check (P)", lambda: self.goto_check(-1))
        add(g, "nextgrain", "Next Grain (])", lambda: self.step_grain(1))
        add(g, "prevgrain", "Previous Grain ([)", lambda: self.step_grain(-1))
        add(g, "first", "First Time (Home)", lambda: self.set_bin(0))
        add(g, "last", "Last Time (End)", lambda: self.set_bin(self.data.n_bins - 1) if self.data else None)
        mb.Append(g, "Go")
        c = wx.Menu()
        add(c, "confirm", "Confirm (Enter)", lambda: self.on_action("confirm"))
        add(c, "onset", "Onset Here (O)", lambda: self.on_action("onset"))
        add(c, "no_onset", "Never Germinated (Shift-O)", lambda: self.on_action("no_onset"))
        add(c, "tip", "Set Tip (T)", lambda: self.on_action("tip"))
        add(c, "path", "Draw Tube (D)", lambda: self.on_action("path"))
        add(c, "no_tube", "No Tube Here", lambda: self.on_action("no_tube"))
        add(c, "burst", "Burst or Gone Here (B)", lambda: self.on_action("burst"))
        c.AppendSeparator()
        add(c, "not_a_grain", "Not a Grain (X)", lambda: self.on_action("not_a_grain"))
        add(c, "clump", "Clump (K)", lambda: self.on_action("clump"))
        add(c, "include", "Include Again", lambda: self.on_action("include"))
        c.AppendSeparator()
        add(c, "undo", "Undo\tCtrl+Z", lambda: self.on_action("undo"))
        add(c, "revert", "Back to Model's Answer (U)", lambda: self.on_action("revert"))
        mb.Append(c, "Grain")
        t = wx.Menu()
        add(t, "legacy", "Legacy Manual Pipeline...", self.on_legacy)
        mb.Append(t, "Tools")
        h = wx.Menu()
        add(h, "keys", "Keyboard Shortcuts", lambda: ShortcutsDialog(self).ShowModal())
        about = h.Append(wx.ID_ABOUT, "About TubeTracker")
        self.Bind(wx.EVT_MENU, lambda e: self.on_about(), about)
        mb.Append(h, "Help")
        self.SetMenuBar(mb)

    def _movie_panel(self, parent):
        p = bg(wx.Panel(parent))
        outer = wx.BoxSizer(wx.VERTICAL)
        head = wx.BoxSizer(wx.HORIZONTAL)
        self.title = wx.StaticText(p, label="")
        self.title.SetFont(theme.font(14, bold=True))
        head.Add(self.title, 0, wx.ALIGN_CENTER_VERTICAL)
        self.line = wx.StaticText(p, label="")
        self.line.SetForegroundColour(theme.muted_fg())
        head.Add(self.line, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 16)
        head.AddStretchSpacer()
        self.warn = wx.StaticText(p, label="")
        self.warn.SetForegroundColour(theme.colour(theme.WARN_TEXT))
        head.Add(self.warn, 0, wx.ALIGN_CENTER_VERTICAL)
        outer.Add(head, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        body = wx.BoxSizer(wx.HORIZONTAL)
        stage = wx.BoxSizer(wx.VERTICAL)
        bar = wx.BoxSizer(wx.HORIZONTAL)
        self.mode_btns = {}
        for i, (key, name) in enumerate(MODES):
            rb = wx.RadioButton(p, label=name, style=wx.RB_GROUP if i == 0 else 0)
            rb.Bind(wx.EVT_RADIOBUTTON, lambda e, k=key: self.set_mode(k))
            bar.Add(rb, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 10)
            self.mode_btns[key] = rb
        self.mode_btns["g"].SetToolTip("This time minus 6 bins earlier: growing tips stand out")
        bar.AddSpacer(12)
        self.ov_boxes = {}
        for key, name in (("grains", "Grains"), ("tubes", "Tubes"), ("names", "Names")):
            cb = wx.CheckBox(p, label=name)
            cb.SetValue(True)
            cb.Bind(wx.EVT_CHECKBOX, lambda e, k=key: self.toggle_overlay(k))
            bar.Add(cb, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 10)
            self.ov_boxes[key] = cb
        bar.AddSpacer(12)
        fit = wx.Button(p, label="Fit", style=wx.BU_EXACTFIT)
        fit.Bind(wx.EVT_BUTTON, lambda e: self.fit())
        bar.Add(fit, 0, wx.ALIGN_CENTER_VERTICAL)
        self.zoom_label = wx.StaticText(p, label="", size=(48, -1))
        self.zoom_label.SetForegroundColour(theme.muted_fg())
        bar.Add(self.zoom_label, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 8)
        bar.AddStretchSpacer()
        bar.Add(Legend(p), 0, wx.ALIGN_CENTER_VERTICAL)
        stage.Add(bar, 0, wx.EXPAND | wx.ALL, 6)
        self.canvas = FieldCanvas(p, self)
        stage.Add(self.canvas, 1, wx.EXPAND)
        tr = wx.BoxSizer(wx.HORIZONTAL)
        self.play_btn = wx.Button(p, label="Play")
        self.play_btn.Bind(wx.EVT_BUTTON, lambda e: self.toggle_play())
        tr.Add(self.play_btn, 0, wx.ALIGN_CENTER_VERTICAL)
        self.slider = wx.Slider(p, minValue=0, maxValue=1, value=0)
        self.slider.Bind(wx.EVT_SLIDER, lambda e: self.set_bin(self.slider.GetValue()))
        tr.Add(self.slider, 1, wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 8)
        self.time_label = wx.StaticText(p, label="", size=(170, -1))
        tr.Add(self.time_label, 0, wx.ALIGN_CENTER_VERTICAL)
        stage.Add(tr, 0, wx.EXPAND | wx.ALL, 6)
        self.timeline = Timeline(p, self)
        stage.Add(self.timeline, 0, wx.EXPAND)
        body.Add(stage, 1, wx.EXPAND)
        self.side = SidePanel(p, self)
        body.Add(self.side, 0, wx.EXPAND)
        outer.Add(body, 1, wx.EXPAND | wx.TOP, 6)
        p.SetSizer(outer)
        return p

    # ================================================================ screens
    def _show(self, which):
        for p in (self.start, self.analysis, self.movie):
            p.Show(p is which)
        self.root.Layout()
        on_movie = which is self.movie
        for key in ("close", "results", "export", "settings", "analyse", "cycle", "fit", "zoomgrain", "zin", "zout",
                    "play", "nextcheck", "prevcheck", "nextgrain", "prevgrain", "first", "last", "confirm", "onset",
                    "no_onset", "tip", "path", "no_tube", "burst", "not_a_grain", "clump", "include", "undo", "revert",
                    "mode_n", "mode_h", "mode_g", "ov_grains", "ov_tubes", "ov_names"):
            self._items[key].Enable(on_movie)
        self._items["close"].Enable(which is not self.start)
        self._items["settings"].Enable(which is not self.start)
        self._items["cancel"].Enable(False)

    def show_start(self):
        self.stop_play()
        self.SetTitle("TubeTracker")
        self.start.refresh([f.listing() for f in find_runs(self.runs_root)], self.jobs.status(), str(self.runs_root))
        self._show(self.start)

    def close_movie(self):
        self.stop_play()
        self.folder = self.data = self.frames = self.reviewer = None
        self.grains, self.sel, self.tool = [], None, None
        if self.results_win:
            self.results_win.Destroy()
            self.results_win = None
        self.show_start()

    # ================================================================ opening
    def on_open_movie(self):
        wild = "Movies|" + ";".join(f"*{s}" for s in MOVIE_SUFFIXES) + "|All files|*"
        with wx.FileDialog(self, "Open Movie", wildcard=wild, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.open_path(dlg.GetPath())

    def on_open_folder(self):
        with wx.DirDialog(self, "Open Analysis Folder", defaultPath=str(self.runs_root),
                          style=wx.DD_DIR_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.open_path(dlg.GetPath())

    def open_path(self, path):
        p = Path(path).expanduser()
        if not p.exists():
            return self.error(f"Not found: {p}")
        if p.is_file() and not is_movie(p) and p.name != "predictions.json":
            return self.error(f"Not a movie: {p.name}")
        folder = folder_for(p, self.runs_root)
        if p.is_dir() and not looks_like_run(folder.root) and not folder.has_analysis():
            return self.error(f"Not an analysis folder: {p}")
        if p.is_file() and is_movie(p):
            setup = folder.load_setup()
            if setup.get("movie") != str(p.resolve()) or not setup.get("n_frames"):
                folder.save_setup({"movie": str(p.resolve()), "movie_name": p.name,
                                   "n_frames": folder.n_frames() or setup.get("n_frames") or movie_frame_count(p)})
        self.close_movie()
        self.folder = folder
        if folder.has_analysis():
            self.load_run(folder)
            if not folder.is_setup():
                wx.CallAfter(self.on_settings)
        else:
            self.show_analysis()
            if not folder.is_setup():
                wx.CallAfter(self.on_settings, True)

    def load_run(self, folder: RunFolder):
        from .imaging import FrameSource
        from .model import RunData

        wx.BeginBusyCursor()
        try:
            self.folder = folder
            self.data = RunData(folder)
            self.frames = FrameSource(folder.cache)
            self.reviewer = Reviewer(self.data)
            self.reviewer.start(background=True)
        except Exception as exc:  # noqa: BLE001 - a broken folder must not take the window down
            self.data = None
            self.error(f"Could not open {folder.root}: {exc}")
            self.show_start()
            return
        finally:
            wx.EndBusyCursor()
        d = self.data
        self.recompute()
        self.b = self.population["t50_bin"] if self.population["t50_bin"] is not None else d.n_bins // 2
        self.slider.SetRange(0, d.n_bins - 1)
        self.sel = None
        self._show(self.movie)
        self.canvas.fit_view()
        self.update_all()
        self.status("Opened " + folder.movie_name())

    def show_analysis(self):
        self.SetTitle(f"TubeTracker: {self.folder.movie_name()}")
        job = self.jobs.job_for(str(self.folder.root))
        self.analysis.show(self.folder.movie_name(), job.to_json() if job else None)
        self._show(self.analysis)
        self._items["cancel"].Enable(bool(job) and job.state in ("running", "queued"))

    # ================================================================ settings and analysis
    def on_settings(self, analyse_after: bool = False):
        f = self.folder
        if f is None:
            return
        analysed = f.has_analysis()
        dlg = SetupDialog(self, f.movie_name(), f.n_frames(), f.load_setup(), load_prefs(self.runs_root),
                          prepared=f.has_cache(), analyse=not analysed and not self._busy(f))
        if dlg.ShowModal() == wx.ID_OK:
            try:
                setup = f.apply_setup(dlg.values())
            except ValueError as exc:
                dlg.Destroy()
                return self.error(str(exc))
            save_prefs(self.runs_root, um_per_px=setup.get("um_per_px"))
            if self.data is not None:
                with self.reviewer.lock:
                    self.data.set_units(setup)
                self.recompute()
                self.update_all()
            elif not analysed:
                self.on_analyse()
        dlg.Destroy()

    def _busy(self, folder) -> bool:
        job = self.jobs.job_for(str(folder.root))
        return bool(job) and job.state in ("running", "queued")

    def on_analyse(self):
        f = self.folder
        movie = f.movie_path()
        if not f.has_cache() and not (movie and movie.exists()):
            return self.error(f"The movie is not at {movie} any more. Open it from where it is now.")
        setup = f.load_setup()
        self.jobs.submit(str(f.root), str(movie) if movie and movie.exists() else None, f.movie_name(),
                         bool(setup.get("flatfield")))
        if self.data is None:
            self.show_analysis()

    def on_analyse_again(self):
        if self.folder is None:
            return
        if wx.MessageBox("Analyse this movie again? The current analysis and its checks are kept in its "
                         "'earlier' folder.", "Analyse Again", wx.OK | wx.CANCEL | wx.ICON_QUESTION, self) != wx.OK:
            return
        folder = self.folder
        self.close_movie()
        self.folder = folder
        self.on_analyse()
        self.show_analysis()

    def on_cancel_analysis(self):
        job = self.jobs.job_for(str(self.folder.root)) if self.folder else None
        if job and job.state in ("running", "queued"):
            self.jobs.cancel(job.id)

    def _job_finished(self, job):
        if self.folder is not None and str(self.folder.root) == job.folder and self.data is None:
            if job.state == "done":
                self.load_run(RunFolder(job.folder))
            else:
                self.show_analysis()
        if self.start.IsShown():
            self.show_start()
        word = {"done": "finished", "failed": "failed", "cancelled": "cancelled"}.get(job.state, job.state)
        self.status(f"Analysis of {job.name} {word}", 1)

    def _poll(self):
        st = self.jobs.status()
        run = next((j for j in st["active"] if j["state"] == "running"), None)
        if run:
            prog = f"grain {run['k']} of {run['n']}" if run["phase"] == "grains" and run["n"] else run["label"]
            self.status(f"Analysing {run['name']}: {prog}", 1)
        if self.analysis.IsShown() and self.folder is not None:
            job = self.jobs.job_for(str(self.folder.root))
            self.analysis.show(self.folder.movie_name(), job.to_json() if job else None)
            self._items["cancel"].Enable(bool(job) and job.state in ("running", "queued"))
        if self.start.IsShown() and st["active"]:
            self.start.refresh([f.listing() for f in find_runs(self.runs_root)], st, str(self.runs_root))
        if self.reviewer is not None and not getattr(self, "_review_seen", False) and \
                (self.reviewer.ready() or self.reviewer.error):
            self._review_seen = True
            self.recompute()
            self.update_side()
            if self.reviewer.error:
                self.status(f"Corrections unavailable: {self.reviewer.error}")

    # ================================================================ state
    def recompute(self):
        d = self.data
        lock = self.reviewer.lock if self.reviewer else None
        if lock:
            lock.acquire()
        try:
            self.grains = d.grains()
            self.population = d.population(self.grains)
            self.events = d.events(self.grains, self.population)
            self.checks = d.checks(self.grains)
            self.summary = d.summary(self.grains, self.population)
        finally:
            if lock:
                lock.release()
        self._by_id = {g["id"]: g for g in self.grains}

    def selected(self):
        return self._by_id.get(self.sel) if self.sel and self.data else None

    def update_all(self):
        d = self.data
        s = d.setup
        name = s.get("sample_id") or self.folder.name
        self.SetTitle(f"TubeTracker: {name}")
        self.title.SetLabel("  ".join(x for x in (name, s.get("genotype"), f"rep {s['replicate']}" if s.get("replicate")
                                                  else "") if x))
        self.line.SetLabel(self.summary["line"])
        warns = d.warnings()
        self.warn.SetLabel("  ·  ".join(w["text"] for w in warns))
        self.warn.SetToolTip("\n\n".join(w.get("detail") or w["text"] for w in warns))
        for key, rb in self.mode_btns.items():
            rb.SetValue(key == self.mode)
        self.update_side(lists=True)
        self.redraw()
        self.movie.Layout()
        if self.results_win:
            self.results_win.refresh()

    def update_side(self, lists: bool = True):
        if self.data is None:
            return
        ready = self.reviewer is not None and self.reviewer.ready()
        note = "" if ready else (f"Corrections unavailable: {self.reviewer.error}" if self.reviewer and self.reviewer.error
                                 else "Preparing corrections...")
        self.side.update_nav(self.checks, self.sel)
        if lists:
            self.side.update_lists(self.checks, self.events, self.data, self.sel)
        self.side.update_grain(self.selected(), self.data, self.b, ready, note)

    def redraw(self):
        if self.data is None:
            return
        d = self.data
        self.slider.SetValue(self.b)
        self.time_label.SetLabel(f"{d.when(d.frame(self.b))}   bin {self.b}/{d.n_bins - 1}")
        self.zoom_label.SetLabel(f"{self.canvas.zoom_percent()}%")
        self.canvas.Refresh()
        self.timeline.Refresh()
        self.side.view.Refresh()
        self.side.curve.Refresh()

    def on_view_changed(self):
        self.zoom_label.SetLabel(f"{self.canvas.zoom_percent()}%")

    # ================================================================ time
    def set_bin(self, b):
        if self.data is None:
            return
        b = int(max(0, min(int(round(b)), self.data.n_bins - 1)))
        if b == self.b:
            return
        if self.tool == "path" and self.path_pts:
            self.set_tool(None)
            self.status("Drawing cancelled: the time changed")
        self.b = b
        self.redraw()
        g = self.selected()
        if g is not None:
            self.side.update_grain(g, self.data, self.b, self.reviewer is not None and self.reviewer.ready())
        self._prefetch()

    def _prefetch(self):
        d, fr = self.data, self.frames
        for k in (1, 2, 3, -1):
            b = self.b + k
            key = (b, self.mode)
            if 0 <= b < d.n_bins and key not in self._pending_prefetch:
                self._pending_prefetch.add(key)
                fut = self.pool.submit(fr.field, b, self.mode)
                fut.add_done_callback(lambda f, key=key: self._pending_prefetch.discard(key))

    def toggle_play(self):
        if self.data is None:
            return
        if self.player.IsRunning():
            return self.stop_play()
        if self.b >= self.data.n_bins - 1:
            self.set_bin(0)
        self.player.Start(int(1000 / self.speed))
        self.play_btn.SetLabel("Pause")

    def stop_play(self):
        if self.player.IsRunning():
            self.player.Stop()
        self.play_btn.SetLabel("Play")

    def _tick(self):
        if self.data is None or self.b >= self.data.n_bins - 1:
            return self.stop_play()
        self.set_bin(self.b + 1)

    def set_mode(self, mode):
        self.mode = mode
        for key, rb in self.mode_btns.items():
            rb.SetValue(key == mode)
        self._items[f"mode_{mode}"].Check(True)
        self.redraw()
        self._prefetch()

    def toggle_overlay(self, key):
        self.overlays[key] = not self.overlays[key]
        self.ov_boxes[key].SetValue(self.overlays[key])
        self._items[f"ov_{key}"].Check(self.overlays[key])
        self.redraw()

    def fit(self):
        if self.data is not None:
            self.canvas.fit_view()
            self.on_view_changed()

    def zoom_to_selected(self):
        g = self.selected()
        if g is not None:
            self.canvas.centre_on(*pos_at(g, self.b))

    # ================================================================ selection, checks, events
    def select(self, gid, zoom: bool = False):
        if self.data is None or gid not in self._by_id:
            return
        self.sel = gid
        if self.tool:
            self.set_tool(None)
        if zoom:
            self.canvas.centre_on(*pos_at(self._by_id[gid], self.b))
        self.side.book.SetSelection(0)
        self.update_side(lists=False)
        self.side.update_lists(self.checks, self.events, self.data, self.sel)
        self.redraw()

    def goto_check(self, direction: int):
        items = self.checks
        if not items:
            return self.status("Nothing to check")
        cur = next((i for i, c in enumerate(items) if c["gid"] == self.sel), None)
        n = len(items)
        for k in range(1, n + 1):
            i = ((cur if cur is not None else (-1 if direction > 0 else 0)) + direction * k) % n
            if not items[i]["done"] or k == n:
                return self.open_check(i)

    def open_check(self, index: int):
        if not 0 <= index < len(self.checks):
            return
        item = self.checks[index]
        self.b = item["bin"]
        self.select(item["gid"], zoom=True)
        self._prefetch()

    def goto_event(self, ev):
        self.b = ev["bin"]
        if ev.get("gid"):
            self.select(ev["gid"], zoom=True)
        else:
            self.redraw()
            self.update_side(lists=False)

    def goto_event_index(self, i):
        if 0 <= i < len(self.events):
            self.goto_event(self.events[i])

    def step_grain(self, direction: int):
        live = [g for g in self.grains if not g["excluded"]]
        if not live:
            return
        i = next((k for k, g in enumerate(live) if g["id"] == self.sel), -1)
        self.select(live[(i + direction) % len(live)]["id"], zoom=True)

    def grain_tip(self, g) -> str:
        st = state_at(g, self.b)
        L = g["L"][self.b]
        words = {"germinated": "germinated", "notyet": "not yet germinated", "never": "never germinated",
                 "lost": "lost", "excluded": "excluded", "unobservable": "not readable"}[st]
        tip = f"{g['id']}: {words}" + (f", {self.data.length_words(L)}" if L > 0.5 else "")
        if needs_check(g):
            tip += "\n" + " · ".join(r["text"] for r in g["check"])
        return tip

    # ================================================================ corrections
    def can_undo(self) -> bool:
        return self.reviewer is not None and self.reviewer.ready() and self.reviewer.undo_depth() > 0

    def set_tool(self, tool):
        self.tool = tool
        self.path_pts = []
        cursor = wx.Cursor(wx.CURSOR_CROSS if tool else wx.CURSOR_DEFAULT)
        self.canvas.SetCursor(cursor)
        self.side.view.SetCursor(cursor)
        if tool == "tip":
            self.status(f"Click the tip of {self.sel}'s tube (Esc cancels)")
        elif tool == "path":
            self.status(f"Click where {self.sel}'s tube leaves the grain, then along it to the tip; Enter saves")
        self.redraw()

    def on_click(self, ref, grain, zoomed: bool = False):
        if self.tool == "tip":
            self.set_tool(None)
            return self.correct("tip", b=self.b, x=ref[0], y=ref[1])
        if self.tool == "path":
            self.path_pts.append(list(ref))
            return self.redraw()
        if grain is not None and grain["id"] != self.sel:
            self.select(grain["id"])

    def on_action(self, key):
        if self.data is None:
            return
        if key == "undo":
            return self.correct("undo")
        if self.selected() is None:
            return self.status("Select a grain first")
        if key in ("tip", "path"):
            return self.set_tool(None if self.tool == key else key)
        if key == "not_a_grain" and self.selected()["excluded"]:
            key = "include"
        if key in ("not_a_grain", "clump"):
            return self.correct("exclude", reason=key)
        if key in ("onset", "no_tube", "burst"):
            return self.correct(key, b=self.b)
        if key == "revert":
            if wx.MessageBox(f"Forget the corrections to {self.sel}?", "Back to Model's Answer",
                             wx.OK | wx.CANCEL, self) != wx.OK:
                return
        return self.correct(key)

    def correct(self, action, **kw):
        if self.reviewer is None or not self.reviewer.ready():
            return self.status("Corrections are not ready yet")
        wx.BeginBusyCursor()
        try:
            done = self.reviewer.act(self.sel, action, **kw)
        except ReviewError as exc:
            return self.status(str(exc))
        finally:
            wx.EndBusyCursor()
        self.recompute()
        if action == "undo" and done.get("gid") and done["gid"] != self.sel:
            self.select(done["gid"], zoom=True)
        self.update_all()
        self.status(f"Saved: {done['message']}")
        if action == "confirm":
            wx.CallLater(250, self.goto_check, 1)

    def finish_path(self):
        pts = list(self.path_pts)
        self.set_tool(None)
        if len(pts) < 2:
            return self.status("Click at least where the tube leaves the grain and its tip")
        self.correct("path", b=self.b, points=pts)

    # ================================================================ results, export, compare
    def on_results(self):
        if self.data is None:
            return
        if self.results_win is None:
            self.results_win = ResultsFrame(self, self)
            self.results_win.Bind(wx.EVT_CLOSE, self._results_closed)
        self.results_win.refresh()
        self.results_win.Show()
        self.results_win.Raise()

    def _results_closed(self, e):
        self.results_win = None
        e.Skip()

    def on_export(self):
        if self.data is None:
            return
        from .exports import export_movie
        wx.BeginBusyCursor()
        try:
            with self.reviewer.lock:
                out = export_movie(self.data)
        except Exception as exc:  # noqa: BLE001 - reported
            return self.error(f"Export failed: {exc}")
        finally:
            wx.EndBusyCursor()
        self.status(f"Exported to {out['folder']}")
        reveal(out["folder"])

    def on_compare(self):
        runs = [f.listing() for f in find_runs(self.runs_root)]
        if not any(r["analysed"] for r in runs):
            return self.status("No analysed movies to compare")
        CompareFrame(self, self, runs).Show()

    def compare_rows(self, folders: list[str]):
        from .exports import compare_rows
        return compare_rows([RunFolder(f) for f in folders])

    def on_export_compare(self, folders: list[str]):
        from .exports import export_summary
        if not folders:
            return
        try:
            out = export_summary([RunFolder(f) for f in folders], self.runs_root / "summary")
        except Exception as exc:  # noqa: BLE001 - reported
            return self.error(f"Export failed: {exc}")
        self.status(f"Exported to {out['folder']}")
        reveal(out["folder"])

    def on_legacy(self):
        subprocess.Popen([sys.executable, "-m", "tubetracker", "--legacy"], cwd=str(REPO))

    def on_about(self):
        info = wx.adv.AboutDialogInfo()
        info.SetName("TubeTracker")
        from sparsetrack import __version__
        info.SetVersion(f"SparseTrack {__version__}")
        info.SetDescription("Pollen germination and tube growth in time-lapse movies.")
        bmp = self.icon_bitmap(96)
        if bmp:
            icon = wx.Icon()
            icon.CopyFromBitmap(bmp)
            info.SetIcon(icon)
        wx.adv.AboutBox(info, self)

    # ================================================================ keys, status, closing
    def _on_key(self, e):
        focus = wx.Window.FindFocus()
        if (self.data is None or not self.movie.IsShown() or e.CmdDown() or e.AltDown()
                or isinstance(focus, (wx.TextCtrl, wx.SpinCtrl, wx.ComboBox))):
            e.Skip()
            return
        code, shift = e.GetKeyCode(), e.ShiftDown()
        in_list = isinstance(focus, (wx.ListCtrl, wx.Slider, wx.Choice))
        if code == wx.WXK_ESCAPE:
            return self.set_tool(None) if self.tool else None
        if code in (wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER):
            if self.tool == "path":
                return self.finish_path()
            if not in_list:
                return self.on_action("confirm")
        if code == wx.WXK_BACK and self.tool == "path" and self.path_pts:
            self.path_pts.pop()
            return self.redraw()
        if not in_list:
            step = 10 if shift else 1
            nav = {wx.WXK_SPACE: self.toggle_play, wx.WXK_LEFT: lambda: self.set_bin(self.b - step),
                   wx.WXK_RIGHT: lambda: self.set_bin(self.b + step), wx.WXK_HOME: lambda: self.set_bin(0),
                   wx.WXK_END: lambda: self.set_bin(self.data.n_bins - 1)}
            if code in nav:
                return nav[code]()
        ch = chr(code) if 32 <= code < 127 else ""
        keys = {"N": lambda: self.goto_check(1), "P": lambda: self.goto_check(-1), "]": lambda: self.step_grain(1),
                "[": lambda: self.step_grain(-1), "C": lambda: self.set_mode({"n": "h", "h": "g", "g": "n"}[self.mode]),
                "G": lambda: self.set_mode("n" if self.mode == "g" else "g"), "F": self.fit, "Z": self.zoom_to_selected,
                "I": lambda: self.toggle_overlay("names"), "=": lambda: self.canvas.zoom_at(1.4),
                "-": lambda: self.canvas.zoom_at(1 / 1.4),
                "O": lambda: self.on_action("no_onset" if shift else "onset"), "T": lambda: self.on_action("tip"),
                "D": lambda: self.on_action("path"), "B": lambda: self.on_action("burst"),
                "X": lambda: self.on_action("not_a_grain"), "K": lambda: self.on_action("clump"),
                "U": lambda: self.on_action("revert")}
        if ch.upper() in keys and (not in_list or ch.upper() in "NP"):
            return keys[ch.upper()]()
        e.Skip()

    def status(self, text: str, field: int = 0):
        self.SetStatusText(text, field)

    def error(self, text: str):
        wx.MessageBox(text, "TubeTracker", wx.OK | wx.ICON_WARNING, self)

    def _on_close(self, e):
        if any(j["state"] == "running" for j in self.jobs.status()["active"]):
            if wx.MessageBox("An analysis is running. Quit and stop it?", "Quit TubeTracker",
                             wx.OK | wx.CANCEL | wx.ICON_QUESTION, self) != wx.OK:
                e.Veto()
                return
        self._shutdown()
        e.Skip()

    def _shutdown(self):
        for t in (self.poll, self.player):
            if t.IsRunning():
                t.Stop()
        self.jobs.shutdown()
        self.pool.shutdown(wait=False, cancel_futures=True)
        dock = getattr(self, "_dock", None)
        if dock is not None:  # a Dock icon left behind would keep the app from quitting
            dock.RemoveIcon()
            dock.Destroy()
            self._dock = None

    def Destroy(self):
        self._shutdown()
        return super().Destroy()

    # ================================================================ for tests and screenshots
    def snapshot(self, path) -> None:
        """Save the window's contents (without its title bar) as a PNG."""
        self.Update()
        src = wx.ClientDC(self.root)
        w, h = self.root.GetClientSize()
        bmp = wx.Bitmap(w, h)
        mem = wx.MemoryDC(bmp)
        mem.Blit(0, 0, w, h, src, 0, 0)
        mem.SelectObject(wx.NullBitmap)
        bmp.SaveFile(str(path), wx.BITMAP_TYPE_PNG)


def reveal(path) -> None:
    """Show a file or folder in the Finder (macOS)."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="TubeTracker", description="Pollen germination movies: analyse, check, export.")
    ap.add_argument("paths", nargs="*", help="a movie or analysis folder to open")
    ap.add_argument("--runs", default=os.environ.get("TUBETRACKER_RUNS", str(DEFAULT_RUNS_ROOT)),
                    help=f"where each movie's analysis folder goes (default {DEFAULT_RUNS_ROOT})")
    ap.add_argument("--legacy", action="store_true", help="the old manual pipeline (Hough grains, tip templates)")
    return ap.parse_args(argv)


def open_window(args: argparse.Namespace) -> MainFrame:
    """The main window, shown, with the Dock icon, opening ``args.paths[0]`` once the event loop runs."""
    frame = MainFrame(args.runs)
    frame.Show()
    if sys.platform == "darwin":
        try:  # the Dock icon
            bmp = frame.icon_bitmap(256)
            if bmp:
                icon = wx.Icon()
                icon.CopyFromBitmap(bmp)
                frame._dock = wx.adv.TaskBarIcon(wx.adv.TBI_DOCK)
                frame._dock.SetIcon(icon, "TubeTracker")
        except Exception:  # noqa: BLE001 - cosmetic
            pass
    if args.paths:
        wx.CallAfter(frame.open_path, args.paths[0])
    return frame


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.legacy:
        from tubetracker.gui import main as legacy_main
        legacy_main()
        return
    app = wx.App(False)
    app.SetAppName("TubeTracker")
    app.SetAppDisplayName("TubeTracker")
    open_window(args)
    app.MainLoop()
