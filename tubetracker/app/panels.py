"""The window's panels: the start screen (movies), the analysis progress of a movie not analysed yet, and the side
panel (the check list navigator, the selected grain or the movie at a glance, the check list, the events)."""

from __future__ import annotations

import time
from pathlib import Path

import wx
import wx.adv

from . import theme
from .canvas import GrainView
from .charts import GrowthCurve
from .overlay import EMERGED, MIN_TUBE_PX, state_at
from .runfolder import REPO

STATE_WORDS = {"germinated": "germinated", "notyet": "not yet germinated", "never": "never germinated",
               "lost": "lost", "excluded": "excluded", "unobservable": "not readable"}
REVIEW_WORDS = {"model": "not checked", "partly checked": "partly checked", "checked": "checked",
                "corrected": "corrected", "excluded": "excluded"}


def bg(win, colour=None):
    win.SetBackgroundColour(colour or theme.panel_bg())
    return win


def fmt_left(s: float) -> str:
    """A time still to go, as roughly as it is known: ``40 s``, ``7 min``, ``1 h 20 min``."""
    s = int(round(s or 0))
    if s < 60:
        return f"{s} s"
    m = int(round(s / 60))
    return f"{m // 60} h {m % 60} min" if m >= 60 else f"{m} min"


def fmt_secs(s: float) -> str:
    s = int(round(s or 0))
    return f"{s // 3600} h {(s % 3600) // 60} min" if s >= 3600 else f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


class StartPanel(wx.Panel):
    """No movie open: open one, or pick one analysed before."""

    COLS = (("Movie", 300), ("Sample", 200), ("Genotype", 100), ("Rep.", 50), ("Status", 150), ("Changed", 130))

    def __init__(self, parent, ctl):
        super().__init__(parent)
        bg(self)
        self.ctl = ctl
        self.runs = []
        outer = wx.BoxSizer(wx.VERTICAL)
        head = wx.BoxSizer(wx.HORIZONTAL)
        icon = ctl.icon_bitmap(40)
        if icon:
            head.Add(wx.StaticBitmap(self, bitmap=icon), 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 10)
        title = wx.StaticText(self, label="TubeTracker")
        title.SetFont(theme.font(20, bold=True))
        head.Add(title, 0, wx.ALIGN_CENTER_VERTICAL)
        outer.Add(head, 0, wx.LEFT | wx.TOP, 24)
        buttons = wx.BoxSizer(wx.HORIZONTAL)
        self.open_movie = wx.Button(self, label="Open Movie...")
        self.open_movie.SetDefault()
        self.open_folder = wx.Button(self, label="Open Analysis Folder...")
        buttons.Add(self.open_movie, 0, wx.RIGHT, 8)
        buttons.Add(self.open_folder, 0)
        outer.Add(buttons, 0, wx.LEFT | wx.TOP, 24)
        self.hint = wx.StaticText(self, label="")
        self.hint.SetForegroundColour(theme.muted_fg())
        outer.Add(self.hint, 0, wx.LEFT | wx.TOP, 24)
        lab = wx.StaticText(self, label="Movies")
        lab.SetFont(theme.font(13, bold=True))
        outer.Add(lab, 0, wx.LEFT | wx.TOP, 18)
        self.list = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for i, (name, width) in enumerate(self.COLS):
            self.list.InsertColumn(i, name, width=width)
        outer.Add(self.list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 24)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.where = wx.StaticText(self, label="")
        self.where.SetForegroundColour(theme.muted_fg())
        row.Add(self.where, 1, wx.ALIGN_CENTER_VERTICAL)
        self.open_btn = wx.Button(self, label="Open")
        self.compare_btn = wx.Button(self, label="Compare Movies...")
        row.Add(self.compare_btn, 0, wx.LEFT, 8)
        row.Add(self.open_btn, 0, wx.LEFT, 8)
        outer.Add(row, 0, wx.EXPAND | wx.ALL, 24)
        self.SetSizer(outer)
        self.open_movie.Bind(wx.EVT_BUTTON, lambda e: ctl.on_open_movie())
        self.open_folder.Bind(wx.EVT_BUTTON, lambda e: ctl.on_open_folder())
        self.open_btn.Bind(wx.EVT_BUTTON, lambda e: self._open_selected())
        self.compare_btn.Bind(wx.EVT_BUTTON, lambda e: ctl.on_compare())
        self.list.Bind(wx.EVT_LIST_ITEM_ACTIVATED, lambda e: self._open_selected())

    def refresh(self, runs: list[dict], jobs: dict, runs_root: str):
        self.runs = runs
        busy = {j["folder"]: j for j in jobs.get("active", [])}
        sel = self.list.GetFirstSelected()
        self.list.DeleteAllItems()
        for i, r in enumerate(runs):
            j = busy.get(r["folder"])
            if j:
                status = (f"analysing {j['k']}/{j['n']}" if j["phase"] == "grains" and j["n"] else
                          "analysing" if j["state"] == "running" else "queued")
            elif r["analysed"]:
                status = "analysed, reviewed" if r["reviewed"] else "analysed"
            else:
                status = "prepared" if r["prepared"] else "not analysed"
            changed = time.strftime("%d %b %H:%M", time.localtime(r["updated"])) if r["updated"] else ""
            self.list.InsertItem(i, r["movie"])
            for c, v in enumerate((r["sample_id"], r["genotype"], r["replicate"], status, changed), start=1):
                self.list.SetItem(i, c, str(v or ""))
        if 0 <= sel < len(runs):
            self.list.Select(sel)
        root = Path(runs_root)
        shown = root.relative_to(REPO) if REPO in root.parents else root
        self.where.SetLabel(f"{len(runs)} in {shown}")
        self.hint.SetLabel("Open a movie to analyse it, or double-click one below to see its analysis." if runs else
                           "Open a movie to analyse it. Its analysis is kept, and listed here.")
        self.open_btn.Enable(bool(runs))
        self.compare_btn.Enable(sum(1 for r in runs if r["analysed"]) >= 1)

    def _open_selected(self):
        i = self.list.GetFirstSelected()
        if 0 <= i < len(self.runs):
            self.ctl.open_path(self.runs[i]["folder"])


class AnalysisPanel(wx.Panel):
    """A movie that has not been analysed: start the analysis, follow it, cancel it."""

    ROWS = ("probe", "prepare", "register", "census", "maps", "speed", "grains", "finish")

    def __init__(self, parent, ctl):
        super().__init__(parent)
        bg(self)
        self.ctl = ctl
        col = wx.BoxSizer(wx.VERTICAL)
        self.title = wx.StaticText(self, label="")
        self.title.SetFont(theme.font(16, bold=True))
        col.Add(self.title, 0, wx.BOTTOM, 18)
        from .jobs import PHASE_LABELS
        grid = wx.FlexGridSizer(cols=2, vgap=6, hgap=24)
        self.rows = {}
        for key in self.ROWS:
            name = wx.StaticText(self, label=PHASE_LABELS[key])
            state = wx.StaticText(self, label="")
            grid.Add(name, 0)
            grid.Add(state, 0)
            self.rows[key] = (name, state)
        self._font = state.GetFont()  # the labels' own font (the system's GUI font is smaller on macOS)
        col.Add(grid, 0, wx.BOTTOM, 18)
        self.message = wx.StaticText(self, label="")
        col.Add(self.message, 0, wx.BOTTOM, 10)
        self.gauge = wx.Gauge(self, range=1000, size=(460, -1))
        col.Add(self.gauge, 0, wx.EXPAND | wx.BOTTOM, 8)
        self.detail = wx.StaticText(self, label="")
        self.detail.SetForegroundColour(theme.muted_fg())
        col.Add(self.detail, 0, wx.BOTTOM, 18)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.start = wx.Button(self, label="Analyse")
        self.cancel = wx.Button(self, label="Cancel")
        self.settings = wx.Button(self, label="Settings...")
        self.back = wx.Button(self, label="Back to Movies")
        self.back.SetToolTip("The list of movies (the analysis goes on meanwhile)")
        for b in (self.start, self.cancel, self.settings, self.back):
            row.Add(b, 0, wx.RIGHT, 8)
        col.Add(row, 0, wx.BOTTOM, 18)
        self.logpane = wx.CollapsiblePane(self, label="Details")
        self.log = wx.TextCtrl(self.logpane.GetPane(), style=wx.TE_MULTILINE | wx.TE_READONLY, size=(620, 200))
        self.log.SetFont(wx.Font(wx.FontInfo(10).Family(wx.FONTFAMILY_TELETYPE)))
        ps = wx.BoxSizer(wx.VERTICAL)
        ps.Add(self.log, 1, wx.EXPAND)
        self.logpane.GetPane().SetSizer(ps)
        col.Add(self.logpane, 0, wx.EXPAND)
        outer = wx.BoxSizer(wx.HORIZONTAL)
        outer.AddStretchSpacer(1)
        outer.Add(col, 0, wx.TOP, 60)
        outer.AddStretchSpacer(2)
        self.SetSizer(outer)
        self.start.Bind(wx.EVT_BUTTON, lambda e: ctl.on_analyse())
        self.cancel.Bind(wx.EVT_BUTTON, lambda e: ctl.on_cancel_analysis())
        self.settings.Bind(wx.EVT_BUTTON, lambda e: ctl.on_settings())
        self.back.Bind(wx.EVT_BUTTON, lambda e: ctl.close_movie())
        self.logpane.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED, lambda e: self.Layout())

    def show(self, name: str, job: dict | None):
        self.title.SetLabel(name)
        running = bool(job) and job["state"] in ("running", "queued")
        self.start.Show(not running)
        self.cancel.Show(running)
        self.start.SetLabel("Analyse Again" if job and job["state"] in ("failed", "cancelled") else "Analyse")
        phases = {p["phase"]: p for p in (job or {}).get("phases", [])}
        plain = self._font
        bold = plain.Bold()
        for key, (name_, state) in self.rows.items():
            p = phases.get(key)
            text = {"done": "done", "skipped": "already done", "running": "..."}.get(p["state"], p["state"]) if p else ""
            if p and p["state"] == "running" and job["state"] in ("failed", "cancelled"):
                text = job["state"]
            if p and p["state"] == "running" and job["n"]:
                text = f"{job['k']} of {job['n']}"
            state.SetLabel(text)
            now = bool(p) and p["state"] == "running"
            name_.SetFont(bold if now else plain)
            name_.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT) if p
                                      else theme.muted_fg())
        if not job:
            message = ("Not analysed yet. The analysis reads the movie, finds the grains and follows every tube. It "
                       "runs on its own:\nother movies can be opened meanwhile.")
        elif job["state"] == "queued":
            message = "Waiting for another analysis to finish."
        elif job["state"] == "failed":
            message = f"Failed: {job['error'] or 'see the log'}"
        elif job["state"] == "cancelled":
            message = "Cancelled."
        else:
            message = job["label"] or ""
        self.message.SetLabel(message)
        if job and job["n"]:
            self.gauge.SetValue(int(1000 * job["k"] / job["n"]))
        elif running:
            self.gauge.Pulse()
        else:
            self.gauge.SetValue(0)
        eta = f"  ·  about {fmt_left(job['eta'])} left" if job and job.get("eta") else ""
        self.detail.SetLabel(f"{fmt_secs(job['elapsed'])}{eta}" if job and job["elapsed"] else "")
        text = "\n".join((job or {}).get("log", []))
        if self.log.GetValue() != text:
            self.log.SetValue(text)
            self.log.ShowPosition(self.log.GetLastPosition())
        self.Layout()


class Overview(wx.Panel):
    """The movie at a glance, shown while no grain is selected: its numbers, the checks left, what to do next."""

    ROWS = (("grains", "Grains"), ("germinated", "Germinated"), ("t50", "T50"), ("growth", "Growth"),
            ("final", "Final length"), ("lost", "Lost partway"), ("checked", "Checked"))
    TIPS = {"germinated": "Counted grains whose tube appeared, and their share",
            "t50": "The time by which half the counted grains had surely germinated",
            "growth": "The median over the tubes of the length each gained between reaching 10% and 90% of its final "
                      "length, over that time",
            "final": "The median tube length at the end (or when its grain was lost)",
            "lost": "Counted grains that burst, drifted out of view or were swept off partway",
            "checked": "Grains a person has checked or corrected"}

    def __init__(self, parent, ctl):
        super().__init__(parent)
        bg(self, theme.window_bg())
        self.ctl = ctl
        s = wx.BoxSizer(wx.VERTICAL)
        title = wx.StaticText(self, label="This movie")
        title.SetFont(theme.font(15, bold=True))
        s.Add(title, 0, wx.LEFT | wx.RIGHT | wx.TOP, 12)
        grid = wx.FlexGridSizer(cols=2, vgap=5, hgap=16)
        self.vals = {}
        for key, name in self.ROWS:
            lab = wx.StaticText(self, label=name)
            lab.SetForegroundColour(theme.muted_fg())
            val = wx.StaticText(self, label="")
            if key in self.TIPS:
                lab.SetToolTip(self.TIPS[key])
                val.SetToolTip(self.TIPS[key])
            grid.Add(lab, 0)
            grid.Add(val, 0)
            self.vals[key] = val
        s.Add(grid, 0, wx.ALL, 12)
        self.todo = wx.StaticText(self, label="")
        s.Add(self.todo, 0, wx.LEFT | wx.RIGHT | wx.TOP, 12)
        self.go = wx.Button(self, label="Start Checking (N)")
        s.Add(self.go, 0, wx.LEFT | wx.RIGHT | wx.TOP, 12)
        self.hint = wx.StaticText(self, label="Click a grain to see it close up. Drag to move the movie, scroll or "
                                              "pinch to zoom. Help (F1) says what everything means.")
        self.hint.SetForegroundColour(theme.muted_fg())
        s.Add(self.hint, 0, wx.ALL, 12)
        self.made = wx.StaticText(self, label="")
        self.made.SetForegroundColour(theme.muted_fg())
        self.made.SetFont(theme.font(11))
        s.Add(self.made, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
        self.SetSizer(s)
        self.go.Bind(wx.EVT_BUTTON, lambda e: self._go())

    def _go(self):
        if any(not c["done"] for c in self.ctl.checks):
            self.ctl.goto_check(1)
        else:
            self.ctl.on_export()

    def update(self, width: int):
        ctl = self.ctl
        d, sm = ctl.data, ctl.summary
        if d is None or sm is None:
            return
        key = (width, sm["line"], sm["reviewed"], d.units.rate_unit, tuple(c["done"] for c in ctl.checks))
        if key == getattr(self, "_shown", None):  # as the time moves, nothing here changes
            return
        self._shown = key
        u = d.units
        n, total = sm["n"], sm["grains"]
        v = self.vals
        v["grains"].SetLabel(f"{total}" + (f"  ({n} counted)" if n < total else ""))
        for key in ("grains",):
            v[key].SetToolTip(sm.get("detail") or "")
        v["germinated"].SetLabel(f"{sm['germinated']} of {n}  ·  {100 * sm['share']:.0f}% by the curve" if sm["by_curve"]
                                 else f"{sm['germinated']} of {n}  ({100 * sm['share']:.0f}%)")
        v["germinated"].SetToolTip(sm.get("detail") or "")
        v["t50"].SetLabel(d.when(sm["t50_frame"]) if sm["t50_frame"] is not None else "not reached")
        v["growth"].SetLabel(f"{u.rate(sm['median_rate']):.3g} {u.rate_unit}" if sm["median_rate"] else "-")
        v["final"].SetLabel(d.length_words(sm["median_final"]) if sm["median_final"] else "-")
        v["lost"].SetLabel(str(sm["lost"]))
        v["checked"].SetLabel(f"{sm['reviewed']} of {total}")
        left = sum(1 for c in ctl.checks if not c["done"])
        if left:
            self.todo.SetLabel(f"{left} grain{'s' if left != 1 else ''} to check, the ones the model is least sure of "
                               f"first.")
            self.go.SetLabel("Start Checking (N)" if left == len(ctl.checks) else "Next Check (N)")
        else:
            self.todo.SetLabel("Every flagged grain has been checked." if ctl.checks else "Nothing flagged for checking.")
            self.go.SetLabel("Export Results")
        self.made.SetLabel(d.made())
        for t in (self.todo, self.hint, self.made):
            t.SetLabel(t.GetLabel().replace("\n", " "))
            t.Wrap(max(width - 28, 200))
        self.Layout()


class SidePanel(wx.Panel):
    """Check list navigator above three pages: the selected grain (the movie at a glance while none is), the check
    list, the events."""

    # the answers by what they are about: (key, label, tooltip); Confirm and Undo sit above them
    ROWS = (("Onset", (("onset", "Here (O)", "The tube is first visible at this time"),
                       ("no_onset", "Never (Shift-O)", "The grain never germinated"))),
            ("Tube", (("tip", "Set tip (T)", "Then click the tube's tip at this time (Esc cancels)"),
                      ("path", "Draw (D)", "Then click where the tube leaves the grain and along it to its tip; Enter "
                                           "saves, Esc cancels"),
                      ("no_tube", "None here", "There is no tube at this time"))),
            ("Grain", (("burst", "Burst (B)", "The grain burst or is gone by this time: nothing is measured after it"),
                       ("not_a_grain", "Not a grain (X)", "Debris, or anything else that is not a pollen grain: left out"),
                       ("clump", "Clump (K)", "Grains stuck together: left out of the counts"))))
    TOOLS = ("tip", "path")

    def __init__(self, parent, ctl):
        super().__init__(parent, size=(392, -1))
        bg(self)
        self.ctl = ctl
        s = wx.BoxSizer(wx.VERTICAL)
        nav = wx.BoxSizer(wx.HORIZONTAL)
        self.nav_label = wx.StaticText(self, label="")
        self.nav_label.SetFont(self.nav_label.GetFont().Bold())
        nav.Add(self.nav_label, 1, wx.ALIGN_CENTER_VERTICAL)
        self.prev = wx.Button(self, label="Previous (P)")
        self.next = wx.Button(self, label="Next (N)")
        self.prev.SetToolTip("The previous grain to check")
        self.next.SetToolTip("The next grain to check")
        nav.Add(self.prev, 0, wx.LEFT, 6)
        nav.Add(self.next, 0, wx.LEFT, 6)
        s.Add(nav, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        self.progress = wx.Gauge(self, range=1000, size=(-1, 6))
        s.Add(self.progress, 0, wx.EXPAND | wx.ALL, 10)
        self.book = wx.Notebook(self)
        self.grain_page = self._grain_page(self.book)
        self.checks = wx.ListCtrl(self.book, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for i, (name, width) in enumerate((("Grain", 58), ("Why", 232), ("Conf.", 50))):
            self.checks.InsertColumn(i, name, width=width)
        self.events = wx.ListCtrl(self.book, style=wx.LC_REPORT | wx.LC_SINGLE_SEL)
        for i, (name, width) in enumerate((("Time", 80), ("Event", 260))):
            self.events.InsertColumn(i, name, width=width)
        self.book.AddPage(self.grain_page, "Grain")
        self.book.AddPage(self.checks, "Checks")
        self.book.AddPage(self.events, "Events")
        s.Add(self.book, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)
        self.SetSizer(s)
        self.prev.Bind(wx.EVT_BUTTON, lambda e: ctl.goto_check(-1))
        self.next.Bind(wx.EVT_BUTTON, lambda e: ctl.goto_check(1))
        # the lists are rebuilt when a grain is opened: open it after the list's own event has finished
        self.checks.Bind(wx.EVT_LIST_ITEM_SELECTED,
                         lambda e: None if self._filling else wx.CallAfter(ctl.open_check, e.GetIndex()))
        self.checks.Bind(wx.EVT_MOTION, self._check_tip)
        self._check_rows, self._tip_row = [], None
        self.events.Bind(wx.EVT_LIST_ITEM_SELECTED,
                         lambda e: None if self._filling else wx.CallAfter(ctl.goto_event_index, e.GetIndex()))
        self._filling = False

    def _grain_page(self, parent):
        page = wx.ScrolledWindow(parent, style=wx.VSCROLL)
        page.SetScrollRate(0, 12)
        bg(page, theme.window_bg())
        outer = wx.BoxSizer(wx.VERTICAL)
        self.overview = Overview(page, self.ctl)
        outer.Add(self.overview, 0, wx.EXPAND)
        self.grain_panel = p = bg(wx.Panel(page), theme.window_bg())
        outer.Add(p, 0, wx.EXPAND)
        s = wx.BoxSizer(wx.VERTICAL)
        head = wx.BoxSizer(wx.HORIZONTAL)
        self.gid = wx.StaticText(p, label="")
        self.gid.SetFont(theme.font(17, bold=True))
        head.Add(self.gid, 0, wx.ALIGN_BOTTOM)
        self.gstate = wx.StaticText(p, label="", style=wx.ST_ELLIPSIZE_END)
        self.gstate.SetMinSize((60, -1))
        self.gstate.SetForegroundColour(theme.muted_fg())
        head.Add(self.gstate, 1, wx.LEFT | wx.ALIGN_BOTTOM, 10)
        self.back = wx.Button(p, label="Movie (Esc)", style=wx.BU_EXACTFIT)
        self.back.SetWindowVariant(wx.WINDOW_VARIANT_SMALL)
        self.back.SetToolTip("Back to the movie's numbers (or click an empty place on the movie)")
        self.back.Bind(wx.EVT_BUTTON, lambda e: self.ctl.deselect())
        head.Add(self.back, 0, wx.LEFT | wx.ALIGN_CENTER_VERTICAL, 6)
        s.Add(head, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        self.reasons = wx.BoxSizer(wx.VERTICAL)  # why to check it: a link to the time and a sentence, each
        self._reasons_key, self._reason_rows, self._active = None, [], None
        s.Add(self.reasons, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        self.view = GrainView(p, self.ctl, size=280)
        s.Add(self.view, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 8)
        zrow = wx.BoxSizer(wx.HORIZONTAL)
        self.now = wx.StaticText(p, label="", style=wx.ST_ELLIPSIZE_END)
        self.now.SetMinSize((60, -1))
        zrow.Add(self.now, 1, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 6)
        self.zoom = wx.Choice(p, choices=["Near", "Wide", "Far"])
        self.zoom.SetSelection(0)
        self.zoom.SetToolTip("How much of the grain's surroundings the close-up shows (or scroll on it)")
        zrow.Add(self.zoom, 0)
        s.Add(zrow, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)
        self.buttons = {}
        top = wx.BoxSizer(wx.HORIZONTAL)
        for key, text, tip, prop in (("confirm", "Confirm (Enter)", "What is shown is right: mark the grain checked "
                                      "and go to the next", 3),
                                     ("undo", "Undo", "Take back the last correction (Cmd-Z)", 1)):
            b = wx.Button(p, label=text)
            b.SetToolTip(tip)
            b.Bind(wx.EVT_BUTTON, lambda e, k=key: self.ctl.on_action(k))
            top.Add(b, prop, wx.RIGHT if key == "confirm" else 0, 6)
            self.buttons[key] = b
        s.Add(top, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        grid = wx.FlexGridSizer(cols=4, vgap=4, hgap=5)
        for c in (1, 2, 3):
            grid.AddGrowableCol(c)
        for name, items in self.ROWS:
            lab = wx.StaticText(p, label=name)
            lab.SetForegroundColour(theme.muted_fg())
            grid.Add(lab, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
            for key, text, tip in items:
                b = (wx.ToggleButton if key in self.TOOLS else wx.Button)(p, label=text)
                b.SetWindowVariant(wx.WINDOW_VARIANT_SMALL)
                b.SetToolTip(tip)
                event = wx.EVT_TOGGLEBUTTON if key in self.TOOLS else wx.EVT_BUTTON
                b.Bind(event, lambda e, k=key: self.ctl.on_action(k))
                grid.Add(b, 0, wx.EXPAND)
                self.buttons[key] = b
            for _ in range(3 - len(items)):
                grid.AddSpacer(0)
        s.Add(grid, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)
        stats = wx.FlexGridSizer(cols=4, vgap=4, hgap=8)
        stats.AddGrowableCol(1)
        stats.AddGrowableCol(3)
        self.stats, self.stat_names = {}, []
        tips = {"onset": "When the tube first showed (absent at one time, visible at the next)",
                "final": "The tube's length at the end, or when its grain was lost",
                "rate": "The length gained between reaching 10% and 90% of the final length, over that time",
                "conf": "How likely the model's length is right where it is least sure, from the reading's own "
                        "history. It orders the checks; it does not make checking safe to skip."}
        for key, name in (("onset", "Onset"), ("final", "Final length"), ("rate", "Growth"), ("conf", "Confidence")):
            lab = wx.StaticText(p, label=name)
            lab.SetForegroundColour(theme.muted_fg())
            lab.SetToolTip(tips[key])
            val = wx.StaticText(p, label="")
            val.SetToolTip(tips[key])
            stats.Add(lab, 0)
            stats.Add(val, 0, wx.EXPAND)
            self.stats[key] = val
            self.stat_names.append(lab)
        s.Add(stats, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 12)
        self.notes = wx.StaticText(p, label="")
        self.notes.SetForegroundColour(theme.muted_fg())
        self.notes.SetFont(theme.font(11))
        s.Add(self.notes, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)
        self.curve = GrowthCurve(p, self.ctl)
        s.Add(self.curve, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)
        self.status = wx.StaticText(p, label="")
        self.status.SetForegroundColour(theme.muted_fg())
        s.Add(self.status, 0, wx.ALL, 8)
        p.SetSizer(s)
        page.SetSizer(outer)
        self.zoom.Bind(wx.EVT_CHOICE, lambda e: self.view.set_zoom(self.zoom.GetSelection()))
        self._zoomed_for = None
        return page

    def fit_zoom(self, g: dict, data) -> None:
        """A newly selected grain's close-up: aimed at it and its tube at its longest, as close as shows both."""
        if g["id"] == self._zoomed_for:
            return
        self._zoomed_for = g["id"]
        last = (g["lost"] - 1) if g.get("lost") is not None else data.n_bins - 1
        self.zoom.SetSelection(self.view.fit(g, max(last, 0)))

    # ---- updates ------------------------------------------------------------------------------------
    def _check_tip(self, e):
        """The whole of a check's reasons, with their sentences, on hover (the list cuts them short)."""
        row, _ = self.checks.HitTest(e.GetPosition())
        if row != self._tip_row:
            self._tip_row = row
            c = self._check_rows[row] if 0 <= row < len(self._check_rows) else None
            self.checks.SetToolTip("\n".join(f"{t}: {d}" if d else t for t, d in zip(c["reasons"], c["details"]))
                                   + ("" if c["isolated"] else "\nIn a clump or at the edge: not counted.")
                                   if c else "")
        e.Skip()

    def update_nav(self, checks: list[dict], sel: str | None):
        left = sum(1 for c in checks if not c["done"])
        cur = next((i for i, c in enumerate(checks) if c["gid"] == sel), None)
        if not checks:
            self.nav_label.SetLabel("Nothing flagged")
        elif cur is None:
            self.nav_label.SetLabel(f"{left} to check" if left else "All checked")
        elif checks[cur]["done"]:  # looked at: it has moved to the end of the list
            self.nav_label.SetLabel(f"Checked  ·  {left} left" if left else "All checked")
        else:  # the ones not yet looked at come first: count on from the ones done
            self.nav_label.SetLabel(f"Check {len(checks) - left + cur + 1} of {len(checks)}")
        self.progress.SetValue(int(1000 * (len(checks) - left) / len(checks)) if checks else 0)
        self.prev.Enable(bool(checks))
        self.next.Enable(bool(checks))
        self.book.SetPageText(1, f"Checks ({left})" if left else "Checks")

    def update_lists(self, checks: list[dict], events: list[dict], data, sel: str | None):
        self._filling = True
        try:
            self.checks.DeleteAllItems()
            muted = theme.muted_fg()
            for i, c in enumerate(checks):
                self.checks.InsertItem(i, c["gid"])
                self.checks.SetItem(i, 1, " · ".join(c["reasons"]) + ("" if c["isolated"] else " (not counted)"))
                self.checks.SetItem(i, 2, "" if c["conf"] is None else f"{100 * c['conf']:.0f}%")
                if c["done"]:
                    self.checks.SetItemTextColour(i, muted)
                if c["gid"] == sel:
                    self.checks.Select(i)
            self._check_rows = checks
            self.events.DeleteAllItems()
            self.events.SetColumnWidth(0, 80 if data.units.timed else 110)
            for i, ev in enumerate(events):
                self.events.InsertItem(i, data.when(data.frame(ev["bin"])))
                self.events.SetItem(i, 1, ev["text"])
        finally:
            self._filling = False

    def _show_reasons(self, g: dict | None, data, b: int) -> None:
        """Why to check the grain: each reason as a link to its time, and the sentence of one of them (the one at the
        time shown, or clicked; the first to begin with). Rebuilt only when the reasons change."""
        reasons = (g.get("check") or []) if g is not None and not g["done"] else []
        key = (g["id"] if g else None, tuple((r["text"], r["detail"], r["bin"]) for r in reasons))
        if key != self._reasons_key:
            self._reasons_key = key
            self.reasons.Clear(delete_windows=True)
            self._reason_rows, self._active = [], None
            p = self.grain_panel
            width = max(self.grain_page.GetClientSize()[0] - 28, 220)
            for i, r in enumerate(reasons):
                link = wx.adv.HyperlinkCtrl(p, label=r["text"], url=f"tubetracker:bin/{r['bin']}",
                                            style=wx.adv.HL_ALIGN_LEFT | wx.NO_BORDER)
                for set_colour in (link.SetNormalColour, link.SetVisitedColour):
                    set_colour(theme.colour(theme.WARN_TEXT))
                link.SetHoverColour(theme.colour(theme.CHECK))
                link.SetBackgroundColour(theme.window_bg())
                link.SetToolTip(f"Go to {data.when(data.frame(r['bin']))}" + (f": {r['detail']}" if r.get("detail")
                                                                                else ""))
                link.Bind(wx.adv.EVT_HYPERLINK, lambda e, i=i: self._go_reason(i))
                text = wx.StaticText(p, label=r.get("detail") or "")
                text.SetForegroundColour(theme.muted_fg())
                text.SetFont(theme.font(11))
                text.Wrap(width)
                self.reasons.Add(link, 0, wx.TOP, 2)
                self.reasons.Add(text, 0, wx.BOTTOM, 3)
                self._reason_rows.append((r["bin"], text))
        hit = next((i for i, (rb, _) in enumerate(self._reason_rows) if rb == b), None)
        self._focus_reason(hit if hit is not None else self._active if self._active is not None else 0)

    def _focus_reason(self, i: int) -> None:
        if i == self._active or not self._reason_rows:
            return
        self._active = i
        for k, (_, text) in enumerate(self._reason_rows):
            text.Show(k == i and bool(text.GetLabel()))
        self.grain_panel.Layout()
        self.grain_page.FitInside()

    def _go_reason(self, i: int) -> None:
        self._focus_reason(i)
        self.ctl.set_bin(self._reason_rows[i][0])

    def show_tool(self, tool: str | None) -> None:
        for key in self.TOOLS:
            self.buttons[key].SetValue(tool == key)

    def update_grain(self, g: dict | None, data, b: int, ready: bool, busy_note: str = ""):
        page = self.grain_page
        shown = g is not None
        self.grain_panel.Show(shown)
        self.overview.Show(not shown)
        if self.book.GetPageText(0) != ("Grain" if shown else "Movie"):
            self.book.SetPageText(0, "Grain" if shown else "Movie")
        width = page.GetClientSize()[0]
        if not shown:
            self._zoomed_for = None  # the next grain chosen is framed afresh
            self._show_reasons(None, data, b)
            self.overview.update(width)
        if shown:
            st = state_at(g, b)
            self.fit_zoom(g, data)
            self.gid.SetLabel(g["id"])
            self.gid.SetToolTip(f"{g['id']}: at x {g['x']:.0f}, y {g['y']:.0f} px in the first frames, "
                                f"{2 * g['r']:.0f} px across")
            where = [] if g["isolated"] else ([f"clump of {g['clump']}, not counted"] if g["clump"] > 1 else
                                              ["at the edge, not counted"])
            if g["excluded"]:
                words = [f"excluded ({g['excluded'].replace('_', ' ')})"]
            else:
                words = [STATE_WORDS[st], *where, REVIEW_WORDS[g["review"]["state"]]]
            self.gstate.SetLabel("  ·  ".join(words))
            self.gstate.SetToolTip("  ·  ".join(words) + ("\nGrains in a clump or at the edge of the field are left out "
                                                         "of the movie's numbers." if where else ""))
            self._show_reasons(g, data, b)
            L = g["L"][b]
            if st == "germinated":
                now = data.length_words(L) if L >= MIN_TUBE_PX else "no tube yet"
            else:
                now = {"notyet": "not germinated yet", "never": "", "lost": "lost", "excluded": "excluded",
                       "unobservable": "not readable"}[st]
            yours = any(t["bin"] == b for t in g.get("human") or [])
            self.now.SetLabel("  ·  ".join(x for x in (data.when(data.frame(b)), now) if x))
            self.now.SetToolTip("The time shown and the tube's length then" + (" (yours)" if yours else ""))
            u = data.units
            if g["status"] == "emerged_within" and g["onset_by"] is not None:
                onset = data.when_range(g["onset_after"], g["onset_by"])
            else:
                onset = {"emerged_at_start": "before start", "no_emergence_by_end": "none",
                         "unobservable": "-"}.get(g["status"], "-")
            grown = g["status"] in EMERGED and not g["excluded"]
            self.stats["onset"].SetLabel(onset + ("  (checked)" if g["review"]["onset"] != "model" else ""))
            frames = (f": not there at frame {int(g['onset_after']):,}, there at frame {int(g['onset_by']):,}"
                      if g["onset_after"] is not None and g["onset_by"] is not None else "")
            self.stats["onset"].SetToolTip(f"When the tube first showed{frames}")
            self.stats["final"].SetLabel(data.length_words(g["final"]) if grown else "-")
            rate = u.rate(g["rate"]) if grown and g["rate"] else None
            self.stats["rate"].SetLabel(f"{rate:.3g} {u.rate_unit}" if rate is not None else "-")
            self.stats["conf"].SetLabel("-" if g["conf"] is None else f"{100 * g['conf']:.0f}%")
            for key, btn in self.buttons.items():
                btn.Enable(ready if key != "undo" else ready and self.ctl.can_undo())
            self.buttons["undo"].SetToolTip(self.ctl.undo_words())
            self.buttons["not_a_grain"].SetLabel("Include (X)" if g["excluded"] else "Not a grain (X)")
            self.show_tool(self.ctl.tool)
            notes = data.notes(g)
            self.notes.SetLabel("Also: " + "; ".join(notes) + "." if notes else "")
            self.notes.Wrap(max(width - 28, 220))
            self.notes.Show(bool(notes))
            self.status.SetLabel(busy_note)
            self.status.Show(bool(busy_note))
        self.view.Refresh()
        self.curve.Refresh()
        self.grain_panel.Layout()
        page.Layout()
        page.FitInside()
