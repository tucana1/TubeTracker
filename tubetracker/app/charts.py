"""Drawn panels: the event timeline under the movie, one grain's growth curve, and the plots of the results and
comparison windows. All on the dark ground of the movie views, drawn with a ``wx.GraphicsContext``."""

from __future__ import annotations

import math

import wx

from . import theme
from .overlay import EMERGED


def nice_ticks(a: float, b: float, n: int) -> list[float]:
    span = max(b - a, 1e-9)
    mag = 10 ** math.floor(math.log10(span / max(n, 1)))
    step = next((m * mag for m in (1, 2, 5, 10) if span / (m * mag) <= n), 10 * mag)
    first = math.ceil(a / step) * step
    return [round(first + k * step, 6) for k in range(int((b - first) / step + 1e-9) + 1)]


def fmt_tick(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 10 or v == int(v) else f"{v:.1f}"


class Drawn(wx.Panel):
    """A panel painted by ``draw(gc, w, h)`` on the dark ground."""

    def __init__(self, parent, size=(-1, -1)):
        super().__init__(parent, size=size, style=wx.FULL_REPAINT_ON_RESIZE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(theme.colour(theme.CANVAS_BG))
        self.Bind(wx.EVT_PAINT, self._paint)

    def _paint(self, e):
        dc = wx.PaintDC(self)
        dc.SetBackground(wx.Brush(theme.colour(theme.CANVAS_BG)))
        dc.Clear()
        w, h = self.GetClientSize()
        if w > 4 and h > 4:
            self.draw(wx.GraphicsContext.Create(dc), w, h)

    def draw(self, gc, w, h):
        pass

    @staticmethod
    def text(gc, s, x, y, rgb=theme.AXIS_TEXT, size=9.0, align="left", bold=False):
        gc.SetFont(theme.font(size, bold), theme.colour(rgb))
        tw, th = gc.GetTextExtent(s)[:2]
        x = x - tw if align == "right" else x - tw / 2 if align == "center" else x
        gc.DrawText(s, x, y - th / 2)

    @staticmethod
    def line(gc, x0, y0, x1, y1, rgb, width=1.0, dashed=False, alpha=255):
        gc.SetPen(theme.pen(rgb, width, wx.PENSTYLE_SHORT_DASH if dashed else wx.PENSTYLE_SOLID, alpha))
        gc.StrokeLine(x0, y0, x1, y1)

    @staticmethod
    def poly(gc, pts, rgb, width=1.5, dashed=False, alpha=255):
        if len(pts) < 2:
            return
        path = gc.CreatePath()
        path.MoveToPoint(*pts[0])
        for p in pts[1:]:
            path.AddLineToPoint(*p)
        gc.SetPen(theme.pen(rgb, width, wx.PENSTYLE_SHORT_DASH if dashed else wx.PENSTYLE_SOLID, alpha))
        gc.StrokePath(path)

    @staticmethod
    def area(gc, top, bottom, rgb, alpha=50):
        path = gc.CreatePath()
        path.MoveToPoint(*top[0])
        for p in top[1:]:
            path.AddLineToPoint(*p)
        for p in reversed(bottom):
            path.AddLineToPoint(*p)
        path.CloseSubpath()
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.SetBrush(wx.Brush(theme.colour(rgb, alpha)))
        gc.FillPath(path)


# ------------------------------------------------------------------------------------------------ timeline
LANES = (("germination", "Germination"), ("lost", "Lost"), ("stall", "Stopped"), ("unsure", "Unsure"))


class Timeline(Drawn):
    """Germination curve (with T50), then one lane per kind of event, then the time axis; the current time as a
    line. Click an event to go to it; click or drag elsewhere to move in time."""

    GUTTER, RIGHT, TOP, CURVE_H, LANE_H = 92, 12, 8, 36, 13

    def __init__(self, parent, ctl):
        super().__init__(parent, size=(-1, 128))
        self.ctl = ctl
        self.SetMinSize((300, 128))
        self.hits = []
        self._scrub = False
        self._tip = None
        self.Bind(wx.EVT_LEFT_DOWN, self.on_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)

    def x_of(self, b, w):
        nb = self.ctl.data.n_bins
        return self.GUTTER + (b + 0.5) / nb * (w - self.GUTTER - self.RIGHT)

    def bin_at(self, x, w):
        nb = self.ctl.data.n_bins
        return int(min(max((x - self.GUTTER) / max(w - self.GUTTER - self.RIGHT, 1) * nb, 0), nb - 1))

    def draw(self, gc, w, h):
        ctl = self.ctl
        if ctl.data is None:
            return
        nb, pop = ctl.data.n_bins, ctl.population
        y0 = self.TOP + self.CURVE_H
        yof = lambda f: y0 - f * self.CURVE_H
        self.text(gc, "Germinated", self.GUTTER - 10, self.TOP + self.CURVE_H / 2, align="right")
        top = [(self.x_of(b, w), yof(pop["possible"][b])) for b in range(nb)]
        bottom = [(self.x_of(b, w), yof(pop["certain"][b])) for b in range(nb)]
        self.area(gc, top, bottom, theme.SELECTED, 45)
        self.poly(gc, bottom, theme.SELECTED, 1.8)
        self.line(gc, self.GUTTER, y0 + 0.5, w - self.RIGHT, y0 + 0.5, theme.GRID)
        if pop.get("t50_bin") is not None:
            x = self.x_of(pop["t50_bin"], w)
            self.line(gc, x, self.TOP, x, y0, theme.SELECTED, 1.0, dashed=True)
            s = f"T50 {ctl.data.when(pop['t50_frame'])}"
            right = x > w - 140
            self.text(gc, s, x + (-6 if right else 6), self.TOP + 7, theme.SELECTED, 9.0, "right" if right else "left")
        lane_top = y0 + 6
        for i, (kind, name) in enumerate(LANES):
            y = lane_top + i * self.LANE_H
            gc.SetPen(wx.TRANSPARENT_PEN)
            gc.SetBrush(wx.Brush(wx.Colour(255, 255, 255, 9 if i % 2 else 16)))
            gc.DrawRectangle(self.GUTTER, y, w - self.GUTTER - self.RIGHT, self.LANE_H)
            self.text(gc, name, self.GUTTER - 10, y + self.LANE_H / 2, align="right")
        lanes = {k: i for i, (k, _) in enumerate(LANES)}
        lane_bottom = lane_top + len(LANES) * self.LANE_H
        self.hits = []
        for ev in ctl.events:
            x = self.x_of(ev["bin"], w)
            if ev["kind"] == "focus":
                self.line(gc, x, self.TOP, x, lane_bottom, theme.FOCUS, 1.2, dashed=True)
                self.hits.append((x, self.TOP + 14, ev))
                continue
            if ev["kind"] not in lanes:
                continue
            y = lane_top + lanes[ev["kind"]] * self.LANE_H + self.LANE_H / 2
            sel = ev.get("gid") == ctl.sel and ctl.sel is not None
            alpha = 255 if sel or ctl.sel is None else 150
            gc.SetBrush(wx.Brush(theme.colour(theme.EVENT[ev["kind"]], alpha)))
            gc.SetPen(wx.Pen(wx.Colour(255, 255, 255), 1) if sel else wx.TRANSPARENT_PEN)
            path = gc.CreatePath()
            if ev["kind"] == "germination":
                path.AddRectangle(x - 1.1, y - 4.5, 2.2, 9)
            elif ev["kind"] == "lost":
                path.MoveToPoint(x - 4.5, y - 4)
                path.AddLineToPoint(x + 4.5, y - 4)
                path.AddLineToPoint(x, y + 4)
                path.CloseSubpath()
            elif ev["kind"] == "stall":
                path.AddRectangle(x - 3.2, y - 3.2, 6.4, 6.4)
            else:
                path.MoveToPoint(x, y - 4.8)
                path.AddLineToPoint(x + 4.8, y)
                path.AddLineToPoint(x, y + 4.8)
                path.AddLineToPoint(x - 4.8, y)
                path.CloseSubpath()
            gc.FillPath(path)
            if sel:
                gc.StrokePath(path)
            self.hits.append((x, y, ev))
        axis_y = lane_bottom + 4
        u = ctl.data.units
        t_end = u.time(ctl.data.frame(nb - 1))
        for t in nice_ticks(0, t_end, max(3, int((w - self.GUTTER) / 90))):
            frame = t * 60 / u.s_per_frame if u.timed else t
            b = (frame - ctl.data.fpb // 2) / ctl.data.fpb
            if b < -0.5 or b > nb - 0.5:
                continue
            x = self.x_of(b, w)
            self.line(gc, x, axis_y, x, axis_y + 4, theme.AXIS_TEXT)
            self.text(gc, fmt_tick(t), x, axis_y + 12, align="center")
        self.text(gc, "min" if u.timed else "frame", self.GUTTER - 10, axis_y + 12, align="right")
        x = self.x_of(ctl.b, w)
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.SetBrush(wx.Brush(theme.colour(theme.SELECTED)))
        gc.DrawRectangle(x - 1, self.TOP - 4, 2, axis_y - self.TOP + 4)

    def hit(self, x, y):
        best, bd = None, 8.0
        for hx, hy, ev in self.hits:
            d = math.hypot(hx - x, (hy - y) * 0.8)
            if d < bd:
                best, bd = ev, d
        return best

    def on_down(self, e):
        if self.ctl.data is None:
            return
        ev = self.hit(e.GetX(), e.GetY())
        if ev:
            self.ctl.goto_event(ev)
            return
        self._scrub = True
        if not self.HasCapture():
            self.CaptureMouse()
        self.ctl.set_bin(self.bin_at(e.GetX(), self.GetClientSize()[0]))

    def on_up(self, e):
        self._scrub = False
        if self.HasCapture():
            self.ReleaseMouse()

    def on_motion(self, e):
        if self.ctl.data is None:
            return
        w = self.GetClientSize()[0]
        if self._scrub and e.LeftIsDown():
            self.ctl.set_bin(self.bin_at(e.GetX(), w))
            return
        ev = self.hit(e.GetX(), e.GetY())
        tip = f"{ev['text']}, {self.ctl.data.when(self.ctl.data.frame(ev['bin']))}" if ev else ""
        if tip != self._tip:
            self._tip = tip
            self.SetToolTip(tip)


# ------------------------------------------------------------------------------------------------ one grain's curve
class GrowthCurve(Drawn):
    """The selected grain's tube length over time: as shown (pink), the model's before your checks (dashed grey),
    the lengths you checked (dots), onset (green line), lost (shaded), now (blue line). Click to go to a time."""

    PAD = (38, 8, 8, 18)  # left, right, top, bottom

    def __init__(self, parent, ctl):
        super().__init__(parent, size=(-1, 104))
        self.ctl = ctl
        self.SetMinSize((200, 104))
        self.SetToolTip("The tube's length over time: pink as shown, dashed grey the model's before your corrections, "
                        "white dots the lengths you gave, green dashed the onset, blue the time shown. Click to go to "
                        "a time.")
        self.Bind(wx.EVT_LEFT_DOWN, self.on_click)

    def draw(self, gc, w, h):
        ctl = self.ctl
        g = ctl.selected()
        if g is None:
            return
        nb, u = ctl.data.n_bins, ctl.data.units
        l, r, t, b_ = self.PAD
        ymax = max(8.0, max(g["L"]), max(g.get("Lm") or [0.0])) * 1.1
        X = lambda b: l + b / max(nb - 1, 1) * (w - l - r)
        Y = lambda px: h - b_ - px / ymax * (h - t - b_)
        scale = u.um_per_px or 1.0
        for tick in nice_ticks(0, ymax * scale, 3):
            y = Y(tick / scale)
            self.line(gc, l, y, w - r, y, theme.GRID)
            self.text(gc, fmt_tick(tick), l - 5, y, align="right", size=8.5)
        self.text(gc, u.length_unit, 2, t + 4, size=8.5)
        self.text(gc, "min" if u.timed else "frame", 2, h - 7, size=8.5)
        t_end = u.time(ctl.data.frame(nb - 1))
        for tick in nice_ticks(0, t_end, 4):
            frame = tick * 60 / u.s_per_frame if u.timed else tick
            b = (frame - ctl.data.fpb // 2) / ctl.data.fpb
            if 0 <= b <= nb - 1:
                self.text(gc, fmt_tick(tick if u.timed else tick / 1000) + ("" if u.timed else "k"), X(b), h - 7,
                          align="center", size=8.5)
        if g.get("lost") is not None:
            gc.SetPen(wx.TRANSPARENT_PEN)
            gc.SetBrush(wx.Brush(theme.colour(theme.STATE["lost"], 30)))
            gc.DrawRectangle(X(g["lost"]), t, w - r - X(g["lost"]), h - t - b_)
        if g.get("onset") is not None and g["status"] in EMERGED:
            self.line(gc, X(g["onset"]), t, X(g["onset"]), h - b_, theme.STATE["germinated"], 1.0, dashed=True)
        if g.get("Lm"):
            self.poly(gc, [(X(b), Y(v)) for b, v in enumerate(g["Lm"])], (100, 114, 139), 1.2, dashed=True)
        self.poly(gc, [(X(b), Y(v)) for b, v in enumerate(g["L"])], theme.TUBE, 2.0)
        for tr in g.get("human") or []:
            if tr["state"] in ("full", "partial", "no_tube"):
                x, y = X(tr["bin"]), Y(0.0 if tr["state"] == "no_tube" else tr["L"])
                gc.SetPen(wx.Pen(theme.colour(theme.CANVAS_BG), 1))
                gc.SetBrush(wx.Brush(wx.Colour(248, 250, 252)))
                gc.DrawEllipse(x - 3.2, y - 3.2, 6.4, 6.4)
        if g.get("unsure") is not None and not g.get("done") and (g.get("conf") or 1) < 0.5:
            x, y = X(g["unsure"]), Y(g["L"][g["unsure"]])
            path = gc.CreatePath()
            path.MoveToPoint(x, y - 4.5)
            path.AddLineToPoint(x + 4.5, y)
            path.AddLineToPoint(x, y + 4.5)
            path.AddLineToPoint(x - 4.5, y)
            path.CloseSubpath()
            gc.SetPen(wx.TRANSPARENT_PEN)
            gc.SetBrush(wx.Brush(theme.colour(theme.CHECK)))
            gc.FillPath(path)
        x = X(ctl.b)
        self.line(gc, x, t - 2, x, h - b_, theme.SELECTED, 1.5)

    def on_click(self, e):
        if self.ctl.selected() is None:
            return
        w = self.GetClientSize()[0]
        l, r = self.PAD[0], self.PAD[1]
        self.ctl.set_bin(round((e.GetX() - l) / max(w - l - r, 1) * (self.ctl.data.n_bins - 1)))


# ------------------------------------------------------------------------------------------------ plots
class Plot(Drawn):
    """A plot with axes. ``series``: dicts with x, y, colour, width, dashed, step, band_y (upper edge of a band
    below y); ``dots``: (x, y, colour, radius); ``vlines``: (x, colour, label). Categorical x with ``categories``."""

    PAD = (54, 14, 12, 38)

    def __init__(self, parent, size=(420, 260)):
        super().__init__(parent, size=size)
        self.SetMinSize((280, 200))
        self.series, self.dots, self.vlines, self.categories = [], [], [], None
        self.xlabel = self.ylabel = ""
        self.xrange = self.yrange = (0.0, 1.0)
        self.yfmt = fmt_tick
        self.on_pick = None  # callback(series index) when a line is clicked
        self.Bind(wx.EVT_LEFT_DOWN, self._click)
        self._last = None

    def set(self, series=(), dots=(), vlines=(), xrange=None, yrange=None, xlabel="", ylabel="", categories=None,
            yfmt=None):
        self.series, self.dots, self.vlines, self.categories = list(series), list(dots), list(vlines), categories
        xs = [v for s in self.series for v in s["x"]] + [d[0] for d in self.dots] or [0.0, 1.0]
        ys = [v for s in self.series for v in s["y"]] + [d[1] for d in self.dots] or [0.0, 1.0]
        self.xrange = xrange or (min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1)
        self.yrange = yrange or (0.0, max(ys) * 1.08 if max(ys) > 0 else 1.0)
        self.xlabel, self.ylabel = xlabel, ylabel
        self.yfmt = yfmt or fmt_tick
        self.Refresh()

    def maps(self, w, h):
        l, r, t, b = self.PAD
        (x0, x1), (y0, y1) = self.xrange, self.yrange
        X = lambda v: l + (v - x0) / ((x1 - x0) or 1) * (w - l - r)
        Y = lambda v: h - b - (v - y0) / ((y1 - y0) or 1) * (h - t - b)
        return X, Y

    def draw(self, gc, w, h):
        l, r, t, b = self.PAD
        X, Y = self.maps(w, h)
        for v in nice_ticks(*self.yrange, 4):
            y = Y(v)
            self.line(gc, l, y, w - r, y, theme.GRID)
            self.text(gc, self.yfmt(v), l - 6, y, align="right")
        if self.categories:
            for i, name in enumerate(self.categories):
                self.text(gc, name, X(i), h - b + 12, align="center")
        else:
            for v in nice_ticks(*self.xrange, max(3, int(w / 90))):
                self.text(gc, fmt_tick(v), X(v), h - b + 12, align="center")
        self.text(gc, self.xlabel, (l + w - r) / 2, h - 9, align="center")
        if self.ylabel:
            gc.PushState()
            gc.Translate(12, (t + h - b) / 2)
            gc.Rotate(-math.pi / 2)
            self.text(gc, self.ylabel, 0, 0, align="center")
            gc.PopState()
        self._last = []
        for s in self.series:
            pts = self._points(s, X, Y)
            if s.get("band_y") is not None:
                upper = self._points({**s, "y": s["band_y"]}, X, Y)
                self.area(gc, upper, pts, s["colour"], 45)
            self.poly(gc, pts, s["colour"], s.get("width", 1.5), s.get("dashed", False), s.get("alpha", 255))
            self._last.append(pts)
        for x, colour_, label in self.vlines:
            self.line(gc, X(x), t, X(x), h - b, colour_, 1.0, dashed=True)
            if label:
                self.text(gc, label, X(x) + 6, t + 8, colour_)
        for x, y, colour_, rad in self.dots:
            gc.SetPen(wx.TRANSPARENT_PEN)
            gc.SetBrush(wx.Brush(theme.colour(colour_)))
            gc.DrawEllipse(X(x) - rad, Y(y) - rad, 2 * rad, 2 * rad)

    @staticmethod
    def _points(s, X, Y):
        if not s.get("step"):
            return [(X(x), Y(y)) for x, y in zip(s["x"], s["y"])]
        pts = []
        for i, (x, y) in enumerate(zip(s["x"], s["y"])):
            if i:
                pts.append((X(x), pts[-1][1]))
            pts.append((X(x), Y(y)))
        return pts

    def _click(self, e):
        if not self.on_pick or not self._last:
            return
        best, bd = None, 10.0
        for i, pts in enumerate(self._last):
            for x, y in pts:
                d = math.hypot(x - e.GetX(), y - e.GetY())
                if d < bd:
                    best, bd = i, d
        if best is not None:
            self.on_pick(best)
