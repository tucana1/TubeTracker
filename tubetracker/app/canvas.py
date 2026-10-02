"""The movie views: the whole field (``FieldCanvas``) and one grain close up (``GrainView``).

Frames come from ``imaging.FrameSource`` as 8-bit numpy images, are scaled to the screen's pixels with OpenCV and
shown with ``wx.Bitmap.FromBuffer``; the grains, tubes, tips and exits are drawn over them with a
``wx.GraphicsContext`` from the grain records (``overlay``), so nothing is written to disk and scrubbing only
re-draws. Both views read the window's state (``ctl``): the grains, the bin, the view mode, the selection, the tool.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import wx

from . import theme
from .overlay import needs_check, path_length, pos_at, route_at, shift, state_at, to_length, tube_at, drift_at


def to_bitmap(img: np.ndarray) -> wx.Bitmap:
    """An 8-bit grey (or RGB) numpy image as a bitmap."""
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else np.ascontiguousarray(img)
    return wx.Bitmap.FromBuffer(rgb.shape[1], rgb.shape[0], rgb)


def stroke(gc, path, rgb, width, dashed=False, halo=True):
    style = wx.PENSTYLE_SHORT_DASH if dashed else wx.PENSTYLE_SOLID
    if halo:
        p = wx.Pen(theme.colour(theme.HALO), 1, style)
        p.SetWidth(max(1, int(round(width + 2.5))))
        p.SetCap(wx.CAP_ROUND)
        p.SetJoin(wx.JOIN_ROUND)
        gc.SetPen(p)
        gc.StrokePath(path)
    p = wx.Pen(theme.colour(rgb), 1, style)
    p.SetWidth(max(1, int(round(width))))
    p.SetCap(wx.CAP_ROUND)
    p.SetJoin(wx.JOIN_ROUND)
    gc.SetPen(p)
    gc.StrokePath(path)


def disc(gc, x, y, r, rgb, edge=theme.HALO):
    gc.SetPen(wx.Pen(theme.colour(edge), 1))
    gc.SetBrush(wx.Brush(theme.colour(rgb)))
    gc.DrawEllipse(x - r, y - r, 2 * r, 2 * r)


def diamond(gc, x, y, r, rgb, edge=theme.HALO):
    """The mark of a reading to check (the timeline marks unsure readings the same way)."""
    path = gc.CreatePath()
    path.MoveToPoint(x, y - r)
    path.AddLineToPoint(x + r, y)
    path.AddLineToPoint(x, y + r)
    path.AddLineToPoint(x - r, y)
    path.CloseSubpath()
    gc.SetPen(wx.Pen(theme.colour(edge), 1))
    gc.SetBrush(wx.Brush(theme.colour(rgb)))
    gc.DrawPath(path)


def label(gc, text, x, y, rgb=(241, 245, 249), bold=False, size=10.5):
    gc.SetFont(theme.font(size, bold), theme.colour(rgb))
    tw, th = gc.GetTextExtent(text)[:2]
    gc.SetPen(wx.TRANSPARENT_PEN)
    gc.SetBrush(wx.Brush(wx.Colour(5, 8, 15, 170)))
    gc.DrawRoundedRectangle(x - 3, y - 1, tw + 6, th + 2, 3)
    gc.DrawText(text, x, y)


def banner(gc, text, w, y=14.0, edge=theme.DRAW):
    """A line across a view: what to click with the tool in hand (at the top), or what just happened."""
    gc.SetFont(theme.font(12.5, bold=True), theme.colour((241, 245, 249)))
    tw, th = gc.GetTextExtent(text)[:2]
    x = max((w - tw) / 2, 16.0)
    gc.SetPen(theme.pen(edge, 1.5))
    gc.SetBrush(wx.Brush(wx.Colour(5, 8, 15, 220)))
    gc.DrawRoundedRectangle(x - 12, y - 7, tw + 24, th + 14, 7)
    gc.DrawText(text, x, y)


def polyline(gc, pts):
    path = gc.CreatePath()
    for i, (x, y) in enumerate(pts):
        path.MoveToPoint(x, y) if i == 0 else path.AddLineToPoint(x, y)
    return path


def draw_grains(gc, ctl, to_c, scale: float, w: float, h: float, zoomed: bool = False) -> None:
    """Every grain as it is at the current bin: outline coloured by state, name, tube to its length, tip, exit,
    and a mark on grains still to check."""
    b, sel, ov = ctl.b, ctl.sel, ctl.overlays
    for g in ctl.grains:
        gx, gy = pos_at(g, b)
        cx, cy = to_c(gx, gy)
        rr = max((g["r"] + 3.5) * scale, 11.0 if zoomed else 5.5)
        if cx < -rr - 150 or cy < -rr - 150 or cx > w + rr + 150 or cy > h + rr + 150:
            continue
        st = state_at(g, b)
        is_sel = g["id"] == sel
        if ov.get("tubes", True) and not g.get("excluded"):
            if is_sel and zoomed:  # the whole route the tube is measured along (the tip tool snaps to it)
                route, _ = route_at(g, b)
                if len(route) > 1:
                    full = shift(to_length(route, max(path_length(route) + 15.0, g["L"][b]), 15.0), drift_at(g, b))
                    stroke(gc, polyline(gc, [to_c(*p) for p in full]), theme.TUBE + (110,), 1.4, dashed=True,
                           halo=False)
            tube = tube_at(g, b)
            if tube:
                pts = [to_c(*p) for p in tube]
                faded = st == "lost"
                stroke(gc, polyline(gc, pts), theme.TUBE + ((140,) if faded else ()), 3.0 if is_sel else 2.2)
                disc(gc, pts[0][0], pts[0][1], 2.6 if is_sel else 2.0, theme.EXIT)
                disc(gc, pts[-1][0], pts[-1][1], 4.2 if is_sel else 3.2, theme.TIP)
        if ov.get("grains", True) or is_sel:
            path = gc.CreatePath()
            path.AddCircle(cx, cy, rr)
            stroke(gc, path, theme.STATE[st], 2.6 if is_sel else 1.8, dashed=st in theme.DASHED)
            if st == "excluded":
                d = rr * 0.55
                x = gc.CreatePath()
                x.MoveToPoint(cx - d, cy - d)
                x.AddLineToPoint(cx + d, cy + d)
                x.MoveToPoint(cx + d, cy - d)
                x.AddLineToPoint(cx - d, cy + d)
                stroke(gc, x, theme.STATE["excluded"], 1.8)
            if is_sel:
                ring = gc.CreatePath()
                ring.AddCircle(cx, cy, rr + 5.5)
                stroke(gc, ring, theme.SELECTED, 2.2)
            if needs_check(g):
                a = math.radians(-45)
                diamond(gc, cx + rr * math.cos(a), cy + rr * math.sin(a), 6.5 if zoomed else 5.0, theme.CHECK)
        if is_sel or (ov.get("names", True) and (scale >= 0.42 or needs_check(g))):
            label(gc, g["id"], cx + rr * 0.78 + 2, cy + rr * 0.55, theme.SELECTED if is_sel else (241, 245, 249),
                  bold=is_sel, size=11.0 if is_sel else 9.5)


def draw_path_tool(gc, pts):
    if not pts:
        return
    stroke(gc, polyline(gc, pts), theme.DRAW, 2.2)
    for i, (x, y) in enumerate(pts):
        disc(gc, x, y, 4.0 if i in (0, len(pts) - 1) else 2.6, theme.EXIT if i == 0 else theme.DRAW)


class FieldCanvas(wx.Panel):
    """The whole field at the current bin, with pan (drag), zoom (scroll, pinch) and clicks."""

    def __init__(self, parent, ctl):
        super().__init__(parent, style=wx.WANTS_CHARS | wx.FULL_REPAINT_ON_RESIZE)
        self.ctl = ctl
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(theme.colour(theme.CANVAS_BG))
        self.SetMinSize((360, 240))
        self.view = None  # (scale, ox, oy): canvas px = (ref - o) * scale
        self.fit = True
        self._drag = None
        self._hover = None
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)
        self.Bind(wx.EVT_LEFT_DCLICK, self.on_dclick)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_wheel)
        if hasattr(wx, "EVT_MAGNIFY"):
            self.Bind(wx.EVT_MAGNIFY, self.on_magnify)

    # ---- view ------------------------------------------------------------------------------------
    def fit_view(self):
        if self.ctl.data is None:
            return
        w, h = self.GetClientSize()
        W, H = self.ctl.data.width, self.ctl.data.height
        if w < 2 or h < 2 or not W:
            return
        s = min(w / W, h / H) * 0.98
        self.view = (s, W / 2 - w / (2 * s), H / 2 - h / (2 * s))
        self.fit = True
        self.Refresh()

    def to_c(self, x, y):
        s, ox, oy = self.view
        return ((x - ox) * s, (y - oy) * s)

    def to_ref(self, cx, cy):
        s, ox, oy = self.view
        return (cx / s + ox, cy / s + oy)

    def zoom_at(self, factor, cx=None, cy=None):
        if self.view is None:
            return
        w, h = self.GetClientSize()
        cx = w / 2 if cx is None else cx
        cy = h / 2 if cy is None else cy
        s, ox, oy = self.view
        rx, ry = cx / s + ox, cy / s + oy
        s2 = float(np.clip(s * factor, 0.1, 16.0))
        self.view = (s2, rx - cx / s2, ry - cy / s2)
        self.fit = False
        self.Refresh()
        self.ctl.on_view_changed()

    def centre_on(self, x, y, span: float = 190.0):
        w, h = self.GetClientSize()
        s = float(np.clip(min(w, h) / span, 0.5, 8.0))
        self.view = (s, x - w / (2 * s), y - h / (2 * s))
        self.fit = False
        self.Refresh()
        self.ctl.on_view_changed()

    def zoom_percent(self) -> int:
        return int(round(self.view[0] * 100)) if self.view else 100

    # ---- painting -------------------------------------------------------------------------------------
    def on_size(self, e):
        if self.fit or self.view is None:
            self.fit_view()
        self.Refresh()
        e.Skip()

    def on_paint(self, e):
        dc = wx.PaintDC(self)
        w, h = self.GetClientSize()
        dc.SetBackground(wx.Brush(theme.colour(theme.CANVAS_BG)))
        dc.Clear()
        ctl = self.ctl
        if ctl.data is None or ctl.frames is None:
            return
        if self.view is None:
            self.fit_view()
            if self.view is None:
                return
        gc = wx.GraphicsContext.Create(dc)
        try:
            self.draw_image(gc, w, h)
            draw_grains(gc, ctl, self.to_c, self.view[0], w, h)
            if ctl.tool == "path" and ctl.path_pts:
                draw_path_tool(gc, [self.to_c(*p) for p in ctl.path_pts])
            if ctl.tool:
                banner(gc, ctl.tool_hint(), w)
            if ctl.notice_text:
                text, ok = ctl.notice_text
                banner(gc, text, w, y=h - 40.0, edge=theme.STATE["germinated"] if ok else theme.CHECK)
        except Exception as exc:  # noqa: BLE001 - a drawing error must not stop the window
            ctl.status(f"Drawing failed: {exc}")

    def draw_image(self, gc, w, h):
        img = self.ctl.frames.field(self.ctl.b, self.ctl.mode)
        H, W = img.shape
        s, ox, oy = self.view
        x0, y0 = max(0, int(math.floor(ox))), max(0, int(math.floor(oy)))
        x1, y1 = min(W, int(math.ceil(ox + w / s)) + 1), min(H, int(math.ceil(oy + h / s)) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        dx, dy = (x0 - ox) * s, (y0 - oy) * s
        dw, dh = (x1 - x0) * s, (y1 - y0) * s
        sf = self.GetContentScaleFactor()
        pw, ph = max(1, int(round(dw * sf))), max(1, int(round(dh * sf)))
        crop = img[y0:y1, x0:x1]
        interp = cv2.INTER_AREA if pw < crop.shape[1] else (cv2.INTER_NEAREST if s * sf > 5 else cv2.INTER_LINEAR)
        small = cv2.resize(crop, (pw, ph), interpolation=interp)
        gc.DrawBitmap(to_bitmap(small), dx, dy, dw, dh)

    # ---- mouse ------------------------------------------------------------------------------------------
    def grain_at(self, cx, cy):
        x, y = self.to_ref(cx, cy)
        best, bd = None, 1e9
        for g in self.ctl.grains:
            gx, gy = pos_at(g, self.ctl.b)
            d = math.hypot(gx - x, gy - y)
            if d < bd:
                best, bd = g, d
        return best if best is not None and bd <= best["r"] + max(6.0, 8.0 / self.view[0]) else None

    def on_down(self, e):
        self.SetFocus()
        self._drag = (e.GetX(), e.GetY(), self.view, False)
        if not self.HasCapture():
            self.CaptureMouse()

    def on_motion(self, e):
        if self._drag and e.LeftIsDown():
            x0, y0, view, moved = self._drag
            dx, dy = e.GetX() - x0, e.GetY() - y0
            if moved or abs(dx) + abs(dy) > 3:
                s, ox, oy = view
                self.view = (s, ox - dx / s, oy - dy / s)
                self.fit = False
                self._drag = (x0, y0, view, True)
                self.Refresh()
            return
        if self.view is None or self.ctl.data is None:
            return
        g = self.grain_at(e.GetX(), e.GetY())
        gid = g["id"] if g else None
        if gid != self._hover:
            self._hover = gid
            self.SetToolTip(self.ctl.grain_tip(g) if g else "")
            if not self.ctl.tool:
                self.SetCursor(wx.Cursor(wx.CURSOR_HAND if g else wx.CURSOR_DEFAULT))

    def on_up(self, e):
        if self.HasCapture():
            self.ReleaseMouse()
        drag, self._drag = self._drag, None
        if drag and drag[3]:
            self.ctl.on_view_changed()
            return
        if self.view is None:
            return
        ref = self.to_ref(e.GetX(), e.GetY())
        self.ctl.on_click(ref, self.grain_at(e.GetX(), e.GetY()))

    def on_dclick(self, e):
        g = self.grain_at(e.GetX(), e.GetY())
        if g:
            self.ctl.select(g["id"], zoom=True)

    def on_right(self, e):
        """A grain's corrections, at the time shown (it is selected first)."""
        if self.view is None or self.ctl.data is None:
            return
        g = self.grain_at(e.GetX(), e.GetY())
        if g is not None:
            self.ctl.select(g["id"])
            self.PopupMenu(self.ctl.grain_menu(), e.GetPosition())

    def on_wheel(self, e):
        rot = e.GetWheelRotation()
        if not rot:
            return
        self.zoom_at(math.exp(rot / max(e.GetWheelDelta(), 1) * 0.12), e.GetX(), e.GetY())

    def on_magnify(self, e):
        self.zoom_at(1.0 + e.GetMagnification(), e.GetX(), e.GetY())


class GrainView(wx.Panel):
    """The selected grain close up at the current bin, following it as it moves."""

    HALVES = (40, 80, 160)

    def __init__(self, parent, ctl, size=360):
        super().__init__(parent, size=(size, size), style=wx.FULL_REPAINT_ON_RESIZE)
        self.ctl = ctl
        self.zoom_index = 0
        self.aim = (0.0, 0.0)  # where to look from the grain's centre (towards its tube), kept as it moves
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetBackgroundColour(theme.colour(theme.CANVAS_BG))
        self.SetMinSize((size, size))
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_UP, self.on_up)
        self.Bind(wx.EVT_MOUSEWHEEL, self.on_wheel)

    def geometry(self):
        g = self.ctl.selected()
        if g is None:
            return None
        w = self.GetClientSize()[0]
        half = self.HALVES[self.zoom_index]
        x, y = pos_at(g, self.ctl.b)
        room = max(half - g["r"] - 4.0, 0.0)  # the grain stays in view
        x, y = x + float(np.clip(self.aim[0], -room, room)), y + float(np.clip(self.aim[1], -room, room))
        cx, cy = round(x * 2) / 2, round(y * 2) / 2
        return {"half": half, "cx": cx, "cy": cy, "z": w / (2 * half), "w": w}

    def to_c(self, G):
        return lambda x, y: ((x - (G["cx"] - G["half"])) * G["z"], (y - (G["cy"] - G["half"])) * G["z"])

    def on_paint(self, e):
        dc = wx.PaintDC(self)
        w, h = self.GetClientSize()
        dc.SetBackground(wx.Brush(theme.colour(theme.CANVAS_BG)))
        dc.Clear()
        ctl = self.ctl
        G = self.geometry()
        if G is None or ctl.frames is None:
            return
        gc = wx.GraphicsContext.Create(dc)
        sf = self.GetContentScaleFactor()
        zoom = min(12.0, round(G["z"] * sf * 4) / 4)
        try:
            img = ctl.frames.crop(ctl.b, G["cx"], G["cy"], G["half"], zoom, ctl.mode, ctl.sel)
        except Exception as exc:  # noqa: BLE001 - a drawing error must not stop the window
            ctl.status(f"Drawing failed: {exc}")
            return
        gc.DrawBitmap(to_bitmap(img), 0, 0, w, w)
        to_c = self.to_c(G)
        draw_grains(gc, ctl, to_c, G["z"], w, h, zoomed=True)
        if ctl.tool == "path" and ctl.path_pts:
            draw_path_tool(gc, [to_c(*p) for p in ctl.path_pts])
        # scale bar: 10 um (20 px without the pixel size)
        um = ctl.data.units.um_per_px
        bar = (10.0 / um if um else 20.0) * G["z"]
        gc.SetPen(wx.Pen(wx.Colour(0, 0, 0, 200), 1))
        gc.SetBrush(wx.Brush(wx.Colour(248, 250, 252)))
        gc.DrawRectangle(w - 12 - bar, h - 15, bar, 4)
        label(gc, "10 µm" if um else "20 px", w - 12 - bar, h - 33, size=9.0)

    def on_up(self, e):
        G = self.geometry()
        if G is None:
            return
        ref = (G["cx"] - G["half"] + e.GetX() / G["z"], G["cy"] - G["half"] + e.GetY() / G["z"])
        best, bd = None, 1e9
        for g in self.ctl.grains:
            x, y = pos_at(g, self.ctl.b)
            d = math.hypot(x - ref[0], y - ref[1])
            if d < bd:
                best, bd = g, d
        self.ctl.on_click(ref, best if best is not None and bd <= best["r"] + 3 else None, zoomed=True)

    def on_wheel(self, e):
        step = -1 if e.GetWheelRotation() > 0 else 1
        self.set_zoom(self.zoom_index + step)

    def set_zoom(self, i):
        self.zoom_index = int(np.clip(i, 0, len(self.HALVES) - 1))
        self.Refresh()

    def fit(self, g: dict, last: int) -> int:
        """Aim at a grain and its tube at its longest (``last``) and pick the closest zoom that shows both; returns
        the zoom picked."""
        x, y = pos_at(g, last)
        rel = [(px - x, py - y) for px, py in (tube_at(g, last) or [])] + [(-g["r"], -g["r"]), (g["r"], g["r"])]
        xs, ys = [q[0] for q in rel], [q[1] for q in rel]
        self.aim = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        need = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 6.0
        i = next((k for k, half in enumerate(self.HALVES) if need <= half), len(self.HALVES) - 1)
        self.set_zoom(i)
        return i
