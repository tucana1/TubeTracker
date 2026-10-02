"""Colours and fonts. The window uses the system's own look; the movie and its charts sit on a dark ground so the
grains, tubes and marks stand out."""

from __future__ import annotations

import wx

CANVAS_BG = (11, 15, 23)
GRID = (38, 50, 77)
AXIS_TEXT = (147, 161, 184)
STATE = {  # grain outline by state
    "germinated": (52, 211, 153), "notyet": (226, 232, 240), "never": (124, 180, 255), "lost": (251, 146, 60),
    "excluded": (248, 113, 113), "unobservable": (168, 162, 158),
}
SELECTED = (56, 189, 248)
CHECK = (251, 191, 36)
TUBE = (244, 114, 182)
TIP = (253, 224, 71)
EXIT = (255, 255, 255)
DRAW = (34, 211, 238)
FOCUS = (192, 132, 252)
HALO = (0, 0, 0, 180)
WARN_TEXT = (180, 110, 0)
EVENT = {"germination": STATE["germinated"], "lost": STATE["lost"], "stall": (203, 213, 225), "check": CHECK,
         "focus": FOCUS, "t50": SELECTED}
DASHED = ("lost", "excluded", "unobservable")


def colour(rgb, alpha: int = 255) -> wx.Colour:
    return wx.Colour(*rgb[:3], rgb[3] if len(rgb) > 3 else alpha)


def pen(rgb, width: float = 1.0, style=wx.PENSTYLE_SOLID, alpha: int = 255) -> wx.Pen:
    p = wx.Pen(colour(rgb, alpha), 1, style)
    p.SetWidth(max(1, int(round(width))))
    return p


def font(size: float | None = None, bold: bool = False) -> wx.Font:
    f = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
    if size:
        f.SetFractionalPointSize(size)
    if bold:
        f = f.Bold()
    return f


def window_bg() -> wx.Colour:
    return wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)


def panel_bg() -> wx.Colour:
    return wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE)


def muted_fg() -> wx.Colour:
    return wx.SystemSettings.GetColour(wx.SYS_COLOUR_GRAYTEXT)
