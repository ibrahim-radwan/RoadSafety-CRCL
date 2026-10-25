#!/usr/bin/env python3
"""
replay_session.py — replay a 3-sensor session in real time, on one clock,
with our GPU detection and tracking live on the radar and the LiDAR.

One window, six panels (each can be maximised: double-click its title, its
button, or keys 1-6; Esc restores):

    Camera          the ZED left image (SVO2 decoded on the GPU by the ZED SDK)
    LiDAR           bird's-eye (forward / 360) or panorama, filling the panel;
                    object boxes coloured by size class, track IDs (L<n>)
    LiDAR 3D        the same sweep as an orbitable point cloud, 3D boxes
    Radar BEV       range-azimuth power on the same metric window as the LiDAR,
                    detections (colour = radial velocity, hollow = multipath
                    ghost), confirmed tracks (R<n>)
    Range-Doppler   range vs radial velocity
    Objects         confirmed tracks of both sensors

Each panel shows the time of the data it displays; for the radar frame on
screen, the pairing errors and flags from sync/pairs.csv are shown.

All sensors follow one clock (CLOCK_MONOTONIC session time from
align_session.py, offsets included). Play/pause, speed, seek and stepping
by radar frame apply to every sensor at once. Choose sensors with --sensors
to review one alone.

Requires: python align_session.py --session <s>   (writes sync/)

KEYS    space play/pause   left/right: previous/next radar frame
        [ ] slower/faster  s: save a screenshot   m: radar moving/all
        1-6 maximise a panel, Esc / 0 restore all

USAGE (conda env)
    python replay/replay_session.py --session sessions/run05
    python replay/replay_session.py --session sessions/run05 --sensors lidar
    python replay/replay_session.py --session sessions/run05 --start 12.5 --speed 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# `conda activate` runs qt-main's activation script, which sets
# QT_XCB_GL_INTEGRATION=none: Qt then has neither GLX nor EGL, and the 3D LiDAR
# view (OpenGL) fails with "Cannot create platform OpenGL context". Re-enable
# Qt's default GL integration for this process, before Qt is loaded. Any other
# value the user sets is respected.
if os.environ.get("QT_XCB_GL_INTEGRATION") == "none":
    del os.environ["QT_XCB_GL_INTEGRATION"]

import numpy as np
from matplotlib import cm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from sources import (LIDAR_COLORS, LIDAR_VIEWS, CameraSource, LidarSource, RadarSource,  # noqa: E402
                     ReplayClock, load_csv, top_extent)

LIDAR_RANGES = [5.0, 10.0, 20.0, 40.0, 80.0]
LIDAR_NEAR = [0.0, 0.3, 0.5, 1.0, 2.0]
SPEEDS = [0.25, 0.5, 1.0, 2.0]
RADAR_C, LIDAR_C, CAMERA_C = (42, 120, 214), (235, 104, 52), (27, 175, 122)   # report palette
RADAR_POINTS = [("detections (ours: CFAR + tracking)", "detections"), ("raw (all range-azimuth cells)", "raw"),
                ("filtered (team pipeline)", "filtered"), ("off", "off")]

STYLE = """
QWidget { background: #15171c; color: #d4d7dd; font-size: 12px; }
QFrame#panel { background: #0e1014; border: 1px solid #2a2e37; border-radius: 4px; }
QFrame#panelHead { background: #1b1e25; border: none; border-bottom: 1px solid #2a2e37;
                   border-top-left-radius: 4px; border-top-right-radius: 4px; }
QLabel#panelTitle { font-weight: 600; color: #e8eaee; background: transparent; }
QLabel#panelInfo, QLabel#hint { color: #8b919c; background: transparent; }
QToolButton { background: transparent; border: none; color: #8b919c; padding: 0 4px; }
QToolButton:hover { color: #ffffff; }
QPushButton { background: #232731; border: 1px solid #343a46; border-radius: 4px; padding: 3px 10px; }
QPushButton:hover { background: #2c3240; }
QPushButton#play { background: #2a78d6; border-color: #2a78d6; color: white; font-weight: 600; min-width: 64px; }
QComboBox, QSpinBox { background: #1d2028; border: 1px solid #343a46; border-radius: 4px; padding: 2px 6px; }
QComboBox QAbstractItemView { background: #1d2028; selection-background-color: #2a78d6; }
QGroupBox { border: 1px solid #2a2e37; border-radius: 4px; margin-top: 8px; padding: 6px 6px 2px 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #8b919c; }
QSlider::groove:horizontal { height: 6px; background: #2a2e37; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #2a78d6; border-radius: 3px; }
QSlider::handle:horizontal { background: #e8eaee; width: 12px; margin: -4px 0; border-radius: 6px; }
QSplitter::handle { background: #15171c; }
QTableWidget { background: #0e1014; gridline-color: #22262e; border: none; }
QHeaderView::section { background: #1b1e25; color: #8b919c; border: none; padding: 3px; }
QCheckBox::indicator { width: 13px; height: 13px; }
"""


class Panel:
    """A titled panel: header (title, info, maximise button) over a body widget."""

    def __init__(self, key: str, title: str, body, on_max):
        from PyQt5 import QtCore, QtWidgets
        self.key = key
        self.frame = QtWidgets.QFrame()
        self.frame.setObjectName("panel")
        lay = QtWidgets.QVBoxLayout(self.frame)
        lay.setContentsMargins(1, 1, 1, 1)
        lay.setSpacing(0)
        head = QtWidgets.QFrame()
        head.setObjectName("panelHead")
        hl = QtWidgets.QHBoxLayout(head)
        hl.setContentsMargins(8, 3, 4, 3)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("panelTitle")
        self.info = QtWidgets.QLabel()
        self.info.setObjectName("panelInfo")
        self.info.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.btn = QtWidgets.QToolButton()
        self.btn.setText("⤢")
        self.btn.setToolTip("maximise / restore (double-click the title)")
        self.btn.clicked.connect(lambda: on_max(self.key))
        hl.addWidget(self.title)
        hl.addStretch(1)
        hl.addWidget(self.info)
        hl.addWidget(self.btn)
        head.mouseDoubleClickEvent = lambda _e: on_max(self.key)
        lay.addWidget(head)
        lay.addWidget(body, stretch=1)

    def set(self, title=None, info=None):
        if title is not None:
            self.title.setText(title)
        if info is not None:
            self.info.setText(info)


def _plot(pg, xlabel=None, ylabel=None):
    w = pg.PlotWidget()
    w.setBackground((14, 16, 20))
    p = w.getPlotItem()
    for ax in ("left", "bottom"):
        a = p.getAxis(ax)
        a.setPen(pg.mkPen((70, 76, 88)))
        a.setTextPen(pg.mkPen((140, 146, 156)))
    if xlabel:
        p.setLabel("bottom", xlabel)
    if ylabel:
        p.setLabel("left", ylabel)
    p.setMenuEnabled(False)
    return w, p


def build_window(session: Path, sensors: list, clock: ReplayClock, srcs: dict, pairs: dict,
                 bev_x, bev_y, rd_axes, args, track=None):
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtGui, QtWidgets
    from perception.lidar import CLASS_COLOURS, CLASSES, O

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"Replay - {session.name}")
    win.setStyleSheet(STYLE)
    root = QtWidgets.QVBoxLayout(win)
    root.setContentsMargins(6, 6, 6, 4)
    root.setSpacing(4)

    panels, items = {}, {}
    lut_hot = pg.colormap.get("inferno").getLookupTable(0.0, 1.0, 256)
    state = {"slider_drag": False, "last_rates": {}, "rate_t": time.monotonic(), "rate_n": {}, "max": None}
    shown = {}

    def toggle_max(key):
        state["max"] = None if state["max"] == key else key
        for k, pnl in panels.items():
            pnl.frame.setVisible(state["max"] is None or k == state["max"])
        for row in (top, bottom):
            row.setVisible(any(panels[k].frame.isVisible() for k in panels if panels[k].frame.parent() is row))

    # ── panels ──────────────────────────────────────────────────────────
    top, bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal), QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    vsplit.addWidget(top)
    vsplit.addWidget(bottom)
    vsplit.setChildrenCollapsible(False)
    root.addWidget(vsplit, stretch=1)

    def bev_plot():
        w, p = _plot(pg, "y  (m, + = left)", "x  (m, forward)")
        img = pg.ImageItem()
        p.addItem(img)
        img.setImage(np.zeros((2, 2), np.float32), autoLevels=False, levels=(0, 1))   # 0.13: image before setRect
        p.setAspectLocked(True)
        p.invertX(True)                                          # +y (left) drawn on the left
        p.showGrid(x=True, y=True, alpha=0.12)
        fov = p.plot([], [], pen=pg.mkPen((255, 80, 80, 160), width=1, style=QtCore.Qt.DashLine))
        tgt = pg.ScatterPlotItem(size=18, symbol="o", pen=pg.mkPen((255, 255, 255), width=2), brush=None)
        tgt.setZValue(20)
        p.addItem(tgt)
        return w, p, img, fov, tgt

    if "camera" in sensors:
        w, p = _plot(pg)
        img = pg.ImageItem()
        p.addItem(img)
        p.invertY(True)
        p.setAspectLocked(True)
        p.hideAxis("left"); p.hideAxis("bottom")
        items["camera"] = (p, img)
        panels["camera"] = Panel("camera", "Camera", w, toggle_max)
        top.addWidget(panels["camera"].frame)
    if "lidar" in sensors:
        w, p, img, fov, tgt = bev_plot()
        boxes = {c: p.plot([], [], pen=pg.mkPen(CLASS_COLOURS[c], width=1.5), connect="finite") for c in CLASSES}
        items["lidar"] = dict(plot=p, img=img, fov=fov, tgt=tgt, boxes=boxes, labels=[], widget=w)
        panels["lidar"] = Panel("lidar", "LiDAR", w, toggle_max)
        top.addWidget(panels["lidar"].frame)
        try:
            from lidar3d import Lidar3DView
            v3 = Lidar3DView(args.lidar_range)
            items["lidar3d"] = v3
            body3 = v3.widget
        except ImportError as exc:                                 # PyOpenGL missing
            body3 = QtWidgets.QLabel(f"3D view unavailable: {exc}")
            body3.setAlignment(QtCore.Qt.AlignCenter)
        panels["lidar3d"] = Panel("lidar3d", "LiDAR 3D", body3, toggle_max)
        top.addWidget(panels["lidar3d"].frame)
    if "radar" in sensors:
        w, p, img, fov, tgt = bev_plot()
        img.setLookupTable(lut_hot)
        dets = pg.ScatterPlotItem(size=7, pen=None)
        dets.setZValue(12)
        ghosts = pg.ScatterPlotItem(size=7, pen=pg.mkPen((170, 170, 170), width=1), brush=None)
        ghosts.setZValue(11)
        rpts = pg.ScatterPlotItem(size=4, pen=None)
        rpts.setZValue(10)
        trk = pg.ScatterPlotItem(size=14, symbol="d", pen=pg.mkPen((255, 255, 255), width=1.5),
                                 brush=pg.mkBrush(42, 120, 214, 200))
        trk.setZValue(15)
        for it in (dets, ghosts, rpts, trk):
            p.addItem(it)
        items["radar_bev"] = dict(plot=p, img=img, fov=fov, tgt=tgt, dets=dets, ghosts=ghosts, pts=rpts, trk=trk,
                                  labels=[], widget=w)
        panels["radar_bev"] = Panel("radar_bev", "Radar BEV", w, toggle_max)
        bottom.addWidget(panels["radar_bev"].frame)
        w, p = _plot(pg, "radial velocity (m/s, + = receding)", "range (m)")
        img = pg.ImageItem()
        img.setLookupTable(lut_hot)
        p.addItem(img)
        (r0, r1), (v0, v1), shape = rd_axes
        img.setImage(np.zeros(shape, np.float32), autoLevels=False, levels=(0, 1))
        img.setRect(QtCore.QRectF(v0, r0, v1 - v0, r1 - r0))
        rd_line = pg.InfiniteLine(angle=0, pen=pg.mkPen((255, 255, 255), width=1, style=QtCore.Qt.DashLine))
        p.addItem(rd_line)
        rd_line.hide()
        rd_dets = pg.ScatterPlotItem(size=6, pen=pg.mkPen((0, 230, 255), width=1), brush=None)
        p.addItem(rd_dets)
        items["radar_rd"] = dict(plot=p, img=img, line=rd_line, dets=rd_dets)
        panels["radar_rd"] = Panel("radar_rd", "Range-Doppler", w, toggle_max)
        bottom.addWidget(panels["radar_rd"].frame)
    table = QtWidgets.QTableWidget(0, 7)
    table.setHorizontalHeaderLabels(["sensor", "track", "class", "range m", "az °", "speed m/s", "v_r m/s"])
    table.verticalHeader().hide()
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    table.setFocusPolicy(QtCore.Qt.NoFocus)
    panels["objects"] = Panel("objects", "Objects (confirmed tracks)", table, toggle_max)
    bottom.addWidget(panels["objects"].frame)
    for sp, sizes in ((top, [5, 5, 5]), (bottom, [5, 5, 3])):
        sp.setChildrenCollapsible(False)
        sp.setSizes([s * 100 for s in sizes[:sp.count()]])
    vsplit.setSizes([560, 400])

    rd_rect = QtCore.QRectF(rd_axes[1][0], rd_axes[0][0], rd_axes[1][1] - rd_axes[1][0],
                            rd_axes[0][1] - rd_axes[0][0])
    extent_r = QtCore.QRectF(-bev_y[-1], bev_x[0], bev_y[-1] - bev_y[0], bev_x[-1] - bev_x[0])
    if "radar_bev" in items:
        items["radar_bev"]["img"].setRect(extent_r)

    # ── view geometry: fill each bird's-eye panel at the selected range ─
    def view_for(key):
        """(mode, R, aspect, px) of a bird's-eye panel: its view box size on screen."""
        vb = items[key]["plot"].getViewBox()
        w_, h_ = max(vb.width(), 10.0), max(vb.height(), 10.0)
        mode = srcs["lidar"].view["mode"] if "lidar" in srcs else "top"
        R = lidar_ctl["range_m"].currentData() if "range_m" in lidar_ctl else args.lidar_range
        return ("top" if mode == "panorama" else mode), float(R), w_ / h_, w_

    def apply_views(force=False):
        for key in ("lidar", "radar_bev"):
            if key not in items:
                continue
            mode, R, asp, px = view_for(key)
            x0, x1, y0, y1 = top_extent({"mode": mode, "range_m": R, "aspect": asp})
            p_ = items[key]["plot"]
            if key == "lidar":
                if srcs["lidar"].view["mode"] == "panorama":
                    continue
                v = srcs["lidar"].view
                if force or abs(v.get("aspect", 0) - asp) > 0.02 * asp or abs(v.get("px", 0) - px) > 40:
                    srcs["lidar"].set_view(aspect=asp, px=px)
            p_.disableAutoRange()
            p_.setRange(xRange=(y0, y1), yRange=(x0, x1), padding=0)
            rr = x1 if mode == "top" else x1
            c, sn = np.cos(np.radians(85)) * rr * 1.5, np.sin(np.radians(85)) * rr * 1.5
            items[key]["fov"].setData([sn, 0, -sn], [c, 0, c])
        if "radar_rd" in items:
            R = lidar_ctl["range_m"].currentData() if "range_m" in lidar_ctl else args.lidar_range
            items["radar_rd"]["plot"].disableAutoRange()
            items["radar_rd"]["plot"].setRange(xRange=(rd_axes[1][0], rd_axes[1][1]), yRange=(0, R), padding=0)

    # ── event strip: where the gaps are; click to jump ──────────────────
    ev, evp = _plot(pg)
    ev.setFixedHeight(54)
    evp.hideAxis("left")
    ev.setMouseEnabled(x=False, y=False)
    evp.setYRange(-0.2, 2.2, padding=0)
    span = clock.t1 - clock.t0
    margin = 0.09 * span                          # labels live left of t = 0
    evp.setXRange(-margin, span, padding=0)
    marks = {"camera_gap": ([], CAMERA_C, 0), "lidar_gap": ([], LIDAR_C, 1)}
    for pr in pairs.values():
        for f in (pr.get("flags") or "").split(";"):
            if f in marks:
                marks[f][0].append(float(pr["t_radar"]) - clock.t0)
    for f, (xs, col, row_) in marks.items():
        if xs:
            evp.plot(np.repeat(xs, 3), np.tile([row_ + 0.05, row_ + 0.95, np.nan], len(xs)),
                     pen=pg.mkPen(col, width=1), connect="finite")
    for text, col, y in (("camera gaps", CAMERA_C, 0.5), ("LiDAR gaps", LIDAR_C, 1.5)):
        lab = pg.TextItem(text, color=col, anchor=(0, 0.5))
        lab.setPos(-margin, y)
        evp.addItem(lab)
    if srcs.get("radar") is not None and getattr(srcs["radar"], "timing_source", "").startswith("NOMINAL"):
        lab = pg.TextItem("radar times NOMINAL (DCA backlog)", color=RADAR_C, anchor=(1, 0.5))
        lab.setPos(clock.t1 - clock.t0, 2.0)
        evp.addItem(lab)
    cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen((220, 220, 220), width=1))
    evp.addItem(cursor)

    def ev_clicked(event):
        pos = evp.vb.mapSceneToView(event.scenePos())
        clock.seek(clock.t0 + float(np.clip(pos.x(), 0, clock.t1 - clock.t0)))
    ev.scene().sigMouseClicked.connect(ev_clicked)
    root.addWidget(ev)

    # ── transport ───────────────────────────────────────────────────────
    bar = QtWidgets.QHBoxLayout()
    root.addLayout(bar)
    b_play = QtWidgets.QPushButton("Play")
    b_play.setObjectName("play")
    b_prev = QtWidgets.QPushButton("◀ frame")
    b_next = QtWidgets.QPushButton("frame ▶")
    for b, tip in ((b_prev, "previous radar frame (←)"), (b_next, "next radar frame (→)"),
                   (b_play, "play / pause (space)")):
        b.setToolTip(tip)
    speed = QtWidgets.QComboBox()
    speed.addItems([f"{s:g}×" for s in SPEEDS])
    speed.setCurrentIndex(SPEEDS.index(args.speed) if args.speed in SPEEDS else 2)
    speed.setToolTip("playback speed ([ and ])")
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setRange(0, 10000)
    t_label = QtWidgets.QLabel()
    t_label.setMinimumWidth(150)
    for w_ in (b_play, b_prev, b_next, speed):
        bar.addWidget(w_)
    bar.addWidget(slider, stretch=1)
    bar.addWidget(t_label)

    # ── settings ────────────────────────────────────────────────────────
    sets = QtWidgets.QHBoxLayout()
    root.addLayout(sets)
    lidar_ctl, rctl, dctl = {}, {}, {}

    def group(title):
        g = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QHBoxLayout(g)
        lay.setContentsMargins(6, 2, 6, 2)
        sets.addWidget(g)
        return lay

    if "lidar" in srcs:
        row = group("LiDAR")
        v = srcs["lidar"].view
        for key, label, options, current in (
                ("mode", "", [("Top - forward", "top"), ("Top - 360°", "top360"), ("Panorama", "panorama")], v["mode"]),
                ("range_m", "range", [(f"{r:g} m", float(r)) for r in LIDAR_RANGES], v["range_m"]),
                ("color", "colour", [("height", "height"), ("intensity", "intensity"), ("density", "density")], v["color"]),
                ("near_m", "hide <", [(f"{r:g} m", float(r)) for r in LIDAR_NEAR], v["near_m"])):
            if label:
                row.addWidget(QtWidgets.QLabel(label))
            box = QtWidgets.QComboBox()
            for text, value in options:
                box.addItem(text, value)
            vals = [o[1] for o in options]
            box.setCurrentIndex(vals.index(current) if current in vals else 0)
            box.currentIndexChanged.connect(lambda _i, k=key, b=box: srcs["lidar"].set_view(**{k: b.currentData()}))
            row.addWidget(box)
            lidar_ctl[key] = box
        lidar_ctl["near_m"].setToolTip("hide returns nearer than this (the LiDAR mount, the vehicle body)")
        lidar_ctl["range_m"].setToolTip("forward extent of both bird's-eye views and the range-Doppler map")
    if "radar" in srcs:
        row = group("Radar")
        moving = QtWidgets.QCheckBox("moving only")
        moving.setChecked(True)
        moving.setToolTip("remove the static scene (zero Doppler) - m")
        row.addWidget(moving)
        rmode = QtWidgets.QComboBox()
        for text, val in RADAR_POINTS:
            rmode.addItem(text, val)
        row.addWidget(rmode)
        row.addWidget(QtWidgets.QLabel("raw >"))
        rthr = QtWidgets.QSpinBox()
        rthr.setRange(3, 40)
        rthr.setValue(15)
        rthr.setSuffix(" dB")
        rthr.setToolTip("raw cells: dB above the frame's median (display only)")
        row.addWidget(rthr)
        rflat = QtWidgets.QCheckBox("z = 0")
        rflat.setChecked(True)
        rflat.setToolTip("draw radar points at z = 0 in 3D: the radar elevation is not verified")
        row.addWidget(rflat)
        rctl.update(mode=rmode, thr=rthr, flat=rflat, moving=moving)
        if srcs["radar"].cloud is None:
            rmode.model().item(2).setEnabled(False)
            rmode.setItemText(2, "filtered (team: run radar_cloud.py first)")
        want = getattr(args, "radar_points", "detections")
        if want == "filtered" and srcs["radar"].cloud is None:
            want = "detections"
        rmode.setCurrentIndex([v_ for _, v_ in RADAR_POINTS].index(want))
        moving.stateChanged.connect(lambda s_: setattr(srcs["radar"], "moving_only", bool(s_)))
    row = group("Detections")
    for key, text, tip in (("lidar_obj", "LiDAR objects", "boxes coloured by size class (heuristic)"),
                           ("tracks", "track IDs", "confirmed tracks: L<n> LiDAR, R<n> radar"),
                           ("target", "LiDAR target", "the single-target reference track (make_report.py)")):
        cb = QtWidgets.QCheckBox(text)
        cb.setChecked(True)
        cb.setToolTip(tip)
        row.addWidget(cb)
        dctl[key] = cb
    legend = QtWidgets.QLabel("  ".join(f"<span style='color:rgb{CLASS_COLOURS[c]}'>■</span> {c}"
                                        for c in CLASSES))
    legend.setTextFormat(QtCore.Qt.RichText)
    row.addWidget(legend)
    sets.addStretch(1)
    if "lidar" in srcs:
        for key in ("range_m", "mode"):
            lidar_ctl[key].currentIndexChanged.connect(lambda _i: apply_views(force=True))
        if "lidar3d" in items:
            lidar_ctl["range_m"].currentIndexChanged.connect(
                lambda _i: items["lidar3d"].set_range(lidar_ctl["range_m"].currentData()))
    elif "radar_bev" in items:
        pass

    info = QtWidgets.QLabel()
    info.setObjectName("hint")
    info.setTextFormat(QtCore.Qt.RichText)
    root.addWidget(info)

    radar_t = srcs["radar"].t if "radar" in srcs else None

    def toggle_play():
        clock.pause() if clock.playing else clock.play()

    def step(direction):
        clock.pause()
        if radar_t is None:
            clock.seek(clock.now() + direction / 30.0)
            return
        k = int(np.searchsorted(radar_t, clock.now(), side="right")) - 1 + direction
        clock.seek(float(radar_t[int(np.clip(k, 0, len(radar_t) - 1))]))

    def save_shot(path=None):
        path = path or str(session / "sync" / f"replay_{time.strftime('%H%M%S')}.png")
        img = win.grab().toImage()
        if "lidar3d" in items and items["lidar3d"].widget.isVisible():
            gl_w = items["lidar3d"].view                     # QWidget.grab cannot read an OpenGL surface
            painter = QtGui.QPainter(img)
            painter.drawImage(gl_w.mapTo(win, QtCore.QPoint(0, 0)), gl_w.grabFramebuffer())
            painter.end()
        img.save(path)
        return path

    b_play.clicked.connect(toggle_play)
    b_prev.clicked.connect(lambda: step(-1))
    b_next.clicked.connect(lambda: step(1))
    speed.currentIndexChanged.connect(lambda i: clock.set_speed(SPEEDS[i]))
    slider.sliderPressed.connect(lambda: state.update(slider_drag=True))
    slider.sliderReleased.connect(lambda: (state.update(slider_drag=False),
                                           clock.seek(clock.t0 + (clock.t1 - clock.t0) * slider.value() / 10000.0)))
    pkeys = list(panels)
    keys = {QtCore.Qt.Key_Space: toggle_play, QtCore.Qt.Key_Left: lambda: step(-1),
            QtCore.Qt.Key_Right: lambda: step(1), QtCore.Qt.Key_S: lambda: print("saved", save_shot()),
            QtCore.Qt.Key_BracketLeft: lambda: speed.setCurrentIndex(max(0, speed.currentIndex() - 1)),
            QtCore.Qt.Key_BracketRight: lambda: speed.setCurrentIndex(min(len(SPEEDS) - 1, speed.currentIndex() + 1)),
            QtCore.Qt.Key_Escape: lambda: state["max"] and toggle_max(state["max"]),
            QtCore.Qt.Key_0: lambda: state["max"] and toggle_max(state["max"])}
    if rctl:
        keys[QtCore.Qt.Key_M] = lambda: rctl["moving"].setChecked(not rctl["moving"].isChecked())
    for i, k in enumerate(pkeys[:6]):
        keys[getattr(QtCore.Qt, f"Key_{i + 1}")] = (lambda kk=k: toggle_max(kk))
    for k, fn in keys.items():
        QtWidgets.QShortcut(QtGui.QKeySequence(k), win, activated=fn)

    def fmt(t):
        return f"{t - clock.t0:7.2f} s"

    def text_pool(key, n, prefix_colour):
        pool = items[key]["labels"]
        while len(pool) < n:
            t_ = pg.TextItem(color=prefix_colour, anchor=(0, 1))
            t_.setZValue(25)
            items[key]["plot"].addItem(t_)
            pool.append(t_)
        return pool

    def changed(name, key):
        if shown.get(name, "init") == key:
            return False
        shown[name] = key
        return True

    def box_lines(objs):
        """Per class: NaN-separated outline polylines of the objects' boxes, in plot coords (y, x)."""
        out = {c: ([], []) for c in CLASSES}
        for ob in objs:
            a = np.radians(ob[O["yaw_deg"]])
            u, v_ = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            hl, hw = ob[O["length"]] / 2, ob[O["width"]] / 2
            c0 = np.array([ob[O["x"]], ob[O["y"]]])
            cs = [c0 + sl * hl * u + sw * hw * v_ for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))]
            xs, ys = out[CLASSES[int(ob[O["cls"]])]]
            xs += [c[1] for c in cs] + [np.nan]
            ys += [c[0] for c in cs] + [np.nan]
        return out

    def update():
        t = clock.now()
        b_play.setText("Pause" if clock.playing else "Play")
        if not state["slider_drag"]:
            slider.blockSignals(True)
            slider.setValue(int(10000 * (t - clock.t0) / max(1e-9, clock.t1 - clock.t0)))
            slider.blockSignals(False)
        t_label.setText(f"{fmt(t)} / {clock.t1 - clock.t0:.1f} s")
        cursor.setPos(t - clock.t0)
        if not state.get("views_set"):
            apply_views(force=True)
            state["views_set"] = True
        # LiDAR target overlay at the displayed time
        tgt = None
        if track is not None and dctl["target"].isChecked():
            j = int(np.searchsorted(track["t"], t))
            for jj in (j - 1, j):
                if 0 <= jj < len(track["t"]) and abs(track["t"][jj] - t) < 0.15 and np.isfinite(track["r"][jj]):
                    tgt = (float(track["r"][jj]), float(track["az"][jj]))
                    break
        for key in ("lidar", "radar_bev"):
            if key in items:
                pano = key == "lidar" and srcs["lidar"].view["mode"] == "panorama"
                if tgt and not pano:
                    items[key]["tgt"].setData([tgt[0] * np.sin(np.radians(tgt[1]))], [tgt[0] * np.cos(np.radians(tgt[1]))])
                elif tgt:
                    items[key]["tgt"].setData([tgt[1]], [0.0])
                else:
                    items[key]["tgt"].setData([], [])
        if "radar_rd" in items:
            if tgt:
                items["radar_rd"]["line"].setPos(tgt[0])
                items["radar_rd"]["line"].show()
            else:
                items["radar_rd"]["line"].hide()
        lines, now = [], time.monotonic()
        for name, src in srcs.items():
            if src.error:
                lines.append(f"<span style='color:#ff6b6b'>{src.error}</span>")
        rows = []                                               # objects table rows

        # camera
        cam = srcs["camera"].get() if "camera" in srcs else None
        if "camera" in srcs and changed("camera", None if cam is None else (cam[0], cam[1])):
            if cam:
                items["camera"][1].setImage(cam[2], autoLevels=False, levels=(0, 255))
                panels["camera"].set("Camera (left)", f"frame {cam[1]}  @ {fmt(cam[0])}")
            else:
                items["camera"][1].setImage(np.zeros((2, 2, 3), np.uint8), autoLevels=False, levels=(0, 255))
                panels["camera"].set("Camera (left)", "no data at this time")

        # LiDAR
        lid = srcs["lidar"].get() if "lidar" in srcs else None
        ld = None if lid is None else lid["det"]
        if "lidar" in srcs:
            if "lidar3d" in items:
                srcs["lidar"].want_labels = items["lidar3d"].wants_labels()
            if changed("lidar", None if lid is None else (lid["t"], lid["version"], ld is not None
                                                          and ld.get("point_label") is not None)):
                it = items["lidar"]
                if lid:
                    it["img"].setImage(lid["rgb"], autoLevels=False, levels=(0, 255))
                    it["img"].setRect(QtCore.QRectF(*lid["ext"]))
                    mode = lid["mode"]
                    if state.get("lidar_mode") != mode:
                        state["lidar_mode"] = mode
                        pano = mode == "panorama"
                        it["plot"].setLabel("bottom", "azimuth  (deg, + = left, 0 = forward)" if pano else "y  (m, + = left)")
                        it["plot"].setLabel("left", "elevation  (deg)" if pano else "x  (m, forward)")
                        it["plot"].setAspectLocked(not pano)
                        if pano:
                            ext = lid["ext"]
                            it["fov"].setData([85, 85, np.nan, -85, -85], [ext[1], ext[1] + ext[3], np.nan,
                                                                           ext[1], ext[1] + ext[3]], connect="finite")
                            it["plot"].autoRange()
                        else:
                            apply_views(force=True)
                    what = {"panorama": "panorama", "top360": "360°", "top": "forward"}[mode]
                    ms = "" if ld is None else f"  |  {ld['ms_total']:.0f} ms GPU"
                    panels["lidar"].set(f"LiDAR {what}", f"{lid['n_shown']:,} pts  @ {fmt(lid['t'])}{ms}")
                    show_obj = ld is not None and dctl["lidar_obj"].isChecked() and mode != "panorama"
                    bl = box_lines(ld["objects"]) if show_obj else {c: ([], []) for c in CLASSES}
                    for c, (xs, ys) in bl.items():
                        it["boxes"][c].setData(xs, ys, connect="finite")
                    trs = ld["tracks"] if (ld is not None and dctl["tracks"].isChecked() and mode != "panorama") else []
                    pool = text_pool("lidar", len(trs), (255, 255, 255))
                    for i_, t_ in enumerate(pool):
                        if i_ < len(trs):
                            t_.setText(f"L{trs[i_][0]}")
                            t_.setPos(trs[i_][2], trs[i_][1])
                            t_.show()
                        else:
                            t_.hide()
                else:
                    it["img"].setImage(np.zeros((2, 2, 3), np.uint8), autoLevels=False, levels=(0, 255))
                    panels["lidar"].set("LiDAR", "no data at this time")
                    for c in CLASSES:
                        it["boxes"][c].setData([], [])
            if ld is not None:
                for tr in ld["tracks"]:
                    tid, x, y, vx, vy, cls, hits = tr
                    rows.append(("LiDAR", f"L{tid}", CLASSES[cls], np.hypot(x, y), np.degrees(np.arctan2(y, x)),
                                 np.hypot(vx, vy), None, LIDAR_C))

        # radar
        rad = srcs["radar"].get() if "radar" in srcs else None
        radar_pts = None
        if "radar" in srcs:
            rb, rr = items["radar_bev"], items["radar_rd"]
            if not rad and changed("radar", None):
                rb["img"].setImage(np.full((2, 2), np.nan, np.float32), autoLevels=False, levels=(0, 1))
                rb["img"].setRect(extent_r)
                rr["img"].setImage(np.zeros((2, 2), np.float32), autoLevels=False, levels=(0, 1))
                rr["img"].setRect(rd_rect)
                panels["radar_bev"].set("Radar BEV", "no data at this time (before the radar's first frame)")
                for k_ in ("dets", "ghosts", "pts", "trk"):
                    rb[k_].setData([], [])
            mode_pts = rctl["mode"].currentData()
            if rad and changed("radar", (rad["k"], mode_pts, rctl["thr"].value(), dctl["tracks"].isChecked())):
                bev = rad["bev"]
                fin = np.isfinite(bev)
                # colour from 3 dB above the noise floor, but at most 30 dB below the frame's peak: a near,
                # strong target's sidelobes (the beam is still wide) do not flood the map
                hi = float(np.nanpercentile(bev, 99.9)) if fin.any() else 1.0
                lo = max(float(np.nanmedian(bev)) + 3, hi - 30) if fin.any() else 0.0
                hi = max(hi, lo + 15)
                rb["img"].setImage(bev, autoLevels=False, levels=(lo, hi))
                rb["img"].setRect(extent_r)     # re-apply: a "no data" placeholder may have set the scale
                panels["radar_bev"].set("Radar BEV", f"frame {rad['k']}  @ {fmt(rad['t'])}  |  "
                                                     f"{srcs['radar'].ms_per_frame:.0f} ms GPU")
                rdi = rad["rd"].T                                    # rows = range, cols = Doppler
                lo2 = float(np.median(rdi)) + 3
                hi2 = max(lo2 + 25, float(np.percentile(rdi, 99.9)))
                rr["img"].setImage(rdi, autoLevels=False, levels=(lo2, hi2))
                rr["img"].setRect(rd_rect)
                from perception.radar import C as RC
                d = rad["det"]
                real = d[:, RC["ghost"]] == 0 if len(d) else np.zeros(0, bool)
                vmax = float(max(abs(srcs["radar"].proc.dop_axis[0]), 1.0))
                if mode_pts == "detections" and len(d):
                    cols = cm.coolwarm(np.clip(0.5 + 0.5 * d[real, RC["v_r"]] / 2.0, 0, 1))
                    rb["dets"].setData(d[real, RC["y"]], d[real, RC["x"]],
                                       brush=[pg.mkBrush(*(255 * c[:3]).astype(int), 235) for c in cols],
                                       pen=pg.mkPen((255, 255, 255, 120), width=0.5))
                    rb["ghosts"].setData(d[~real, RC["y"]], d[~real, RC["x"]])
                    rr["dets"].setData(d[:, RC["v_r"]], d[:, RC["range"]])
                    radar_pts = (np.c_[d[real, RC["x"]], d[real, RC["y"]], d[real, RC["z"]]], cols,
                                 f"{int(real.sum())} detections + {int((~real).sum())} multipath ghosts "
                                 "(colour: radial velocity, blue approaching / red receding)")
                else:
                    rb["dets"].setData([], [])
                    rb["ghosts"].setData([], [])
                    rr["dets"].setData([], [])
                rb["pts"].setData([], [])
                vr_ = lidar_ctl["range_m"].currentData() if "range_m" in lidar_ctl else args.lidar_range
                if mode_pts == "raw":
                    xyz, val = srcs["radar"].raw_points(rad["ra"], rctl["thr"].value(), vr_)
                    rb["pts"].setData(xyz[:, 1], xyz[:, 0], brush=pg.mkBrush(0, 230, 255, 200), size=3)
                    radar_pts = (xyz, cm.plasma(np.clip(val / 30.0, 0, 1)),
                                 f"raw: {len(xyz):,} cells > {rctl['thr'].value()} dB above floor")
                elif mode_pts == "filtered":
                    fp = srcs["radar"].filtered_points(rad["k"])
                    if fp is not None:
                        xyz, dop, pw = fp
                        cols = cm.coolwarm(np.clip(0.5 + 0.5 * dop / vmax, 0, 1))
                        rb["pts"].setData(xyz[:, 1], xyz[:, 0], size=8,
                                          brush=[pg.mkBrush(*(255 * c[:3]).astype(int), 240) for c in cols],
                                          pen=pg.mkPen((255, 255, 255), width=1))
                        radar_pts = (xyz, cols, f"team filtered: {len(xyz)} detections")
                trs = rad["tracks"] if dctl["tracks"].isChecked() else []
                rb["trk"].setData([t_[2] for t_ in trs], [t_[1] for t_ in trs])
                pool = text_pool("radar_bev", len(trs), (140, 190, 255))
                for i_, t_ in enumerate(pool):
                    if i_ < len(trs):
                        t_.setText(f"R{trs[i_][0]}")
                        t_.setPos(trs[i_][2], trs[i_][1])
                        t_.show()
                    else:
                        t_.hide()
                state["radar_pts"] = radar_pts
            radar_pts = state.get("radar_pts") if rad else None
            if rad:
                for tr in rad["tracks"]:
                    tid, x, y, vx, vy, vr, hits = tr
                    rows.append(("radar", f"R{tid}", "moving", np.hypot(x, y), np.degrees(np.arctan2(y, x)),
                                 np.hypot(vx, vy), vr, RADAR_C))

        # 3D
        if "lidar3d" in items and items["lidar3d"].widget.isVisible() and "lidar" in srcs:
            v3 = items["lidar3d"]
            pts = srcs["lidar"].get_points()
            if pts:
                v3.show_sweep(f"sweep @ {fmt(pts[0])}", pts[1], tgt, pts[2], pts[3])
            if ld is not None and dctl["lidar_obj"].isChecked():
                cols3 = [np.r_[np.array(CLASS_COLOURS[CLASSES[int(c)]]) / 255.0, 1.0] for c in ld["objects"][:, O["cls"]]]
                v3.set_object_classes(ld["objects"][:, O["cls"]].astype(int))
                v3.show_objects(ld["objects"], cols3, ld["tracks"] if dctl["tracks"].isChecked() else [])
            else:
                v3.show_objects(None, None, [])
            if radar_pts is not None:
                xyz = radar_pts[0]
                if rctl.get("flat") is not None and rctl["flat"].isChecked():
                    xyz = np.c_[xyz[:, :2], np.zeros(len(xyz), np.float32)]     # sensor plane
                v3.show_radar(xyz, radar_pts[1], size=9 if rctl["mode"].currentData() != "raw" else 4)
            else:
                v3.show_radar(None, None)
            if pts:
                panels["lidar3d"].set(None, f"{len(pts[1]):,} pts  @ {fmt(pts[0])}")

        # objects table (refresh ~5 times/s)
        if now - state.get("table_t", 0) > 0.2:
            state["table_t"] = now
            rows.sort(key=lambda r_: r_[3])
            table.setRowCount(len(rows))
            for i_, (sen, tid, cls, rng, az, spd, vr, col) in enumerate(rows):
                vals = [sen, tid, cls, f"{rng:5.1f}", f"{az:+5.1f}", f"{spd:4.1f}", "" if vr is None else f"{vr:+4.1f}"]
                for j_, v_ in enumerate(vals):
                    item = table.item(i_, j_)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem()
                        table.setItem(i_, j_, item)
                    item.setText(v_)
                    item.setForeground(QtGui.QColor(*col) if j_ == 0 else QtGui.QColor(212, 215, 221))
            panels["objects"].set(None, f"{len(rows)} confirmed")

        # info line
        if radar_pts is not None:
            lines.append("radar points: " + radar_pts[2])
        if tgt:
            lines.append(f"LiDAR target {tgt[0]:.2f} m at {tgt[1]:+.1f}° ({'left' if tgt[1] > 0 else 'right'})")
        if rad:
            pr = pairs.get(rad["k"])
            if pr:
                flag = f" <b style='color:#ff6b6b'>[{pr['flags']}]</b>" if pr["flags"] else ""
                lines.append(f"radar frame {rad['k']}: camera frame {pr['camera_frame_index']} "
                             f"({float(pr['camera_error_ms'] or 0):+.1f} ms), LiDAR sweep {pr['lidar_sweep']} "
                             f"({float(pr['lidar_error_ms'] or 0):+.1f} ms){flag}")
        if now - state["rate_t"] >= 1.0:
            for name, src in srcs.items():
                state["last_rates"][name] = (src.produced - state["rate_n"].get(name, src.produced)) / (now - state["rate_t"])
                state["rate_n"][name] = src.produced
            state["rate_t"] = now
        if state["last_rates"]:
            lines.append("updates/s: " + "  ".join(f"{n} {r:.1f}" for n, r in state["last_rates"].items())
                         + f"  |  speed {clock.speed:g}×  |  {'playing' if clock.playing else 'paused'}")
        info.setText("   •   ".join(lines))

    # re-fit the bird's-eye views when panels are resized
    class _Resize(QtCore.QObject):
        def eventFilter(self, obj, ev_):
            if ev_.type() == QtCore.QEvent.Resize:
                QtCore.QTimer.singleShot(0, apply_views)
            return False
    rf = _Resize(win)
    for key in ("lidar", "radar_bev"):
        if key in items:
            items[key]["widget"].installEventFilter(rf)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(int(1000 / 30))
    win._keep = (timer, save_shot, update, toggle_max, rf)        # keep references alive
    return win


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a 3-sensor session")
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--sensors", default="camera,lidar,radar", help="comma list: any of camera,lidar,radar")
    ap.add_argument("--start", type=float, default=0.0, help="start time (s from session start)")
    ap.add_argument("--speed", type=float, default=1.0, choices=SPEEDS)
    ap.add_argument("--bev-range", type=float, default=80.0,
                    help="radar BEV computed out to this forward extent (m); the view follows the LiDAR range")
    ap.add_argument("--bev-res", type=float, default=0.2, help="BEV cell size (m)")
    ap.add_argument("--camera-width", type=int, default=960)
    ap.add_argument("--lidar-view", default="top", choices=LIDAR_VIEWS,
                    help="top (forward, matches the radar), top360, or panorama")
    ap.add_argument("--lidar-range", type=float, default=20.0, choices=LIDAR_RANGES)
    ap.add_argument("--lidar-color", default="height", choices=LIDAR_COLORS)
    ap.add_argument("--lidar-near", type=float, default=0.5, choices=LIDAR_NEAR,
                    help="hide returns nearer than this (mount, vehicle body)")
    ap.add_argument("--play", action="store_true", help="start playing immediately")
    ap.add_argument("--maximise", choices=["camera", "lidar", "lidar3d", "radar_bev", "radar_rd", "objects"],
                    help="start with this panel maximised")
    ap.add_argument("--lidar-3d", action="store_true", help=argparse.SUPPRESS)   # 3D is a panel now: always shown
    ap.add_argument("--radar-points", default="detections", choices=["detections", "raw", "filtered", "off"],
                    help="detections = our CFAR detections + tracks (live, GPU); raw = every range-azimuth cell "
                         "above a threshold; filtered = the team pipeline's detections (radar_cloud.py once)")
    ap.add_argument("--no-rx-phase", action="store_true",
                    help="radar: without the in-situ RX phase correction (calibration/radar_rx_phase.json)")
    ap.add_argument("--no-detect", action="store_true", help="LiDAR: no object detection (display only)")
    # testing / automation
    ap.add_argument("--screenshot", type=Path, help="save a screenshot after --screenshot-after s, then exit")
    ap.add_argument("--screenshot-after", type=float, default=5.0)
    ap.add_argument("--size", default="1800x1000", help="window size WxH")
    a = ap.parse_args()

    s = a.session
    if not (s / "sync").is_dir():
        raise SystemExit(f"{s}/sync missing - run: python align_session.py --session {s}")
    sensors = [x.strip() for x in a.sensors.split(",") if x.strip()]
    meta = json.loads((s / "session.json").read_text())
    sensors = [x for x in sensors if x in meta.get("sensors", [])]
    if not sensors:
        raise SystemExit("none of the requested sensors are in this session")
    offsets = json.loads((s / "sync" / "alignment.json").read_text()).get("offsets", {}) \
        if (s / "sync" / "alignment.json").is_file() else {}

    bev_x = np.arange(a.bev_res / 2, a.bev_range, a.bev_res)
    bev_y = np.arange(-a.bev_range / 2 + a.bev_res / 2, a.bev_range / 2, a.bev_res)
    clock = ReplayClock(0.0, 1.0)
    srcs = {}
    if "camera" in sensors:
        srcs["camera"] = CameraSource(clock, s, a.camera_width)
    if "radar" in sensors:
        srcs["radar"] = RadarSource(clock, s, bev_x, bev_y, rx_phase=not a.no_rx_phase)
    if "lidar" in sensors:
        srcs["lidar"] = LidarSource(clock, s, offsets.get("lidar_ms", 0.0) / 1e3,
                                    view=dict(mode=a.lidar_view, range_m=a.lidar_range, color=a.lidar_color,
                                              near_m=a.lidar_near))
        srcs["lidar"].detect = not a.no_detect
    spans = [src.span() for src in srcs.values()]
    clock.t0, clock.t1 = min(x[0] for x in spans), max(x[1] for x in spans)
    clock.seek(clock.t0 + a.start)
    clock.set_speed(a.speed)

    pairs = {}
    if (s / "sync" / "pairs.csv").is_file():
        pairs = {int(r["radar_idx"]): r for r in load_csv(s / "sync" / "pairs.csv")}
    rd_axes = ((0.0, 0.0), (0.0, 0.0), (1, 1))
    if "radar" in srcs:
        pr = srcs["radar"].proc
        rd_axes = ((0.0, float(pr.range_axis[-1])), (float(pr.dop_axis[0]), float(pr.dop_axis[-1])),
                   (len(pr.range_axis), len(pr.dop_axis)))

    from PyQt5 import QtCore, QtWidgets
    app = QtWidgets.QApplication(sys.argv[:1])
    track = None
    if (s / "sync" / "timing_lidar_track.npz").is_file():
        d = np.load(s / "sync" / "timing_lidar_track.npz")
        track = {"t": d["t"], "r": d["r"], "az": d["az"]}
    else:
        print("no sync/timing_lidar_track.npz: run make_report.py once to enable the LiDAR target overlay")
    win = build_window(s, sensors, clock, srcs, pairs, bev_x, bev_y, rd_axes, a, track)
    w_, h_ = (int(v) for v in a.size.lower().split("x"))
    win.resize(w_, h_)
    for src in srcs.values():
        src.start()
    if a.play or a.screenshot:
        clock.play()
    if a.maximise:
        QtCore.QTimer.singleShot(200, lambda: win._keep[3](a.maximise))
    if a.screenshot:
        def shoot():
            win._keep[2]()
            win._keep[1](str(a.screenshot))
            print(f"screenshot -> {a.screenshot}")
            app.quit()
        QtCore.QTimer.singleShot(int(a.screenshot_after * 1000), shoot)
    win.show()
    code = app.exec_()
    for src in srcs.values():
        src.stop()
    time.sleep(0.3)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
