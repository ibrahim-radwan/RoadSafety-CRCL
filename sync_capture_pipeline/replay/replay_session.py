#!/usr/bin/env python3
"""
replay_session.py — replay a 3-sensor session in real time, on one clock.

    Camera      the ZED left image (SVO2 decoded on the GPU by the ZED SDK)
    LiDAR BEV   bird's-eye point density per sweep, radar FOV outlined
    Radar BEV   range-azimuth power on the SAME metric grid as the LiDAR BEV
    Radar R-D   range-Doppler map
    Each panel shows the time of the data it displays; for the radar frame
    on screen, the pairing errors and flags from sync/pairs.csv are shown.

All sensors follow one clock (CLOCK_MONOTONIC session time from
align_session.py, offsets included). Play/pause, speed, seek and stepping
by radar frame apply to every sensor at once. Choose sensors with --sensors
to review one alone.

Requires: python align_session.py --session <s>   (writes sync/)

KEYS    space play/pause   left/right: previous/next radar frame
        [ ] slower/faster  s: save a screenshot   m: radar moving/all

USAGE (conda env)
    python replay/replay_session.py --session sessions/run05
    python replay/replay_session.py --session sessions/run05 --sensors lidar
    python replay/replay_session.py --session sessions/run05 --start 12.5 --speed 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib import cm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sources import (LIDAR_COLORS, LIDAR_VIEWS, CameraSource, LidarSource, RadarSource,  # noqa: E402
                     ReplayClock, load_csv)

LIDAR_RANGES = [5.0, 10.0, 20.0, 40.0, 80.0]
LIDAR_NEAR = [0.0, 0.3, 0.5, 1.0, 2.0]

SPEEDS = [0.25, 0.5, 1.0, 2.0]


RADAR_C, LIDAR_C, CAMERA_C = (42, 120, 214), (235, 104, 52), (27, 175, 122)   # report palette


def build_window(session: Path, sensors: list, clock: ReplayClock, srcs: dict, pairs: dict,
                 bev_x, bev_y, rd_axes, args, track=None):
    import pyqtgraph as pg
    from PyQt5 import QtCore, QtGui, QtWidgets

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
    win = QtWidgets.QWidget()
    win.setWindowTitle(f"Replay - {session.name}")
    root = QtWidgets.QVBoxLayout(win)
    grid = pg.GraphicsLayoutWidget()
    root.addWidget(grid, stretch=1)

    panels = {}
    lut_hot = pg.colormap.get("inferno").getLookupTable(0.0, 1.0, 256)

    def image_panel(row, col, title, lut=None, xlabel=None, ylabel=None, rowspan=1, colspan=1):
        p = grid.addPlot(row=row, col=col, rowspan=rowspan, colspan=colspan, title=title)
        img = pg.ImageItem()
        if lut is not None:
            img.setLookupTable(lut)
        p.addItem(img)
        if xlabel:
            p.setLabel("bottom", xlabel)
        if ylabel:
            p.setLabel("left", ylabel)
        return p, img

    # BEV geometry: x forward (up on screen), y left (left on screen).
    extent = QtCore.QRectF(-bev_y[-1], bev_x[0], bev_y[-1] - bev_y[0], bev_x[-1] - bev_x[0])
    fov_r = bev_x[-1]
    fov_line = np.array([[-np.sin(np.radians(85)) * fov_r, np.cos(np.radians(85)) * fov_r], [0, 0],
                         [np.sin(np.radians(85)) * fov_r, np.cos(np.radians(85)) * fov_r]])
    # Top row: one view per sensor; bottom row: radar range-Doppler across.
    tops = [x for x in ("camera", "lidar", "radar") if x in sensors]
    lidar_fov = None
    for col, name in enumerate(tops):
        if name == "camera":
            p, img = image_panel(0, col, "Camera (left)")
            p.invertY(True)                     # image rows run downwards
            p.setAspectLocked(True)
            p.hideAxis("left"); p.hideAxis("bottom")
            panels["camera"] = (p, img)
        else:
            title = "LiDAR" if name == "lidar" else "Radar BEV (range-azimuth, dB)"
            p, img = image_panel(0, col, title, None if name == "lidar" else lut_hot,
                                 "y  (m, + = left)", "x  (m, forward)")
            # pyqtgraph 0.13 needs an image before setRect: start with a blank one.
            img.setImage(np.zeros((len(bev_x), len(bev_y)), np.float32), autoLevels=False, levels=(0, 1))
            img.setRect(extent)                 # BEV rows = x (forward), columns = y
            p.setAspectLocked(True)
            p.invertX(True)                     # +y (left) drawn on the left
            fov_item = p.plot(fov_line[:, 0], fov_line[:, 1],
                              pen=pg.mkPen((255, 80, 80), width=1, style=QtCore.Qt.DashLine))
            panels["lidar" if name == "lidar" else "radar_bev"] = (p, img)
            if name == "lidar":
                lidar_fov = fov_item
    if "radar" in sensors:
        p2, img2 = image_panel(1, 0, "Radar range-Doppler (dB)", lut_hot, "velocity (m/s)", "range (m)",
                               colspan=len(tops))
        (r0, r1), (v0, v1), shape = rd_axes
        img2.setImage(np.zeros(shape, np.float32), autoLevels=False, levels=(0, 1))
        img2.setRect(QtCore.QRectF(v0, r0, v1 - v0, r1 - r0))
        panels["radar_rd"] = (p2, img2)
        grid.ci.layout.setRowStretchFactor(0, 3)
        grid.ci.layout.setRowStretchFactor(1, 2)

    rd_rect = QtCore.QRectF(rd_axes[1][0], rd_axes[0][0], rd_axes[1][1] - rd_axes[1][0],
                            rd_axes[0][1] - rd_axes[0][0])

    # LiDAR target overlay (sync/timing_lidar_track.npz, written by make_report.py): a circle on
    # both bird's-eye views and a line at its range on the range-Doppler map.
    overlays = {}
    for name in ("lidar", "radar_bev"):
        if name in panels:
            sc = pg.ScatterPlotItem(size=18, symbol="o", pen=pg.mkPen((255, 255, 255), width=2), brush=None)
            panels[name][0].addItem(sc)
            overlays[name] = sc
    if "radar_bev" in panels:
        rpts = pg.ScatterPlotItem(size=5, pen=None)
        rpts.setZValue(10)                            # above the BEV image
        panels["radar_bev"][0].addItem(rpts)
        overlays["radar_pts"] = rpts
    if "radar_rd" in panels:
        rd_line = pg.InfiniteLine(angle=0, pen=pg.mkPen((255, 255, 255), width=1, style=QtCore.Qt.DashLine))
        panels["radar_rd"][0].addItem(rd_line)
        rd_line.hide()
        overlays["radar_rd"] = rd_line

    def sync_views(range_m, mode="top"):
        """Radar views follow the LiDAR range choice (radar BEV is computed out to --bev-range)."""
        R = float(range_m)
        xr = (-R, R) if mode == "top360" else (-R / 2, R / 2)
        for name in ("radar_bev", "lidar"):         # identical window on both bird's-eye views
            if name in panels:
                p_ = panels[name][0]
                p_.disableAutoRange()
                p_.setRange(xRange=xr, yRange=(0, R), padding=0)
        if "radar_rd" in panels:
            panels["radar_rd"][0].disableAutoRange()
            panels["radar_rd"][0].setRange(xRange=(rd_axes[1][0], rd_axes[1][1]), yRange=(0, R), padding=0)

    sync_views(args.lidar_range, args.lidar_view)

    # ── event strip: where the gaps are; click to jump ──────────────────
    ev = pg.PlotWidget()
    ev.setFixedHeight(64)
    ev.hideAxis("left")
    ev.setMouseEnabled(x=False, y=False)
    ev.setMenuEnabled(False)
    ev.setYRange(-0.2, 2.2, padding=0)
    span = clock.t1 - clock.t0
    margin = 0.09 * span                          # labels live left of t = 0
    ev.setXRange(-margin, span, padding=0)
    marks = {"camera_gap": ([], CAMERA_C, 0), "lidar_gap": ([], LIDAR_C, 1)}
    for pr in pairs.values():
        for f in (pr.get("flags") or "").split(";"):
            if f in marks:
                marks[f][0].append(float(pr["t_radar"]) - clock.t0)
    for f, (xs, col, row_) in marks.items():
        if xs:
            xx = np.repeat(xs, 3)
            yy = np.tile([row_ + 0.05, row_ + 0.95, np.nan], len(xs))
            ev.plot(xx, yy, pen=pg.mkPen(col, width=1), connect="finite")
    for text, col, y in (("camera gaps", CAMERA_C, 0.5), ("LiDAR gaps", LIDAR_C, 1.5)):
        lab = pg.TextItem(text, color=col, anchor=(0, 0.5))
        lab.setPos(-margin, y)
        ev.addItem(lab)
    if radar_nominal := (srcs.get("radar") is not None and getattr(srcs["radar"], "timing_source", "").startswith("NOMINAL")):
        lab = pg.TextItem("radar times NOMINAL (DCA backlog)", color=RADAR_C, anchor=(1, 0.5))
        lab.setPos(clock.t1 - clock.t0, 2.0)
        ev.addItem(lab)
    cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen((220, 220, 220), width=1))
    ev.addItem(cursor)

    def ev_clicked(event):
        pos = ev.getPlotItem().vb.mapSceneToView(event.scenePos())
        clock.seek(clock.t0 + float(np.clip(pos.x(), 0, clock.t1 - clock.t0)))
    ev.scene().sigMouseClicked.connect(ev_clicked)
    root.addWidget(ev)

    # ── controls ────────────────────────────────────────────────────────
    bar = QtWidgets.QHBoxLayout()
    root.addLayout(bar)
    b_play = QtWidgets.QPushButton("Play")
    b_prev = QtWidgets.QPushButton("< radar frame")
    b_next = QtWidgets.QPushButton("radar frame >")
    speed = QtWidgets.QComboBox()
    speed.addItems([f"{s:g}x" for s in SPEEDS])
    speed.setCurrentIndex(SPEEDS.index(args.speed) if args.speed in SPEEDS else 2)
    moving = QtWidgets.QCheckBox("radar: moving only")
    moving.setChecked(True)
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setRange(0, 10000)
    t_label = QtWidgets.QLabel()
    t_label.setMinimumWidth(170)
    for w in (b_play, b_prev, b_next, speed, moving):
        bar.addWidget(w)
    bar.addWidget(slider, stretch=1)
    bar.addWidget(t_label)
    lidar_ctl = {}
    if "lidar" in srcs:
        row = QtWidgets.QHBoxLayout()
        root.addLayout(row)
        row.addWidget(QtWidgets.QLabel("LiDAR view:"))
        v = srcs["lidar"].view
        for key, label, options, current in (
                ("mode", "", [("Top - forward", "top"), ("Top - 360\u00b0", "top360"), ("Panorama", "panorama")], v["mode"]),
                ("range_m", "range", [(f"{r:g} m", float(r)) for r in LIDAR_RANGES], v["range_m"]),
                ("color", "colour", [("height", "height"), ("intensity", "intensity"), ("density", "density")], v["color"]),
                ("near_m", "hide nearer than", [(f"{r:g} m", float(r)) for r in LIDAR_NEAR], v["near_m"])):
            if label:
                row.addWidget(QtWidgets.QLabel(label))
            box = QtWidgets.QComboBox()
            for text, value in options:
                box.addItem(text, value)
            box.setCurrentIndex(max(0, [o[1] for o in options].index(current) if current in [o[1] for o in options] else 0))
            box.currentIndexChanged.connect(lambda _i, k=key, b=box: srcs["lidar"].set_view(**{k: b.currentData()}))
            if key in ("range_m", "mode"):
                box.currentIndexChanged.connect(lambda _i: sync_views(lidar_ctl["range_m"].currentData(),
                                                                       lidar_ctl["mode"].currentData()))
            row.addWidget(box)
            lidar_ctl[key] = box
        b3d = QtWidgets.QPushButton("3D LiDAR")
        b3d.setToolTip("open a 3D point-cloud window that follows the replay")
        row.addWidget(b3d)
        row.addStretch(1)
    rctl = {}
    if "radar" in srcs:
        rrow = QtWidgets.QHBoxLayout()
        root.addLayout(rrow)
        rrow.addWidget(QtWidgets.QLabel("Radar points:"))
        rmode = QtWidgets.QComboBox()
        rmode.addItem("off", "off")
        rmode.addItem("raw (all range-azimuth cells, z = 0)", "raw")
        rmode.addItem("filtered (team pipeline: CFAR + filters)", "filtered")
        rrow.addWidget(rmode)
        rrow.addWidget(QtWidgets.QLabel("raw threshold"))
        rthr = QtWidgets.QSpinBox()
        rthr.setRange(3, 40)
        rthr.setValue(15)
        rthr.setSuffix(" dB above floor")
        rrow.addWidget(rthr)
        rflat = QtWidgets.QCheckBox("filtered: z = 0 (ignore elevation)")
        rflat.setToolTip("the team pipeline's elevation estimate is not yet verified")
        rrow.addWidget(rflat)
        rnote = QtWidgets.QLabel()
        rrow.addWidget(rnote)
        rrow.addStretch(1)
        rctl.update(mode=rmode, thr=rthr, note=rnote, flat=rflat)
        if srcs["radar"].cloud is None:
            rmode.model().item(2).setEnabled(False)
            rnote.setText("  (filtered: run  radar_cloud.py --session <s>  first)")
        want = getattr(args, "radar_points", "off")
        if want == "filtered" and srcs["radar"].cloud is None:
            want = "off"
        rmode.setCurrentIndex(["off", "raw", "filtered"].index(want))
    w3d = {"win": None}

    def open_3d():
        if "lidar" not in srcs:
            return
        if w3d["win"] is None:
            try:
                from lidar3d import Lidar3DWindow
            except ImportError as exc:                              # PyOpenGL missing
                print(f"3D view unavailable: {exc}")
                return
            rng = lidar_ctl["range_m"].currentData() if "range_m" in lidar_ctl else args.lidar_range
            w3d["win"] = Lidar3DWindow(f"3D LiDAR - {session.name}", rng)
        w3d["win"].win.show()
        w3d["win"].win.raise_()
    if "lidar" in srcs:
        b3d.clicked.connect(open_3d)
        lidar_ctl["range_m"].currentIndexChanged.connect(
            lambda _i: w3d["win"] and w3d["win"].set_range(lidar_ctl["range_m"].currentData()))
    info = QtWidgets.QLabel()
    info.setTextFormat(QtCore.Qt.RichText)
    root.addWidget(info)

    radar_t = srcs["radar"].t if "radar" in srcs else None
    state = {"slider_drag": False, "last_rates": {}, "rate_t": time.monotonic(), "rate_n": {}}
    shown = {}                                   # last item drawn per panel

    def toggle_play():
        if clock.playing:
            clock.pause()
        else:
            clock.play()

    def step(direction):
        clock.pause()
        if radar_t is None:
            clock.seek(clock.now() + direction / 30.0)
            return
        k = int(np.searchsorted(radar_t, clock.now(), side="right")) - 1 + direction
        clock.seek(float(radar_t[int(np.clip(k, 0, len(radar_t) - 1))]))

    def set_speed(i):
        clock.set_speed(SPEEDS[i])

    def set_moving(state_):
        if "radar" in srcs:
            srcs["radar"].moving_only = bool(state_)

    def slider_moved(v):
        clock.seek(clock.t0 + (clock.t1 - clock.t0) * v / 10000.0)

    def save_shot(path=None):
        path = path or str(session / "sync" / f"replay_{time.strftime('%H%M%S')}.png")
        win.grab().save(path)
        if w3d["win"] is not None and w3d["win"].win.isVisible():
            w3d["win"].grab(path.rsplit(".", 1)[0] + "_3d.png")
        return path

    b_play.clicked.connect(toggle_play)
    b_prev.clicked.connect(lambda: step(-1))
    b_next.clicked.connect(lambda: step(1))
    speed.currentIndexChanged.connect(set_speed)
    moving.stateChanged.connect(set_moving)
    slider.sliderPressed.connect(lambda: state.update(slider_drag=True))
    slider.sliderReleased.connect(lambda: (state.update(slider_drag=False), slider_moved(slider.value())))

    keys = {QtCore.Qt.Key_Space: toggle_play, QtCore.Qt.Key_Left: lambda: step(-1),
            QtCore.Qt.Key_Right: lambda: step(1), QtCore.Qt.Key_S: lambda: print("saved", save_shot()),
            QtCore.Qt.Key_BracketLeft: lambda: speed.setCurrentIndex(max(0, speed.currentIndex() - 1)),
            QtCore.Qt.Key_BracketRight: lambda: speed.setCurrentIndex(min(len(SPEEDS) - 1, speed.currentIndex() + 1)),
            QtCore.Qt.Key_M: lambda: moving.setChecked(not moving.isChecked())}
    for k, fn in keys.items():
        QtWidgets.QShortcut(QtGui.QKeySequence(k), win, activated=fn)

    def fmt(t):
        return f"{t - clock.t0:7.2f} s"

    def update():
        t = clock.now()
        b_play.setText("Pause" if clock.playing else "Play")
        if not state["slider_drag"]:
            slider.blockSignals(True)
            slider.setValue(int(10000 * (t - clock.t0) / max(1e-9, clock.t1 - clock.t0)))
            slider.blockSignals(False)
        t_label.setText(f"{fmt(t)} / {clock.t1 - clock.t0:.1f} s")
        cursor.setPos(t - clock.t0)
        # LiDAR target overlay at the displayed time
        tgt = None
        if track is not None:
            j = int(np.searchsorted(track["t"], t))
            for jj in (j - 1, j):
                if 0 <= jj < len(track["t"]) and abs(track["t"][jj] - t) < 0.15 and np.isfinite(track["r"][jj]):
                    tgt = (float(track["r"][jj]), float(track["az"][jj]))
                    break
        for name in ("lidar", "radar_bev"):
            if name in overlays:
                mode_now = srcs["lidar"].view["mode"] if "lidar" in srcs else "top"
                if tgt and not (name == "lidar" and mode_now == "panorama"):
                    r_, a_ = tgt
                    overlays[name].setData([r_ * np.sin(np.radians(a_))], [r_ * np.cos(np.radians(a_))])
                elif tgt and name == "lidar":
                    overlays[name].setData([tgt[1]], [0.0])
                else:
                    overlays[name].setData([], [])
        if "radar_rd" in overlays:
            if tgt:
                overlays["radar_rd"].setPos(tgt[0])
                overlays["radar_rd"].show()
            else:
                overlays["radar_rd"].hide()
        lines, now = [], time.monotonic()
        for name, src in srcs.items():
            if src.error:
                lines.append(f"<span style='color:#d33'>{src.error}</span>")
        def changed(name, item):
            """Redraw a panel only when its data changed (identity: time, index)."""
            key = None if item is None else ((item[0], item[2], item[5]) if name == "lidar" else (item[0], item[1]))
            if shown.get(name, "init") == key:
                return False
            shown[name] = key
            return True

        def blank(panel, title):
            p, img = panels[panel]
            img.setImage(np.zeros((2, 2), np.float32), autoLevels=False, levels=(0, 1))
            if panel in ("lidar", "radar_bev"):
                img.setRect(extent)
            p.setTitle(title)

        cam = srcs.get("camera").get() if "camera" in srcs else None
        if "camera" in srcs and changed("camera", cam):
            if cam:
                panels["camera"][1].setImage(cam[2], autoLevels=False, levels=(0, 255))
                panels["camera"][0].setTitle(f"Camera frame {cam[1]}  @ {fmt(cam[0])}")
            else:
                blank("camera", "Camera - no data at this time")
        lid = srcs.get("lidar").get() if "lidar" in srcs else None
        if "lidar" in srcs and changed("lidar", lid):
            if lid:
                t_l, rgb, n_shown, ext, mode, _ = lid
                p_l, img_l = panels["lidar"]
                img_l.setImage(rgb, autoLevels=False, levels=(0, 255))
                img_l.setRect(QtCore.QRectF(*ext))
                if state.get("lidar_mode") != (mode, ext):          # view changed: axes + FOV line
                    state["lidar_mode"] = (mode, ext)
                    pano = mode == "panorama"
                    p_l.setLabel("bottom", "azimuth  (deg, + = left, 0 = forward)" if pano else "y  (m, + = left)")
                    p_l.setLabel("left", "elevation  (deg)" if pano else "x  (m, forward)")
                    p_l.setAspectLocked(not pano)
                    if lidar_fov is not None:
                        if pano:                                     # radar FOV: azimuth +/-85 deg
                            lidar_fov.setData([85, 85, np.nan, -85, -85], [ext[1], ext[1] + ext[3], np.nan,
                                                                          ext[1], ext[1] + ext[3]], connect="finite")
                        else:
                            rr = ext[3] if mode == "top" else ext[3] / 2
                            c, sn = np.cos(np.radians(85)) * rr, np.sin(np.radians(85)) * rr
                            lidar_fov.setData([sn, 0, -sn], [c, 0, c])
                    if pano:
                        p_l.autoRange()
                    else:
                        sync_views(ext[3] if mode == "top" else ext[3] / 2, mode)
                what = {"panorama": "panorama", "top360": "top view 360\u00b0", "top": "top view forward"}[mode]
                p_l.setTitle(f"LiDAR {what}  ({n_shown:,} pts)  @ {fmt(t_l)}")
            else:
                blank("lidar", "LiDAR - no data at this time")
        rad = srcs.get("radar").get() if "radar" in srcs else None
        if "radar" in srcs and not rad and changed("radar", rad):
            blank("radar_bev", "Radar - no data at this time (before the radar's first frame)")
            blank("radar_rd", "Radar range-Doppler")
            panels["radar_rd"][1].setRect(rd_rect)
        if rad and changed("radar", rad):
            t_r, k, rd, bev = rad[:4]
            # colour scale relative to this frame's noise floor (median): an empty scene stays dark
            fin = np.isfinite(bev)
            lo = float(np.nanmedian(bev)) + 3 if fin.any() else 0.0
            hi = max(lo + 20, float(np.nanpercentile(bev, 99.8))) if fin.any() else 1.0
            panels["radar_bev"][1].setImage(bev, autoLevels=False, levels=(lo, hi))
            panels["radar_bev"][1].setRect(extent)   # re-apply: a 2x2 "no data" placeholder may have set the scale
            panels["radar_bev"][0].setTitle(f"Radar BEV  frame {k}  @ {fmt(t_r)}  ({srcs['radar'].ms_per_frame:.1f} ms GPU)")
            rdi = rd.T                                           # rows = range, cols = Doppler
            lo2 = float(np.median(rdi)) + 3
            hi2 = max(lo2 + 25, float(np.percentile(rdi, 99.9)))
            panels["radar_rd"][1].setImage(rdi, autoLevels=False, levels=(lo2, hi2))
            panels["radar_rd"][1].setRect(rd_rect)
        # radar point cloud (raw: live from this frame; filtered: team pipeline, pre-computed)
        radar_pts = None
        if rad and rctl and rctl["mode"].currentData() != "off":
            vr = lidar_ctl["range_m"].currentData() if "range_m" in lidar_ctl else args.lidar_range
            if rctl["mode"].currentData() == "raw" and len(rad) > 4:
                xyz, val = srcs["radar"].raw_points(rad[4], rctl["thr"].value(), vr)
                rgba = cm.plasma(np.clip(val / 30.0, 0, 1))
                radar_pts = (xyz, rgba, f"raw: {len(xyz):,} cells > {rctl['thr'].value()} dB above floor")
            elif rctl["mode"].currentData() == "filtered":
                fp = srcs["radar"].filtered_points(rad[1])
                if fp is not None:
                    xyz, dop, pw = fp
                    if rctl["flat"].isChecked():
                        xyz = np.c_[xyz[:, :2], np.zeros(len(xyz), np.float32)]
                    vmax = float(max(abs(srcs["radar"].proc.dop_axis[0]), 1.0))
                    rgba = cm.coolwarm(np.clip(0.5 + 0.5 * dop / vmax, 0, 1))
                    radar_pts = (xyz, rgba, f"filtered: {len(xyz)} detections (colour: radial velocity, "
                                            f"blue − / red +, ±{vmax:.1f} m/s)")
        if "radar_pts" in overlays:
            if radar_pts is not None and len(radar_pts[0]):
                xyz, rgba, _ = radar_pts
                if rctl["mode"].currentData() == "raw":     # raw cells sit on the BEV's bright cells:
                    overlays["radar_pts"].setData(xyz[:, 1], xyz[:, 0], brush=pg.mkBrush(0, 230, 255, 200),
                                                  size=3)    # one contrasting colour on this view
                else:
                    overlays["radar_pts"].setData(xyz[:, 1], xyz[:, 0], size=8,
                                                  brush=[pg.mkBrush(*(255 * c[:3]).astype(int), 240) for c in rgba],
                                                  pen=pg.mkPen((255, 255, 255), width=1))
            else:
                overlays["radar_pts"].setData([], [])
        if radar_pts is not None:
            lines.append("Radar points " + radar_pts[2])
        if tgt:
            lines.append(f"LiDAR target: {tgt[0]:.2f} m at {tgt[1]:+.1f}\u00b0 ({'left' if tgt[1] > 0 else 'right'})"
                         " - circle on both bird's-eye views, dashed line on range-Doppler")
        if rad:
            k = rad[1]
            pr = pairs.get(k)
            if pr:
                flag = f" <b style='color:#d33'>[{pr['flags']}]</b>" if pr["flags"] else ""
                lines.append(f"radar frame {k}: camera frame {pr['camera_frame_index']} "
                             f"({float(pr['camera_error_ms'] or 0):+.1f} ms), LiDAR sweep {pr['lidar_sweep']} "
                             f"({float(pr['lidar_error_ms'] or 0):+.1f} ms){flag}")
        # achieved product rates (real-time check)
        if now - state["rate_t"] >= 1.0:
            for name, src in srcs.items():
                state["last_rates"][name] = (src.produced - state["rate_n"].get(name, src.produced)) / (now - state["rate_t"])
                state["rate_n"][name] = src.produced
            state["rate_t"] = now
        if state["last_rates"]:
            lines.append("updates/s: " + "  ".join(f"{n} {r:.1f}" for n, r in state["last_rates"].items())
                         + f"  |  speed {clock.speed:g}x  |  {'playing' if clock.playing else 'paused'}")
        if w3d["win"] is not None and w3d["win"].win.isVisible():
            pts = srcs["lidar"].get_points()
            w3d["win"].show_sweep(f"sweep @ {fmt(pts[0])}" if pts else "no sweep yet",
                                  pts[1] if pts else None, tgt)
            w3d["win"].show_radar(None if radar_pts is None else radar_pts[0],
                                  None if radar_pts is None else radar_pts[1],
                                  size=9 if rctl and rctl["mode"].currentData() == "filtered" else 4)
        info.setText("<br>".join(lines))

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(int(1000 / 30))
    win._keep = (timer, save_shot, update, open_3d)        # keep references alive
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
    ap.add_argument("--lidar-range", type=float, default=40.0, choices=LIDAR_RANGES)
    ap.add_argument("--lidar-color", default="height", choices=LIDAR_COLORS)
    ap.add_argument("--lidar-near", type=float, default=0.5, choices=LIDAR_NEAR,
                    help="hide returns nearer than this (mount, vehicle body)")
    ap.add_argument("--play", action="store_true", help="start playing immediately")
    ap.add_argument("--lidar-3d", action="store_true", help="open the 3D LiDAR window at start")
    ap.add_argument("--radar-points", default="off", choices=["off", "raw", "filtered"],
                    help="radar point cloud: raw = every range-azimuth cell above a threshold (live, z = 0); "
                         "filtered = the team pipeline's detections (needs radar_cloud.py once per session)")
    # testing / automation
    ap.add_argument("--screenshot", type=Path, help="save a screenshot after --screenshot-after s, then exit")
    ap.add_argument("--screenshot-after", type=float, default=5.0)
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
        srcs["radar"] = RadarSource(clock, s, bev_x, bev_y)
    if "lidar" in sensors:
        srcs["lidar"] = LidarSource(clock, s, offsets.get("lidar_ms", 0.0) / 1e3,
                                    view=dict(mode=a.lidar_view, range_m=a.lidar_range, color=a.lidar_color,
                                              near_m=a.lidar_near))
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
    win.resize(1600, 900)
    for src in srcs.values():
        src.start()
    if a.play or a.screenshot:
        clock.play()
    if a.lidar_3d:
        win._keep[3]()
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
