"""3D LiDAR view for the replay (pyqtgraph.opengl, needs PyOpenGL).

A separate window that follows the replay clock: it shows the sweep the 2D
LiDAR panel is showing, as a point cloud you can orbit (left drag), pan
(middle drag / ctrl+drag) and zoom (wheel).

Coordinates are the LiDAR's: x forward, y left, z up. The default camera sits
behind and above the sensors looking forward, so vehicle-left is on the left
as in every other replay view. Drawn as well: a 1 m ground grid, the radar's
±60° azimuth field of view, and the LiDAR target (white) when a target track
exists (sync/timing_lidar_track.npz).
"""
from __future__ import annotations

import numpy as np
from matplotlib import cm


class Lidar3DWindow:
    def __init__(self, title: str, range_m: float = 40.0, ground_z: float = -1.0):
        import pyqtgraph.opengl as gl
        from PyQt5 import QtWidgets

        self.gl = gl
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle(title)
        lay = QtWidgets.QVBoxLayout(self.win)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((12, 12, 14))
        lay.addWidget(self.view, stretch=1)
        bar = QtWidgets.QHBoxLayout()
        lay.addLayout(bar)
        self.label = QtWidgets.QLabel()
        bar.addWidget(self.label, stretch=1)
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(lambda: self.reset_camera())
        bar.addWidget(reset)
        self.colour = QtWidgets.QComboBox()
        self.colour.addItems(["colour: height", "colour: intensity", "colour: range"])
        self.colour.currentIndexChanged.connect(lambda _i: self._redraw())
        bar.addWidget(self.colour)

        self.range_m = float(range_m)
        self.ground_z = ground_z
        self.grid = grid = gl.GLGridItem()
        grid.setSize(200, 200)
        grid.setSpacing(1, 1)
        grid.translate(0, 0, ground_z)
        grid.setColor((60, 60, 66, 255))
        self.view.addItem(grid)
        self.fov = gl.GLLinePlotItem(width=1.5, color=(0.16, 0.47, 0.84, 1.0), antialias=True, mode="lines")
        self.view.addItem(self.fov)
        self.cloud = gl.GLScatterPlotItem(size=2.0, pxMode=True)
        self.view.addItem(self.cloud)
        self.target = gl.GLScatterPlotItem(size=16.0, pxMode=True, color=(1, 1, 1, 0.9))
        self.view.addItem(self.target)
        self.radar = gl.GLScatterPlotItem(size=6.0, pxMode=True)     # radar points (raw or filtered)
        self.view.addItem(self.radar)
        self._pts = None
        self._key = None
        self._ground_known = False
        self.set_range(range_m)
        self.reset_camera()
        self.win.resize(1100, 750)

    # ── view ────────────────────────────────────────────────────────────
    def set_range(self, range_m: float):
        self.range_m = float(range_m)
        R, z = self.range_m, self.ground_z
        s = np.sin(np.radians(60)) * R
        c = np.cos(np.radians(60)) * R
        self.fov.setData(pos=np.array([[0, 0, z], [c, s, z], [0, 0, z], [c, -s, z]], dtype=np.float32))

    def reset_camera(self):
        from pyqtgraph import Vector
        R = self.range_m
        self.view.opts["center"] = Vector(R * 0.35, 0, 0)
        # azimuth 180: camera behind the sensor (at -x) looking along +x -> +y (left) on screen-left
        self.view.setCameraPosition(distance=R * 1.1, elevation=22, azimuth=180)

    # ── data ────────────────────────────────────────────────────────────
    def show_sweep(self, t_label: str, pts: np.ndarray | None, target=None):
        """pts: N x 5 (x, y, z, range, intensity). target: (range_m, azimuth_deg) or None."""
        key = (t_label, None if pts is None else len(pts))
        if not self._ground_known and pts is not None and len(pts) > 1000:
            # ground level from the data (mount height is not recorded): low points 3-15 m ahead
            ahead = (pts[:, 0] > 3) & (pts[:, 0] < 15) & (np.abs(pts[:, 1]) < 4)
            if ahead.sum() > 200:
                self.ground_z = float(np.percentile(pts[ahead, 2], 3))
                self.grid.resetTransform()
                self.grid.translate(0, 0, self.ground_z)
                self.set_range(self.range_m)
                self._ground_known = True
        if key != self._key:
            self._key = key
            self._pts = pts
            self._redraw()
            n = 0 if pts is None else len(pts)
            self.label.setText(f"  {t_label}   {n:,} points (valid returns)   "
                               "left drag: orbit   wheel: zoom   middle/ctrl drag: pan")
        if target is None:
            self.target.setData(pos=np.zeros((0, 3), np.float32))
        else:
            r, az = target
            self.target.setData(pos=np.array([[r * np.cos(np.radians(az)), r * np.sin(np.radians(az)),
                                               self.ground_z + 1.0]],
                                             dtype=np.float32))

    def show_radar(self, xyz, rgba, size=6.0):
        """Radar points in the LiDAR frame (radar assumed co-located with the LiDAR)."""
        if xyz is None or not len(xyz):
            self.radar.setData(pos=np.zeros((0, 3), np.float32))
            return
        self.radar.setData(pos=np.ascontiguousarray(xyz, dtype=np.float32),
                           color=np.ascontiguousarray(rgba, dtype=np.float32), size=float(size))

    def _redraw(self):
        pts = self._pts
        if pts is None or not len(pts):
            self.cloud.setData(pos=np.zeros((0, 3), np.float32))
            return
        keep = pts[:, 3] <= self.range_m * 1.6
        p = pts[keep]
        mode = self.colour.currentIndex()
        if mode == 0:
            v = np.clip((p[:, 2] - self.ground_z + 0.2) / 3.0, 0, 1)          # height: ground .. +3 m
            rgba = cm.turbo(v)
        elif mode == 1:
            v = np.clip(p[:, 4] / 120.0, 0, 1)
            rgba = cm.viridis(v)
        else:
            v = np.clip(p[:, 3] / max(1.0, self.range_m), 0, 1)
            rgba = cm.viridis(1 - v)
        self.cloud.setData(pos=np.ascontiguousarray(p[:, :3], dtype=np.float32),
                           color=np.ascontiguousarray(rgba, dtype=np.float32))

    def grab(self, path: str):
        self.view.grabFramebuffer().save(path)
