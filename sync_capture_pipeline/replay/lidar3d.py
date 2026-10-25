"""3D LiDAR view for the replay (pyqtgraph.opengl, needs PyOpenGL).

A panel of the replay window that follows the replay clock: it shows the
sweep the 2D LiDAR panel is showing, as a point cloud you can orbit (left
drag), pan (middle drag / ctrl+drag) and zoom (wheel).

Coordinates are the LiDAR's: x forward, y left, z up. The default camera sits
behind and above the sensors looking forward, so vehicle-left is on the left
as in every other replay view. Drawn as well: a 1 m ground grid at the ground
height the LiDAR detector measures, the radar's +/-60 deg azimuth field of
view, the LiDAR target (white) when a target track exists
(sync/timing_lidar_track.npz), the LiDAR objects as 3D boxes coloured by size
class, confirmed track IDs, and the radar points.
"""
from __future__ import annotations

import numpy as np
from matplotlib import cm

COLOUR_MODES = ("height", "objects", "intensity", "range")


class Lidar3DView:
    """Owns a QWidget (self.widget) to place in a layout."""

    def __init__(self, range_m: float = 40.0, ground_z: float = -1.0):
        import pyqtgraph.opengl as gl
        from PyQt5 import QtWidgets

        self.gl = gl
        self.widget = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.widget)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((14, 16, 20))
        lay.addWidget(self.view, stretch=1)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(4, 0, 4, 2)
        lay.addLayout(bar)
        self.label = QtWidgets.QLabel()
        self.label.setObjectName("hint")
        bar.addWidget(self.label, stretch=1)
        self.colour = QtWidgets.QComboBox()
        self.colour.addItems([f"colour: {m}" for m in COLOUR_MODES])
        self.colour.setToolTip("objects: points of each detected object in its class colour, ground dimmed")
        self.colour.currentIndexChanged.connect(lambda _i: self._redraw())
        bar.addWidget(self.colour)
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(lambda: self.reset_camera())
        bar.addWidget(reset)

        self.range_m = float(range_m)
        self.ground_z = ground_z
        self.grid = grid = gl.GLGridItem()
        grid.setSize(200, 200)
        grid.setSpacing(1, 1)
        grid.translate(0, 0, ground_z)
        grid.setColor((48, 52, 60, 255))
        self.view.addItem(grid)
        self.fov = gl.GLLinePlotItem(width=1.5, color=(0.16, 0.47, 0.84, 1.0), antialias=True, mode="lines")
        self.view.addItem(self.fov)
        self.cloud = gl.GLScatterPlotItem(size=2.0, pxMode=True)
        self.view.addItem(self.cloud)
        self.boxes = gl.GLLinePlotItem(width=1.5, antialias=True, mode="lines")
        self.view.addItem(self.boxes)
        self.target = gl.GLScatterPlotItem(size=16.0, pxMode=True, color=(1, 1, 1, 0.9))
        self.view.addItem(self.target)
        self.radar = gl.GLScatterPlotItem(size=6.0, pxMode=True)     # radar points (raw, detections, filtered)
        self.view.addItem(self.radar)
        self._texts = []                                             # pooled track-ID labels
        self._pts = self._lab = None
        self._key = None
        self.set_range(range_m)
        self.reset_camera()

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

    def wants_labels(self) -> bool:
        return self.widget.isVisible() and COLOUR_MODES[self.colour.currentIndex()] == "objects"

    def _set_ground(self, gz):
        if gz is None or not np.isfinite(gz) or abs(gz - self.ground_z) < 0.05:
            return
        self.ground_z = float(gz)
        self.grid.resetTransform()
        self.grid.translate(0, 0, self.ground_z)
        self.set_range(self.range_m)

    # ── data ────────────────────────────────────────────────────────────
    def show_sweep(self, t_label: str, pts, target=None, labels=None, ground_z=None):
        """pts: N x 5 (x, y, z, range, intensity); labels: per point (-1 ground, 0 other, k object k)
        or None. target: (range_m, azimuth_deg) or None."""
        self._set_ground(ground_z)
        key = (t_label, None if pts is None else len(pts), labels is not None)
        if key != self._key:
            self._key = key
            self._pts, self._lab = pts, labels
            self._redraw()
            n = 0 if pts is None else len(pts)
            self.label.setText(f"{t_label}   {n:,} points   drag: orbit   wheel: zoom   ctrl+drag: pan")
        if target is None:
            self.target.setData(pos=np.zeros((0, 3), np.float32))
        else:
            r, az = target
            self.target.setData(pos=np.array([[r * np.cos(np.radians(az)), r * np.sin(np.radians(az)),
                                               self.ground_z + 1.0]], dtype=np.float32))

    def show_radar(self, xyz, rgba, size=6.0):
        """Radar points in the LiDAR frame (radar assumed co-located with the LiDAR)."""
        if xyz is None or not len(xyz):
            self.radar.setData(pos=np.zeros((0, 3), np.float32))
            return
        self.radar.setData(pos=np.ascontiguousarray(xyz, dtype=np.float32),
                           color=np.ascontiguousarray(rgba, dtype=np.float32), size=float(size))

    def show_objects(self, objects, colours, tracks=None):
        """objects: K x OBJ_COLS (perception.lidar), colours: K x 4 (0..1); tracks: [(id, x, y, ...)]."""
        from perception.lidar import O
        lim = 1.3 * self.range_m                                     # only what the view shows: no far clutter
        if objects is not None and len(objects):
            near = objects[:, O["range"]] <= lim
            objects, colours = objects[near], [c for c, k in zip(colours, near) if k]
        tracks = [t for t in (tracks or []) if np.hypot(t[1], t[2]) <= lim]
        if objects is None or not len(objects):
            self.boxes.setData(pos=np.zeros((0, 3), np.float32))
        else:
            segs, cols = [], []
            edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
            for ob, c in zip(objects, colours):
                a = np.radians(ob[O["yaw_deg"]])
                hl, hw = ob[O["length"]] / 2, ob[O["width"]] / 2
                # minAreaRect angle is measured on the (x, y) grid: length along (cos a, sin a)
                u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
                base = [np.r_[ob[O["x"]], ob[O["y"]]] + sl * hl * u + sw * hw * v
                        for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
                z0, z1 = ob[O["z_min"]], ob[O["z_max"]]
                corners = [np.r_[b, z0] for b in base] + [np.r_[b, z1] for b in base]
                for i, j in edges:
                    segs += [corners[i], corners[j]]
                    cols += [c, c]
            self.boxes.setData(pos=np.asarray(segs, np.float32), color=np.asarray(cols, np.float32))
        tracks = tracks or []
        while len(self._texts) < len(tracks):
            t = self.gl.GLTextItem(color=(255, 255, 255, 230))
            self.view.addItem(t)
            self._texts.append(t)
        for i, t in enumerate(self._texts):
            if i < len(tracks):
                tid, x, y = tracks[i][:3]
                t.setData(pos=(float(x), float(y), self.ground_z + 2.3), text=f"L{tid}")
                t.setVisible(True)
            else:
                t.setVisible(False)

    def _redraw(self):
        pts = self._pts
        if pts is None or not len(pts):
            self.cloud.setData(pos=np.zeros((0, 3), np.float32))
            return
        keep = pts[:, 3] <= self.range_m * 1.6
        p = pts[keep]
        mode = COLOUR_MODES[self.colour.currentIndex()]
        if mode == "objects" and self._lab is not None:
            from perception.lidar import CLASS_COLOURS, CLASSES
            lab = self._lab[keep]
            rgba = np.tile(np.array([0.33, 0.35, 0.40, 0.5], np.float32), (len(p), 1))   # other non-ground
            rgba[lab == -1] = (0.20, 0.21, 0.24, 0.35)                                  # ground
            self._obj_lab = lab
            cls = getattr(self, "_obj_cls", None)
            if cls is not None and len(cls):
                pal = np.array([np.r_[np.array(CLASS_COLOURS[c]) / 255.0, 1.0] for c in CLASSES], np.float32)
                k = lab > 0
                idx = np.clip(lab[k] - 1, 0, len(cls) - 1)
                rgba[k] = pal[cls[idx]]
        elif mode in ("height", "objects"):
            v = np.clip((p[:, 2] - self.ground_z + 0.2) / 3.0, 0, 1)          # height: ground .. +3 m
            rgba = cm.turbo(v)
        elif mode == "intensity":
            v = np.clip(p[:, 4] / 120.0, 0, 1)
            rgba = cm.viridis(v)
        else:
            v = np.clip(p[:, 3] / max(1.0, self.range_m), 0, 1)
            rgba = cm.viridis(1 - v)
        self.cloud.setData(pos=np.ascontiguousarray(p[:, :3], dtype=np.float32),
                           color=np.ascontiguousarray(rgba, dtype=np.float32))

    def set_object_classes(self, cls):
        """Size class index per object (1-based object labels map to cls[label - 1])."""
        changed = getattr(self, "_obj_cls", None) is None or not np.array_equal(self._obj_cls, cls)
        self._obj_cls = None if cls is None else np.asarray(cls, int)
        if changed and COLOUR_MODES[self.colour.currentIndex()] == "objects":
            self._redraw()

    def grab(self, path: str):
        self.view.grabFramebuffer().save(path)
