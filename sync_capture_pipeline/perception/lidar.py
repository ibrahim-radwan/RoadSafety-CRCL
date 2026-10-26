"""
GPU LiDAR object detection for the RS-Fairy (PyTorch + OpenCV), one sweep at
a time.

    1. valid returns only (range > 0: the RS-Fairy fills no-return directions
       with zero-range points), beyond near_m (mount, vehicle body), within
       max_range_m
    2. local ground, no single plane (slopes, kerbs, driveways): the lowest
       return of each ground_cell_m cell; a cell more than max_step_m above
       its 3 x 3 neighbourhood minimum is lifted (under a car body) and takes
       that minimum instead; empty cells are filled from wider neighbourhoods
    3. obstacle points: height above that ground in (ground_thr_m, max_height_m],
       in columns (cell_m) that are GROUNDED - something in the column reaches
       down to within grounded_m of the ground. Overhead structures with free
       space below (carport roofs, canopies, signs, branches) drop out; posts,
       walls, people and vehicles stay
    4. bird's-eye occupancy grid (cell_m), closed by one cell, 8-connected
       components (OpenCV) -> objects
    5. per object: point count, oriented box (min-area rectangle of its grid
       cells: centre x, y, length along yaw_deg, width), height above ground, and a size class (pedestrian /
       two-wheeler / vehicle / other / structure). Size classes are heuristics
       on the box, not a learned classifier.

Pipeline frame: x forward, y left, z up (the LiDAR's own frame).
"""
from __future__ import annotations

import time

import numpy as np

OBJ_COLS = ("x", "y", "z", "length", "width", "height", "yaw_deg", "n_points", "cls", "range", "az_deg",
            "z_min", "z_max")
O = {k: i for i, k in enumerate(OBJ_COLS)}
CLASSES = ("pedestrian", "two-wheeler", "vehicle", "other", "structure")
# Tracker settings for LiDAR objects (structures are not tracked)
LIDAR_TRACKER = dict(gate_m=1.0, meas_std_m=0.15, accel_std=4.0, max_misses=3, merge_m=0.5)
CLASS_COLOURS = {"pedestrian": (255, 214, 10), "two-wheeler": (0, 200, 255), "vehicle": (255, 70, 160),
                 "other": (200, 200, 200), "structure": (110, 110, 120)}


def box_corners(objects: np.ndarray) -> np.ndarray:
    """Footprint corners [K, 4, 2] (x, y) of oriented boxes (OBJ_COLS rows), in order around the box:
    the length runs along (cos yaw, sin yaw)."""
    ob = np.asarray(objects, np.float32).reshape(-1, len(OBJ_COLS))
    a = np.radians(ob[:, O["yaw_deg"]])
    u = np.stack([np.cos(a), np.sin(a)], axis=1) * (ob[:, O["length"]] / 2)[:, None]
    v = np.stack([-np.sin(a), np.cos(a)], axis=1) * (ob[:, O["width"]] / 2)[:, None]
    c = ob[:, [O["x"], O["y"]]]
    sl = np.array([1, 1, -1, -1], np.float32)[None, :, None]
    sw = np.array([1, -1, -1, 1], np.float32)[None, :, None]
    return c[:, None, :] + sl * u[:, None, :] + sw * v[:, None, :]


def size_class(length: float, width: float, height: float) -> int:
    """Box-size heuristic: length >= width, height above ground (m)."""
    if length > 8.0 or height > 4.5:
        return CLASSES.index("structure")
    if 0.9 <= height <= 2.3 and length <= 1.3 and width <= 1.0:      # a walking person: ~0.6-1.2 m stride
        return CLASSES.index("pedestrian")
    if 0.9 <= height <= 2.3 and 1.3 < length <= 2.5 and width <= 1.2:
        return CLASSES.index("two-wheeler")
    if height >= 1.0 and 2.5 < length <= 7.0 and 1.3 <= width <= 3.0:
        return CLASSES.index("vehicle")
    return CLASSES.index("other")


class LidarDetector:
    def __init__(self, device: str = "cuda", near_m: float = 0.5, max_range_m: float = 60.0, cell_m: float = 0.2,
                 ground_cell_m: float = 1.0, ground_thr_m: float = 0.2, max_height_m: float = 3.0,
                 grounded_m: float = 1.0, min_points: int = 8, min_height_m: float = 0.3,
                 max_step_m: float = 0.35, cluster: str = "density", core_k: float = 200.0, core_max: int = 8,
                 core_close: bool = False):
        import torch
        self.torch = torch
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_step = max_step_m
        self.near, self.max_range, self.cell = near_m, max_range_m, cell_m
        self.gcell, self.g_thr, self.h_max, self.grounded = ground_cell_m, ground_thr_m, max_height_m, grounded_m
        self.min_points, self.min_height = min_points, min_height_m
        self.n_grid = int(np.ceil(2 * max_range_m / cell_m))
        self.n_ggrid = int(np.ceil(2 * max_range_m / ground_cell_m))
        self.cluster, self.core_close = cluster, core_close
        if cluster == "density":
            # core-cell threshold per grid cell: round(core_k / r^2) points, clamped to 1..core_max
            # (LiDAR point density falls with range squared)
            c = (np.arange(self.n_grid) + 0.5) * cell_m - max_range_m
            r = np.hypot(c[:, None], c[None, :])
            self.core_thr = torch.as_tensor(np.clip(np.round(core_k / np.maximum(r, 0.5) ** 2), 1, core_max),
                                            dtype=torch.int32, device=self.dev).view(-1)

    def _ground_map(self, cmin):
        """Cell minima (inf = empty) -> ground height per cell. A cell keeps its own minimum when
        it is within max_step_m of its 3x3 neighbourhood minimum (ground, also on slopes);
        otherwise (lifted: under a car body, a dense object) it takes the neighbourhood minimum.
        Empty cells are filled from wider windows."""
        torch = self.torch
        F = torch.nn.functional
        n = self.n_ggrid
        nb = -F.max_pool2d(-cmin.view(1, 1, n, n), 3, stride=1, padding=1)[0, 0]
        own = cmin.view(n, n)
        g = torch.where(torch.isfinite(own) & (own - nb <= self.max_step), own, nb)
        for k in (5, 9, 17):                                 # fill empty cells from wider neighbourhoods
            empty = ~torch.isfinite(g)
            if not bool(empty.any()):
                break
            wide = -F.max_pool2d(-g.view(1, 1, n, n), k, stride=1, padding=k // 2)[0, 0]
            g = torch.where(empty, wide, g)
        fin = torch.isfinite(g)
        if not bool(fin.all()):
            g = torch.where(fin, g, g[fin].median() if bool(fin.any()) else torch.zeros_like(g))
        return g.view(-1)

    def detect(self, pts: np.ndarray, labels: bool = True):
        """pts: N x >=5 (x, y, z, range, intensity, ...). Returns dict:
        objects (K x len(OBJ_COLS)), point_label (if labels; per input point: -2 invalid/near/far,
        -1 ground, 0 other non-ground (overhead, high, not in an object), k > 0 object k), ground_z (median
        ground height near the sensor), ms (timings)."""
        import cv2
        torch = self.torch
        t0 = time.perf_counter()
        P = torch.as_tensor(np.ascontiguousarray(pts[:, :4], dtype=np.float32), device=self.dev)
        r = P[:, 3]
        valid = torch.isfinite(r) & (r > max(self.near, 0.05)) & (P[:, 0].abs() < self.max_range) \
            & (P[:, 1].abs() < self.max_range)
        vi = torch.nonzero(valid, as_tuple=True)[0]
        x, y, z = P[vi, 0], P[vi, 1], P[vi, 2]
        # local ground
        gi = ((x + self.max_range) / self.gcell).long().clamp(0, self.n_ggrid - 1)
        gj = ((y + self.max_range) / self.gcell).long().clamp(0, self.n_ggrid - 1)
        gc = gi * self.n_ggrid + gj
        cmin = torch.full((self.n_ggrid * self.n_ggrid,), float("inf"), device=self.dev)
        cmin = cmin.scatter_reduce(0, gc, z, reduce="amin", include_self=True)
        gmap = self._ground_map(cmin)
        hag = z - gmap[gc]
        ground = hag <= self.g_thr
        # grounded columns on the fine grid
        fi = ((x + self.max_range) / self.cell).long().clamp(0, self.n_grid - 1)
        fj = ((y + self.max_range) / self.cell).long().clamp(0, self.n_grid - 1)
        fc = fi * self.n_grid + fj
        ng = ~ground
        col_low = torch.full((self.n_grid * self.n_grid,), float("inf"), device=self.dev)
        col_low = col_low.scatter_reduce(0, fc[ng], hag[ng], reduce="amin", include_self=True)
        obst = ng & (hag <= self.h_max) & (col_low[fc] <= self.grounded)
        t1 = time.perf_counter()
        # occupancy grid -> connected components
        oi = torch.nonzero(obst, as_tuple=True)[0]
        foc = fc[oi]
        occ = torch.zeros(self.n_grid * self.n_grid, dtype=torch.uint8, device=self.dev)
        occ[foc] = 1
        occ_np = occ.view(self.n_grid, self.n_grid).cpu().numpy()
        if self.cluster == "density":
            # Clusters grow through dense ("core") cells only; a sparse cell joins a neighbouring
            # cluster but cannot connect two. A single stray point (a mixed pixel at a depth edge)
            # no longer bridges a person and a wall.
            cnt = torch.bincount(foc, minlength=self.n_grid * self.n_grid)
            core_np = (cnt >= self.core_thr).to(torch.uint8).view(self.n_grid, self.n_grid).cpu().numpy() & occ_np
            if self.core_close:                  # bridge a one-cell gap between dense cells only
                core_np = cv2.morphologyEx(core_np, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            n_lab, lab = cv2.connectedComponents(core_np, connectivity=8)
            border = (occ_np > 0) & (core_np == 0)
            grown = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
            lab = np.where(border, grown, lab).astype(np.int32)
        else:
            closed = cv2.morphologyEx(occ_np, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            n_lab, lab = cv2.connectedComponents(closed, connectivity=8)
        t2 = time.perf_counter()
        K = int(n_lab)
        lab_t = torch.as_tensor(lab.astype(np.int32), device=self.dev).view(-1)
        plab = lab_t[foc].long()                                       # component of each obstacle point
        cnt = torch.bincount(plab, minlength=K)
        oz, oh = z[oi], hag[oi]
        stats = []
        for v, red, init in ((oz, "amin", 1e3), (oz, "amax", -1e3), (oh, "amax", -1e3)):
            stats.append(torch.full((K,), init, device=self.dev).scatter_reduce(0, plab, v, reduce=red,
                                                                                include_self=True))
        S = torch.stack([cnt.float()] + stats).cpu().numpy()          # one transfer
        cnt_np, zmin, zmax, hmax = S
        keep = (cnt_np >= self.min_points) & (hmax >= self.min_height)
        keep[0] = False                                                # label 0 = background
        # oriented boxes from each component's grid cells (CPU, few cells)
        cells_i, cells_j = np.nonzero(occ_np)                         # occupied cells (before closing)
        cl = lab[cells_i, cells_j]
        order = np.argsort(cl, kind="stable")
        bounds = np.searchsorted(cl[order], np.arange(K + 1))
        objs, remap = [], np.zeros(K, np.int64)
        for k in np.flatnonzero(keep):
            sel = order[bounds[k]:bounds[k + 1]]
            q = np.c_[cells_i[sel], cells_j[sel]].astype(np.float32) * self.cell - self.max_range + self.cell / 2
            (cx, cy), (s0, s1), ang = cv2.minAreaRect(q)                # side s0 along ang, s1 along ang + 90
            s0, s1 = s0 + self.cell, s1 + self.cell                    # cell centres -> cell edges
            length, width, yaw = (s0, s1, ang) if s0 >= s1 else (s1, s0, ang + 90.0)
            yaw = (yaw + 90.0) % 180.0 - 90.0                          # length axis direction, -90..90 deg
            length, width, height = float(length), float(width), float(hmax[k])
            remap[k] = len(objs) + 1
            objs.append([cx, cy, 0.5 * (zmin[k] + zmax[k]), length, width, height, yaw, cnt_np[k],
                         size_class(length, width, height), np.hypot(cx, cy), np.degrees(np.arctan2(cy, cx)),
                         zmin[k], zmax[k]])
        objects = np.asarray(objs, np.float32).reshape(-1, len(OBJ_COLS))
        point_label = None
        if labels:                                                     # per-point labels (for display)
            lab_pts = torch.zeros(len(vi), dtype=torch.int32, device=self.dev)
            lab_pts[ground] = -1
            lab_pts[oi] = torch.as_tensor(remap.astype(np.int32), device=self.dev)[plab]
            full = torch.full((len(pts),), -2, dtype=torch.int32, device=self.dev)
            full[vi] = lab_pts
            point_label = full.cpu().numpy()
        near = (x.abs() < 5) & (y.abs() < 5)
        ground_z = float(gmap[gc[near]].median()) if bool(near.any()) else float("nan")
        t3 = time.perf_counter()
        return {"objects": objects, "point_label": point_label, "ground_z": ground_z,
                "ms": {"ground": 1e3 * (t1 - t0), "grid": 1e3 * (t2 - t1), "objects": 1e3 * (t3 - t2)}}


_DETECTOR = {}


def detect_sweep(pts: np.ndarray):
    """(objects, ground_z, ms) for one sweep with a default LidarDetector, one per process: the
    per-sweep function for perception.core.lidar_io.decode_parallel's worker processes."""
    import time
    if "d" not in _DETECTOR:
        _DETECTOR["d"] = LidarDetector()
    t0 = time.perf_counter()
    o = _DETECTOR["d"].detect(pts, labels=False)
    return o["objects"], o["ground_z"], 1e3 * (time.perf_counter() - t0)
