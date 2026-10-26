"""
Replay sources: a shared clock and one thread per sensor.

Every source follows the same ReplayClock (CLOCK_MONOTONIC session time) and
keeps its LATEST product ready for the GUI. The GUI never waits on a sensor.

    CameraSource  ZED SDK replays the SVO2 (decoded on the GPU); the frame
                  shown is the last one captured at or before the clock.
    RadarSource   each radar frame is processed on the GPU (perception/radar.py):
                  range-Doppler and bird's-eye maps, CFAR detections, tracks.
    LidarSource   rs_driver decodes lidar.pcap through a named pipe that a
                  feeder thread fills in step with the clock (pause, speed
                  and seek work; rs_driver alone can only play a file from
                  the start). Products per sweep: the 2D view, the points
                  for the 3D view, and objects + tracks (perception/lidar.py).

Times come from sync/ (align_session.py), so offsets applied there apply
here too.
"""
from __future__ import annotations

import csv
import json
import os
import select
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
sys.path.insert(0, str(PIPELINE))
from perception.core.conventions import world_azimuth                          # noqa: E402
from perception.core.lidar_io import (decode_config, lidar_pcap_index, lidar_to_mono,  # noqa: E402,F401
                                      load_wrapper, sweep_time)
from perception.core.radar_io import FramePrefetcher                           # noqa: E402
from perception.core.session import Session                                    # noqa: E402


def load_csv(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── clock ───────────────────────────────────────────────────────────────────
class ReplayClock:
    """Session time on CLOCK_MONOTONIC: play/pause/speed/seek.

    The state lives in a shared-memory array, so a sensor running in its own
    PROCESS follows exactly the same clock (CLOCK_MONOTONIC is system-wide).
    Pass `clock.state` to the child and rebuild with ReplayClock(state=...).
    """
    T0, T1, T, WALL, PLAYING, SPEED, GEN = range(7)

    def __init__(self, t0: float = 0.0, t1: float = 1.0, state=None):
        import multiprocessing as mp
        self.state = state if state is not None else mp.get_context("spawn").Array(
            "d", [t0, t1, t0, 0.0, 0.0, 1.0, 0.0])

    def _get(self, i):
        return self.state[i]

    t0 = property(lambda self: self._get(self.T0), lambda self, v: self.state.__setitem__(self.T0, v))
    t1 = property(lambda self: self._get(self.T1), lambda self, v: self.state.__setitem__(self.T1, v))
    playing = property(lambda self: bool(self._get(self.PLAYING)))
    speed = property(lambda self: self._get(self.SPEED))
    seek_gen = property(lambda self: int(self._get(self.GEN)))   # bumped on every jump; sources re-sync

    def now(self) -> float:
        st = self.state
        with st.get_lock():
            if not st[self.PLAYING]:
                return st[self.T]
            t = st[self.T] + (time.monotonic() - st[self.WALL]) * st[self.SPEED]
            if t >= st[self.T1]:
                st[self.T], st[self.PLAYING] = st[self.T1], 0.0
                return st[self.T1]
            return t

    def play(self):
        st = self.state
        with st.get_lock():
            if not st[self.PLAYING]:
                if st[self.T] >= st[self.T1]:
                    st[self.T] = st[self.T0]
                    st[self.GEN] += 1
                st[self.WALL], st[self.PLAYING] = time.monotonic(), 1.0

    def pause(self):
        t = self.now()
        with self.state.get_lock():
            self.state[self.T], self.state[self.PLAYING] = t, 0.0

    def set_speed(self, speed: float):
        t = self.now()
        with self.state.get_lock():
            self.state[self.T], self.state[self.WALL], self.state[self.SPEED] = t, time.monotonic(), speed

    def seek(self, t: float):
        st = self.state
        with st.get_lock():
            st[self.T] = min(max(t, st[self.T0]), st[self.T1])
            st[self.WALL] = time.monotonic()
            st[self.GEN] += 1


class Source(threading.Thread):
    """Base: a daemon thread that keeps `latest` = (time, product) current."""

    name_ = "source"

    def __init__(self, clock: ReplayClock):
        super().__init__(name=self.name_, daemon=True)
        self.clock = clock
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.produced = 0                   # products made (rate measured by the GUI)

    def publish(self, item):
        with self.lock:
            self.latest = item
            self.produced += 1

    def get(self):
        with self.lock:
            return self.latest

    def stop(self, timeout_s: float = 5.0):
        """Stop and wait for the thread: a source that wraps a driver (the LiDAR) must have shut it down
        before Python exits, or the driver's threads call back into a finalising interpreter."""
        self.stop_evt.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout_s)


# ── camera ──────────────────────────────────────────────────────────────────
def _camera_worker(clock_state, session: str, width: int, height: int, shm_name: str, meta, err, stop_evt):
    """Runs in its own PROCESS: the ZED SDK's grab() holds the GIL for a whole
    frame decode, which in a thread starved every other source and the GUI.
    Publishes the latest frame into shared memory; meta = [seq, t, frame]."""
    from multiprocessing import shared_memory
    try:
        import pyzed.sl as sl
        session = Path(session)
        clock = ReplayClock(state=clock_state)
        rows = load_csv(session / "sync" / "camera_timeline.csv")
        idx = np.array([int(r["frame_index"]) for r in rows])
        tt = np.array([float(r["t_ref"]) for r in rows])
        shm = shared_memory.SharedMemory(name=shm_name)
        out = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)
        init = sl.InitParameters()
        init.set_from_svo_file(str(session / "camera" / "camera.svo2"))
        init.svo_real_time_mode = False
        init.depth_mode = sl.DEPTH_MODE.NONE
        init.optional_settings_path = str(session / "camera" / "calibration")
        init.sdk_verbose = 0
        cam = sl.Camera()
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("cannot open camera.svo2")
        rt, mat = sl.RuntimeParameters(), sl.Mat()
        res = sl.Resolution(width, height)
        pos, gen = -1, None
        while not stop_evt.is_set():
            k = int(np.searchsorted(tt, clock.now(), side="right")) - 1   # last frame at or before now
            if k < 0:                                        # before the camera's first frame
                if meta[2] != -1.0:
                    with meta.get_lock():
                        meta[0] += 1
                        meta[2] = -1.0
                time.sleep(0.01)
                continue
            want = int(idx[k])
            if want == pos and gen == clock.seek_gen:
                time.sleep(0.004)
                continue
            gen = clock.seek_gen
            # Decode forward to catch up (~70 fps available); seek only on a real
            # jump. Seeking an H.264 stream re-decodes from the last keyframe,
            # so seeking whenever slightly behind made playback fall further behind.
            if not (0 < want - pos <= 45):
                cam.set_svo_position(want)
                pos = want - 1
            while pos < want and not stop_evt.is_set():
                if int(cam.grab(rt).value) > 0:
                    break
                pos = cam.get_svo_position()
            cam.retrieve_image(mat, sl.VIEW.LEFT, sl.MEM.CPU, res)
            rgb = mat.get_data()[..., 2::-1]                 # BGRA -> RGB
            with meta.get_lock():
                out[:] = rgb
                meta[0] += 1
                meta[1], meta[2] = float(tt[k]), float(pos)
        mat.free()
        cam.close()
        del out
        shm.close()
    except Exception as exc:
        err.value = str(exc).encode()[:299]


class CameraSource:
    """Camera replay in a child process (see _camera_worker); same interface
    as the thread-based sources."""
    name_ = "camera"

    def __init__(self, clock, session: Path, width: int = 960):
        import multiprocessing as mp
        self.clock, self.session, self.width = clock, session, width
        rows = load_csv(session / "sync" / "camera_timeline.csv")
        self.t = np.array([float(r["t_ref"]) for r in rows])
        calib = json.loads((session / "camera" / "calibration" / "camera.json").read_text())
        left = calib["rectified"]["left"]
        self.height = int(round(left["height"] * width / left["width"]))
        self.ctx = mp.get_context("spawn")
        self.meta = self.ctx.Array("d", [0.0, 0.0, -1.0])
        self.err = self.ctx.Array("c", 300)
        self.stop_evt = self.ctx.Event()
        self.shm = None
        self.proc = None
        self._seq, self._latest = 0, None

    def span(self):
        return float(self.t[0]), float(self.t[-1])

    @property
    def error(self):
        e = self.err.value.decode(errors="replace")
        return f"camera: {e}" if e else None

    @property
    def produced(self):
        return int(self.meta[0])

    def start(self):
        from multiprocessing import shared_memory
        self.shm = shared_memory.SharedMemory(create=True, size=self.height * self.width * 3)
        self.img = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self.shm.buf)
        self.proc = self.ctx.Process(target=_camera_worker, name="replay-camera", daemon=True,
                                     args=(self.clock.state, str(self.session), self.width, self.height,
                                           self.shm.name, self.meta, self.err, self.stop_evt))
        self.proc.start()

    def get(self):
        with self.meta.get_lock():
            seq = int(self.meta[0])
            if seq and seq != self._seq:
                self._seq = seq
                self._latest = None if self.meta[2] < 0 else \
                    (float(self.meta[1]), int(self.meta[2]), self.img.copy())
        return self._latest

    def stop(self):
        self.stop_evt.set()
        if self.proc is not None:
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                self.proc.terminate()
        if self.shm is not None:
            del self.img
            self.shm.close()
            self.shm.unlink()
            self.shm = None


# ── radar ───────────────────────────────────────────────────────────────────
def bev_lookup(proc, bev_x, bev_y):
    """Bird's-eye lookup for a radar processor: each (x forward, y left) cell of the grid -> its
    (range bin, processing-azimuth bin) and whether it is inside the field of view (+-85 deg)."""
    import torch
    X, Y = np.meshgrid(bev_x, bev_y, indexing="ij")
    az_world = np.degrees(np.arctan2(Y, X))
    ri = np.round(np.hypot(X, Y) / proc.range_res).astype(int)
    ai = np.clip(np.searchsorted(proc.az_axis, world_azimuth(az_world)), 0, proc.n_az - 1)
    valid = (X > 0) & (ri < proc.n_range) & (np.abs(az_world) <= 85.0)
    return (torch.as_tensor(np.where(valid, ri, 0), device=proc.dev),
            torch.as_tensor(np.where(valid, ai, 0), device=proc.dev), torch.as_tensor(valid, device=proc.dev))


class RadarProcessor:
    """Radar products for the session report (report/collect.py): range-Doppler and a bird's-eye
    map per frame. It is perception.radar.RadarDetector without detections, without TDM Doppler
    compensation and with the .cfg calibration only (no in-situ correction), as the report has
    always computed them. The processing azimuth is positive to the right; the bird's-eye lookup
    converts it (perception/core/conventions.py)."""

    MIRROR_AZIMUTH = True                     # kept for the report's metadata: world az = - processing az

    def __init__(self, bev_x, bev_y, cfg_path: Path, device="cuda"):
        from perception.radar import RadarDetector
        self.det = RadarDetector(cfg_path, device=device, rx_phase_deg=None, doppler_comp=False)
        self.cfg, self.torch, self.dev = self.det.cfg, self.det.torch, self.det.dev
        self.range_axis, self.az_axis, self.dop_axis = self.det.range_axis, self.det.az_axis, self.det.dop_axis
        self.mirror_azimuth = self.MIRROR_AZIMUTH
        self.bev_ri, self.bev_ai, self.bev_valid = bev_lookup(self.det, bev_x, bev_y)

    def process(self, raw: np.ndarray, moving_only: bool = True):
        """(range-Doppler [doppler, range] dB, bird's-eye [x, y] dB, NaN outside the FOV); also
        sets last_ra, the range-azimuth map [az, range] dB."""
        torch = self.torch
        out = self.det.process(raw, moving_only, detect=False)
        ra = out["ra"]
        self.last_ra = ra.cpu().numpy()
        bev = ra[self.bev_ai, self.bev_ri]
        bev = torch.where(self.bev_valid, bev, torch.full_like(bev, float("nan")))
        return out["rd"].cpu().numpy(), bev.cpu().numpy()


class RadarSource(Source):
    """Radar replay: our GPU detector (perception/radar.py) on every frame the clock reaches,
    with read-ahead, clustering and a Doppler-aware tracker. Publishes a dict:
        t, k            frame time (t_ref) and index
        rd              range-Doppler map [doppler, range] dB
        bev             bird's-eye power on the (bev_x, bev_y) grid, NaN outside the FOV
        ra              range-azimuth map [az, range] dB (processing azimuth order)
        det             detections (perception.radar.DET_COLS), multipath ghosts flagged
        tracks          [(id, x, y, vx, vy, v_r, hits, micro-Doppler features or None,
                         accumulated points [N, 5])] reported tracks
    """
    name_ = "radar"
    PREFETCH = 12                       # frames read ahead (slow disks: NTFS/FUSE, USB)

    def __init__(self, clock, session: Path, bev_x, bev_y, moving_only=True, rx_phase=True):
        super().__init__(clock)
        sys.path.insert(0, str(PIPELINE))
        from perception.radar import RadarDetector, rx_phase_for
        self.session = session
        sess = Session(session)
        self.timeline = sess.radar
        self.t, self.files = self.timeline.t, self.timeline.files
        variant, how = sess.firmware
        rx = rx_phase_for(variant) if rx_phase else None
        print(f"[radar] firmware {variant} ({how}); in-situ RX phase correction: "
              f"{'on' if rx is not None else 'off'}")
        self.proc = RadarDetector(sess.cfg_path, rx_phase_deg=rx)
        self.bev_ri, self.bev_ai, self.bev_valid = bev_lookup(self.proc, bev_x, bev_y)
        self.timing_source = sess.alignment.get("radar", {}).get("timing_source", "")
        self.moving_only = moving_only
        self.show_static = False             # stationary-reflector detections (experimental), on request
        self.ms_per_frame = 0.0
        # Raw point cloud: one point per range-azimuth cell (world azimuth = - processing azimuth), z = 0
        az = np.radians(world_azimuth(self.proc.az_axis))
        rr = np.asarray(self.proc.range_axis)
        self.ra_x = (np.cos(az)[:, None] * rr[None, :]).astype(np.float32)     # [az, range]
        self.ra_y = (np.sin(az)[:, None] * rr[None, :]).astype(np.float32)
        self.ra_r = np.broadcast_to(rr[None, :], self.ra_x.shape)
        self.prefetch = FramePrefetcher(self.timeline, self.PREFETCH)

    def raw_points(self, ra, thr_db: float, max_range: float):
        """Cells more than thr_db above this frame's median, within max_range: (xyz N x 3, dB above floor)."""
        rel = ra - np.median(ra)
        m = (rel > thr_db) & (self.ra_r <= max_range) & (self.ra_r > 0.5)
        return np.c_[self.ra_x[m], self.ra_y[m], np.zeros(int(m.sum()), np.float32)], rel[m]

    def span(self):
        return float(self.t[0]), float(self.t[-1])

    def stop(self, timeout_s: float = 5.0):
        super().stop(timeout_s)
        self.prefetch.stop()

    def run(self):
        sys.path.insert(0, str(PIPELINE))
        from perception.radar import RADAR_TRACKER, cluster_detections
        from perception.tracking import Tracker
        from perception.track_features import TrackFeatures
        tracker = Tracker(**RADAR_TRACKER)
        feats = TrackFeatures(self.proc.range_axis, self.proc.az_axis, self.proc.dop_axis)
        torch = self.proc.torch
        try:                                     # any failure, startup included, shows as self.error
            self.prefetch.start()
            k_done, mode, gen = -1, None, None
            while not self.stop_evt.is_set():
                k = int(np.searchsorted(self.t, self.clock.now(), side="right")) - 1
                self.prefetch.want(k)
                if k < 0:                               # before the radar's first frame
                    if k_done != -1:
                        k_done = -1
                        self.publish(None)
                    time.sleep(0.01)
                    continue
                if k == k_done and mode == self.moving_only:
                    time.sleep(0.004)
                    continue
                if gen != self.clock.seek_gen or k < k_done or k > k_done + 8:
                    tracker.reset()                     # a jump: tracks start over
                    feats.reset()
                gen = self.clock.seek_gen
                raw = self.prefetch.get(k)
                t0 = time.perf_counter()
                out = self.proc.process(raw, self.moving_only, static=self.show_static)
                bev = out["ra"][self.bev_ai, self.bev_ri]
                bev = torch.where(self.bev_valid, bev, torch.full_like(bev, float("nan")))
                rd, bev, ra = out["rd"].cpu().numpy(), bev.cpu().numpy(), out["ra"].cpu().numpy()
                det = out["detections"]
                lab, ob = cluster_detections(det)
                tk = float(self.t[k])
                tracker.update(tk, ob[:, :2], list(ob[:, 4]), radial=ob[:, 4])
                fs = feats.update(tk, tracker.tracks, det, lab, out["p_az"])
                tracks = [(tr.id, tr.x[0], tr.x[1], tr.x[2], tr.x[3], float(tr.payload or 0.0), tr.hits,
                           fs.get(tr.id), feats.accumulated(tr.id, tk, tr.x[2:]))
                          for tr in tracker.reported()]
                self.ms_per_frame = (time.perf_counter() - t0) * 1e3
                k_done, mode = k, self.moving_only
                self.publish({"t": float(self.t[k]), "k": k, "rd": rd, "bev": bev, "ra": ra, "det": det,
                              "tracks": tracks, "static": out.get("static_detections")})
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = f"radar: {exc}"


# ── LiDAR ───────────────────────────────────────────────────────────────────
# ── LiDAR view rendering ────────────────────────────────────────────────────
LIDAR_VIEWS = ("top", "top360", "panorama")
LIDAR_COLORS = ("height", "intensity", "density")
_LUTS = {}


def _lut(name: str) -> np.ndarray:
    if name not in _LUTS:
        import matplotlib
        cmap = matplotlib.colormaps[name]
        _LUTS[name] = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    return _LUTS[name]


def _paint(cells: np.ndarray, values: np.ndarray, n: int, keep: str, lut: np.ndarray, lo=None, hi=None):
    """One value per cell (keep='max' or 'min'), coloured through lut; empty cells black."""
    order = np.argsort(values if keep == "max" else -values, kind="stable")
    img = np.full(n, np.nan, dtype=np.float32)
    img[cells[order]] = values[order]             # later writes win -> max (or min)
    have = np.isfinite(img)
    if lo is None:
        lo, hi = np.percentile(values, [2, 98]) if len(values) else (0.0, 1.0)
    idx = np.clip((img[have] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb = np.zeros((n, 3), dtype=np.uint8)
    rgb[have] = lut[idx]
    return rgb


def _fill_gaps(rgb: np.ndarray) -> np.ndarray:
    """Fill empty (black) cells from their 3x3 neighbours so sparse returns
    read as continuous surfaces; occupied cells keep their own colour."""
    import cv2
    grown = cv2.dilate(rgb, np.ones((3, 3), np.uint8))
    empty = ~rgb.any(axis=2)
    out = rgb.copy()
    out[empty] = grown[empty]
    return out


def top_extent(view: dict):
    """(x0, x1, y0, y1) of a top view: forward range R (top: 0..R; top360: -R..R) and the
    lateral extent that fills the panel's aspect (view["aspect"] = width / height on screen)."""
    R, asp = float(view["range_m"]), max(0.3, float(view.get("aspect", 1.0)))
    if view["mode"] == "top360":
        return -R, R, -R * asp, R * asp
    return 0.0, R, -R * asp / 2, R * asp / 2


def render_lidar(pts: np.ndarray, view: dict, ground_z=None):
    """pts: N x 6 (x, y, z, range, intensity, dt). Returns (rgb HxWx3, extent, n_shown).
    extent = (left, bottom, width, height) in plot coordinates: top views use
    plot-x = y (left +) and plot-y = x (forward); the panorama uses plot-x =
    azimuth (deg, left +) and plot-y = elevation (deg). Top views fill the
    panel (view["aspect"]) at about view["px"] cells across; colour by height
    runs from the ground (ground_z, when known) to 3 m above it."""
    x, y, z, r, inten = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], pts[:, 4]
    keep = r >= view["near_m"]                    # drop the mount / vehicle body
    x, y, z, r, inten = x[keep], y[keep], z[keep], r[keep], inten[keep]
    if view["mode"] == "panorama":
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arcsin(np.clip(z / np.maximum(r, 1e-6), -1, 1)))
        res, a0, a1, e0, e1 = 0.5, -180.0, 180.0, -20.0, 70.0
        nw, nh = int((a1 - a0) / res), int((e1 - e0) / res)
        ia, ie = ((az - a0) / res).astype(np.int64), ((el - e0) / res).astype(np.int64)
        ok = (ia >= 0) & (ia < nw) & (ie >= 0) & (ie < nh)
        cells = ie[ok] * nw + ia[ok]
        if view["color"] == "intensity":
            rgb = _paint(cells, inten[ok], nw * nh, "max", _lut("viridis"))
        else:                                     # nearest return per cell, coloured by distance
            d = np.log(r[ok] + 0.1)
            rgb = _paint(cells, d, nw * nh, "min", _lut("turbo_r"))
        return _fill_gaps(rgb.reshape(nh, nw, 3)), (a0, e0, a1 - a0, e1 - e0), int(ok.sum())
    x0, x1, y0, y1 = top_extent(view)
    res = max(x1 - x0, y1 - y0) / float(np.clip(view.get("px", 500), 250, 900))   # ~ one cell per pixel
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    ix, iy = ((x - x0) / res).astype(np.int64), ((y - y0) / res).astype(np.int64)
    ok = (x >= x0) & (y >= y0) & (ix < nx) & (iy < ny)
    cells = ix[ok] * ny + iy[ok]
    if view["color"] == "density":
        h = np.log1p(np.bincount(cells, minlength=nx * ny)).astype(np.float32)
        h[h == 0] = np.nan
        vals = h[np.isfinite(h)]
        hi = float(np.percentile(vals, 99)) if len(vals) else 1.0
        rgb = _paint(np.flatnonzero(np.isfinite(h)), vals, nx * ny, "max", _lut("viridis"), 0.0, hi)
    elif view["color"] == "intensity":
        rgb = _paint(cells, inten[ok], nx * ny, "max", _lut("viridis"))
    elif ground_z is not None and np.isfinite(ground_z):  # highest point per cell, ground .. +3 m
        rgb = _paint(cells, z[ok], nx * ny, "max", _lut("turbo"), ground_z - 0.1, ground_z + 3.0)
    else:
        rgb = _paint(cells, z[ok], nx * ny, "max", _lut("turbo"))
    return _fill_gaps(rgb.reshape(nx, ny, 3)), (y0, x0, y1 - y0, x1 - x0), int(ok.sum())


class LidarSource(Source):
    name_ = "lidar"
    LOOKAHEAD_S = 0.15      # a sweep is emitted after its last packet; feed ahead of the clock
    PREROLL_S = 0.3         # start decoding this much before a seek point

    def __init__(self, clock, session: Path, offset_s=0.0, view=None):
        super().__init__(clock)
        self.session = session
        self.offset_s = offset_s
        self.view = dict(mode="top", range_m=40.0, color="height", near_m=0.5, version=0, aspect=1.0, px=500)
        self.view.update(view or {})
        self.detect = True                  # objects + tracks per sweep (perception/lidar.py)
        self.want_labels = False            # per-point labels, only while the 3D view shows them
        self._det_cache = (None, None)      # (sweep time, detection result)
        self._tracker, self._trk_gen, self._trk_t = None, None, None
        self.vlock = threading.Lock()
        self.pcap = np.memmap(session / "lidar" / "lidar.pcap", dtype=np.uint8, mode="r")
        idx = lidar_pcap_index(session, self.pcap)
        self.off, self.ln, self.ht, self.port, self.lts, self.mts = (idx[k] for k in
                                                                     ("off", "ln", "ht", "port", "lts", "mts"))
        self.to_mono = lidar_to_mono(idx, offset_s)
        self.sweeps = []                    # (t_mono, raw points N x 6), last ~1.5 s
        self.slock = threading.Lock()

    def span(self):
        return float(self.ht[0]) / 1e9 + self.offset_s, float(self.ht[-1]) / 1e9 + self.offset_s

    def set_view(self, **changes):
        """Change the LiDAR view; the current sweep is re-rendered at once."""
        with self.vlock:
            self.view.update(changes)
            self.view["version"] += 1

    def _detect(self, sweep):
        """Objects and confirmed tracks for a sweep, once per sweep (view changes reuse it)."""
        if self._det_cache[0] == sweep[0] and (not self.want_labels or self._det_cache[1].get("point_label")
                                               is not None):
            return self._det_cache[1]
        sys.path.insert(0, str(PIPELINE))
        from perception.lidar import CLASSES, LIDAR_TRACKER, O, LidarDetector
        from perception.tracking import Tracker
        if self._tracker is None:
            self._detector = LidarDetector(near_m=max(0.05, float(self.view["near_m"])))
            self._tracker = Tracker(**LIDAR_TRACKER)
        t = sweep[0]
        tic = time.perf_counter()
        res = self._detector.detect(sweep[1], labels=self.want_labels)
        if self._det_cache[0] == t:             # same sweep again, now with labels: tracks unchanged
            res["tracks"], res["ms_total"] = self._det_cache[1]["tracks"], 1e3 * (time.perf_counter() - tic)
            self._det_cache = (t, res)
            return res
        if self._trk_gen != self.clock.seek_gen or (self._trk_t is not None and not 0 < t - self._trk_t < 0.5):
            self._tracker.reset()               # a seek or a jump: tracks start over
        self._trk_gen, self._trk_t = self.clock.seek_gen, t
        ob = res["objects"]
        mov = ob[ob[:, O["cls"]] != CLASSES.index("structure")]
        self._tracker.update(t, mov[:, [O["x"], O["y"]]], list(mov[:, O["cls"]]))
        res["tracks"] = [(tr.id, tr.x[0], tr.x[1], tr.x[2], tr.x[3], int(tr.payload), tr.hits, tr.motion)
                         for tr in self._tracker.reported()]
        res["ms_total"] = 1e3 * (time.perf_counter() - tic)
        self._det_cache = (t, res)
        return res

    def _render(self, sweep):
        with self.vlock:
            view = dict(self.view)
        det = None
        if self.detect:
            try:
                det = self._detect(sweep)
            except Exception as exc:            # detection must never stop the replay
                self.error = f"lidar detection: {exc}"
        rgb, extent, shown = render_lidar(sweep[1], view, None if det is None else det["ground_z"])
        # Raw points for the 3D view: valid returns only (the RS-Fairy fills
        # no-return directions with zero range), beyond the near filter.
        p = sweep[1]
        ok = np.isfinite(p[:, 3]) & (p[:, 3] > max(0.05, float(view["near_m"])))
        lab = None if det is None or det.get("point_label") is None else det["point_label"][ok]
        with self.vlock:
            self.points = (sweep[0], np.ascontiguousarray(p[ok, :5]), lab,
                           None if det is None else det["ground_z"])
        return {"t": sweep[0], "rgb": rgb, "n_shown": shown, "ext": extent, "mode": view["mode"],
                "version": view["version"], "det": det}

    def get_points(self):
        """(time, N x 5 [x, y, z, range, intensity], point labels or None, ground z or None)
        of the sweep last published, or None."""
        with self.vlock:
            return getattr(self, "points", None)

    def run(self):
        tmp = Path(tempfile.mkdtemp(prefix="replay_lidar_"))   # decode config + FIFOs
        try:
            Lidar = load_wrapper()
            cfg_path = decode_config(self.session, tmp / "decode.yaml")
            while not self.stop_evt.is_set():
                gen = self.clock.seek_gen
                self._play_from(self.clock.now(), tmp, cfg_path, gen, Lidar)
                # returns on seek, stop or end of data
                while (not self.stop_evt.is_set() and gen == self.clock.seek_gen):
                    time.sleep(0.05)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = f"lidar: {exc}"
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    def _play_from(self, t_start, tmp, cfg_path, gen, Lidar):
        # Start 0.3 s early: the first sweep after a restart is partial, so
        # without a pre-roll nothing would be ready to show at a paused seek.
        host_start = (t_start - self.offset_s - self.PREROLL_S) * 1e9
        first = int(np.searchsorted(self.ht, host_start))
        if first >= len(self.ht):
            return
        difops = np.where((self.port[:first] == 7788))[0]
        fifo = tmp / f"feed_{gen}.pcap"
        os.mkfifo(fifo)
        with self.slock:
            self.sweeps = []

        def on_cloud(points, ts, seq):
            p = np.asarray(points)
            if ts <= 0 or len(p) == 0:
                return
            item = (float(self.to_mono(sweep_time(p, ts))), p.astype(np.float32, copy=True))
            with self.slock:
                self.sweeps.append(item)
                self.sweeps = self.sweeps[-15:]

        lidar = Lidar(str(cfg_path))
        lidar.on_pointcloud(on_cloud)
        done = threading.Event()

        feed_stop = threading.Event()

        def feeder():
            # Non-blocking writes with a short timeout, so a stop or seek is
            # noticed within 0.1 s even when rs_driver is not reading. Closing
            # our end gives rs_driver a normal end-of-file, after which it can
            # be stopped cleanly.
            fd = None
            try:
                while fd is None and not (feed_stop.is_set() or self.stop_evt.is_set()):
                    try:
                        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)    # ENXIO until rs_driver opens it
                    except OSError:
                        time.sleep(0.02)
                if fd is None:
                    return

                def send(data):
                    view = memoryview(data)
                    while view and not feed_stop.is_set():
                        if select.select([], [fd], [], 0.1)[1]:
                            try:
                                view = view[os.write(fd, view):]
                            except BlockingIOError:
                                pass

                send(bytes(self.pcap[:24]))
                if len(difops):                      # rs_driver waits for device info first
                    d = difops[-1]
                    send(bytes(self.pcap[self.off[d]:self.off[d] + self.ln[d]]))
                i = first
                while (i < len(self.off) and not feed_stop.is_set() and not self.stop_evt.is_set()
                       and gen == self.clock.seek_gen):
                    limit = (self.clock.now() - self.offset_s + self.LOOKAHEAD_S) * 1e9
                    j = int(np.searchsorted(self.ht, limit, side="right"))
                    if j > i:
                        send(bytes(self.pcap[self.off[i]:self.off[j - 1] + self.ln[j - 1]]))
                        i = j
                    else:
                        time.sleep(0.01)
            except OSError:
                pass
            finally:
                if fd is not None:
                    os.close(fd)
                done.set()

        feed_thread = threading.Thread(target=feeder, daemon=True, name="lidar-feeder")
        feed_thread.start()
        lidar.start(pcap_path=str(fifo), pcap_rate=100.0)
        while not done.is_set() and not self.stop_evt.is_set() and gen == self.clock.seek_gen:
            # publish the newest sweep whose boresight time has been reached
            with self.slock:
                ready = [s for s in self.sweeps if s[0] <= self.clock.now()]
            latest = self.latest
            if ready and (latest is None or ready[-1][0] != latest["t"] or latest["version"] != self.view["version"]
                          or (self.want_labels and latest["det"] is not None
                              and latest["det"].get("point_label") is None)):
                self.publish(self._render(ready[-1]))        # new sweep, or the view changed
            time.sleep(0.01)
        feed_stop.set()
        feed_thread.join(timeout=2.0)        # our end closed -> rs_driver sees end-of-file
        lidar.stop()
        try:
            fifo.unlink()
        except OSError:
            pass
