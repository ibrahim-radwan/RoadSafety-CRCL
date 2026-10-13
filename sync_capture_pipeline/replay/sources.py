"""
Replay sources: a shared clock and one thread per sensor.

Every source follows the same ReplayClock (CLOCK_MONOTONIC session time) and
keeps its LATEST product ready for the GUI. The GUI never waits on a sensor.

    CameraSource  ZED SDK replays the SVO2 (decoded on the GPU); the frame
                  shown is the last one captured at or before the clock.
    RadarSource   each radar frame is processed on the GPU with PyTorch:
                  range FFT -> virtual array -> Doppler FFT -> azimuth FFT.
                  Products: range-Doppler map and a bird's-eye (BEV) map.
    LidarSource   rs_driver decodes lidar.pcap through a named pipe that a
                  feeder thread fills in step with the clock (pause, speed
                  and seek work; rs_driver alone can only play a file from
                  the start). Product: BEV point density, per sweep.

Times come from sync/ (align_session.py), so offsets applied there apply
here too.
"""
from __future__ import annotations

import csv
import json
import os
import select
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
ROOT = PIPELINE.parent
RADAR_PKG = ROOT / "cascaded_radar_capture_pipeline"
sys.path.insert(0, str(PIPELINE))
import radar_firmware                     # noqa: E402  (session -> firmware .cfg)
LIDAR_WRAPPER = ROOT / "lidar_standalone_capture" / "rs_driver" / "python_wrapper_2"
MSOP_ID = b"\x55\xaa\x05\x5a"


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

    def stop(self):
        self.stop_evt.set()


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
class RadarProcessor:
    """GPU (PyTorch) radar products for one raw frame, using the repo's own
    configuration, channel geometry and antenna calibration."""

    # The processing's azimuth sign is MIRRORED relative to the world: on
    # freewalk_ref_02 (2026-10-11) a walker 3 m to the right at 8 m (camera and
    # LiDAR agree) came out on the left, and vice versa. The raw data is fine;
    # the TX/RX -> azimuth table in radar_config is unverified. Until the team's
    # antenna calibration settles it, the BEV lookup flips the sign.
    MIRROR_AZIMUTH = True

    def __init__(self, bev_x, bev_y, cfg_path: Path, device="cuda", mirror_azimuth: bool = None):
        import torch
        sys.path.insert(0, str(RADAR_PKG))
        from processing.radar_config import parse_calibration, parse_cfg_file
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):       # the parser is verbose
            cfg = parse_cfg_file(str(cfg_path))
            corr = parse_calibration(str(cfg_path), cfg.num_tx, cfg.num_rx)
        self.cfg, self.torch = cfg, torch
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_tx, self.n_rx, self.n_loops, self.n_s = cfg.num_tx, cfg.num_rx, cfg.num_loops, cfg.num_samples
        self.n_dev, self.rx_per_dev = cfg.num_devices, cfg.num_rx_per_dev
        self.corr = torch.as_tensor(np.asarray(corr), dtype=torch.complex64, device=self.dev)
        # Uniform azimuth array: virtual channels placed by their az position;
        # overlapping positions are averaged.
        ch_az, _, _ = cfg.build_channel_table()
        n_pos = max(c["az"] for c in ch_az) + 1
        self.n_pos = n_pos
        self.ch_tx = torch.tensor([c["tx"] for c in ch_az], device=self.dev)
        self.ch_rx = torch.tensor([c["rx"] for c in ch_az], device=self.dev)
        self.ch_pos = torch.tensor([c["az"] for c in ch_az], device=self.dev)
        counts = np.bincount([c["az"] for c in ch_az], minlength=n_pos).astype(np.float32)
        self.pos_w = torch.as_tensor(1.0 / np.maximum(counts, 1), device=self.dev)
        self.win_r = torch.hann_window(self.n_s, periodic=False, device=self.dev)
        self.win_d = torch.hann_window(self.n_loops, periodic=False, device=self.dev)
        self.win_a = torch.hann_window(n_pos, periodic=False, device=self.dev)
        self.n_az = cfg.az_fft_size
        self.range_axis = np.arange(cfg.n_range) * cfg.range_res_m
        self.az_axis = np.asarray(cfg.az_axis_deg)              # deg, ascending
        self.dop_axis = np.asarray(cfg.doppler_axis)
        # BEV lookup: each (x forward, y left) cell -> (range bin, az bin).
        X, Y = np.meshgrid(bev_x, bev_y, indexing="ij")
        r = np.hypot(X, Y)
        az = np.degrees(np.arctan2(Y, X))
        self.mirror_azimuth = self.MIRROR_AZIMUTH if mirror_azimuth is None else mirror_azimuth
        az_lookup = -az if self.mirror_azimuth else az
        ri = np.round(r / cfg.range_res_m).astype(int)
        ai = np.clip(np.searchsorted(self.az_axis, az_lookup), 0, self.n_az - 1)
        valid = (X > 0) & (ri < cfg.n_range) & (np.abs(az) <= 85.0)
        self.bev_ri = torch.as_tensor(np.where(valid, ri, 0), device=self.dev)
        self.bev_ai = torch.as_tensor(np.where(valid, ai, 0), device=self.dev)
        self.bev_valid = torch.as_tensor(valid, device=self.dev)
        self.iq_swap = bool(cfg.iq_swap)

    def process(self, raw: np.ndarray, moving_only: bool = True):
        torch = self.torch
        x = torch.as_tensor(raw, device=self.dev).view(self.n_tx * self.n_loops, self.n_dev * self.rx_per_dev,
                                                      self.n_s, 2).float()
        iq = torch.complex(x[..., 1], x[..., 0]) if self.iq_swap else torch.complex(x[..., 0], x[..., 1])
        # TDM: chirp = loop * n_tx + tx  ->  [tx, loop, rx, sample]
        cube = iq.view(self.n_loops, self.n_tx, self.n_rx, self.n_s).permute(1, 0, 2, 3)
        cube = cube * self.corr[:, None, :, None]
        rng = torch.fft.fft(cube * self.win_r, dim=-1)                       # range
        virt = rng[self.ch_tx, :, self.ch_rx, :]                              # [ch, loop, range]
        arr = torch.zeros((self.n_pos, self.n_loops, rng.shape[-1]), dtype=virt.dtype, device=self.dev)
        arr.index_add_(0, self.ch_pos, virt)
        arr = arr * self.pos_w[:, None, None]                                 # uniform array
        if moving_only:
            arr = arr - arr.mean(dim=1, keepdim=True)                         # remove static (zero-Doppler)
        dop = torch.fft.fftshift(torch.fft.fft(arr * self.win_d[None, :, None], dim=1), dim=1)
        rd = 10 * torch.log10((dop.abs() ** 2).sum(dim=0) + 1e-6)             # [doppler, range]
        az = torch.fft.fftshift(torch.fft.fft(dop * self.win_a[:, None, None], n=self.n_az, dim=0), dim=0)
        ra = 10 * torch.log10((az.abs() ** 2).sum(dim=1) + 1e-6)              # [az, range]
        self.last_ra = ra.cpu().numpy()          # range-azimuth power, for the raw radar point cloud
        bev = ra[self.bev_ai, self.bev_ri]
        bev = torch.where(self.bev_valid, bev, torch.full_like(bev, float("nan")))
        return rd.cpu().numpy(), bev.cpu().numpy()


class RadarSource(Source):
    name_ = "radar"

    def __init__(self, clock, session: Path, bev_x, bev_y, moving_only=True):
        super().__init__(clock)
        self.session = session
        rows = load_csv(session / "sync" / "radar_timeline.csv")
        self.t = np.array([float(r["t_ref"]) for r in rows])
        self.files = [r["filename"] for r in rows]
        variant, how = radar_firmware.session_variant(session)
        print(f"[radar] firmware {variant} ({how})")
        self.proc = RadarProcessor(bev_x, bev_y, radar_firmware.cfg_for_variant(variant))
        al_path = session / "sync" / "alignment.json"
        self.timing_source = (json.loads(al_path.read_text()).get("radar", {}).get("timing_source", "")
                              if al_path.is_file() else "")
        self.moving_only = moving_only
        self.ms_per_frame = 0.0
        # Raw point cloud: one point per range-azimuth cell, world azimuth = the
        # processing's azimuth mirrored (MIRROR_AZIMUTH), z = 0 (no elevation here).
        az = np.radians(-self.proc.az_axis if self.proc.mirror_azimuth else self.proc.az_axis)
        r = np.asarray(self.proc.range_axis)
        self.ra_x = (np.cos(az)[:, None] * r[None, :]).astype(np.float32)     # [az, range]
        self.ra_y = (np.sin(az)[:, None] * r[None, :]).astype(np.float32)
        self.ra_r = np.broadcast_to(r[None, :], self.ra_x.shape)
        # Filtered point cloud: the team's process_frame, pre-computed by radar_cloud.py
        self.cloud = None
        cp = session / "sync" / "radar_cloud_filtered.npz"
        if cp.is_file():
            d = np.load(cp)
            if len(d["start"]) == len(self.files) + 1:
                self.cloud = {k: d[k] for k in ("start", "xyz", "doppler", "power")}

    def raw_points(self, ra, thr_db: float, max_range: float):
        """Cells more than thr_db above this frame's median, within max_range: (xyz N x 3, dB above floor)."""
        rel = ra - np.median(ra)
        m = (rel > thr_db) & (self.ra_r <= max_range) & (self.ra_r > 0.5)
        return np.c_[self.ra_x[m], self.ra_y[m], np.zeros(int(m.sum()), np.float32)], rel[m]

    def filtered_points(self, k: int):
        """(xyz N x 3, doppler, power) of radar frame k from the team pipeline, or None if not computed."""
        if self.cloud is None:
            return None
        a, b = int(self.cloud["start"][k]), int(self.cloud["start"][k + 1])
        return self.cloud["xyz"][a:b], self.cloud["doppler"][a:b], self.cloud["power"][a:b]

    def span(self):
        return float(self.t[0]), float(self.t[-1])

    def run(self):
        try:
            k_done, mode = -1, None
            while not self.stop_evt.is_set():
                k = int(np.searchsorted(self.t, self.clock.now(), side="right")) - 1
                if k < 0:                               # before the radar's first frame
                    if k_done != -1:
                        k_done = -1
                        self.publish(None)
                    time.sleep(0.01)
                    continue
                if k == k_done and mode == self.moving_only:
                    time.sleep(0.004)
                    continue
                raw = np.fromfile(self.session / "radar" / self.files[k], dtype=np.int16)
                t0 = time.perf_counter()
                rd, bev = self.proc.process(raw, self.moving_only)
                self.ms_per_frame = (time.perf_counter() - t0) * 1e3
                k_done, mode = k, self.moving_only
                self.publish((float(self.t[k]), k, rd, bev, self.proc.last_ra))
        except Exception as exc:
            self.error = f"radar: {exc}"


# ── LiDAR ───────────────────────────────────────────────────────────────────
# ── LiDAR view rendering ────────────────────────────────────────────────────
LIDAR_VIEWS = ("top", "top360", "panorama")
LIDAR_COLORS = ("height", "intensity", "density")
_LUTS = {}


def _lut(name: str) -> np.ndarray:
    if name not in _LUTS:
        import matplotlib
        cmap = matplotlib.cm.get_cmap(name)
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


def render_lidar(pts: np.ndarray, view: dict):
    """pts: N x 6 (x, y, z, range, intensity, dt). Returns (rgb HxWx3, extent, n_shown).
    extent = (left, bottom, width, height) in plot coordinates: top views use
    plot-x = y (left +) and plot-y = x (forward); the panorama uses plot-x =
    azimuth (deg, left +) and plot-y = elevation (deg)."""
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
    R = float(view["range_m"])
    x0, x1, y0, y1 = (-R, R, -R, R) if view["mode"] == "top360" else (0.0, R, -R / 2, R / 2)
    res = max(x1 - x0, y1 - y0) / 350.0           # ~350 cells across the longer side
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
    else:                                         # highest point per cell, coloured by height
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
        self.view = dict(mode="top", range_m=40.0, color="height", near_m=0.5, version=0)
        self.view.update(view or {})
        self.vlock = threading.Lock()
        self.pcap = np.memmap(session / "lidar" / "lidar.pcap", dtype=np.uint8, mode="r")
        self._index()
        sys.path.insert(0, str(PIPELINE))
        import align_session as al
        a, b, l0, _, _ = al.envelope_fit(self.lts, self.mts / 1e9)
        self.to_mono = lambda tl: a * (np.asarray(tl) - l0) + b + offset_s
        self.sweeps = []                    # (t_mono, raw points N x 6), last ~1.5 s
        self.slock = threading.Lock()

    def _index(self):
        """Packet table: byte offset, length, host CLOCK_MONOTONIC ns, port.
        Cached in sync/ - building it reads the whole PCAP once."""
        cache = self.session / "sync" / "lidar_pcap_index.npz"
        if cache.is_file():
            z = np.load(cache)
            self.off, self.ln, self.ht, self.port, self.lts, self.mts = (z[k] for k in
                                                                         ("off", "ln", "ht", "port", "lts", "mts"))
            return
        buf = self.pcap
        off, ln, ht, port, lts, mts = [], [], [], [], [], []
        o, n = 24, len(buf)
        while o + 16 <= n:
            sec, nsec, incl, _ = struct.unpack_from("<IIII", buf, o)
            if o + 16 + incl > n:
                break
            t = sec * 1_000_000_000 + nsec
            p = struct.unpack_from("!H", buf, o + 16 + 36)[0]
            off.append(o); ln.append(16 + incl); ht.append(t); port.append(p)
            pl = o + 16 + 42
            if bytes(buf[pl:pl + 4]) == MSOP_ID:
                lts.append(int.from_bytes(bytes(buf[pl + 20:pl + 26]), "big")
                           + int.from_bytes(bytes(buf[pl + 26:pl + 30]), "big") * 1e-6)
                mts.append(t)
            o += 16 + incl
        self.off, self.ln, self.ht, self.port = (np.array(off), np.array(ln), np.array(ht, dtype=np.int64),
                                                 np.array(port))
        self.lts, self.mts = np.array(lts), np.array(mts, dtype=np.int64)
        np.savez(cache, off=self.off, ln=self.ln, ht=self.ht, port=self.port, lts=self.lts, mts=self.mts)

    def span(self):
        return float(self.ht[0]) / 1e9 + self.offset_s, float(self.ht[-1]) / 1e9 + self.offset_s

    def set_view(self, **changes):
        """Change the LiDAR view; the current sweep is re-rendered at once."""
        with self.vlock:
            self.view.update(changes)
            self.view["version"] += 1

    def _render(self, sweep):
        with self.vlock:
            view = dict(self.view)
        rgb, extent, shown = render_lidar(sweep[1], view)
        # Raw points for the 3D view: valid returns only (the RS-Fairy fills
        # no-return directions with zero range), beyond the near filter.
        p = sweep[1]
        ok = np.isfinite(p[:, 3]) & (p[:, 3] > max(0.05, float(view["near_m"])))
        with self.vlock:
            self.points = (sweep[0], np.ascontiguousarray(p[ok, :5]))
        return (sweep[0], rgb, shown, extent, view["mode"], view["version"])

    def get_points(self):
        """(time, N x 5 [x, y, z, range, intensity]) of the sweep last published, or None."""
        with self.vlock:
            return getattr(self, "points", None)

    def run(self):
        tmp = Path(tempfile.mkdtemp(prefix="replay_lidar_"))   # decode config + FIFOs
        try:
            sys.path.insert(0, str(LIDAR_WRAPPER))
            import yaml
            from lidar import Lidar
            cfg = yaml.safe_load((self.session / "lidar" / "config_used.yaml").read_text())
            cfg["lidar"]["split_angle"] = 180.0
            cfg_path = tmp / "decode.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            while not self.stop_evt.is_set():
                gen = self.clock.seek_gen
                self._play_from(self.clock.now(), tmp, cfg_path, gen, Lidar)
                # returns on seek, stop or end of data
                while (not self.stop_evt.is_set() and gen == self.clock.seek_gen):
                    time.sleep(0.05)
        except Exception as exc:
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
            fwd = (p[:, 0] > 0) & (np.abs(p[:, 1]) <= 0.09 * p[:, 0])          # within ~5 deg ahead
            tb = ts + float(np.median(p[fwd, 5])) if fwd.any() else ts + 0.05
            item = (float(self.to_mono(tb)), p.astype(np.float32, copy=True))
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
            if ready and (latest is None or ready[-1][0] != latest[0] or latest[5] != self.view["version"]):
                self.publish(self._render(ready[-1]))        # new sweep, or the view changed
            time.sleep(0.01)
        feed_stop.set()
        feed_thread.join(timeout=2.0)        # our end closed -> rs_driver sees end-of-file
        lidar.stop()
        try:
            fifo.unlink()
        except OSError:
            pass
