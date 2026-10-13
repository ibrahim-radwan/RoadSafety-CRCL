"""
streaming.py  —  Live streaming pipeline for AWR2243-2X cascade radar
=====================================================================

Stage 1 of the offline → live streaming transition.

Architecture
------------

    FileMonitor (producer thread, in capture/dca.py)
        │  on_frame_saved(path, frame_idx, frame_bytes)
        ▼
    FrameQueue  (thread-safe, drop-oldest at depth=2)
        │  get() → raw bytes
        ▼
    StreamingProcessor (consumer thread)
        │  read_frame_from_bytes() → complex64 cube
        │  process_frame()         → ProcessingResult
        ▼
    ResultQueue (thread-safe, depth=1)
        │  get() → ProcessingResult
        ▼
    LiveViewer  (main thread, pyqtgraph non-blocking)
        │  update_plots() at display refresh rate

Design principles
-----------------
- FileMonitor is NOT modified.  It already has an on_frame_saved callback
  which is the correct hook for streaming.  No changes to capture/ code.
- process_frame() is NOT modified.  It takes a numpy array and returns
  ProcessingResult.  It is already stateless with respect to frame source.
- The queue depth is intentionally shallow (2 frames max).  If the consumer
  falls behind, the oldest unprocessed frame is dropped and a counter
  increments.  Frames are never accumulated.
- The LiveViewer uses pyqtgraph which uses Qt OpenGL for hardware-accelerated
  rendering.  On a machine with a GPU this happens automatically — no
  explicit CUDA or CuPy code is needed for the display layer.
- Warm-up frame discarding is handled by FileMonitor before on_frame_saved
  fires.  The streaming pipeline receives only post-warmup frames.

Bugs fixed in this revision (vs. original streaming.py)
---------------------------------------------------------
FIX 1 — Range axis shows ~250 m instead of ~16 m (critical)
  Root cause: pyqtgraph ImageItem was rendered in pixel-index space (0..N)
  rather than physical space (0..16.8 m) because no coordinate transform
  was applied after setImage().  matplotlib's imshow(extent=...) does this
  implicitly; pyqtgraph requires an explicit QTransform.
  Fix: _apply_axis_transform() builds and applies a QTransform after every
  setImage() call, mapping pixel indices to physical axis coordinates
  (metres, m/s, degrees).  PlotItem view ranges are also explicitly set to
  the physical axis extents so auto-ranging does not override the transform.

FIX 1 (continued) — CFAR markers missing or misplaced
  Root cause: CFAR scatter markers were placed at physical coordinates (e.g.
  r=3.5 m) while the RA ImageItem was in pixel-index space (row 53).  The
  markers existed but were off-screen relative to the image.
  Fix: Once Fix 1's transform is applied, the ImageItem and the scatter
  markers share the same physical coordinate system.  Markers now land on
  the correct pixels automatically.

FIX 2 — Frame counter shows display tick count, not processor frame index
  Root cause: self._frame_count was incremented on every QTimer tick
  (every 50 ms) regardless of whether a new frame had been processed.
  At 2600 ms processing time, the counter advanced ~52× faster than actual
  frame output, showing a number unrelated to the true processed frame index.
  Fix: self._frame_count now reads processor.frames_processed directly.
  LiveViewer.run() stores the processor reference as self._processor before
  the QTimer starts.  The status bar also now shows the drop count.

FIX 4 — Range/Doppler bins mirrored (range=34→222, dop=27→37)
  Root cause: The offline pipeline calls apply_raw_iq_order_override(cfg,
  "i-first") inside main.py, forcing cfg.iq_swap=False regardless of
  what adcbufCfg specifies.  live_streaming_test.py did not call this
  function, so cfg.iq_swap was set from adcbufCfg SampleSwap buf[2]=1
  → iq_swap=True.  This conjugated every IQ sample, which flips both
  FFT spectra simultaneously: range bin k → (N-k) and Doppler bin d →
  (M-d).  For N=256, k=34: 256-34=222.  For M=64, d=27: 64-27=37.
  Both match the observed streaming log values exactly.
  Fix: run_live() now accepts raw_iq_order="i-first" (default) and
  applies apply_raw_iq_order_override() before constructing the
  processor, matching the offline pipeline's behaviour unconditionally.


  Root cause: FolderFeeder used frame_rate_hz=4.0 (250 ms interval) while
  process_frame() took ~2600 ms, so ~10 frames were fed and dropped for
  every one processed.  This caused the streaming pipeline to process a
  non-representative subset of the folder's frames.
  Fix: run_live() now accepts auto_rate=True (default).  Before starting
  FolderFeeder, it measures process_frame() latency on one real frame and
  sets the feed rate to latency_headroom_factor (80%) of the actual
  processing speed.  For a 2600 ms latency this gives ~0.31 fps feed rate,
  matching the processor and eliminating drops.

Dependencies
------------
  pip install pyqtgraph pyqt5 numpy

Usage
-----
  from processing.streaming import run_live
  from processing.radar_config import parse_cfg_file, parse_calibration

    cfg  = parse_cfg_file("firmware/r017m/cascade_r017m.cfg")
    corr = parse_calibration("firmware/r017m/cascade_r017m.cfg", cfg.num_tx, cfg.num_rx)

  # Development mode — validate against existing frames (auto_rate=True by default):
  run_live(cfg, corr,
           frame_folder=Path("capture/frames/moving_person_3-4m_frames_Truth"),
           loop=True, quiet=False)

  # Live hardware mode — wire to FileMonitor:
  fq = run_live(cfg, corr)
  on_frame_saved = make_on_frame_saved(fq)
  run_capture(settings, on_frame_saved=on_frame_saved, ...)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# PyOpenGL compatibility fix — must be done before ANY OpenGL import.
# PyOpenGL_accelerate has a numpy ABI mismatch with NumPy >= 1.20 that
# causes "numpy.dtype size changed" ValueError in GLScatterPlotItem.
# Disabling the accelerator here forces pure-Python OpenGL bindings.
# ---------------------------------------------------------------------------
import os as _os
_os.environ.setdefault("PYOPENGL_PLATFORM", "")
try:
    import OpenGL as _ogl
    _ogl.USE_ACCELERATE = False
except ImportError:
    pass

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Callable

import numpy as np


def _make_qapp(pg, title):
    """Create Qt's application without exposing our --display option to Qt."""
    original_argv = sys.argv
    filtered_argv = [original_argv[0]]
    skip_next = False
    for argument in original_argv[1:]:
        if skip_next:
            skip_next = False
        elif argument == "--display":
            skip_next = True
        elif not argument.startswith("--display="):
            filtered_argv.append(argument)

    sys.argv = filtered_argv
    try:
        return pg.mkQApp(title)
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# FrameQueue — thread-safe, drop-oldest, shallow
# ---------------------------------------------------------------------------

class FrameQueue:
    """
    Thread-safe queue with a hard depth limit and drop-oldest semantics.

    When the producer is faster than the consumer (e.g. a burst of frames
    arrives while process_frame is running) the oldest unprocessed frame
    is dropped and a counter increments.  This keeps the consumer operating
    on the most recent data rather than falling arbitrarily far behind.

    max_depth=2 means at most one frame is waiting while another is being
    processed.  At 4 fps the processing window is 256 ms; process_frame
    takes roughly 50-150 ms on CPU.  Depth 2 gives one frame of headroom.
    """

    def __init__(self, max_depth: int = 2):
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=max_depth)
        self.dropped: int = 0          # frames dropped due to queue full
        self.received: int = 0         # frames received from producer
        self._lock = threading.Lock()

    def put(self, frame_bytes: bytes) -> None:
        """Non-blocking put. Drops oldest frame if queue is full."""
        with self._lock:
            self.received += 1
            if self._q.full():
                try:
                    self._q.get_nowait()   # discard oldest
                    self.dropped += 1
                except queue.Empty:
                    pass
            try:
                self._q.put_nowait(frame_bytes)
            except queue.Full:
                self.dropped += 1         # race: still full after get

    def get(self, timeout: float = 0.1) -> Optional[bytes]:
        """Blocking get with timeout. Returns None on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def status(self) -> str:
        return (f"FrameQueue: received={self.received}  "
                f"dropped={self.dropped}  "
                f"qsize={self._q.qsize()}")


# ---------------------------------------------------------------------------
# Frame reader — bytes → complex64 cube
# ---------------------------------------------------------------------------

def _read_frame_from_bytes(
    raw_bytes: bytes,
    cfg: "RadarConfig",         # type: ignore[name-defined]
) -> np.ndarray:
    """
    Convert a raw frame byte buffer to a complex64 cube.

    Shape: [num_tx, num_loops, num_rx, num_samples]

    This mirrors exactly what radar_reader.read_frame() does when reading
    from a frame_*.bin file.  Keeping the logic here (rather than calling
    read_frame with a temp file) avoids disk I/O in the hot path.

    The reshape assumes device-major, rx-within-device-minor LVDS interleave
    (TI SWRU546 §3.2).  This is the same assumption validated in
    diag_frame_stationarity.py and confirmed correct for this capture setup.
    """
    n_int16 = cfg.chirps_per_frame * cfg.num_rx * cfg.num_samples * 2
    expected_bytes = n_int16 * 2

    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Frame byte length mismatch: got {len(raw_bytes)}, "
            f"expected {expected_bytes} "
            f"(chirps={cfg.chirps_per_frame} rx={cfg.num_rx} "
            f"samples={cfg.num_samples})"
        )

    raw = np.frombuffer(raw_bytes, dtype=np.int16).copy()
    raw = raw.reshape(
        cfg.chirps_per_frame,
        cfg.num_devices,
        cfg.num_rx_per_dev,
        cfg.num_samples,
        2,
    )

    if cfg.iq_swap:
        i_ch = raw[..., 1].astype(np.float32)
        q_ch = raw[..., 0].astype(np.float32)
    else:
        i_ch = raw[..., 0].astype(np.float32)
        q_ch = raw[..., 1].astype(np.float32)

    cdata = (i_ch + 1j * q_ch).astype(np.complex64)
    cdata = cdata.reshape(cfg.chirps_per_frame, cfg.num_rx, cfg.num_samples)

    # TDM separation: chirp_idx = loop * num_tx + tx
    return np.stack(
        [cdata[t::cfg.num_tx] for t in range(cfg.num_tx)],
        axis=0,
    ).astype(np.complex64)


# ---------------------------------------------------------------------------
# ResultQueue — single-slot, always fresh
# ---------------------------------------------------------------------------

class ResultQueue:
    """
    Single-slot result buffer.  Always holds the most recent ProcessingResult.

    The display thread reads this at its own refresh rate.  If the display
    is slower than the processor, it still always shows the latest result
    rather than queued stale results.  If the display is faster, it simply
    redraws the same result.
    """

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, result) -> None:
        with self._lock:
            self._result = result
        self._event.set()

    def get(self, timeout: float = 0.1):
        """Returns most recent result, or None on timeout."""
        if self._event.wait(timeout):
            with self._lock:
                r = self._result
            self._event.clear()
            return r
        return None

    def peek(self):
        """Non-blocking read of the current result without clearing the event."""
        with self._lock:
            return self._result


# ---------------------------------------------------------------------------
# StreamingProcessor — consumer thread
# ---------------------------------------------------------------------------

class StreamingProcessor(threading.Thread):
    """
    Consumer thread: reads raw frame bytes from FrameQueue, converts to a
    complex cube, calls process_frame(), and pushes the result to ResultQueue.

    Runs as a daemon thread so it exits when the main thread exits.
    Stop by calling stop() or setting stop_event.
    """

    def __init__(
        self,
        frame_queue: FrameQueue,
        result_queue: ResultQueue,
        cfg: "RadarConfig",             # type: ignore[name-defined]
        correction: np.ndarray,
        *,
        quiet: bool = True,
    ):
        super().__init__(daemon=True, name="StreamingProcessor")
        self._fq = frame_queue
        self._rq = result_queue
        self._cfg = cfg
        self._correction = correction
        self._quiet = quiet
        self.stop_event = threading.Event()
        self.frames_processed: int = 0
        self.frames_errored: int = 0
        # ── P0-6 subject-proximity gate ──────────────────────────────────────
        # Current tracked subject position (x, y) in metres, or None.
        # Written by LiveViewer (GUI thread) after each accumulator update,
        # read here (worker thread) and handed to process_frame(track_hint=).
        # A plain tuple assignment is atomic under the GIL, so no lock is
        # needed; the value may be one frame stale, which is immaterial
        # because the gate radius (0.90 m) far exceeds one frame of motion
        # (0.30 m at 1.2 m/s).
        self.track_hint = None
        self._process_times: list[float] = []   # rolling last-10 latencies

    def run(self) -> None:
        from .radar_processing import process_frame   # local import avoids circular

        # Pre-allocate CuPy memory pool before the first frame is processed.
        # This eliminates the period-3 ~300ms GC synchronization spikes caused
        # by the pool allocator calling cudaDeviceSynchronize() when it runs
        # out of a required block size mid-frame.
        # warmup_gpu_memory_pool is a no-op when GPU is unavailable.
        try:
            from .gpu_accel import warmup_gpu_memory_pool
            warmup_gpu_memory_pool(self._cfg)
        except Exception as _wu_err:
            print(f"[STREAM] GPU warmup skipped: {_wu_err}", flush=True)

        while not self.stop_event.is_set():
            raw = self._fq.get(timeout=0.1)
            if raw is None:
                continue
            try:
                t0 = time.perf_counter()
                cube = _read_frame_from_bytes(raw, self._cfg)
                result = process_frame(
                    cube, self._cfg, self._correction,
                    verbose=False,          # never verbose in streaming
                    track_hint=self.track_hint,   # P0-6 subject gate
                )
                dt = time.perf_counter() - t0
                self._process_times.append(dt)
                if len(self._process_times) > 10:
                    self._process_times.pop(0)
                self.frames_processed += 1
                self._rq.put(result)
                if not self._quiet:
                    from .radar_plots import _det_counts, _det_label
                    rd, ra, cloud = _det_counts(result)
                    avg_ms = 1000 * sum(self._process_times) / len(self._process_times)
                    print(
                        f"[STREAM] frame={self.frames_processed:4d}  "
                        f"{_det_label(rd, ra, cloud)}  "
                        f"proc={dt*1000:.0f}ms  avg={avg_ms:.0f}ms  "
                        f"dropped={self._fq.dropped}",
                        flush=True,
                    )
            except Exception as exc:
                self.frames_errored += 1
                print(f"[STREAM] frame error (#{self.frames_errored}): {exc}", flush=True)

    def stop(self) -> None:
        self.stop_event.set()

    def avg_latency_ms(self) -> float:
        if not self._process_times:
            return 0.0
        return 1000 * sum(self._process_times) / len(self._process_times)


# ---------------------------------------------------------------------------
# LiveViewer — pyqtgraph non-blocking display
# ---------------------------------------------------------------------------

class LiveViewer:
    """
    Non-blocking live display using pyqtgraph.

    Two display modes selected via the `display_mode` parameter:

    "heatmap"    (default) — three side-by-side heatmaps:
                   [Range-Doppler] [Range-Elevation] [Range-Azimuth]
                   with CFAR detection marker overlay on the RA plot.

    "pointcloud" — full-window interactive 3D scatter plot (pyqtgraph GL).
                   Each detected point is rendered as a sphere coloured by
                   Doppler velocity (blue = approaching, red = receding,
                   white = stationary).  Requires CFAR to be enabled.
                   Uses pyqtgraph.opengl (GLViewWidget + GLScatterPlotItem).
    """

    def __init__(
        self,
        result_queue: ResultQueue,
        cfg: "RadarConfig",             # type: ignore[name-defined]
        *,
        pause_ms: int = 50,
        title: str = "AWR2243-2X Live",
        display_mode: str = "heatmap",  # "heatmap" | "pointcloud"
    ):
        self._rq = result_queue
        self._cfg = cfg
        self._pause_ms = pause_ms
        self._title = title
        self._display_mode = display_mode.lower().strip()
        self._win = None
        self._plots: dict = {}
        self._last_result = None
        self._frame_count = 0
        self._processor: Optional[StreamingProcessor] = None

        # Rolling vmax history for colour scale stabilisation (heatmap mode)
        self._vmax_history: list[float] = []
        self._vmax_window = 8

        # Point cloud history — accumulate last N frames for persistence effect
        self._cloud_history: list[np.ndarray] = []
        self._cloud_history_frames = int(
            getattr(cfg, 'pointcloud_history_frames', 5)
        )

        # Phase 2: motion-compensated temporal accumulator
        # Active when cfg.temporal_accum_en=True; falls back to naive concat otherwise
        _accum_en = bool(getattr(cfg, 'temporal_accum_en', False))
        if _accum_en:
            try:
                from processing.accumulate import TemporalAccumulator
                self._accumulator = TemporalAccumulator(
                    window             = int(getattr(cfg, 'accum_window', 16)),
                    max_jump_m         = float(getattr(cfg, 'accum_max_centroid_jump_m', 0.6)),
                    alpha              = float(getattr(cfg, 'accum_alpha', 0.3)),
                    frame_rate_hz      = float(getattr(cfg, 'frame_rate_hz', 4.0)),
                    dbscan_eps         = float(getattr(cfg, 'dbscan_eps', 0.80)),
                    dbscan_min_samples = int(getattr(cfg, 'dbscan_min_samples', 3)),
                    max_cluster_radius = float(getattr(cfg, 'max_cluster_radius', 0.65)),
                    subject_frame_en   = bool(getattr(cfg, 'subject_frame_en', False)),
                    gait_features_en   = bool(getattr(cfg, 'gait_features_en', False)),
                    dop_limb_thr       = float(getattr(cfg, 'dop_limb_thr', 0.30)),
                )
            except ImportError:
                self._accumulator = None
        else:
            self._accumulator = None

        # Phase 6: micro-Doppler spectrogram buffer
        _md_en = bool(getattr(cfg, 'microdoppler_en', False))
        if _md_en:
            try:
                from processing.microdoppler import MicroDopplerBuffer
                self._mdb = MicroDopplerBuffer(
                    time_frames   = int(getattr(cfg, 'spectrogram_frames', 64)),
                    v_max         = float(getattr(cfg, 'spectrogram_v_max', 1.5)),
                    n_dop_bins    = 40,
                    frame_rate_hz = float(getattr(cfg, 'frame_rate_hz', 4.0)),
                )
            except ImportError:
                self._mdb = None
        else:
            self._mdb = None

        # ── Centroid-trail ring buffer (Panel 2) ─────────────────────────────
        # Initialised here, not in _build_pointcloud_window(), so the state
        # exists in every display mode and cannot raise AttributeError.
        # Written at self._trail_ptr; read time-ordered via np.roll (see
        # _update_pointcloud). Audit P0-4b: reading in raw array order
        # scrambles the fade once the buffer wraps.
        self._TRAIL_LEN = 24
        self._trail_x   = np.full(self._TRAIL_LEN, np.nan, dtype=np.float32)
        self._trail_y   = np.full(self._TRAIL_LEN, np.nan, dtype=np.float32)
        self._trail_ptr = 0

    def _reset_trail(self) -> None:
        """Clear the centroid trail (call on track loss or replay restart)."""
        self._trail_x[:] = np.nan
        self._trail_y[:] = np.nan
        self._trail_ptr  = 0

    def _smooth_vmax(self, raw_vmax: float) -> float:
        self._vmax_history.append(raw_vmax)
        if len(self._vmax_history) > self._vmax_window:
            self._vmax_history.pop(0)
        return float(np.median(self._vmax_history))

    def _build_window(self):
        """Build the pyqtgraph window. Branches on self._display_mode."""
        if self._display_mode == "focused":
            self._build_focused_window()
        elif self._display_mode == "pointcloud":
            self._build_pointcloud_window()
        elif self._display_mode == "volume":
            self._build_volume_window()
        else:
            self._build_heatmap_window()


    # ══════════════════════════════════════════════════════════════════════════
    #  FOCUSED DISPLAY  —  one panel, only what the pipeline reliably knows
    # ══════════════════════════════════════════════════════════════════════════
    #  The 5-panel view drew RAW POINTS, which is exactly the operator
    #  complaint: "the points are just visualized maybe close to each other".
    #  It also devoted two panels to the two things measurement showed are NOT
    #  reliable on this hardware:
    #      body-shape ellipse : inter-subject dx_std 0.0037 m, below resolution
    #      spectrogram/cadence: unvalidated, no ground truth
    #
    #  This view answers only the questions the pipeline can answer, each
    #  backed by a measured result on the 12 recorded sessions:
    #      presence   detection 97.7-100% on all 10 valid sessions
    #      position   cluster centroid, 82 jumps / ~3400 frames = 2.4%
    #      direction  4/4 sessions correct (Stage 4.4 zero-crossing check)
    #      speed      dop_centroid, full +/-2.498 m/s range captured
    #      activity   limb_frac 20-32% unsaturated after Stage 4.2
    #
    #  It deliberately shows NOTHING about identity, body shape or subject
    #  count. Showing a body outline the radar cannot resolve would invite
    #  conclusions the data does not support.
    def _build_focused_window(self):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        import numpy as _np      # hoisted: used by both plots below
        pg.setConfigOptions(antialias=True)
        self._win = QtWidgets.QWidget()
        self._win.setWindowTitle("AWR2243-2X  —  Subject Tracking")
        self._win.resize(1080, 760)
        self._win.setStyleSheet("background-color:#0d0f12;")

        root = QtWidgets.QVBoxLayout(self._win)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # ── Status strip: the answer, in words, before any graphics ──────────
        self._f_status = QtWidgets.QLabel("WAITING FOR DATA")
        self._f_status.setAlignment(QtCore.Qt.AlignCenter)
        self._f_status.setStyleSheet(
            "color:#7a8290; font-family:'Segoe UI',sans-serif; font-size:26px;"
            "font-weight:600; letter-spacing:2px; padding:10px;"
            "background-color:#14171c; border-radius:6px;")
        root.addWidget(self._f_status)

        # ── Metric row ───────────────────────────────────────────────────────
        row = QtWidgets.QHBoxLayout(); row.setSpacing(10)
        self._f_metrics = {}
        # POINTS is first deliberately. It is the cascade metric: a 2-chip
        # 6TX x 8RX array gives 48 virtual channels and 20-28 points on a
        # subject, where a single-chip 3TX x 4RX device typically gives one.
        # Showing only a centroid made the view indistinguishable from
        # single-chip output and discarded the justification for the hardware.
        for key, title in (('points', 'POINTS ON SUBJECT'),
                           ('range',  'RANGE'), ('lateral', 'LATERAL'),
                           ('speed',  'SPEED'), ('activity', 'ACTIVITY')):
            box = QtWidgets.QWidget()
            box.setStyleSheet("background-color:#14171c; border-radius:6px;")
            bl = QtWidgets.QVBoxLayout(box)
            bl.setContentsMargins(14, 10, 14, 10); bl.setSpacing(2)
            cap = QtWidgets.QLabel(title)
            cap.setStyleSheet("color:#5c6470; font-family:'Segoe UI',sans-serif;"
                              "font-size:11px; letter-spacing:1.5px;")
            val = QtWidgets.QLabel("--")
            val.setStyleSheet("color:#e8ecf1; font-family:'Segoe UI',sans-serif;"
                              "font-size:27px; font-weight:600;")
            bl.addWidget(cap); bl.addWidget(val)
            self._f_metrics[key] = val
            row.addWidget(box)
        root.addLayout(row)

        # ── Top-down plot: the subject, not a point cloud ────────────────────
        plot = pg.PlotWidget()
        plot.setBackground('#14171c')
        plot.setLabel('bottom', 'Lateral offset', units='m',
                      **{'color': '#5c6470', 'font-size': '11pt'})
        plot.setLabel('left', 'Range', units='m',
                      **{'color': '#5c6470', 'font-size': '11pt'})
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setXRange(-4, 4); plot.setYRange(0, 12)
        plot.getAxis('bottom').setPen('#2a2f38')
        plot.getAxis('left').setPen('#2a2f38')
        plot.getAxis('bottom').setTextPen('#5c6470')
        plot.getAxis('left').setTextPen('#5c6470')
        # ── Subject detail plot ──────────────────────────────────────────────
        # The scene plot spans 8 m x 12 m, so a 0.5 m body cluster occupies
        # ~3% of its width - the individual returns are invisible at that
        # scale. Showing 20-28 points on a subject IS the cascade advantage
        # over a single-chip device, so it needs a view at body scale.
        # Two panels with clearly separate jobs ("where" and "what") is not
        # the five-panel ambiguity the operators objected to.
        detail = pg.PlotWidget()
        detail.setBackground('#14171c')
        detail.setLabel('bottom', 'Lateral (m, subject-relative)',
                        **{'color': '#5c6470', 'font-size': '10pt'})
        detail.showGrid(x=True, y=True, alpha=0.12)
        detail.setAspectLocked(True)
        detail.getAxis('bottom').setPen('#2a2f38')
        detail.getAxis('left').setPen('#2a2f38')
        detail.getAxis('bottom').setTextPen('#5c6470')
        detail.getAxis('left').setTextPen('#5c6470')
        detail.setXRange(-0.9, 0.9); detail.setYRange(-0.9, 0.9)

        # 0.45 m human-width reference ring, so the point spread can be read
        # against a known scale rather than judged by eye.
        _rt = _np.linspace(0, 2 * _np.pi, 64)
        detail.plot((0.225 * _np.cos(_rt)).tolist(),
                    (0.225 * _np.sin(_rt)).tolist(),
                    pen=pg.mkPen('#2f3641', width=1, style=QtCore.Qt.DashLine))
        self._f_dhull = detail.plot([], [], pen=pg.mkPen('#4da3ff', width=1),
                                    brush=pg.mkBrush(77, 163, 255, 26))
        self._f_dpts = pg.ScatterPlotItem(pxMode=True); detail.addItem(self._f_dpts)
        self._f_dtext = pg.TextItem(text='', color='#5c6470', anchor=(0, 0))
        self._f_dtext.setPos(-0.86, 0.86); detail.addItem(self._f_dtext)

        split = QtWidgets.QHBoxLayout(); split.setSpacing(10)
        split.addWidget(plot, stretch=3)
        split.addWidget(detail, stretch=2)
        root.addLayout(split, stretch=1)

        # Radar origin.
        plot.plot([0], [0], symbol='t', symbolSize=13,
                  symbolBrush=pg.mkBrush('#5c6470'), symbolPen=None, pen=None)

        # Range arcs, NOT field-of-view lines. The +/-60 deg FoV edges were
        # drawn here first and were actively misleading: the plot is 8 m wide
        # by 12 m deep, so the FoV edges exit the SIDE of the view at y~2.3 m
        # and read as a low ceiling the subject is standing above. At the
        # working ranges (3-9 m) the FoV is nowhere near binding - +/-60 deg at
        # 9 m is +/-15.6 m lateral, far outside the plotted area. Range arcs
        # give distance context without implying a boundary that is not there.
        _th = _np.linspace(_np.radians(-70), _np.radians(70), 80)
        for _r in (3.0, 6.0, 9.0, 12.0):
            plot.plot((_r * _np.sin(_th)).tolist(), (_r * _np.cos(_th)).tolist(),
                      pen=pg.mkPen('#232830', width=1))

        # Trail (oldest faint -> newest bright), then the subject on top.
        self._f_trail = pg.ScatterPlotItem(pxMode=True); plot.addItem(self._f_trail)
        # Cluster membership shading, then the member points themselves.
        # Order matters: the hull must sit UNDER the points so the points
        # stay legible.
        self._f_hull = plot.plot([], [], pen=pg.mkPen('#4da3ff', width=1),
                                 fillLevel=None,
                                 brush=pg.mkBrush(77, 163, 255, 26))
        self._f_pts = pg.ScatterPlotItem(pxMode=True); plot.addItem(self._f_pts)
        self._f_marker = pg.ScatterPlotItem(pxMode=True); plot.addItem(self._f_marker)
        self._f_arrow = plot.plot([], [], pen=pg.mkPen('#4da3ff', width=3))
        # anchor to the RIGHT of the text so it sits left of the marker,
        # clear of the direction arrow which extends up/right.
        self._f_label = pg.TextItem(text='', color='#8b93a1', anchor=(1.25, 0.5))
        plot.addItem(self._f_label)

        # Provenance line - keeps the demo honest about what is NOT shown
        note = QtWidgets.QLabel(
            "Each dot is a radar return on the subject, shaded by radial "
            "velocity (pale = moving with the torso, bright = moving faster "
            "or slower).   Colour is a velocity scale, NOT body-part "
            "identification - no anatomy assignment is established.   "
            "Body shape, subject identity and cadence are NOT shown: "
            "measured below this radar's resolution on the recorded sessions.")
        note.setAlignment(QtCore.Qt.AlignCenter)
        note.setStyleSheet("color:#4a515c; font-family:'Segoe UI',sans-serif;"
                           "font-size:11px; padding-top:2px;")
        root.addWidget(note)

        self._f_trail_x = _np.full(40, _np.nan, dtype=_np.float32)
        self._f_trail_y = _np.full(40, _np.nan, dtype=_np.float32)
        self._f_trail_ptr = 0
        self._win.show()

    def _build_heatmap_window(self):
        """Build the three-heatmap layout (RD / RE / RA)."""
        import pyqtgraph as pg
        pg.setConfigOptions(antialias=True, background='k', foreground='w')

        self._win = pg.GraphicsLayoutWidget(title=self._title, show=True)
        self._win.resize(1400, 700)

        p_rd = self._win.addPlot(title="Range–Doppler",       row=0, col=0)
        p_re = self._win.addPlot(title="Range–Elevation",     row=0, col=1)
        p_ra = self._win.addPlot(title="Range–Azimuth (FFT)", row=0, col=2)

        self._status_label = pg.LabelItem(justify='left')
        self._win.addItem(self._status_label, row=1, col=0, colspan=3)

        for p in (p_rd, p_re, p_ra):
            p.setLabel('left', 'Range (m)')
        p_rd.setLabel('bottom', 'Doppler (m/s)')
        p_re.setLabel('bottom', 'Elevation (°)')
        p_ra.setLabel('bottom', 'Azimuth (°)')

        cm = pg.colormap.get('jet', source='matplotlib')

        def _make_image(plot):
            img = pg.ImageItem()
            plot.addItem(img)
            bar = pg.ColorBarItem(values=(60, 110), colorMap=cm, width=10)
            bar.setImageItem(img, insert_in=plot)
            return img, bar

        img_rd, bar_rd = _make_image(p_rd)
        img_re, bar_re = _make_image(p_re)
        img_ra, bar_ra = _make_image(p_ra)

        cfar_scatter = pg.ScatterPlotItem(
            symbol='+', size=14, pen=pg.mkPen('w', width=2), brush=None
        )
        p_ra.addItem(cfar_scatter)

        self._plots = {
            'p_rd': p_rd, 'p_re': p_re, 'p_ra': p_ra,
            'img_rd': img_rd, 'img_re': img_re, 'img_ra': img_ra,
            'bar_rd': bar_rd, 'bar_re': bar_re, 'bar_ra': bar_ra,
            'cfar_scatter': cfar_scatter,
        }

    def _build_volume_window(self):
        """Build a single software-rendered 3D Range/Doppler/Azimuth viewer.

        The Jetson image used by this project can report an unsupported OpenGL
        version and terminate the process while pyqtgraph creates a GL
        context. Matplotlib's Agg canvas provides the same 3D coordinates and
        interaction without requiring an OpenGL context.
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from pyqtgraph.Qt import QtCore, QtWidgets

        self._app_win = QtWidgets.QMainWindow()
        self._app_win.setWindowTitle(self._title)
        self._app_win.resize(1250, 850)

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        figure = plt.Figure(figsize=(12, 8), facecolor='#090b10')
        volume_ax = figure.add_subplot(111, projection='3d')
        volume_ax.set_facecolor('#090b10')
        volume_canvas = FigureCanvasQTAgg(figure)
        layout.addWidget(volume_canvas, stretch=1)

        controls = QtWidgets.QWidget()
        control_row = QtWidgets.QHBoxLayout(controls)
        control_row.setContentsMargins(8, 4, 8, 4)
        threshold_label = QtWidgets.QLabel('SNR threshold: percentile')
        threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        threshold_slider.setRange(9000, 9999)
        threshold_slider.setValue(int(float(getattr(self._cfg, 'volume_percentile', 99.0)) * 100))
        threshold_slider.setFixedWidth(300)
        threshold_value = QtWidgets.QLabel('99.00%')
        voxel_count = QtWidgets.QLabel('voxels: 0')
        axis_label = QtWidgets.QLabel(
            'X: Range (m)   Y: Doppler velocity (m/s)   Z: Azimuth (deg)'
        )
        control_row.addWidget(threshold_label)
        control_row.addWidget(threshold_slider)
        control_row.addWidget(threshold_value)
        control_row.addSpacing(18)
        control_row.addWidget(voxel_count)
        control_row.addSpacing(24)
        control_row.addWidget(axis_label)
        control_row.addStretch()
        layout.addWidget(controls, stretch=0)

        status = QtWidgets.QLabel('Waiting for RDA volume...')
        status.setStyleSheet('color:#b8c0cc; background:#090b10; padding:4px;')
        layout.addWidget(status, stretch=0)
        self._app_win.setCentralWidget(root)
        self._app_win.show()
        self._win = self._app_win
        self._plots = {
            'volume_canvas': volume_canvas, 'volume_ax': volume_ax,
            'volume_status': status, 'volume_threshold': threshold_slider,
            'volume_threshold_label': threshold_value,
            'volume_voxel_count': voxel_count,
        }

    def _update_volume(self, result):
        """Render one thresholded Range/Doppler/Azimuth volume panel."""
        import matplotlib.pyplot as plt

        r_ax = result.range_axis
        d_ax = result.doppler_axis
        a_ax = getattr(result, 'az_axis', None)
        volume = getattr(result, 'RDA', None)
        if volume is None or r_ax is None or d_ax is None or a_ax is None:
            self._plots['volume_status'].setText(
                'RDA volume unavailable; enable volume_en for this display mode.'
            )
            return

        finite = volume[np.isfinite(volume)]
        if finite.size == 0:
            return
        threshold_pct = self._plots['volume_threshold'].value() / 100.0
        self._plots['volume_threshold_label'].setText(f'{threshold_pct:.2f}%')
        threshold = float(np.percentile(finite, threshold_pct))
        vmax = float(np.percentile(finite, 99.9))
        if vmax <= threshold:
            vmax = threshold + 1.0
        dop_idx, range_idx, az_idx = np.where(volume >= threshold)
        max_voxels = int(getattr(self._cfg, 'volume_max_voxels', 50000))
        if len(dop_idx) > max_voxels:
            values = volume[dop_idx, range_idx, az_idx]
            keep = np.argpartition(values, -max_voxels)[-max_voxels:]
            dop_idx, range_idx, az_idx = dop_idx[keep], range_idx[keep], az_idx[keep]
            values = values[keep]
        else:
            values = volume[dop_idx, range_idx, az_idx]

        volume_ax = self._plots['volume_ax']
        volume_ax.clear()
        volume_ax.set_facecolor('#090b10')
        volume_ax.set_xlabel('Range (m)', color='white')
        volume_ax.set_ylabel('Doppler velocity (m/s)', color='white')
        volume_ax.set_zlabel('Azimuth (deg)', color='white')
        volume_ax.tick_params(colors='white')
        volume_ax.set_xlim(float(r_ax[0]), float(r_ax[-1]))
        volume_ax.set_ylim(float(d_ax[0]), float(d_ax[-1]))
        volume_ax.set_zlim(float(a_ax[0]), float(a_ax[-1]))
        volume_ax.set_title('Range × Doppler × Azimuth volume', color='white')
        volume_ax.grid(True, color='#586174', alpha=0.25)

        if len(values):
            positions = np.column_stack((
                r_ax[range_idx], d_ax[dop_idx], a_ax[az_idx]
            )).astype(np.float32)
            norm = np.clip((values - threshold) / max(vmax - threshold, 1e-6), 0, 1)
            colors = plt.get_cmap('inferno')(norm)
            colors[:, 3] = np.clip(0.25 + 0.75 * norm, 0.25, 1.0)
            sizes = 4.0 + 20.0 * norm
            volume_ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=colors, s=sizes, depthshade=False, linewidths=0,
            )
        self._plots['volume_canvas'].draw_idle()
        self._plots['volume_voxel_count'].setText(f'voxels: {len(values):,}')
        self._plots['volume_status'].setText(
            f'Frame {self._frame_count} | voxels {len(values)} / {volume.size} | '
            f'threshold {threshold:.1f} dB | X range (m) / Y Doppler (m/s) / Z azimuth (deg)'
        )

    def _build_pointcloud_window(self):
        """
        Build the point cloud display with three synchronized panels.

        DESIGN RATIONALE
        ────────────────
        The AWR2243-2X has a fundamental aperture asymmetry:
          Azimuth : 86 virtual elements → 0.70°/bin → 3.7 cm at 3 m  ✓ reliable
          Range   : 6.6 cm resolution                                  ✓ reliable
          Elevation: 16 virtual elements → 3.58°/bin → 19 cm at 3 m  ✗ noisy

        The elevation (z) coordinate is unreliable for body-shape work:
        ±1-2 bin noise = ±19-38 cm at 3 m, and the radar mounting height
        shifts all z values by an unknown offset (negative if radar is above
        the target centre of mass).

        The two reliable spatial dimensions are X (lateral, from azimuth) and
        Y (forward/range).  Body WIDTH is visible as X-spread; body DEPTH as
        Y-spread.  A top-down overhead view therefore directly shows the body
        footprint — shoulder-to-shoulder in X, front-to-back in Y.

        THREE-PANEL LAYOUT
        ──────────────────
        LEFT   3D scatter (GLViewWidget) — full spatial context, live orbit
        CENTRE Top-down  (X–Y) — body footprint with shoulder-width oval reference
        RIGHT  Doppler profile (X–Doppler) — lateral position vs velocity,
               shows trunk (low Dop) vs swinging limbs (high Dop)

        CONTROLS (toolbar row below panels)
        ─────────────────────────────────────
        Radar height slider (0–2 m): shifts z by +h_radar so the 3D view
        shows absolute above-ground heights rather than radar-relative heights.
        History slider (1–20 frames): temporal persistence window.
        """
        try:
            import OpenGL as _ogl2
            _ogl2.USE_ACCELERATE = False
        except ImportError:
            pass

        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        from pyqtgraph.Qt import QtWidgets, QtCore

        pg.setConfigOptions(antialias=True)

        self._app_win = QtWidgets.QMainWindow()
        self._app_win.setWindowTitle(self._title)
        self._app_win.resize(1800, 920)

        root    = QtWidgets.QWidget()
        vroot   = QtWidgets.QVBoxLayout(root)
        vroot.setContentsMargins(4, 4, 4, 4)
        vroot.setSpacing(4)

        # ── Panel row ─────────────────────────────────────────────────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # ── LEFT: 3D GL scatter ──────────────────────────────────────────────
        gl_view = gl.GLViewWidget()
        gl_view.setBackgroundColor((12, 12, 12, 255))
        gl_view.setCameraPosition(distance=20, elevation=25, azimuth=180)

        # Ground grid at z=0
        grid = gl.GLGridItem()
        grid.setSize(20, 20, 1)
        grid.setSpacing(1, 1, 1)
        grid.setColor((70, 70, 70, 100))
        gl_view.addItem(grid)

        # ── RADAR HEIGHT FLOOR PLANE ─────────────────────────────────────────
        # A second translucent plane at z = radar_height shows the radar
        # mounting level so stakeholders can see which points are above/below
        # the radar. Updated when the height slider changes.
        _radar_h_init = float(getattr(self._cfg, 'radar_height_m', 0.0))
        radar_floor = gl.GLGridItem()
        radar_floor.setSize(16, 16, 1)
        radar_floor.setSpacing(2, 2, 1)
        radar_floor.setColor((50, 100, 180, 60))   # dim blue — radar level
        radar_floor.translate(0, 8, _radar_h_init)
        gl_view.addItem(radar_floor)

        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        gl_view.addItem(axis)

        radar_dot = gl.GLScatterPlotItem(
            pos=np.array([[0,0,0]], dtype=np.float32),
            color=np.array([[1,1,1,1]], dtype=np.float32),
            size=8, pxMode=True)
        gl_view.addItem(radar_dot)

        scatter_3d = gl.GLScatterPlotItem(
            pos=np.zeros((1,3), dtype=np.float32),
            color=np.ones((1,4), dtype=np.float32),
            size=10, pxMode=True)
        scatter_3d.setGLOptions('translucent')
        gl_view.addItem(scatter_3d)
        splitter.addWidget(gl_view)

        # ── CENTRE: top-down X–Y body footprint ─────────────────────────────
        xy_plot = pg.PlotWidget(background='#0c0c0c')
        xy_plot.setLabel('bottom', 'X  lateral (m)')
        xy_plot.setLabel('left',   'Y  range (m)')
        xy_plot.setTitle('Top-down  (X–Y)  ← body width →',
                         color='#777', size='9pt')
        xy_plot.setXRange(-2.0, 2.0, padding=0)
        xy_plot.setYRange( 0.0, 12.0, padding=0)
        xy_plot.showGrid(x=True, y=True, alpha=0.12)
        xy_plot.setAspectLocked(False)
        xy_plot.getAxis('bottom').setTextPen('#555')
        xy_plot.getAxis('left').setTextPen('#555')

        # Radar position marker
        xy_plot.plot([0], [0],
                     symbol='o', symbolSize=10,
                     symbolBrush=pg.mkBrush(255,255,255,180),
                     symbolPen=None, pen=None)

        # Human body footprint reference oval — fixed size, drawn once
        _body_oval_pen = pg.mkPen(color=(70,70,120), width=1.5,
                                   style=QtCore.Qt.DashLine)
        _shoulder_t = np.linspace(0, 2*np.pi, 64)
        _shoulder_x = 0.225 * np.cos(_shoulder_t)
        _shoulder_y = 0.15  * np.sin(_shoulder_t)
        oval_item = xy_plot.plot(_shoulder_x, _shoulder_y, pen=_body_oval_pen)

        # ── DYNAMIC CLUSTER-EXTENT oval ─────────────────────────────────────
        # Amber outline drawn at the ACTUAL DBSCAN cluster radius (rx, ry).
        #
        # NOTE (audit P0-1): this oval shows cluster EXTENT only. It is NOT a
        # subject-count indicator. Measured counter-example: B_S01 is a single
        # subject walking laterally with rx=0.398 m, LARGER than the two-subject
        # outdoor session at rx=0.257 m. Lateral motion inflates rx for one body,
        # so no rx threshold can separate 1 subject from 2.
        # Real multi-subject detection requires retaining all DBSCAN clusters
        # (audit Stage 3); until then the viewer makes no subject-count claim.
        _dyn_oval_pen = pg.mkPen(color=(255,180,50), width=2.0)
        dyn_oval_item = xy_plot.plot([], [], pen=_dyn_oval_pen)

        # ── CENTROID TRAIL — fading opacity, oldest→newest ───────────────────
        trail_scatter = pg.ScatterPlotItem(pxMode=True)
        xy_plot.addItem(trail_scatter)

        # Range annotation
        xy_range_text = pg.TextItem(text='', color='#555', anchor=(0, 1))
        xy_range_text.setPos(-1.9, 11.8)
        xy_plot.addItem(xy_range_text)

        scatter_xy = pg.ScatterPlotItem(size=10, pxMode=True)
        xy_plot.addItem(scatter_xy)
        splitter.addWidget(xy_plot)

        # ── RIGHT: X vs Doppler lateral profile ─────────────────────────────
        dop_max_val = float(getattr(self._cfg, 'dop_max', 2.5)) or 2.5
        dop_plot = pg.PlotWidget(background='#0c0c0c')
        dop_plot.setLabel('bottom', 'X  lateral (m)')
        dop_plot.setLabel('left',   'Doppler (m/s)')
        dop_plot.setTitle('Lateral × Doppler  (trunk/limb discriminator)',
                          color='#777', size='9pt')
        dop_plot.setXRange(-2.0, 2.0, padding=0)
        dop_plot.setYRange(-dop_max_val, dop_max_val, padding=0)
        dop_plot.showGrid(x=True, y=True, alpha=0.12)
        dop_plot.getAxis('bottom').setTextPen('#555')
        dop_plot.getAxis('left').setTextPen('#555')

        # Zero-doppler reference line
        dop_plot.addLine(y=0, pen=pg.mkPen(color=(60,60,60), width=1))
        # Stationary threshold band
        dop_thr = float(getattr(self._cfg, 'min_abs_doppler_mps',
                                0.5 * getattr(self._cfg, 'dop_res', 0.08)))
        dop_plot.addLine(y= dop_thr, pen=pg.mkPen(color=(50,50,50), width=1,
                                                    style=QtCore.Qt.DotLine))
        dop_plot.addLine(y=-dop_thr, pen=pg.mkPen(color=(50,50,50), width=1,
                                                    style=QtCore.Qt.DotLine))

        scatter_dop = pg.ScatterPlotItem(size=10, pxMode=True)
        dop_plot.addItem(scatter_dop)

        # Region labels
        trunk_text = pg.TextItem('← trunk (low dop)', color='#444', anchor=(0,0))
        trunk_text.setPos(-1.9, dop_thr * 0.2)
        dop_plot.addItem(trunk_text)
        limb_text = pg.TextItem('← limbs (high dop)', color='#444', anchor=(0,1))
        limb_text.setPos(-1.9, dop_max_val * 0.85)
        dop_plot.addItem(limb_text)

        splitter.addWidget(dop_plot)
        splitter.setSizes([700, 580, 520])

        # ── Phase 4: body-frame panel (subject-centric ±0.5m) ───────────────
        _sf_en  = bool(getattr(self._cfg, 'subject_frame_en', False))
        _env_en = bool(getattr(self._cfg, 'body_envelope_en', False))
        if _sf_en:
            sf_plot = pg.PlotWidget(background='#0c0c0c')
            sf_plot.setLabel('bottom', 'dx  lateral (m)')
            sf_plot.setLabel('left',   'dy  range (m)')
            _panel_title = ('Body-frame + envelope  (±0.5 m)' if _env_en
                            else 'Body-frame  (subject-centric ±0.5 m)')
            sf_plot.setTitle(_panel_title, color='#777', size='9pt')
            sf_plot.setXRange(-0.5, 0.5, padding=0)
            sf_plot.setYRange(-0.5, 0.5, padding=0)
            sf_plot.showGrid(x=True, y=True, alpha=0.15)
            sf_plot.setAspectLocked(True)
            sf_plot.getAxis('bottom').setTextPen('#555')
            sf_plot.getAxis('left').setTextPen('#555')
            # Reference crosshair at origin
            sf_plot.addLine(x=0, pen=pg.mkPen(color=(50,50,50), width=1))
            sf_plot.addLine(y=0, pen=pg.mkPen(color=(50,50,50), width=1))
            # Body-width reference box: ±0.225m in dx, ±0.15m in dy
            _bw_pen = pg.mkPen(color=(70,70,120), width=1.5,
                                style=QtCore.Qt.DashLine)
            sf_plot.plot([-0.225,-0.225, 0.225, 0.225,-0.225],
                         [-0.150, 0.150, 0.150,-0.150,-0.150], pen=_bw_pen)
            scatter_sf = pg.ScatterPlotItem(size=10, pxMode=True)
            sf_plot.addItem(scatter_sf)
            # Phase 5: PCA ellipse overlay (drawn on top of scatter)
            if _env_en:
                _env_pen     = pg.mkPen(color=(255, 180, 50), width=2.0)
                envelope_item = sf_plot.plot([], [], pen=_env_pen)
            else:
                envelope_item = None
            sf_text = pg.TextItem(text='', color='#555', anchor=(0,1))
            sf_text.setPos(-0.48, 0.48)
            sf_plot.addItem(sf_text)
            splitter.addWidget(sf_plot)
            splitter.setSizes([600, 500, 450, 400])
        else:
            scatter_sf    = None
            sf_text       = None
            envelope_item = None

        # ── Phase 6: Doppler-time spectrogram panel ───────────────────────
        _md_en = bool(getattr(self._cfg, 'microdoppler_en', False))
        if _md_en:
            import pyqtgraph as pg
            md_plot = pg.PlotWidget(background='#0c0c0c')
            md_plot.setLabel('bottom', 'time  (frames, newest right)')
            md_plot.setLabel('left',   'Doppler  (m/s)')
            md_plot.setTitle('Micro-Doppler spectrogram',
                             color='#777', size='9pt')
            md_plot.showGrid(x=False, y=True, alpha=0.15)
            md_plot.getAxis('bottom').setTextPen('#555')
            md_plot.getAxis('left').setTextPen('#555')
            _v_max = float(getattr(self._cfg, 'spectrogram_v_max', 1.5))
            _T     = int(getattr(self._cfg, 'spectrogram_frames', 64))
            md_plot.setXRange(0, _T, padding=0)
            md_plot.setYRange(-_v_max, _v_max, padding=0)

            # Reference lines: ±dop_limb_thr (limb boundary)
            _thr = float(getattr(self._cfg, 'dop_limb_thr', 0.30))
            _lp  = pg.mkPen(color=(80,80,160), width=1,
                             style=QtCore.Qt.DashLine)
            md_plot.addLine(y= _thr, pen=_lp)
            md_plot.addLine(y=-_thr, pen=_lp)
            md_plot.addLine(y=0,     pen=pg.mkPen(color=(50,50,50), width=1))

            # ImageItem for the spectrogram heatmap
            md_img = pg.ImageItem(axisOrder='col-major')
            # Colour map: dark→cyan→white (intensity)
            _cmap = pg.colormap.get('CET-L6', source='colorcet') \
                    if hasattr(pg.colormap, 'get') else None
            if _cmap is not None:
                md_img.setColorMap(_cmap)
            md_plot.addItem(md_img)

            # Torso line overlay (dop_centroid history)
            torso_pen   = pg.mkPen(color=(255,200,50), width=2)
            torso_curve = md_plot.plot([], [], pen=torso_pen)

            # Cadence annotation
            cad_text = pg.TextItem(text='', color='#EFD080', anchor=(0,1))
            cad_text.setPos(2, _v_max * 0.92)
            md_plot.addItem(cad_text)

            splitter.addWidget(md_plot)
            splitter.setSizes([500, 430, 380, 340, 300])
        else:
            md_img      = None
            torso_curve = None
            cad_text    = None

        vroot.addWidget(splitter, stretch=1)

        # ── Control bar ───────────────────────────────────────────────────────
        ctrl_bar  = QtWidgets.QWidget()
        ctrl_bar.setStyleSheet("background:#111; border-top:1px solid #222;")
        ctrl_row  = QtWidgets.QHBoxLayout(ctrl_bar)
        ctrl_row.setContentsMargins(8, 4, 8, 4)
        ctrl_row.setSpacing(20)

        def _lbl(text):
            l = QtWidgets.QLabel(text)
            l.setStyleSheet("color:#888; font-size:11px; font-family:monospace;")
            return l

        # Radar height slider
        ctrl_row.addWidget(_lbl("Radar height (m):"))
        h_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        h_slider.setMinimum(0); h_slider.setMaximum(200)
        h_slider.setValue(80)   # default 0.80 m
        h_slider.setFixedWidth(160)
        h_slider.setStyleSheet("QSlider::handle {background:#378ADD;}")
        h_val_lbl = QtWidgets.QLabel("0.80 m")
        h_val_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        h_slider.valueChanged.connect(
            lambda v: h_val_lbl.setText(f"{v/100:.2f} m"))
        ctrl_row.addWidget(h_slider)
        ctrl_row.addWidget(h_val_lbl)

        # History slider
        ctrl_row.addWidget(_lbl("Accumulate (frames):"))
        acc_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        acc_slider.setMinimum(1); acc_slider.setMaximum(30)
        acc_slider.setValue(self._cloud_history_frames)
        acc_slider.setFixedWidth(120)
        acc_slider.setStyleSheet("QSlider::handle {background:#1D9E75;}")
        acc_val_lbl = QtWidgets.QLabel(str(self._cloud_history_frames))
        acc_val_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        def _on_acc(v):
            self._cloud_history_frames = v
            acc_val_lbl.setText(str(v))
        acc_slider.valueChanged.connect(_on_acc)
        ctrl_row.addWidget(acc_slider)
        ctrl_row.addWidget(acc_val_lbl)

        ctrl_row.addStretch()

        # Doppler legend
        for text, colour in [("● approaching", "#378ADD"),
                              ("● stationary",  "#cccccc"),
                              ("● receding",    "#E24B4A")]:
            l = QtWidgets.QLabel(text)
            l.setStyleSheet(f"color:{colour}; font-size:11px;")
            ctrl_row.addWidget(l)

        vroot.addWidget(ctrl_bar, stretch=0)

        # ── Status bar ────────────────────────────────────────────────────────
        status_label = QtWidgets.QLabel("")
        status_label.setStyleSheet(
            "color:#bbb; background:#0a0a0a; padding:3px 8px; "
            "font-family:monospace; font-size:11px; "
            "border-top:1px solid #1a1a1a;")
        vroot.addWidget(status_label, stretch=0)

        self._app_win.setCentralWidget(root)
        self._app_win.show()
        self._win = self._app_win

        self._plots = {
            'gl_view':         gl_view,
            'scatter_3d':      scatter_3d,
            'radar_floor':     radar_floor,        # blue plane at radar height
            'scatter_xy':      scatter_xy,
            'scatter_dop':     scatter_dop,
            'oval_item':       oval_item,
            'dyn_oval_item':   dyn_oval_item,      # cluster-extent oval
            'trail_scatter':   trail_scatter,       # centroid trail
            'xy_range_text':   xy_range_text,
            'status_label':    status_label,
            'h_slider':        h_slider,
            'scatter_sf':      scatter_sf,
            'sf_text':         sf_text,
            'envelope_item':   envelope_item,
            'md_img':          md_img,
            'torso_curve':     torso_curve,
            'cad_text':        cad_text,
        }

    def _update(self):
        """Called by QTimer on every refresh tick."""
        result = self._rq.get(timeout=0.001)
        if result is None:
            result = self._rq.peek()
        if result is None:
            return

        self._last_result = result
        self._frame_count = (
            self._processor.frames_processed if self._processor is not None else 0
        )

        if self._display_mode == "focused":
            self._update_focused(result)
        elif self._display_mode == "pointcloud":
            self._update_pointcloud(result)
        elif self._display_mode == "volume":
            self._update_volume(result)
        else:
            self._update_heatmap(result)


    def _update_focused(self, result):
        """Render the focused view. Every quantity here is one the pipeline
        was measured to produce reliably; nothing is inferred or smoothed for
        appearance."""
        import numpy as np
        import pyqtgraph as pg

        # Drive the accumulator here too. It was previously advanced ONLY
        # inside _update_pointcloud, so a display mode that did not call that
        # method would show an empty tracker with no error - the cluster would
        # simply never exist.
        if self._accumulator is not None:
            xyz = getattr(result, 'xyz_cloud', None)
            if xyz is not None and len(xyz) > 0:
                accum_result = self._accumulator.update(xyz, self._frame_count)
                if self._processor is not None:
                    self._processor.track_hint = list(
                        getattr(accum_result, 'track_hints', None)
                        or [(float(accum_result.predicted_x),
                             float(accum_result.predicted_y))])
            else:
                accum_result = self._accumulator._build_result(compensated=True)
            self._last_accum_result = accum_result

        acc = getattr(self, '_last_accum_result', None)
        clu = getattr(acc, 'cluster', None) if acc is not None else None
        gait = getattr(acc, 'gait', None) if acc is not None else None
        tracked = clu is not None and clu.cluster_found

        if not tracked:
            # Keep raw CFAR-confirmed points visible while DBSCAN acquires a
            # temporal cluster. The processing log can report cloud points
            # even when this display branch previously cleared them all.
            raw_xyz = getattr(result, 'xyz_cloud', None)
            if raw_xyz is not None and len(raw_xyz) > 0:
                raw_x = raw_xyz[:, 0].astype(float)
                raw_y = raw_xyz[:, 1].astype(float)
                raw_dop = raw_xyz[:, 4].astype(float)
                raw_cx = float(np.median(raw_x))
                raw_cy = float(np.median(raw_y))
                raw_speed = float(np.median(np.abs(raw_dop)))
                self._f_status.setText("DETECTION (TRACKING)")
                self._f_status.setStyleSheet(
                    "color:#4da3ff; font-family:'Segoe UI',sans-serif; "
                    "font-size:26px; font-weight:600; letter-spacing:2px; "
                    "padding:10px; background-color:#14171c; border-radius:6px;")
                self._f_pts.setData(
                    x=raw_x.tolist(), y=raw_y.tolist(), size=9,
                    pen=None, brush=pg.mkBrush(77, 163, 255, 210))
                self._f_dpts.setData(
                    x=(raw_x - raw_cx).tolist(),
                    y=(raw_y - raw_cy).tolist(), size=15,
                    pen=None, brush=pg.mkBrush(77, 163, 255, 210))
                self._f_marker.setData(
                    x=[raw_cx], y=[raw_cy], size=30, symbol='o',
                    brush=None, pen=pg.mkPen('#4da3ff', width=2))
                self._f_metrics['points'].setText(str(len(raw_xyz)))
                self._f_metrics['range'].setText(f"{raw_cy:.1f} m")
                self._f_metrics['lateral'].setText(f"{raw_cx:+.2f} m")
                self._f_metrics['speed'].setText(f"{raw_speed:.2f} m/s")
                self._f_metrics['activity'].setText("acquiring")
                self._f_dtext.setText(f"{len(raw_xyz)} raw returns   tracking")
                self._f_hull.setData([], [])
                self._f_dhull.setData([], [])
                self._f_label.setText(f"{raw_cy:.1f} m")
                self._f_label.setPos(raw_cx, raw_cy)
                return

            self._f_status.setText("NO SUBJECT DETECTED")
            self._f_status.setStyleSheet(
                "color:#7a8290; font-family:'Segoe UI',sans-serif; font-size:26px;"
                "font-weight:600; letter-spacing:2px; padding:10px;"
                "background-color:#14171c; border-radius:6px;")
            for v in self._f_metrics.values():
                v.setText("--")
            self._f_marker.setData([], []); self._f_arrow.setData([], [])
            self._f_label.setText('')
            # Clear the trail. If it persisted across a dropout and the subject
            # reappeared elsewhere, the fade would connect the two positions and
            # draw a path that was never walked.
            self._f_trail_x[:] = np.nan
            self._f_trail_y[:] = np.nan
            self._f_trail_ptr = 0
            self._f_trail.setData([], [])
            self._f_pts.setData([], []); self._f_hull.setData([], [])
            self._f_dpts.setData([], []); self._f_dhull.setData([], [])
            self._f_dtext.setText('')
            return

        cx, cy = float(clu.centroid_x), float(clu.centroid_y)

        # Speed and direction come from dop_centroid - the signed radial
        # velocity of the tracked cluster. Stage 4.4 verified the sign on
        # 4/4 sessions with data.
        v_rad = float(gait.dop_centroid) if (gait and gait.valid) else 0.0
        spread = float(gait.dop_spread) if (gait and gait.valid) else 0.0

        # Activity: standing still still returns small |v| from breathing and
        # sway, so use the Doppler SPREAD as well - a walking subject has
        # limbs at a range of velocities, a standing one does not.
        DEAD = 0.20
        if abs(v_rad) < DEAD and spread < 0.25:
            activity, acol = "STANDING", '#c9a227'
        elif v_rad > DEAD:
            activity, acol = "WALKING AWAY", '#1f9d55'
        elif v_rad < -DEAD:
            activity, acol = "APPROACHING", '#e8613c'
        else:
            activity, acol = "MOVING", '#4da3ff'

        self._f_status.setText(activity)
        self._f_status.setStyleSheet(
            f"color:{acol}; font-family:'Segoe UI',sans-serif; font-size:26px;"
            "font-weight:600; letter-spacing:2px; padding:10px;"
            "background-color:#14171c; border-radius:6px;")

        # ── Cluster member points — the cascade advantage made visible ───────
        # These are the ACTUAL DBSCAN cluster members, not a decoration. The
        # count is the headline metric: 20-28 returns on one subject is what
        # 48 virtual channels buy over a single-chip device.
        n_pts_sub = 0
        cloud = getattr(clu, 'cloud', None)
        if cloud is not None and len(cloud):
            px = cloud[:, 0].astype(float)
            py = cloud[:, 1].astype(float)
            pd = cloud[:, 4].astype(float)
            n_pts_sub = len(px)

            # Shade by |dop - torso velocity|. Points moving WITH the torso are
            # pale; points moving faster or slower are brighter. This is a
            # velocity scale only - nothing here assigns body parts.
            dev = np.abs(pd - v_rad)
            span = max(float(dev.max()), 1e-3)
            norm = np.clip(dev / span, 0.0, 1.0)
            brushes = [pg.mkBrush(int(120 + 135 * t), int(180 - 40 * t),
                                  int(255 - 150 * t), int(110 + 130 * t))
                       for t in norm]
            self._f_pts.setData(x=px.tolist(), y=py.tolist(),
                                size=9, pen=None, brush=brushes)
            # Same points, subject-relative, at body scale.
            self._f_dpts.setData(x=(px - cx).tolist(), y=(py - cy).tolist(),
                                 size=15, pen=None, brush=brushes)
            self._f_dtext.setText(
                f"{n_pts_sub} returns   spread "
                f"{px.max()-px.min():.2f} x {py.max()-py.min():.2f} m")

            # Convex hull, so the points read as ONE subject rather than a
            # loose scatter. That ambiguity was the original operator
            # complaint: "cannot tell if the points are for the same person
            # or multiple persons".
            if n_pts_sub >= 3:
                try:
                    # Andrew's monotone chain convex hull. Sorting points by
                    # angle about the centroid (the first attempt) produces a
                    # STAR polygon for any concave arrangement, which rendered
                    # as scribble rather than a boundary. A true hull is the
                    # shape that reads as "these points are one subject".
                    pts2 = np.column_stack([px, py])
                    o0 = np.lexsort((pts2[:, 1], pts2[:, 0]))
                    sp = pts2[o0]

                    def _cross(o, a, b):
                        return ((a[0]-o[0])*(b[1]-o[1])
                                - (a[1]-o[1])*(b[0]-o[0]))

                    lower = []
                    for q in sp:
                        while len(lower) >= 2 and _cross(lower[-2], lower[-1], q) <= 0:
                            lower.pop()
                        lower.append(q)
                    upper = []
                    for q in sp[::-1]:
                        while len(upper) >= 2 and _cross(upper[-2], upper[-1], q) <= 0:
                            upper.pop()
                        upper.append(q)
                    hull = np.array(lower[:-1] + upper[:-1])
                    if len(hull) < 3:
                        raise ValueError('degenerate hull')
                    hx = np.append(hull[:, 0], hull[0, 0])
                    hy = np.append(hull[:, 1], hull[0, 1])
                    self._f_hull.setData(hx.tolist(), hy.tolist())
                    self._f_dhull.setData((hx - cx).tolist(), (hy - cy).tolist())
                except Exception:
                    self._f_hull.setData([], []); self._f_dhull.setData([], [])
            else:
                self._f_hull.setData([], []); self._f_dhull.setData([], [])
        else:
            self._f_pts.setData([], []); self._f_hull.setData([], [])
            self._f_dpts.setData([], []); self._f_dhull.setData([], [])
            self._f_dtext.setText('')

        self._f_metrics['points'].setText(f"{n_pts_sub}")
        self._f_metrics['range'].setText(f"{cy:.1f} m")
        self._f_metrics['lateral'].setText(f"{cx:+.2f} m")
        self._f_metrics['speed'].setText(f"{abs(v_rad):.2f} m/s")
        self._f_metrics['activity'].setText(
            "walking" if activity not in ("STANDING",) else "still")

        # Trail, time-ordered (np.roll) - reading a ring buffer in array order
        # scrambles the fade once it wraps (the P0-4b defect).
        L = len(self._f_trail_x)
        idx = self._f_trail_ptr % L
        self._f_trail_x[idx] = cx; self._f_trail_y[idx] = cy
        self._f_trail_ptr = (idx + 1) % L
        tx = np.roll(self._f_trail_x, -self._f_trail_ptr)
        ty = np.roll(self._f_trail_y, -self._f_trail_ptr)
        ok = ~(np.isnan(tx) | np.isnan(ty))
        if ok.any():
            n = int(ok.sum())
            alphas = np.linspace(20, 110, n) if n > 1 else np.array([110])
            self._f_trail.setData(
                x=tx[ok].tolist(), y=ty[ok].tolist(), size=6, pen=None,
                brush=[pg.mkBrush(77, 163, 255, int(a)) for a in alphas])

        # Subject marker: ONE symbol, not a scatter of raw points. Size is
        # fixed - it is a position marker, not a body outline, because the
        # radar cannot resolve body width (dx_std 0.0037 m inter-subject).
        # Centroid ring: an OUTLINE, not a filled disc. A filled marker sat
        # on top of the member points and hid the very thing the cascade
        # exists to produce.
        self._f_marker.setData(
            x=[cx], y=[cy], size=34, symbol='o',
            brush=None, pen=pg.mkPen(acol, width=3))

        # Direction arrow along the radial, length scaled by speed
        if abs(v_rad) >= DEAD:
            rng = max(cy, 1e-3)
            ux, uy = cx / rng, cy / rng            # unit vector from radar
            sgn = 1.0 if v_rad > 0 else -1.0
            ln = min(0.4 + abs(v_rad) * 0.8, 2.0)
            self._f_arrow.setData([cx, cx + sgn * ux * ln],
                                  [cy, cy + sgn * uy * ln])
        else:
            self._f_arrow.setData([], [])

        self._f_label.setText(f"{cy:.1f} m")
        self._f_label.setPos(cx, cy)

    def _update_heatmap(self, result):
        """Heatmap display update — RD / RE / RA images + CFAR scatter."""
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtGui
        from .radar_plots import _det_counts, _det_label, compute_global_scale

        _, raw_vmax = compute_global_scale(result, self._cfg)
        vmax = self._smooth_vmax(raw_vmax)
        vmin = vmax - float(getattr(self._cfg, 'db_span', 24.0))

        def _apply_axis_transform(img_item, x_axis, y_axis):
            if x_axis is None or y_axis is None:
                return
            x0 = float(x_axis[0])
            y0 = float(y_axis[0])
            dx = float(x_axis[-1] - x_axis[0]) / max(len(x_axis) - 1, 1)
            dy = float(y_axis[-1] - y_axis[0]) / max(len(y_axis) - 1, 1)
            tr = QtGui.QTransform()
            tr.translate(x0, y0)
            tr.scale(dx, dy)
            img_item.setTransform(tr)

        def _set_image(img_item, bar_item, data, x_axis, y_axis):
            if data is None or data.size == 0:
                return
            safe = np.where(np.isfinite(data), data, vmin).astype(np.float32)
            img_item.setImage(safe, autoLevels=False, levels=(vmin, vmax))
            _apply_axis_transform(img_item, x_axis, y_axis)
            bar_item.setLevels((vmin, vmax))

        r_ax  = result.range_axis
        d_ax  = result.doppler_axis
        el_ax = result.el_axis
        az_ax = result.az_axis

        _set_image(self._plots['img_rd'], self._plots['bar_rd'],
                   result.RD_mti, d_ax, r_ax)
        _set_image(self._plots['img_re'], self._plots['bar_re'],
                   result.RE.T if result.RE is not None else None, el_ax, r_ax)
        _set_image(self._plots['img_ra'], self._plots['bar_ra'],
                   result.RA.T if result.RA is not None else None, az_ax, r_ax)

        if r_ax is not None and d_ax is not None and result.RD_mti is not None:
            r_max = float(getattr(self._cfg, 'max_range_m', 0) or r_ax[-1])
            self._plots['p_rd'].setXRange(float(d_ax[0]),  float(d_ax[-1]),  padding=0)
            self._plots['p_rd'].setYRange(float(r_ax[0]),  r_max,            padding=0)
        if r_ax is not None and el_ax is not None and result.RE is not None:
            r_max = float(getattr(self._cfg, 'max_range_m', 0) or r_ax[-1])
            self._plots['p_re'].setXRange(float(el_ax[0]), float(el_ax[-1]), padding=0)
            self._plots['p_re'].setYRange(float(r_ax[0]),  r_max,            padding=0)
        if r_ax is not None and az_ax is not None and result.RA is not None:
            r_max = float(getattr(self._cfg, 'max_range_m', 0) or r_ax[-1])
            self._plots['p_ra'].setXRange(float(az_ax[0]), float(az_ax[-1]), padding=0)
            self._plots['p_ra'].setYRange(float(r_ax[0]),  r_max,            padding=0)

        ra_det = result.ra_det
        if ra_det is not None and ra_det.any() and r_ax is not None and az_ax is not None:
            det_r_idx, det_az_idx = np.where(ra_det)
            self._plots['cfar_scatter'].setData(
                x=az_ax[det_az_idx].tolist(),
                y=r_ax[det_r_idx].tolist(),
            )
        else:
            self._plots['cfar_scatter'].setData([], [])

        rd, ra, cloud = _det_counts(result)
        self._status_label.setText(
            f"<span style='color:#aaffaa'>Frame {self._frame_count} "
            f"(processed)</span>  |  "
            f"{_det_label(rd, ra, cloud)}  |  "
            f"<span style='color:#aaaaff'>{vmin:.0f}–{vmax:.0f} dB</span>  |  "
            f"<span style='color:#ffaaaa'>dropped="
            f"{self._processor._fq.dropped if self._processor else 0}</span>"
        )

    def _update_pointcloud(self, result):
        """
        Update all three display panels from xyz_cloud.

        COLOUR FIX (obs 1 — all green):
          Old formula used r=clip(t,0,1), g=clip(1-|t|,0,1), b=clip(-t,0,1).
          This maps ANY moderate Doppler to greenish because g peaks at t=0
          and is still 0.80 at t=±0.2 (dop=0.5 m/s).  For a walking person
          (0.3–1.5 m/s radial) all points come out green.
          New formula is a proper white-centred diverging scheme:
            stationary (|t|<thr) → white
            receding   (t>0)     → lerp white→red
            approaching(t<0)     → lerp white→blue

        OVAL FIX (obs 2 — moving oval):
          The oval was redrawn every frame at the median cloud x,y, causing it
          to jump around and look like part of the data cloud.  It is now drawn
          once at the last-detected range in a fixed STATIC position (x=0) so
          it stays centred at radar boresight as a reference outline, not a
          tracking marker.  Range is shown by text only.

        X-DOPPLER PANEL (obs 3 — limbs vs trunk):
          This is correct and expected.  Walking subject: torso → low Doppler
          (near y=0), swinging arms/legs → higher Doppler (away from y=0).
          Labels added to make this explicit.
        """
        import pyqtgraph as pg

        xyz = result.xyz_cloud

        # ── Phase 2: motion-compensated accumulation ──────────────────────────
        # When temporal_accum_en=True: use TemporalAccumulator (registered, dense)
        # Otherwise: use existing naive ring-buffer concat (backward compatible)
        if self._accumulator is not None:
            if xyz is not None and len(xyz) > 0:
                accum_result = self._accumulator.update(xyz, self._frame_count)
                # Publish the tracked position for the P0-6 gate. Prefer the
                # DBSCAN cluster centroid (body-only) over the EMA of the raw
                # cloud, which is itself contaminated by the returns the gate
                # exists to remove.
                if self._processor is not None:
                    # Use the LAG-CORRECTED prediction, not the raw centroid.
                    # centroid_* derives from the EMA and trails the subject by
                    # ~2.3 frames (0.58 m at 0.25 m/frame), which ate most of the
                    # 0.90 m gate radius and produced a 34.6% bypass rate.
                    # Stage 3.3: pass EVERY confirmed track, so the gate does
                    # not delete the second subject that 3.1/3.2 now retain.
                    self._processor.track_hint = list(
                        getattr(accum_result, 'track_hints', None)
                        or [(float(accum_result.predicted_x),
                             float(accum_result.predicted_y))])
                pts = accum_result.cloud if len(accum_result.cloud) > 0 else None
                _n_accum_frames = accum_result.n_frames
                _vel_y = accum_result.velocity_y
            else:
                accum_result = self._accumulator._build_result(compensated=True)
                pts = accum_result.cloud if len(accum_result.cloud) > 0 else None
                _n_accum_frames = accum_result.n_frames
                _vel_y = 0.0
            self._last_accum_result = accum_result  # Phase 3: expose cluster
        else:
            # Legacy naive concat (default when temporal_accum_en=False)
            if xyz is not None and len(xyz) > 0:
                self._cloud_history.append(xyz)
            if len(self._cloud_history) > self._cloud_history_frames:
                self._cloud_history.pop(0)
            pts = (np.concatenate(self._cloud_history, axis=0)
                   if self._cloud_history else None)
            _n_accum_frames = len(self._cloud_history)
            _vel_y = 0.0

        dop_max  = float(getattr(self._cfg, 'dop_max', 2.5)) or 2.5
        h_radar  = self._plots['h_slider'].value() / 100.0
        dop_thr  = float(getattr(self._cfg, 'min_abs_doppler_mps',
                                 0.5 * getattr(self._cfg, 'dop_res', 0.08)))

        def _colors_diverging(dop_mps, pwr_db):
            """
            White-centred diverging colour map:
              approaching (dop < -dop_thr) → lerp white → blue  [0.2, 0.5, 1.0]
              stationary  (|dop| <= dop_thr)→ white              [1.0, 1.0, 1.0]
              receding    (dop > +dop_thr)  → lerp white → red   [1.0, 0.2, 0.2]
            Alpha is scaled by power so brighter returns are more opaque.
            """
            t      = np.clip(dop_mps / dop_max, -1.0, 1.0)   # [-1, 1]
            # Receding (t>0): white→red
            r_out  = np.ones_like(t)                           # r stays 1
            g_out  = np.where(t >= 0, 1.0 - 0.8 * t,          # green drops for receding
                                      1.0 + 0.5 * t)          # green drops for approaching
            b_out  = np.where(t >= 0, 1.0 - 0.8 * t,          # blue drops for receding
                                      1.0)                     # blue stays for approaching
            r_out  = np.where(t < 0,  1.0 + 0.8 * t, r_out)   # red drops for approaching

            r_out  = np.clip(r_out, 0, 1).astype(np.float32)
            g_out  = np.clip(g_out, 0, 1).astype(np.float32)
            b_out  = np.clip(b_out, 0, 1).astype(np.float32)

            p_rng  = max(float(pwr_db.max()) - float(pwr_db.min()), 1.0)
            alpha  = np.clip(0.4 + 0.6*(pwr_db - pwr_db.min())/p_rng,
                             0.25, 1.0).astype(np.float32)
            return np.stack([r_out, g_out, b_out, alpha], axis=1)

        n_pts = 0
        if pts is not None and len(pts) > 0:
            x       = pts[:, 0].astype(np.float32)
            y       = pts[:, 1].astype(np.float32)
            z_raw   = pts[:, 2].astype(np.float32)
            pwr_db  = pts[:, 3].astype(np.float32)
            dop_mps = pts[:, 4].astype(np.float32)
            rng_m   = pts[:, 5].astype(np.float32)
            n_pts   = len(pts)

            z        = z_raw + h_radar
            colors_f = _colors_diverging(dop_mps, pwr_db)

            p_rng    = max(float(pwr_db.max()) - float(pwr_db.min()), 1.0)
            sz_arr   = np.clip(7 + 9*(pwr_db - pwr_db.min())/p_rng, 5, 20)
            sz_mean  = float(sz_arr.mean())

            brushes  = [pg.mkBrush(int(c[0]*255), int(c[1]*255),
                                   int(c[2]*255), int(c[3]*255))
                        for c in colors_f]

            # ── LEFT: 3D scatter ──────────────────────────────────────────────
            self._plots['scatter_3d'].setData(
                pos=np.stack([x, y, z], axis=1),
                color=colors_f,
                size=sz_mean,
            )

            # ── CENTRE: X-Y top-down body footprint ───────────────────────────
            # Point SIZE encodes elevation: higher z = larger dot
            # This gives stakeholders a sense of body height structure in 2D
            z_raw_pts = pts[:, 2].astype(np.float32)
            z_corr    = z_raw_pts + h_radar
            # Normalise z to [6, 18] px: z near 0 = small, z near +1.5m = large
            z_norm    = np.clip((z_corr + 0.5) / 2.0, 0.0, 1.0)
            xy_sizes  = (6 + 12 * z_norm).astype(np.float32)

            self._plots['scatter_xy'].setData(
                x=x.tolist(), y=y.tolist(),
                brush=brushes, pen=None,
                size=[int(s) for s in xy_sizes],
            )

            # Oval: Phase 3 — track cluster centroid when DBSCAN is available,
            # otherwise fall back to raw median (Phase 2) or boresight (legacy).
            _accum_result = getattr(self, '_last_accum_result', None)
            _cluster      = (getattr(_accum_result, 'cluster', None)
                             if _accum_result is not None else None)

            # NOTE (audit P0-1): a previous build inferred "multi-subject" from
            # `cluster_pts > accum_pts`. That condition is unreachable — DBSCAN's
            # n_pts counts a SUBSET of the cloud it was given, so cluster_pts is
            # always <= accum_pts. The apparent inversion in the audit summary
            # came from comparing two averages with different denominators
            # (accum divided by ALL frames, cluster divided by DETECTED frames).
            # The detector, its red oval and its banner have been removed rather
            # than left in place claiming a signal they cannot produce.
            # Real multi-subject support = audit Stage 3 (retain all clusters).

            if _cluster is not None and _cluster.cluster_found:
                oval_cx = float(_cluster.centroid_x)
                oval_cy = float(_cluster.centroid_y)
                oval_rx = max(float(_cluster.radius_x), 0.10)
                oval_ry = max(float(_cluster.radius_y), 0.07)
                cluster_label = (f"cluster: {_cluster.n_pts}pts "
                                 f"({_cluster.n_noise} noise) "
                                 f"r={oval_rx:.2f}×{oval_ry:.2f}m")
            else:
                oval_cx = 0.0
                oval_cy = float(np.median(rng_m))
                oval_rx, oval_ry = 0.225, 0.150
                cluster_label = "no cluster"

            t_o = np.linspace(0, 2*np.pi, 64)

            # ── Fixed reference oval (dashed blue — nominal body size) ────────
            self._plots['oval_item'].setData(
                oval_cx + 0.225 * np.cos(t_o),
                oval_cy + 0.150 * np.sin(t_o),
            )

            # ── Cluster-extent oval — actual DBSCAN rx/ry, always amber ───────
            # Shows measured cluster size only. Makes no subject-count claim.
            self._plots['dyn_oval_item'].setData(
                oval_cx + oval_rx * np.cos(t_o),
                oval_cy + oval_ry * np.sin(t_o),
            )

            # ── Centroid trail — fading dots, oldest faint → newest bright ────
            # Audit P0-4b fix: the ring buffer must be read in TIME order.
            # Writing at self._trail_ptr means the time-ordered sequence is
            # np.roll(buf, -ptr); reading raw array order scrambles the fade
            # once the buffer wraps (after _TRAIL_LEN frames).
            if not (np.isnan(oval_cx) or np.isnan(oval_cy)):
                L   = self._TRAIL_LEN
                idx = self._trail_ptr % L
                self._trail_x[idx] = oval_cx
                self._trail_y[idx] = oval_cy
                self._trail_ptr    = (idx + 1) % L

                tx_ord = np.roll(self._trail_x, -self._trail_ptr)
                ty_ord = np.roll(self._trail_y, -self._trail_ptr)
                valid  = ~(np.isnan(tx_ord) | np.isnan(ty_ord))
                n_valid = int(valid.sum())
                if n_valid:
                    tx = tx_ord[valid]      # oldest → newest
                    ty = ty_ord[valid]
                    alphas = (np.linspace(30, 200, n_valid)
                              if n_valid > 1 else np.array([200]))
                    trail_brushes = [pg.mkBrush(255, 180, 50, int(a))
                                     for a in alphas]
                    self._plots['trail_scatter'].setData(
                        x=tx.tolist(), y=ty.tolist(),
                        brush=trail_brushes, pen=None, size=5)

            self._plots['xy_range_text'].setText(
                f"range {oval_cy:.2f} m  |  lateral {oval_cx:+.2f} m  "
                f"|  {n_pts} pts  |  {cluster_label}"
            )

            # ── Update radar-height floor plane in 3D panel ───────────────────
            _rf = self._plots.get('radar_floor')
            if _rf is not None:
                try:
                    _rf.resetTransform()
                    _rf.translate(0, 8, h_radar)
                except Exception:
                    pass

            # ── RIGHT: X-Doppler lateral profile ─────────────────────────────
            self._plots['scatter_dop'].setData(
                x=x.tolist(), y=dop_mps.tolist(),
                brush=brushes, pen=None,
                size=max(10, int(sz_mean)),
            )

            # ── BODY-FRAME panel (Phase 4+5, optional) ───────────────────────
            if self._plots.get('scatter_sf') is not None:
                _accum_result = getattr(self, '_last_accum_result', None)
                sf_cloud = getattr(_accum_result, 'subject_cloud', None) \
                           if _accum_result is not None else None
                cluster  = getattr(_accum_result, 'cluster', None) \
                           if _accum_result is not None else None

                if sf_cloud is not None and len(sf_cloud) > 0 and \
                        cluster is not None and cluster.cluster_found:
                    # Build a body-local view of the FULL accumulated cloud
                    # (not just the DBSCAN cluster) so all 64 pts are visible.
                    # Translate every accumulated pt by the cluster centroid.
                    full_cloud = _accum_result.cloud
                    cx_c = cluster.centroid_x
                    cy_c = cluster.centroid_y
                    cz_c = float(np.median(cluster.cloud[:, 2])) \
                           if len(cluster.cloud) > 0 else 0.0

                    all_dx = (full_cloud[:, 0] - cx_c).astype(np.float32)
                    all_dy = (full_cloud[:, 1] - cy_c).astype(np.float32)

                    # Determine cluster membership by proximity (eps gate)
                    dist2 = np.sqrt((full_cloud[:, 0] - cx_c)**2 +
                                    (full_cloud[:, 1] - cy_c)**2)
                    _eps = float(getattr(self._cfg, 'dbscan_eps', 0.80))
                    is_body = dist2 <= _eps * 1.5   # generous gate

                    # Colour: cluster pts = cyan-blue, noise pts = dim grey
                    # Alpha=160 makes dots semi-transparent so overlapping
                    # pts are distinguishable by accumulated brightness.
                    _C_BODY  = (80, 180, 255, 160)   # cyan-blue, semi-transparent
                    _C_NOISE = (80,  80,  80, 100)   # dim grey, more transparent
                    sf_brushes = [
                        pg.mkBrush(*(_C_BODY if b else _C_NOISE))
                        for b in is_body
                    ]

                    self._plots['scatter_sf'].setData(
                        x=all_dx.tolist(), y=all_dy.tolist(),
                        brush=sf_brushes, pen=None, size=8)

                    # Phase 5: PCA ellipse fitted to cluster pts in body-local coords
                    if self._plots.get('envelope_item') is not None:
                        try:
                            from processing.accumulate import fit_body_ellipse
                            env = fit_body_ellipse(sf_cloud,
                                                   max_dx=0.50, max_dy=0.50)
                            if env is not None:
                                ex = np.append(env['x'], env['x'][0])
                                ey = np.append(env['y'], env['y'][0])
                                self._plots['envelope_item'].setData(
                                    x=ex.tolist(), y=ey.tolist())
                                self._plots['sf_text'].setText(
                                    f"accum={len(full_cloud)} pts  "
                                    f"cluster={int(is_body.sum())}  "
                                    f"noise={int((~is_body).sum())}  "
                                    f"env {env['rx']:.2f}×{env['ry']:.2f}m"
                                    f"  θ={env['angle_deg']:.0f}°")
                            else:
                                self._plots['envelope_item'].setData(x=[], y=[])
                                self._plots['sf_text'].setText(
                                    f"accum={len(full_cloud)} pts  "
                                    f"cluster={int(is_body.sum())}")
                        except Exception:
                            self._plots['envelope_item'].setData(x=[], y=[])
                            self._plots['sf_text'].setText(
                                f"accum={len(full_cloud)} pts")
                    else:
                        dx_std = float(np.std(sf_cloud[:, 0]))
                        dy_std = float(np.std(sf_cloud[:, 1]))
                        self._plots['sf_text'].setText(
                            f"accum={len(full_cloud)} pts  "
                            f"dx_std={dx_std:.2f}m  dy_std={dy_std:.2f}m")
                else:
                    self._plots['scatter_sf'].setData(x=[], y=[])
                    if self._plots.get('envelope_item') is not None:
                        self._plots['envelope_item'].setData(x=[], y=[])
                    self._plots['sf_text'].setText('no body cluster')

            # ── Phase 6: spectrogram update ───────────────────────────────
            if self._mdb is not None and self._plots.get('md_img') is not None:
                _ar  = getattr(self, '_last_accum_result', None)
                _gf  = getattr(_ar, 'gait', None) if _ar is not None else None
                _cl  = getattr(_ar, 'cluster', None) if _ar is not None else None

                # Extract cluster Doppler values
                _dops = None
                if _cl is not None and _cl.cluster_found and len(_cl.cloud) > 0:
                    _dops = _cl.cloud[:, 4].astype(np.float32)

                _lf  = _gf.limb_frac   if (_gf is not None and _gf.valid) else float('nan')
                _dc  = _gf.dop_centroid if (_gf is not None and _gf.valid) else float('nan')

                self._mdb.update(_dops, limb_frac=_lf, dop_centroid=_dc)

                # Render spectrogram
                spec  = self._mdb.spectrogram   # [n_dop_bins, T] float32
                _T    = spec.shape[1]
                _vmax = self._mdb.v_max
                self._plots['md_img'].setImage(
                    spec,
                    rect=(0, -_vmax, _T, 2*_vmax),
                    levels=(0.0, 1.0),
                    autoLevels=False)

                # Torso line
                torso = self._mdb.torso_line   # [T] float32
                t_idx = np.arange(_T, dtype=np.float32)
                valid_t = ~np.isnan(torso)
                if valid_t.any():
                    self._plots['torso_curve'].setData(
                        x=t_idx[valid_t].tolist(),
                        y=torso[valid_t].tolist())
                else:
                    self._plots['torso_curve'].setData(x=[], y=[])

                # Cadence annotation
                cad = self._mdb.cadence_hz
                if not np.isnan(cad):
                    self._plots['cad_text'].setText(
                        f"cadence ≈ {cad:.1f} Hz  ({1/cad:.1f}s/stride)")
                elif self._mdb.n_frames_filled < 16:
                    self._plots['cad_text'].setText(
                        f"filling… ({self._mdb.n_frames_filled}/{_T} frames)")
                else:
                    self._plots['cad_text'].setText("cadence: no clear peak")

        else:
            self._plots['scatter_3d'].setData(
                pos=np.zeros((1,3), dtype=np.float32),
                color=np.zeros((1,4), dtype=np.float32),
                size=1,
            )
            self._plots['scatter_xy'].setData(x=[], y=[])
            self._plots['scatter_dop'].setData(x=[], y=[])
            self._plots['xy_range_text'].setText('no detection')

        # ── Status bar ────────────────────────────────────────────────────────
        dropped = self._processor._fq.dropped if self._processor else 0
        accum_label = (f"accum×{_n_accum_frames}(motion-comp)"
                       if self._accumulator is not None
                       else f"{self._cloud_history_frames}f-naive×{_n_accum_frames}")
        vel_label = f"  vy={_vel_y:+.2f}m/s" if abs(_vel_y) > 0.05 else ""

        # Phase 6: gait feature overlay in status bar
        _gait_label = ""
        if self._accumulator is not None:
            _ar = getattr(self, '_last_accum_result', None)
            _gf = getattr(_ar, 'gait', None) if _ar is not None else None
            if _gf is not None and _gf.valid:
                _gait_label = (f"  |  vc={_gf.dop_centroid:+.2f}m/s"
                               f"  σ={_gf.dop_spread:.2f}"
                               f"  limb={_gf.limb_frac:.0%}")

        self._plots['status_label'].setText(
            f"Frame {self._frame_count}  |  "
            f"pts ({accum_label}): {n_pts}{vel_label}"
            f"{_gait_label}  |  "
            f"radar h: {h_radar:.2f} m  |  dropped: {dropped}  |  "
            f"CENTRE = top-down footprint (oval = shoulder ref)  ·  "
            f"RIGHT = lateral × Doppler (trunk near y=0, limbs away)  ·  "
            f"● red = receding  ● white = stationary  ● blue = approaching"
        )


    def run(self, processor: StreamingProcessor) -> None:
        """
        Start the Qt event loop.  Blocks until the window is closed.
        Must be called from the main thread.
        """
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        app = _make_qapp(pg, self._title)
        self._build_window()
        self._processor = processor

        timer = QtCore.QTimer()
        timer.timeout.connect(self._update)
        timer.start(self._pause_ms)

        # Attach close handler — works for both GraphicsLayoutWidget and QMainWindow
        self._win.closeEvent = lambda ev: (processor.stop(), ev.accept())

        app.exec_()
        processor.stop()

        print(
            f"\n[STREAM] Session ended.  "
            f"Frames processed={processor.frames_processed}  "
            f"errored={processor.frames_errored}  "
            f"dropped={processor._fq.dropped}  "
            f"avg_latency={processor.avg_latency_ms():.0f}ms",
            flush=True,
        )


# ---------------------------------------------------------------------------
# on_frame_saved callback factory
# ---------------------------------------------------------------------------

def make_on_frame_saved(
    frame_queue: FrameQueue,
) -> Callable[[Optional[Path], int, bytes], None]:
    """
    Return a callback compatible with FileMonitor's on_frame_saved parameter.

    FileMonitor now passes raw bytes directly as the third argument:
        callback(out_path: Optional[Path], frame_idx: int, frame_data: bytes)

    In streaming mode (out_dir=None), out_path is None and the raw bytes are
    the only source of frame data.  In record mode (out_dir set), out_path
    is a valid Path but we still use the bytes directly to avoid a disk
    read-back that would add latency and require the file to be fully flushed.

    Using the bytes directly (rather than reading from disk) is both faster
    and works correctly in both streaming and record modes.
    """
    def _callback(out_path: Optional[Path], frame_idx: int, frame_data: bytes) -> None:
        try:
            if not frame_data:
                print(
                    f"[STREAM] on_frame_saved: empty frame data for "
                    f"frame {frame_idx}",
                    flush=True,
                )
                return
            frame_queue.put(frame_data)
        except Exception as exc:
            print(f"[STREAM] on_frame_saved error: {exc}", flush=True)

    return _callback


# ---------------------------------------------------------------------------
# Folder-based frame feeder (for development / Stage 1 validation)
# ---------------------------------------------------------------------------

class FolderFeeder(threading.Thread):
    """
    Simulates FileMonitor by feeding frames from an existing folder of
    frame_*.bin files into a FrameQueue at the configured frame rate.

    Used during Stage 1 validation to drive the streaming pipeline from
    the existing offline capture without hardware, verifying that the
    streaming path produces identical ProcessingResults to the offline path.

    frame_rate_hz: frames per second to feed (default 4.0, matching firmware)
    """

    def __init__(
        self,
        frame_folder: Path,
        frame_queue: FrameQueue,
        *,
        frame_rate_hz: float = 4.0,
        loop: bool = False,
    ):
        super().__init__(daemon=True, name="FolderFeeder")
        self._folder = Path(frame_folder)
        self._fq = frame_queue
        self._interval = 1.0 / max(frame_rate_hz, 0.1)
        self._loop = loop
        self.stop_event = threading.Event()
        self.frames_fed: int = 0

    def run(self) -> None:
        frames = sorted(self._folder.glob("frame_*.bin"))
        if not frames:
            print(f"[FOLDER_FEEDER] No frame_*.bin in {self._folder}", flush=True)
            return
        print(
            f"[FOLDER_FEEDER] Feeding {len(frames)} frames at "
            f"{1/self._interval:.1f} fps from {self._folder}",
            flush=True,
        )
        while not self.stop_event.is_set():
            for path in frames:
                if self.stop_event.is_set():
                    return
                try:
                    raw = path.read_bytes()
                    self._fq.put(raw)
                    self.frames_fed += 1
                except Exception as exc:
                    print(f"[FOLDER_FEEDER] {path.name}: {exc}", flush=True)
                time.sleep(self._interval)
            if not self._loop:
                break
        print(
            f"[FOLDER_FEEDER] Done. Fed {self.frames_fed} frames.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_live(
    cfg: "RadarConfig",                     # type: ignore[name-defined]
    correction: np.ndarray,
    *,
    frame_folder: Optional[Path] = None,
    frame_rate_hz: float = 4.0,
    auto_rate: bool = True,
    latency_headroom_factor: float = 0.8,
    raw_iq_order: str = "i-first",
    pause_ms: int = 50,
    quiet: bool = True,
    title: str = "AWR2243-2X Live",
    loop: bool = False,
    display_mode: str = "heatmap",          # "heatmap" | "pointcloud"
) -> FrameQueue:
    """
    Start the streaming pipeline and live display.

    Parameters
    ----------
    cfg          : RadarConfig (from parse_cfg_file)
    correction   : calibration array (from parse_calibration)
    frame_folder : if set, feed from existing frame_*.bin files (dev mode).
                   if None, returns the FrameQueue for the caller to wire
                   to FileMonitor.on_frame_saved via make_on_frame_saved().
    frame_rate_hz: frame feed rate for folder mode (default 4.0).
                   Ignored when auto_rate=True.
    auto_rate    : Fix 3 — when True (default) and frame_folder is set,
                   measures process_frame latency on the first frame and
                   sets the FolderFeeder interval to match.  Prevents the
                   10× drop rate caused when feed_rate >> processing_rate.
    latency_headroom_factor : feed at this fraction of processing speed
                   (default 0.8 = 80%) to absorb GIL and scheduling jitter.
    pause_ms     : display refresh interval in ms (default 50 → ~20 fps)
    quiet        : suppress per-frame StreamingProcessor logs
    title        : window title
    loop         : loop folder feed until window closes

    Returns
    -------
    FrameQueue   : wire this to FileMonitor via make_on_frame_saved() when
                   frame_folder is None (live hardware mode).

    Live hardware mode example
    --------------------------
        fq = run_live(cfg, correction)   # starts processor + display threads
        on_frame_saved = make_on_frame_saved(fq)
        run_capture(settings, on_frame_saved=on_frame_saved, ...)

    Development / validation mode example
    --------------------------------------
        run_live(cfg, correction,
                 frame_folder=Path("capture/frames"),
                 loop=True)
    """
    fq = FrameQueue(max_depth=2)
    rq = ResultQueue()

    # Fix 4: apply IQ order override to match the offline pipeline.
    # The offline pipeline calls apply_raw_iq_order_override(cfg, "i-first")
    # inside main.py before any processing.  Without this, cfg.iq_swap is set
    # from adcbufCfg SampleSwap (buf[2]=1 → iq_swap=True in this cfg file),
    # which conjugates all complex samples and flips both FFT spectra:
    #   range bin 34 → 222  (= 256 - 34)
    #   dop   bin 27 → 37   (= 64  - 27)
    # Applying "i-first" forces iq_swap=False regardless of adcbufCfg, matching
    # the DCA1000EVM hardware behaviour which repacks data as [I,Q] irrespective
    # of the SampleSwap register (documented in radar_reader.py).
    if raw_iq_order != "cfg":
        from .radar_config import apply_raw_iq_order_override
        cfg = apply_raw_iq_order_override(cfg, raw_iq_order)
        print(
            f"[STREAM] IQ order: {raw_iq_order!r} applied  "
            f"(iq_swap={cfg.iq_swap})",
            flush=True,
        )

    processor = StreamingProcessor(fq, rq, cfg, correction, quiet=quiet)
    viewer    = LiveViewer(rq, cfg, pause_ms=pause_ms, title=title,
                           display_mode=display_mode)

    processor.start()

    if frame_folder is not None:
        # Fix 3: auto-measure processing latency before starting FolderFeeder.
        # Read the first available frame, run process_frame once, and use the
        # measured latency to set the feed rate.  This prevents the scenario
        # where frame_rate_hz=4.0 but process_frame takes 2600 ms, causing
        # the queue to drop ~10 frames per processed frame.
        effective_rate_hz = frame_rate_hz
        if auto_rate:
            _frames = sorted(Path(frame_folder).glob("frame_*.bin"))
            if _frames:
                try:
                    import time as _time
                    from .radar_processing import process_frame as _pf
                    from .radar_reader import read_frame as _rf
                    _t0  = _time.perf_counter()
                    _raw = _frames[0].read_bytes()
                    _cube = _read_frame_from_bytes(_raw, cfg)
                    _pf(_cube, cfg, correction, verbose=False)
                    _latency_s = _time.perf_counter() - _t0
                    # Feed at latency_headroom_factor of processing speed so
                    # the processor is never faster than the feeder.
                    _safe_rate = latency_headroom_factor / max(_latency_s, 0.01)
                    effective_rate_hz = _safe_rate
                    print(
                        f"[STREAM] auto_rate: process_frame latency={_latency_s*1000:.0f}ms  "
                        f"→ feed rate capped at {effective_rate_hz:.2f} fps "
                        f"(headroom={latency_headroom_factor:.0%})",
                        flush=True,
                    )
                except Exception as _exc:
                    print(
                        f"[STREAM] auto_rate: latency probe failed ({_exc}); "
                        f"using frame_rate_hz={frame_rate_hz:.2f}",
                        flush=True,
                    )

        feeder = FolderFeeder(
            Path(frame_folder), fq,
            frame_rate_hz=effective_rate_hz,
            loop=loop,
        )
        feeder.start()
    else:
        feeder = None

    # viewer.run() blocks until the window is closed.
    # Must be called from the main thread (Qt requirement).
    try:
        viewer.run(processor)
    finally:
        processor.stop()
        if feeder is not None:
            feeder.stop_event.set()

    return fq



# ---------------------------------------------------------------------------
# Stage 2: CaptureThread — runs run_capture in a background thread
# ---------------------------------------------------------------------------

class CaptureThread(threading.Thread):
    """
    Runs run_capture() in a background thread so the Qt event loop can occupy
    the main thread (required by Qt on Windows and most platforms).

    The capture thread signals the display to close when capture ends by
    calling the optional on_capture_done callback.  The default callback
    closes the pyqtgraph window via a thread-safe Qt signal.

    Attributes
    ----------
    result : CaptureResult | None
        Set when run_capture() returns normally.
    error  : Exception | None
        Set if run_capture() raises.
    """

    def __init__(
        self,
        settings,                                       # CaptureSettings
        frame_queue: FrameQueue,
        *,
        out_dir=None,
        max_frames: int = 0,
        skip_load: bool = False,
        sync_wait: float = 60.0,
        uart_port: Optional[str] = None,
        frame_bytes_override: int = 0,
        warmup_frames: int = 0,
        frame_stats: bool = False,
        diag: bool = False,
        on_capture_done: Optional[Callable] = None,
    ):
        super().__init__(daemon=True, name="CaptureThread")
        self._settings            = settings
        self._fq                  = frame_queue
        self._out_dir             = out_dir
        self._max_frames          = max_frames
        self._skip_load           = skip_load
        self._sync_wait           = sync_wait
        self._uart_port           = uart_port
        self._frame_bytes_override= frame_bytes_override
        self._warmup_frames       = warmup_frames
        self._frame_stats         = frame_stats
        self._diag                = diag
        self._on_done             = on_capture_done
        self.result               = None
        self.error: Optional[Exception] = None

    def run(self) -> None:
        # Import here to avoid circular dependency; capture is a sibling package.
        try:
            from capture.dca import run_capture
            from .streaming import make_on_frame_saved
        except ImportError:
            from capture.dca import run_capture

        on_saved = make_on_frame_saved(self._fq)
        try:
            self.result = run_capture(
                self._settings,
                out_dir=self._out_dir,
                max_frames=self._max_frames,
                skip_load=self._skip_load,
                sync_wait=self._sync_wait,
                uart_port=self._uart_port,
                frame_bytes_override=self._frame_bytes_override,
                on_frame_saved=on_saved,
                file_monitor_verbose=True,
                warmup_frames=self._warmup_frames,
                frame_stats=self._frame_stats,
                diag=self._diag,
            )
            print(
                f"\n[CAPTURE] Done: {self.result.frame_idx} frames  "
                f"raw={self.result.total_bytes:,} B",
                flush=True,
            )
        except KeyboardInterrupt:
            print("\n[CAPTURE] Stopped by user (KeyboardInterrupt).", flush=True)
        except Exception as exc:
            self.error = exc
            print(f"\n[CAPTURE] ERROR: {exc}", flush=True)
        finally:
            # Signal the display to close after capture ends.
            if self._on_done is not None:
                try:
                    self._on_done()
                except Exception:
                    pass


def run_live_hardware(
    cfg: "RadarConfig",                     # type: ignore[name-defined]
    correction: np.ndarray,
    settings,
    *,
    raw_iq_order: str = "i-first",
    out_dir=None,
    max_frames: int = 0,
    skip_load: bool = False,
    sync_wait: float = 60.0,
    uart_port: Optional[str] = None,
    frame_bytes_override: int = 0,
    warmup_frames: int = 0,
    frame_stats: bool = False,
    diag: bool = False,
    pause_ms: int = 50,
    quiet: bool = True,
    title: str = "AWR2243-2X  Live Hardware",
    display_mode: str = "heatmap",          # "heatmap" | "pointcloud"
) -> Optional["CaptureResult"]:           # type: ignore[name-defined]
    """
    Stage 2 entry point: live hardware capture with real-time display.

    Starts the radar capture and live DSP display simultaneously.
    Blocks until the display window is closed or the capture ends.

    Architecture
    ------------
    Main thread  : Qt event loop (app.exec_())  ← required by Qt on Windows
    CaptureThread: run_capture() → FileMonitor → on_frame_saved → FrameQueue
    StreamingProcessor: FrameQueue → process_frame → ResultQueue
    QTimer (50ms): ResultQueue → LiveViewer._update()

    Parameters
    ----------
    cfg            : RadarConfig from parse_cfg_file
    correction     : calibration array from parse_calibration
    settings       : CaptureSettings from create_capture_settings
    raw_iq_order   : IQ byte order — "i-first" (default), "q-first", or "cfg"
    out_dir        : frame output directory (default: settings.out_dir)
    max_frames     : stop after N frames (0 = run until window closed / Ctrl-C)
    skip_load      : True if firmware already running
    sync_wait      : seconds to wait for firmware init after load
    uart_port      : COM port for firmware stop command
    frame_bytes_override : override frame size (0 = derive from header)
    warmup_frames  : discard first N frames (firmware calibration settling)
    frame_stats    : log per-frame ADC magnitude statistics
    diag           : keep CCS debugger attached longer during firmware load
    pause_ms       : display refresh interval in ms (default 50 → ~20 fps)
    quiet          : suppress per-frame StreamingProcessor log lines
    title          : window title

    Returns
    -------
    CaptureResult or None if capture was stopped before completing.

    Usage
    -----
    from processing.radar_config import parse_cfg_file, parse_calibration
    from capture.config import create_capture_settings
    from processing.streaming import run_live_hardware

    cfg      = parse_cfg_file("firmware/r017m/cascade_r017m.cfg")
    corr     = parse_calibration("firmware/r017m/cascade_r017m.cfg",
                                  cfg.num_tx, cfg.num_rx)
    settings = create_capture_settings(uart_port="COM20", max_frames=0)

    run_live_hardware(cfg, corr, settings,
                      skip_load=False, sync_wait=60, warmup_frames=2)
    """
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    fq = FrameQueue(max_depth=2)
    rq = ResultQueue()

    # Apply IQ order override — same as development mode.
    if raw_iq_order != "cfg":
        from .radar_config import apply_raw_iq_order_override
        cfg = apply_raw_iq_order_override(cfg, raw_iq_order)
        print(
            f"[STREAM] IQ order: {raw_iq_order!r} applied  (iq_swap={cfg.iq_swap})",
            flush=True,
        )

    processor = StreamingProcessor(fq, rq, cfg, correction, quiet=quiet)
    viewer    = LiveViewer(rq, cfg, pause_ms=pause_ms, title=title,
                           display_mode=display_mode)

    processor.start()

    # Build Qt app and window before starting capture so the display is ready
    # the moment the first frame arrives.
    app = _make_qapp(pg, title)
    viewer._build_window()

    # QTimer drives the display refresh on the main thread.
    timer = QtCore.QTimer()

    def _close_window():
        """Thread-safe window close — called from CaptureThread when done."""
        QtCore.QMetaObject.invokeMethod(
            timer, "stop", QtCore.Qt.QueuedConnection
        )
        if viewer._win is not None:
            QtCore.QMetaObject.invokeMethod(
                viewer._win, "close", QtCore.Qt.QueuedConnection
            )

    # Wire the processor into the viewer before starting the timer.
    viewer._processor = processor
    timer.timeout.connect(viewer._update)
    timer.start(pause_ms)

    viewer._win.closeEvent = lambda ev: (processor.stop(), ev.accept())

    # Start the capture thread — it calls run_capture which blocks until done.
    capture_thread = CaptureThread(
        settings,
        fq,
        out_dir=out_dir,
        max_frames=max_frames,
        skip_load=skip_load,
        sync_wait=sync_wait,
        uart_port=uart_port,
        frame_bytes_override=frame_bytes_override,
        warmup_frames=warmup_frames,
        frame_stats=frame_stats,
        diag=diag,
        on_capture_done=_close_window,
    )
    capture_thread.start()

    print(
        f"[STREAM] Live hardware mode started.\n"
        f"         Close the display window or press Ctrl-C to stop capture.",
        flush=True,
    )

    # Fix A: Ctrl-C / SIGINT handling for Qt on Windows.
    # On Windows, app.exec_() holds the GIL and Python never sees the SIGINT
    # unless we explicitly yield control back to Python.  The standard fix is:
    # 1. Install a signal handler that calls app.quit() cleanly.
    # 2. Add a 100 ms heartbeat QTimer that does nothing — this wakes Python
    #    from the Qt event loop regularly so the signal handler can fire.
    import signal

    def _sigint_handler(sig, frame):
        print("\n[STREAM] Ctrl-C received — stopping.", flush=True)
        processor.stop()
        app.quit()

    signal.signal(signal.SIGINT, _sigint_handler)

    # Heartbeat timer: fires every 100 ms to let Python process signals.
    heartbeat = QtCore.QTimer()
    heartbeat.timeout.connect(lambda: None)   # no-op — just wakes the loop
    heartbeat.start(100)

    # Block main thread in the Qt event loop until window is closed.
    app.exec_()
    heartbeat.stop()

    print(
        f"\n[STREAM] Session ended.  "
        f"Frames processed={processor.frames_processed}  "
        f"errored={processor.frames_errored}  "
        f"dropped={fq.dropped}  "
        f"avg_latency={processor.avg_latency_ms():.0f}ms",
        flush=True,
    )
    return capture_thread.result


__all__ = [
    "FrameQueue",
    "ResultQueue",
    "StreamingProcessor",
    "LiveViewer",
    "FolderFeeder",
    "CaptureThread",
    "make_on_frame_saved",
    "run_live",
    "run_live_hardware",
]
