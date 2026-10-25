"""
radar_io.py - raw radar frames: the original .bin reader (read_frame, from the old chain's
radar_reader.py, moved here in merge step 2) and the session layer (RadarTimeline,
FramePrefetcher) used by the replay and the batch tools.

Read raw .bin frames from the DCA1000 captured for the AWR2243-2X cascade.

Binary layout (per frame)
--------------------------
  Each chirp in TDM sequence (chirp_idx = loop_idx × num_tx + tx_idx):
    [Dev0_RX0 … Dev0_RX3]  [Dev1_RX0 … Dev1_RX3]
         └── NUM_SAMPLES × IQ int16 per RX ──┘
             IQ order: I,Q (SampleSwap=0) or Q,I (SampleSwap=1)
             Note: DCA1000 firmware may repack data to I,Q regardless of
             SampleSwap — check your capture setup.  Default iq_swap=False.

Output shape
------------
    complex64  [num_tx, num_loops, num_rx, num_samples]
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .radar_config import RadarConfig


def read_frame(bin_path: str, cfg: RadarConfig, frame_idx: int = 0) -> np.ndarray:
    """
    Read one frame from a DCA1000 .bin file.

    Returns
    -------
    cube : complex64  [num_tx, num_loops, num_rx, num_samples]
    """
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    num_rx           = cfg.num_rx
    num_tx           = cfg.num_tx
    num_loops        = cfg.num_loops
    num_samples      = cfg.num_samples
    chirps_per_frame = cfg.chirps_per_frame
    i16_per_frame    = chirps_per_frame * num_rx * num_samples * 2   # × 2 for IQ

    file_size = os.path.getsize(bin_path)
    n_frames  = file_size // (i16_per_frame * 2)

    if n_frames == 0:
        raise ValueError(
            f"File too small for one frame. "
            f"Expected ≥{i16_per_frame * 2} bytes, got {file_size}.")
    if frame_idx >= n_frames:
        raise IndexError(
            f"frame_idx={frame_idx} out of range; file has {n_frames} frame(s).")

    raw = np.fromfile(bin_path, dtype=np.int16,
                      count=i16_per_frame,
                      offset=frame_idx * i16_per_frame * 2)

    if len(raw) < i16_per_frame:
        raise RuntimeError(
            f"Short read: expected {i16_per_frame} int16, got {len(raw)}.")

    # ── Reshape: [chirps, num_devices, rx_per_dev, num_samples, IQ] ─────────
    raw = raw.reshape(chirps_per_frame,
                      cfg.num_devices,
                      cfg.num_rx_per_dev,
                      num_samples,
                      2)   # 2 = IQ

    # IQ order from adcbufCfg SampleSwap
    if cfg.iq_swap:   # SampleSwap=1 → Q,I
        i_ch = raw[..., 1].astype(np.float32)
        q_ch = raw[..., 0].astype(np.float32)
    else:             # SampleSwap=0 → I,Q  (default / DCA1000 repacked)
        i_ch = raw[..., 0].astype(np.float32)
        q_ch = raw[..., 1].astype(np.float32)

    cdata = (i_ch + 1j * q_ch).astype(np.complex64)  # [chirps, devices, rx/dev, samples]

    # Flatten [devices, rx_per_dev] → [num_rx]
    cdata = cdata.reshape(chirps_per_frame, num_rx, num_samples)

    # ── TDM separation: chirp_idx = loop * num_tx + tx ───────────────────────
    cube = np.stack([cdata[t::num_tx] for t in range(num_tx)], axis=0)
    # shape: [num_tx, num_loops, num_rx, num_samples]
    return cube.astype(np.complex64)


def count_frames(bin_path: str, cfg: RadarConfig) -> int:
    if not os.path.isfile(bin_path):
        return 0
    i16_per_frame = cfg.chirps_per_frame * cfg.num_rx * cfg.num_samples * 2
    return os.path.getsize(bin_path) // (i16_per_frame * 2)


def read_all_frames(bin_path: str, cfg: RadarConfig,
                    max_frames: Optional[int] = None) -> np.ndarray:
    """
    Read up to max_frames frames.

    Returns
    -------
    frames : complex64  [n_frames, num_tx, num_loops, num_rx, num_samples]
    """
    n = count_frames(bin_path, cfg)
    if max_frames is not None:
        n = min(n, max_frames)
    if n == 0:
        raise ValueError("No complete frames found.")
    return np.stack([read_frame(bin_path, cfg, i) for i in range(n)], axis=0)


# ── raw frames of a recorded session ─────────────────────────────────────────
# Layout: int16 LE, I first, [loop 64][tx 6][dev 2][rx 4][sample 256][I,Q]; one file per frame,
# listed with its time in <session>/sync/radar_timeline.csv (align_session.py).

def read_raw(path) -> np.ndarray:
    """One radar frame file as int16 (the layout above), unconverted."""
    return np.fromfile(str(path), dtype=np.int16)


class RadarTimeline:
    """The radar frames of a session in order: t_ref (s) and file name per frame."""

    def __init__(self, session):
        import csv
        from pathlib import Path
        self.session = Path(session)
        self.folder = self.session / "radar"
        with open(self.session / "sync" / "radar_timeline.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.t = np.array([float(r["t_ref"]) for r in rows])
        self.files = [r["filename"] for r in rows]

    def __len__(self):
        return len(self.files)

    def path(self, k: int):
        return self.folder / self.files[k]

    def raw(self, k: int) -> np.ndarray:
        return read_raw(self.path(k))

    def index_at(self, t: float) -> int:
        """The last frame at or before t (-1 before the first frame)."""
        return int(np.searchsorted(self.t, t, side="right")) - 1


class FramePrefetcher:
    """Reads radar frame files ahead of a consumer in a background thread. Slow disks (NTFS through
    FUSE: 78 ms median per 3 MB frame) otherwise cap the frame rate.

        pf = FramePrefetcher(timeline); pf.start()
        pf.want(k)          # the consumer is at frame k: keep k .. k + ahead read
        raw = pf.get(k)     # from the cache, or read now
    """

    def __init__(self, timeline: RadarTimeline, ahead: int = 12):
        import threading
        self.tl, self.ahead = timeline, ahead
        self._cache, self._want = {}, 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="radar-prefetch")

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def want(self, k: int):
        self._want = max(int(k), 0)

    def get(self, k: int) -> np.ndarray:
        raw = self._cache.get(k)
        return raw if raw is not None else self.tl.raw(k)

    def _run(self):
        import time
        while not self._stop.is_set():
            want = self._want
            todo = [k for k in range(want, min(want + self.ahead, len(self.tl))) if k not in self._cache]
            if not todo:
                time.sleep(0.005)
                continue
            try:
                raw = self.tl.raw(todo[0])
            except OSError:
                time.sleep(0.05)
                continue
            for old in [j for j in list(self._cache) if j < self._want - 2 or j > self._want + 2 * self.ahead]:
                self._cache.pop(old, None)
            self._cache[todo[0]] = raw
