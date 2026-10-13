"""
radar_reader.py
===============
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
