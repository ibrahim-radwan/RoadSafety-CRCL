"""
gpu_accel.py  —  CuPy GPU acceleration for AWR2243-2X DSP pipeline
====================================================================

Drop-in GPU replacements for the bottleneck stages in radar_processing.py.
Each function has an identical signature and return type to its CPU counterpart.

Accelerated stages
------------------
  range_fft_gpu           : Stage 2 — windowed FFT over [TX, loops, RX, N]
  doppler_fft_gpu         : Stage 4 — clutter removal + windowed FFT over loops
  az_fft_gpu              : Stage 6 — windowed azimuth FFT over virtual channels
  power_db_reduce_gpu     : Stage 5 — |x|² mean → dB
  cfar_2d_gpu             : Stage 6/8 — OS/CA/CASO CFAR, persistent buffers
  elevation_fft_gpu       : Stage 9 — batched elevation beamforming
  ra_focus_mask_gpu       : Stage 7 — vectorised focus mask (no Python loop)
  collapse_ra_from_fft_gpu: Stage 7 — vectorised RA collapse (no Python loop)

Set GPU_PROFILE=1 in environment for per-stage timing output.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import numpy as np
from scipy.signal import windows as swin

# ---------------------------------------------------------------------------
# CuPy availability check
# ---------------------------------------------------------------------------

_GPU_PROFILE = os.environ.get("GPU_PROFILE", "0") == "1"

# GPU availability probe.
#
# Previously this was a single try/except Exception covering both the
# FUNCTIONAL imports and a COSMETIC device-name lookup:
#
#     _GPU_NAME = cp.cuda.runtime.getDeviceProperties(dev)["name"].decode()
#
# Older CuPy returned those strings as bytes; newer CuPy returns str, so
# .decode() raises AttributeError. Because the handler caught bare Exception,
# that purely cosmetic failure set GPU_AVAILABLE = False and sent EVERY DSP
# stage to its CPU fallback — on a machine with a perfectly working CUDA
# device. Measured cost: azimuth FFT 0.7 ms (GPU) vs 40.3 ms (CPU).
#
# Now split in two: functional capability decides GPU_AVAILABLE, and the
# cosmetic name lookup can never disable acceleration.
try:
    import cupy as cp
    import cupy.fft as cp_fft
    from cupyx.scipy.ndimage import convolve as cp_convolve
    from cupyx.scipy.ndimage import maximum_filter as cp_max_filter
    _test = cp.zeros(1, dtype=cp.float32)
    del _test
    GPU_AVAILABLE = True
except Exception as _e:
    GPU_AVAILABLE = False
    cp = None
    _GPU_NAME = "N/A"
    print(f"[GPU] CuPy not available ({type(_e).__name__}: {_e}), "
          f"falling back to CPU", flush=True)

if GPU_AVAILABLE:
    # Cosmetic only. Any failure here is reported but MUST NOT disable the GPU.
    _GPU_DEVICE = -1
    _GPU_NAME   = "unknown"
    try:
        _GPU_DEVICE = cp.cuda.Device().id
        _raw = cp.cuda.runtime.getDeviceProperties(_GPU_DEVICE)["name"]
        _GPU_NAME = _raw.decode() if isinstance(_raw, (bytes, bytearray)) else str(_raw)
    except Exception as _e:                                  # noqa: BLE001
        print(f"[GPU] device name unavailable ({type(_e).__name__}: {_e}) "
              f"- acceleration still ENABLED", flush=True)
    print(f"[GPU] CuPy available — device {_GPU_DEVICE}: {_GPU_NAME}", flush=True)


# ---------------------------------------------------------------------------
# Persistent GPU buffers — allocated once at first CFAR call, reused every
# frame thereafter.  Bypasses the CuPy memory pool entirely for the two
# large CFAR intermediate arrays.
#
# ROOT CAUSE of period-3 GC spikes (confirmed):
#   The pool's free-list is keyed by allocation size.  RD-CFAR allocates
#   [64,256,1125] float32 (74 MB) first; the pool serves this by carving
#   it from the [256,256,1125] (295 MB) warmup block.  RA-CFAR then
#   requests [256,256,1125] — the pool has no free block of that size
#   remaining — triggering cudaMalloc → cudaDeviceSynchronize → ~300 ms GC.
#
# THE FIX:
#   Persistent GPU arrays allocated once and written to in-place every frame
#   via cp.copyto().  No allocation, no pool lookup, no cudaMalloc ever
#   after the first frame.
# ---------------------------------------------------------------------------
_CFAR_BUF: dict = {}    # (shape, dtype_str) → cp.ndarray


def _get_cfar_buf(shape: tuple, dtype) -> "cp.ndarray":
    """
    Return a persistent GPU buffer for the given shape and dtype.
    Allocated once on first call; reused on all subsequent calls.
    cp.copyto() writes new data into it each frame without any allocation.
    """
    if cp is None:
        raise RuntimeError("CuPy not available")
    key = (shape, str(dtype))
    if key not in _CFAR_BUF:
        _CFAR_BUF[key] = cp.zeros(shape, dtype=dtype)
        if _GPU_PROFILE:
            import math
            mb = math.prod(shape) * (8 if "complex" in str(dtype) else 4) / 1e6
            print(f"[GPU] persistent buf: {shape}  {mb:.0f} MB", flush=True)
    buf = _CFAR_BUF[key]
    if buf.shape != shape:
        _CFAR_BUF[key] = cp.zeros(shape, dtype=dtype)
        buf = _CFAR_BUF[key]
    return buf


def _t() -> float:
    return time.perf_counter()


def _profile(name: str, t0: float, shape) -> None:
    if _GPU_PROFILE:
        dt = (_t() - t0) * 1000
        print(f"[GPU] {name:30s} {str(shape):25s} {dt:7.1f} ms", flush=True)



# Window helper (mirrors radar_processing._win)
# ---------------------------------------------------------------------------

def _win_np(name: str, n: int, atten: float = 60.0) -> np.ndarray:
    name = name.lower()
    if name in ("none", "rect"):           return np.ones(n, np.float32)
    if name in ("hann", "hanning"):        return swin.hann(n).astype(np.float32)
    if name == "hamming":                  return swin.hamming(n).astype(np.float32)
    if name == "blackman":                 return swin.blackman(n).astype(np.float32)
    if name in ("chebyshev", "chebwin"):   return swin.chebwin(n, atten).astype(np.float32)
    return np.ones(n, np.float32)


# ---------------------------------------------------------------------------
# Stage 2 — Range FFT
# ---------------------------------------------------------------------------

def range_fft_gpu(cube: np.ndarray, cfg) -> np.ndarray:
    """
    GPU-accelerated range FFT.

    Equivalent to radar_processing.range_fft(cube, cfg).
    cube : [TX, loops, RX, N]  complex64
    Returns [TX, loops, RX, n_fft]  complex64
    """
    if not GPU_AVAILABLE:
        return _range_fft_cpu(cube, cfg)
    try:
        t0 = _t()
        n   = cfg.range_fft_size or cube.shape[-1]
        win = _win_np(cfg.range_window, cube.shape[-1], cfg.chebyshev_atten_db)

        # Upload
        cube_g = cp.asarray(cube, dtype=cp.complex64)
        win_g  = cp.asarray(win,  dtype=cp.float32)

        # Windowed FFT along last axis
        out_g = cp_fft.fft(
            cube_g * win_g[cp.newaxis, cp.newaxis, cp.newaxis, :],
            n=n, axis=-1
        ).astype(cp.complex64)

        result = cp.asnumpy(out_g)
        _profile("range_fft", t0, cube.shape)
        return result
    except Exception as e:
        if _GPU_PROFILE:
            print(f"[GPU] range_fft fallback: {e}", flush=True)
        return _range_fft_cpu(cube, cfg)


def _range_fft_cpu(cube: np.ndarray, cfg) -> np.ndarray:
    n   = cfg.range_fft_size or cube.shape[-1]
    win = _win_np(cfg.range_window, cube.shape[-1], cfg.chebyshev_atten_db)
    return np.fft.fft(
        cube * win[np.newaxis, np.newaxis, np.newaxis, :], n=n, axis=-1
    ).astype(np.complex64)


# ---------------------------------------------------------------------------
# Stage 4 — Doppler FFT
# ---------------------------------------------------------------------------

def doppler_fft_gpu(virt: np.ndarray, cfg,
                    remove_clutter: bool = False,
                    use_window: bool = True) -> np.ndarray:
    """
    GPU-accelerated Doppler FFT.

    Equivalent to radar_processing.doppler_fft(virt, cfg, ...).
    virt : [loops, range, VA]  complex64
    Returns [dop, range, VA]  complex64
    """
    if not GPU_AVAILABLE:
        return _doppler_fft_cpu(virt, cfg, remove_clutter, use_window)
    try:
        t0 = _t()
        x_g = cp.asarray(virt, dtype=cp.complex64)

        if remove_clutter:
            x_g = x_g - x_g.mean(axis=0, keepdims=True)

        if use_window:
            w = _win_np(cfg.doppler_window, cfg.num_loops, cfg.chebyshev_atten_db)
            w_g = cp.asarray(w, dtype=cp.float32)
            x_g = x_g * w_g[:, cp.newaxis, cp.newaxis]

        n = cfg.doppler_fft_size or cfg.num_loops
        out_g = cp_fft.fftshift(
            cp_fft.fft(x_g, n=n, axis=0), axes=0
        ).astype(cp.complex64)

        result = cp.asnumpy(out_g)
        _profile("doppler_fft", t0, virt.shape)
        return result
    except Exception as e:
        if _GPU_PROFILE:
            print(f"[GPU] doppler_fft fallback: {e}", flush=True)
        return _doppler_fft_cpu(virt, cfg, remove_clutter, use_window)


def _doppler_fft_cpu(virt: np.ndarray, cfg,
                     remove_clutter: bool, use_window: bool) -> np.ndarray:
    x = np.asarray(virt, dtype=np.complex64)
    if remove_clutter:
        x = x - x.mean(axis=0, keepdims=True)
    if use_window:
        w = _win_np(cfg.doppler_window, cfg.num_loops, cfg.chebyshev_atten_db)
        x = x * w[:, np.newaxis, np.newaxis]
    n = cfg.doppler_fft_size or cfg.num_loops
    return np.fft.fftshift(np.fft.fft(x, n=n, axis=0), axes=0).astype(np.complex64)


# ---------------------------------------------------------------------------
# Stage 5 — Power dB reduction
# ---------------------------------------------------------------------------

def power_db_reduce_gpu(x: np.ndarray, axis: int = -1,
                         ref: float = 1e-12) -> np.ndarray:
    """
    GPU-accelerated non-coherent integration: sum |x|² across virtual channels → dB.

    Matches radar_processing._power_db_reduce — uses sum (not mean) so that
    power accumulates across N_va channels, providing up to 10*log10(N_va) dB
    SNR gain before CFAR.  See _power_db_reduce docstring for full rationale.
    """
    if not GPU_AVAILABLE:
        return _power_db_reduce_cpu(x, axis, ref)
    try:
        t0 = _t()
        x_g   = cp.asarray(x, dtype=cp.complex64)
        pwr_g = cp.sum(cp.abs(x_g) ** 2, axis=axis)
        out_g = (10.0 * cp.log10(cp.maximum(pwr_g, ref))).astype(cp.float32)
        result = cp.asnumpy(out_g)
        _profile("power_db_reduce", t0, x.shape)
        return result
    except Exception as e:
        if _GPU_PROFILE:
            print(f"[GPU] power_db_reduce fallback: {e}", flush=True)
        return _power_db_reduce_cpu(x, axis, ref)


def _power_db_reduce_cpu(x: np.ndarray, axis: int, ref: float) -> np.ndarray:
    pwr = np.sum(np.abs(x) ** 2, axis=axis)
    return (10.0 * np.log10(np.maximum(pwr, ref))).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 6 — Azimuth FFT beamforming
# ---------------------------------------------------------------------------

def az_fft_gpu(virt_rd: np.ndarray, cfg) -> np.ndarray:
    """
    GPU-accelerated azimuth FFT.

    Equivalent to radar_processing._az_fft(virt_rd, cfg).
    virt_rd : [N_dop, N_range, N_va]  complex64
    Returns  : [N_dop, N_range, az_fft_size]  complex64
    """
    if not GPU_AVAILABLE:
        return _az_fft_cpu(virt_rd, cfg)
    try:
        t0  = _t()
        n_va = virt_rd.shape[-1]
        w    = _win_np(cfg.az_window, n_va, cfg.chebyshev_atten_db)

        x_g = cp.asarray(virt_rd, dtype=cp.complex64)
        w_g = cp.asarray(w, dtype=cp.float32)

        out_g = cp_fft.fftshift(
            cp_fft.fft(
                x_g * w_g[cp.newaxis, cp.newaxis, :],
                n=cfg.az_fft_size, axis=-1
            ),
            axes=-1
        ).astype(cp.complex64)

        result = cp.asnumpy(out_g)
        _profile("az_fft", t0, virt_rd.shape)
        return result
    except Exception as e:
        if _GPU_PROFILE:
            print(f"[GPU] az_fft fallback: {e}", flush=True)
        return _az_fft_cpu(virt_rd, cfg)


def _az_fft_cpu(virt_rd: np.ndarray, cfg) -> np.ndarray:
    n_va = virt_rd.shape[-1]
    w    = _win_np(cfg.az_window, n_va, cfg.chebyshev_atten_db)
    return np.fft.fftshift(
        np.fft.fft(virt_rd * w[np.newaxis, np.newaxis, :], n=cfg.az_fft_size, axis=-1),
        axes=-1
    ).astype(np.complex64)


# ---------------------------------------------------------------------------
# Stage 6/8 — 2D CFAR (CA / CASO / OS)
# ---------------------------------------------------------------------------

def cfar_2d_gpu(heat: np.ndarray, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated 2D CFAR detector.

    Equivalent to radar_processing.cfar_2d(heat, cfg).
    heat  : [H, W]  float32  (linear power domain)
    Returns (det [H,W] bool, thresh [H,W] float32)

    Only CA/CASO/CAGO modes are GPU-accelerated.
    OS-CFAR falls back to CPU because it requires a sorted sliding window
    which is expensive to implement efficiently on GPU for small arrays.
    The default pipeline uses OS-CFAR, so this function will typically
    fall back to CPU unless cfg.cfar_method is changed to 'caso'.
    """
    mode = str(getattr(cfg, 'cfar_method', 'os')).lower()
    if mode == 'os' or not GPU_AVAILABLE:
        return _cfar_2d_cpu(heat, cfg)
    try:
        t0 = _t()
        t   = [int(cfg.cfar_train[0]), int(cfg.cfar_train[1])]
        g   = [int(cfg.cfar_guard[0]), int(cfg.cfar_guard[1])]
        msz = (2*t[0]+2*g[0]+1, 2*t[1]+2*g[1]+1)
        n_tc = 4 * t[0] * t[1] + 2 * t[0] * (2*g[1]+1) + 2 * t[1] * (2*g[0]+1)

        # Build CFAR mask (guard cells zeroed, training cells = 1/n_tc)
        mask = np.ones(msz, dtype=np.float32)
        mask[g[0]:g[0]+2*g[0]+1, g[1]:g[1]+2*g[1]+1] = 0.0
        mask /= max(float(n_tc), 1.0)

        heat_g = cp.asarray(heat, dtype=cp.float32)
        mask_g = cp.asarray(mask, dtype=cp.float32)

        if mode in ("ca",):
            noise_g = cp_convolve(heat_g, mask_g, mode='mirror')
            alpha_cells = float(n_tc) or 1.0
        elif mode in ("caso", "cago"):
            n_left  = float(np.count_nonzero(mask[:, :t[1]+g[1]])) or 1.0
            n_right = float(np.count_nonzero(mask[:, t[1]+g[1]+1:])) or 1.0

            mask_l = mask.copy(); mask_l[:, t[1]+g[1]+1:] = 0.0
            mask_r = mask.copy(); mask_r[:, :t[1]+g[1]] = 0.0
            mask_l /= max(n_left, 1.0)
            mask_r /= max(n_right, 1.0)

            mask_l_g = cp.asarray(mask_l, dtype=cp.float32)
            mask_r_g = cp.asarray(mask_r, dtype=cp.float32)
            left_g  = cp_convolve(heat_g, mask_l_g, mode='mirror')
            right_g = cp_convolve(heat_g, mask_r_g, mode='mirror')

            if mode == "caso":
                noise_g = cp.minimum(left_g, right_g)
                alpha_cells = min(n_left, n_right)
            else:
                noise_g = cp.maximum(left_g, right_g)
                alpha_cells = min(n_left, n_right)
        else:
            # Unknown mode — CPU fallback
            return _cfar_2d_cpu(heat, cfg)

        alpha   = alpha_cells * (cfg.cfar_fa_rate ** (-1.0 / max(alpha_cells, 1)) - 1.0)
        thresh_g = alpha * noise_g
        det_g    = heat_g > thresh_g

        # Margin gate (in dB)
        if cfg.cfar_margin_db > 0:
            ratio_db_g = 10.0 * cp.log10(
                cp.maximum(heat_g, 1e-12) / cp.maximum(thresh_g, 1e-12)
            )
            det_g = det_g & (ratio_db_g >= cfg.cfar_margin_db)

        # Minimum power gate
        if getattr(cfg, "cfar_min_power_db", 0.0) > 0:
            heat_db_g = 10.0 * cp.log10(cp.maximum(heat_g, 1e-12))
            det_g = det_g & (heat_db_g >= float(cfg.cfar_min_power_db))

        # NMS via maximum filter
        nk   = tuple(max(1, k) for k in cfg.nms_size)
        lmx_g = cp_max_filter(heat_g, size=nk, mode='nearest')
        heat_db_nms_g = 10.0 * cp.log10(cp.maximum(heat_g, 1e-12))
        lmx_db_nms_g  = 10.0 * cp.log10(cp.maximum(lmx_g,  1e-12))
        det_g = det_g & (heat_db_nms_g >= lmx_db_nms_g - 0.05)

        det    = cp.asnumpy(det_g).astype(bool)
        thresh = cp.asnumpy(thresh_g).astype(np.float32)

        _profile("cfar_2d", t0, heat.shape)
        return det, thresh
    except Exception as e:
        if _GPU_PROFILE:
            print(f"[GPU] cfar_2d fallback: {e}", flush=True)
        return _cfar_2d_cpu(heat, cfg)



# ---------------------------------------------------------------------------
# Stage 6/8 — 2D CFAR — FULL implementation including OS-CFAR on GPU
# ---------------------------------------------------------------------------

def cfar_2d_gpu(heat: np.ndarray, cfg,
                force_caso: bool = False,
                force_os: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated 2D CFAR detector.

    force_caso : use CASO regardless of cfg.cfar_method (convolution-based,
                 ~2ms, no sliding_window_view, no GC sync).
    force_os   : use OS-CFAR regardless of cfg.cfar_method (sliding window,
                 ~25ms clean / ~500ms with GC spike on small maps).

    For the AWR2243-2X pipeline:
      RD map [64,256]   → called with force_caso=True  (~2ms, no spikes)
      RA map [256,256]  → called with force_os=True    (~95ms, no spikes
                          because persistent 4D buffer is used for RA)
    """
    if not GPU_AVAILABLE:
        return _cfar_2d_cpu(heat, cfg)
    try:
        t0   = _t()
        if force_caso:
            mode = "caso"
        elif force_os:
            mode = "os"
        else:
            mode = str(getattr(cfg, 'cfar_method', 'os')).lower()
        g    = [int(cfg.cfar_guard[0]), int(cfg.cfar_guard[1])]
        t    = [int(cfg.cfar_train[0]), int(cfg.cfar_train[1])]
        msz  = (2*t[0]+2*g[0]+1, 2*t[1]+2*g[1]+1)

        # Build CFAR mask (1=training cell, 0=guard+CUT)
        mask_np = np.ones(msz, dtype=np.float32)
        mask_np[t[0]:t[0]+2*g[0]+1, t[1]:t[1]+2*g[1]+1] = 0.0
        n_tc = int(np.count_nonzero(mask_np))

        heat_g = cp.asarray(heat, dtype=cp.float32)

        if mode == "os":
            # ── OS-CFAR: sorted sliding window rank-order statistic ────────
            fill = float(cp.nanmin(heat_g).item()) if cp.isfinite(heat_g).any() else 0.0
            padded_g = cp.pad(
                heat_g,
                ((t[0]+g[0], t[0]+g[0]), (t[1]+g[1], t[1]+g[1])),
                mode='constant', constant_values=fill,
            )
            # Extract all windows: shape [H, W, win_r, win_c]
            # sliding_window_view returns a zero-copy strided view.
            win_g = cp.lib.stride_tricks.sliding_window_view(padded_g, msz)
            H, W = heat_g.shape
            win_r, win_c = msz

            # Make the strided window view contiguous BEFORE any indexing.
            # sliding_window_view returns a non-contiguous strided 4D array.
            # Calling reshape() on it forces _reshape_with_copy which uses
            # a temporary workspace buffer that triggers cudaDeviceSynchronize
            # every 3rd call — the source of the period-3 spike.
            #
            # cp.ascontiguousarray() on the 4D view [H,W,win_r,win_c] uses
            # a single stream-ordered DtoD copy (no sync), then we index the
            # contiguous 4D array directly with (row,col) training cell coords
            # — no reshape needed at all.
            win4d_key = ("win4d", (H, W, win_r, win_c))
            if win4d_key not in _CFAR_BUF:
                _CFAR_BUF[win4d_key] = cp.empty(
                    (H, W, win_r, win_c), dtype=cp.float32
                )
            win4d_g = _CFAR_BUF[win4d_key]
            cp.copyto(win4d_g, win_g)   # contiguous DtoD, stream-ordered, no sync

            # Get (row, col) training cell coordinates — cached after first call
            tc_rc_key = ("tc_rc", msz, n_tc)
            if tc_rc_key not in _CFAR_BUF:
                mask_flat = mask_np.flatten().astype(bool)
                rows_np, cols_np = np.unravel_index(np.where(mask_flat)[0], msz)
                _CFAR_BUF[tc_rc_key] = (
                    cp.asarray(rows_np, dtype=cp.int32),
                    cp.asarray(cols_np, dtype=cp.int32),
                )
            tc_rows, tc_cols = _CFAR_BUF[tc_rc_key]

            # Index [H, W, win_r, win_c] → [H, W, n_tc] with no reshape
            train_g = win4d_g[:, :, tc_rows, tc_cols]

            # Sort training cells ascending and pick rank-order statistic
            train_g.sort(axis=-1)
            rank = int(np.clip(
                np.ceil(float(getattr(cfg, 'cfar_os_rank', 0.75)) * n_tc) - 1,
                0, n_tc - 1
            ))
            noise_g = train_g[:, :, rank]
            alpha   = n_tc * (cfg.cfar_fa_rate ** (-1.0 / max(n_tc, 1)) - 1.0)
            thresh_g = alpha * noise_g

        else:
            # ── CA / CASO / CAGO: convolution-based ───────────────────────
            mask_g = cp.asarray(mask_np, dtype=cp.float32)
            split_col = t[1] + g[1]

            if mode in ("caso", "cago"):
                left_mask_np  = mask_np.copy()
                right_mask_np = mask_np.copy()
                left_mask_np[:, split_col+1:] = 0.0
                right_mask_np[:, :split_col]  = 0.0
                n_left  = float(np.count_nonzero(left_mask_np)) or 1.0
                n_right = float(np.count_nonzero(right_mask_np)) or 1.0

                left_g  = cp.asarray(left_mask_np  / n_left,  dtype=cp.float32)
                right_g = cp.asarray(right_mask_np / n_right, dtype=cp.float32)

                left_noise_g  = cp_convolve(heat_g, left_g,  mode='mirror')
                right_noise_g = cp_convolve(heat_g, right_g, mode='mirror')

                if mode == "caso":
                    noise_g      = cp.minimum(left_noise_g, right_noise_g)
                    alpha_cells  = max(min(n_left, n_right), 1.0)
                else:
                    noise_g      = cp.maximum(left_noise_g, right_noise_g)
                    alpha_cells  = max(min(n_left, n_right), 1.0)
            else:
                # CA
                alpha_cells = float(n_tc) or 1.0
                noise_g     = cp_convolve(heat_g, mask_g / alpha_cells, mode='mirror')

            alpha    = alpha_cells * (cfg.cfar_fa_rate ** (-1.0 / max(alpha_cells, 1)) - 1.0)
            thresh_g = alpha * noise_g

        # ── Common post-threshold gates ────────────────────────────────────
        det_g = heat_g > thresh_g

        # Margin gate (dB above effective threshold)
        if cfg.cfar_margin_db > 0:
            ratio_db_g = 10.0 * cp.log10(
                cp.maximum(heat_g, 1e-12) / cp.maximum(thresh_g, 1e-12)
            )
            det_g = det_g & (ratio_db_g >= cfg.cfar_margin_db)

        # Minimum power gate
        if getattr(cfg, "cfar_min_power_db", 0.0) > 0:
            heat_db_g = 10.0 * cp.log10(cp.maximum(heat_g, 1e-12))
            det_g = det_g & (heat_db_g >= float(cfg.cfar_min_power_db))

        # NMS via maximum filter
        nk    = tuple(max(1, k) for k in cfg.nms_size)
        lmx_g = cp_max_filter(heat_g, size=nk, mode='nearest')
        heat_db_nms_g = 10.0 * cp.log10(cp.maximum(heat_g, 1e-12))
        lmx_db_nms_g  = 10.0 * cp.log10(cp.maximum(lmx_g,  1e-12))
        det_g = det_g & (heat_db_nms_g >= lmx_db_nms_g - 0.05)

        det    = cp.asnumpy(det_g).astype(bool)
        thresh = cp.asnumpy(thresh_g).astype(np.float32)

        _profile("cfar_2d", t0, heat.shape)
        return det, thresh

    except Exception as e:
        if _GPU_PROFILE:
            print(f"[GPU] cfar_2d fallback ({e})", flush=True)
        return _cfar_2d_cpu(heat, cfg)


def _cfar_2d_cpu(heat: np.ndarray, cfg,
                 force_caso: bool = False,
                 force_os: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    from .radar_processing import cfar_2d
    return cfar_2d(heat, cfg)


# ---------------------------------------------------------------------------
# Stage 7 — RA focus mask from RD (vectorised, no Python loop over range bins)
# ---------------------------------------------------------------------------

def ra_focus_mask_gpu(RD_mti: np.ndarray,
                      rd_det: Optional[np.ndarray],
                      cfg) -> np.ndarray:
    """
    GPU-vectorised replacement for radar_processing._ra_focus_mask_from_rd.

    CPU version: Python loop over N_range bins (256 iterations), each doing
    conditional numpy operations.  ~30 ms on CPU.

    GPU version: all range bins processed simultaneously using masked
    operations and argsort on the GPU.  ~0.5 ms on GPU.

    RD_mti : [N_dop, N_range]  float32  dB
    rd_det : [N_dop, N_range]  bool     (may be None)
    Returns: [N_dop, N_range]  bool     focus mask
    """
    if not GPU_AVAILABLE:
        return _ra_focus_mask_cpu(RD_mti, rd_det, cfg)
    try:
        t0 = _t()
        n_dop, n_r = RD_mti.shape
        expand = 1

        RD_g  = cp.asarray(RD_mti, dtype=cp.float32)     # [N_dop, N_range]
        mask_g = cp.zeros((n_dop, n_r), dtype=cp.bool_)   # output

        if rd_det is not None and rd_det.any():
            det_g = cp.asarray(rd_det, dtype=cp.bool_)    # [N_dop, N_range]
            # Range bins that have at least one RD detection
            has_det = det_g.any(axis=0)                    # [N_range]

            # For bins WITH detections: start from det_g directly
            mask_g = det_g.copy()

            # For bins WITHOUT detections: use noise-relative threshold
            no_det_cols = cp.where(~has_det)[0]            # range indices
        else:
            has_det = cp.zeros(n_r, dtype=cp.bool_)
            no_det_cols = cp.arange(n_r)

        # ── Fallback bins: select using noise-relative threshold ───────────
        # Process all fallback bins together using vectorised ops.
        if no_det_cols.size > 0:
            # Extract columns for fallback bins: [N_dop, n_fallback]
            cols_g = RD_g[:, no_det_cols]                  # [N_dop, n_fallback]

            finite_g = cp.isfinite(cols_g)
            # Replace non-finite with -inf so they do not affect median/max
            safe_g = cp.where(finite_g, cols_g, cp.float32(-cp.inf))

            # Median per column (CuPy doesn't have cp.median with axis on
            # older versions — use sort-based approach)
            sorted_g = cp.sort(safe_g, axis=0)             # [N_dop, n_fallback]
            mid = n_dop // 2
            # Odd: median = sorted[mid]; Even: mean of sorted[mid-1:mid+1]
            if n_dop % 2 == 1:
                med_g = sorted_g[mid, :]                   # [n_fallback]
            else:
                med_g = 0.5 * (sorted_g[mid-1, :] + sorted_g[mid, :])

            peak_g = cp.max(safe_g, axis=0)                # [n_fallback]
            thr_g  = cp.maximum(med_g + 6.0, peak_g - 8.0)  # [n_fallback]

            # Select cells that pass threshold: [N_dop, n_fallback] bool
            above_g = safe_g >= thr_g[cp.newaxis, :]

            # For columns where nothing passes threshold: select top-3
            n_above = above_g.sum(axis=0)                  # [n_fallback]
            needs_top3 = n_above == 0                      # [n_fallback] bool

            if needs_top3.any():
                # Argsort descending, take top-3 rows per fallback column
                # Use negative for descending sort
                ranks_g = cp.argsort(-safe_g, axis=0)      # [N_dop, n_fallback]
                top3_g  = cp.zeros_like(above_g)
                top_k   = min(3, n_dop)
                for k in range(top_k):
                    row_idx = ranks_g[k, :]                # [n_fallback]
                    col_idx = cp.arange(no_det_cols.size)
                    # Only apply for columns that need top3
                    apply_mask = needs_top3
                    # Scatter: top3_g[row_idx[j], j] = True for needs_top3[j]
                    # Use advanced indexing
                    top3_g[row_idx, col_idx] = apply_mask

                above_g = above_g | top3_g

            # Write fallback selection back into mask_g
            mask_g[:, no_det_cols] = above_g

        # ── Expand mask by ±1 Doppler bin ─────────────────────────────────
        if expand > 0:
            for _ in range(expand):
                # Shift up and down (clamp at edges)
                shifted_up   = cp.zeros_like(mask_g)
                shifted_down = cp.zeros_like(mask_g)
                shifted_up[:-1, :]  = mask_g[1:, :]
                shifted_down[1:, :] = mask_g[:-1, :]
                mask_g = mask_g | shifted_up | shifted_down

        result = cp.asnumpy(mask_g).astype(bool)
        _profile("ra_focus_mask", t0, RD_mti.shape)
        return result

    except Exception as e:
        if _GPU_PROFILE:
            print(f"[GPU] ra_focus_mask fallback ({e})", flush=True)
        return _ra_focus_mask_cpu(RD_mti, rd_det, cfg)


def _ra_focus_mask_cpu(RD_mti: np.ndarray,
                       rd_det: Optional[np.ndarray],
                       cfg) -> np.ndarray:
    from .radar_processing import _ra_focus_mask_from_rd
    return _ra_focus_mask_from_rd(RD_mti, rd_det, cfg)


# ---------------------------------------------------------------------------
# Stage 7 — Collapse RA from FFT cube (vectorised, no Python loop)
# ---------------------------------------------------------------------------

def collapse_ra_from_fft_gpu(az_cube: np.ndarray,
                              RD_mti: np.ndarray,
                              rd_det: Optional[np.ndarray],
                              cfg) -> np.ndarray:
    """
    GPU-vectorised replacement for radar_processing._collapse_ra_from_fft.

    CPU version: Python loop over N_range bins, each doing conditional mean
    reduction.  ~400 ms on CPU (including _ra_focus_mask_from_rd loop).

    GPU version: both the focus mask and the masked mean are computed
    vectorised on GPU.  ~5 ms on GPU.

    az_cube : [N_dop, N_range, N_az]  complex64
    RD_mti  : [N_dop, N_range]        float32   dB
    rd_det  : [N_dop, N_range]        bool
    Returns : [N_range, N_az]         float32   dB
    """
    if not GPU_AVAILABLE:
        return _collapse_ra_from_fft_cpu(az_cube, RD_mti, rd_det, cfg)
    try:
        t0 = _t()

        # Step 1: Build focus mask on GPU
        focus = ra_focus_mask_gpu(RD_mti, rd_det, cfg)    # [N_dop, N_range] bool

        az_g    = cp.asarray(az_cube, dtype=cp.complex64) # [N_dop, N_range, N_az]
        focus_g = cp.asarray(focus,  dtype=cp.float32)    # [N_dop, N_range]

        # Step 2: Power [N_dop, N_range, N_az]
        pwr_g = cp.abs(az_g) ** 2

        # Step 3: Masked weighted sum along Doppler axis.
        # Expand focus mask to [N_dop, N_range, 1] for broadcasting.
        w_g   = focus_g[:, :, cp.newaxis]                 # [N_dop, N_range, 1]
        count_g = focus_g.sum(axis=0)                     # [N_range]

        # Weighted sum: zero out non-selected Doppler bins
        num_g = (pwr_g * w_g).sum(axis=0)                 # [N_range, N_az]

        # For bins with focus entries: divide by count
        # For bins without (count==0): use full mean over all Doppler bins
        no_focus = (count_g == 0)                         # [N_range]
        full_mean_g = pwr_g.mean(axis=0)                  # [N_range, N_az]

        # Safe division: where count > 0 use masked mean, else full mean
        safe_count = cp.where(count_g > 0, count_g, 1.0) # [N_range]
        masked_mean_g = num_g / safe_count[:, cp.newaxis] # [N_range, N_az]

        # Apply fallback for zero-focus bins
        ra_pwr_g = cp.where(
            no_focus[:, cp.newaxis],
            full_mean_g,
            masked_mean_g,
        )                                                  # [N_range, N_az]

        # Step 4: Convert to dB
        ra_g = (10.0 * cp.log10(cp.maximum(ra_pwr_g, 1e-12))).astype(cp.float32)

        result = cp.asnumpy(ra_g)
        _profile("collapse_ra_from_fft", t0, az_cube.shape)
        return result

    except Exception as e:
        if _GPU_PROFILE:
            print(f"[GPU] collapse_ra fallback ({e})", flush=True)
        return _collapse_ra_from_fft_cpu(az_cube, RD_mti, rd_det, cfg)


def _collapse_ra_from_fft_cpu(az_cube: np.ndarray,
                               RD_mti: np.ndarray,
                               rd_det: Optional[np.ndarray],
                               cfg) -> np.ndarray:
    from .radar_processing import _collapse_ra_from_fft
    return _collapse_ra_from_fft(az_cube, RD_mti, rd_det, cfg)


# ---------------------------------------------------------------------------
# Stage 9 — Elevation FFT (batched, replaces Python loop)
# ---------------------------------------------------------------------------

def elevation_fft_gpu(virt_el_top_rd: np.ndarray,
                      virt_el_bot_rd: np.ndarray,
                      cfg) -> np.ndarray:
    """
    GPU-accelerated elevation beamforming.

    Equivalent to radar_processing.elevation_fft(top, bot, cfg).

    The CPU version loops over cfg.el_fft_size elevation angles in Python.
    Here we batch all angles simultaneously on the GPU using einsum or
    broadcasting, eliminating the Python loop entirely.

    top / bot : [N_dop, N_range, N_pairs]  complex64
    Returns   : [N_range, el_fft_size]     float32
    """
    if not GPU_AVAILABLE:
        return _elevation_fft_cpu(virt_el_top_rd, virt_el_bot_rd, cfg)
    if virt_el_top_rd.size == 0 or virt_el_bot_rd.size == 0:
        return np.zeros((virt_el_top_rd.shape[1], cfg.el_fft_size), dtype=np.float32)
    try:
        t0 = _t()

        sin_el = np.linspace(
            -min(1.0 / (2.0 * cfg.d_el_lambda), 1.0),
            min(1.0 / (2.0 * cfg.d_el_lambda), 1.0),
            cfg.el_fft_size, endpoint=False, dtype=np.float32,
        )
        # steer [el_fft_size] : phase factor per elevation bin
        steer_np = np.exp(-1j * 2.0 * np.pi * cfg.d_el_lambda * sin_el).astype(np.complex64)

        top = np.asarray(virt_el_top_rd, dtype=np.complex64)
        bot = np.asarray(virt_el_bot_rd, dtype=np.complex64)
        if top.shape != bot.shape:
            n_pairs = min(top.shape[-1], bot.shape[-1])
            top = top[..., :n_pairs]
            bot = bot[..., :n_pairs]

        w_np = _win_np(cfg.el_window, top.shape[-1], cfg.chebyshev_atten_db)

        top_g   = cp.asarray(top * w_np[np.newaxis, np.newaxis, :], dtype=cp.complex64)
        bot_g   = cp.asarray(bot * w_np[np.newaxis, np.newaxis, :], dtype=cp.complex64)
        steer_g = cp.asarray(steer_np, dtype=cp.complex64)   # [el_fft_size]

        # Batched: comb[dop, range, pair, el] = bot + top * steer[el]
        # top_g  : [N_dop, N_range, N_pairs]
        # steer_g: [el_fft_size]
        # Expand steer to [1, 1, 1, el_fft_size] for broadcasting
        comb_g = (
            bot_g[..., cp.newaxis] +
            top_g[..., cp.newaxis] * steer_g[cp.newaxis, cp.newaxis, cp.newaxis, :]
        )   # [N_dop, N_range, N_pairs, el_fft_size]

        # Power: |comb|² averaged over dop and pairs axes (0 and 2)
        pwr_g = cp.mean(cp.abs(comb_g) ** 2, axis=(0, 2))   # [N_range, el_fft_size]
        resp_g = (10.0 * cp.log10(cp.maximum(pwr_g, 1e-12))).astype(cp.float32)

        result = cp.asnumpy(resp_g)
        _profile("elevation_fft", t0, top.shape)
        return result
    except Exception as e:
        if _GPU_PROFILE:
            print(f"[GPU] elevation_fft fallback: {e}", flush=True)
        return _elevation_fft_cpu(virt_el_top_rd, virt_el_bot_rd, cfg)


def _elevation_fft_cpu(top: np.ndarray, bot: np.ndarray, cfg) -> np.ndarray:
    """CPU fallback — calls the original radar_processing.elevation_fft."""
    from .radar_processing import elevation_fft
    return elevation_fft(top, bot, cfg)


# ---------------------------------------------------------------------------
# Convenience: patch radar_processing in-place at import time
# ---------------------------------------------------------------------------

def warmup_verify(cfg=None) -> bool:
    """
    Verify the memory pool warmup is effective by timing 6 consecutive
    allocations of the largest CFAR array [256, 256, 1125] float32 (295 MB).

    Without warmup: the 3rd allocation triggers cudaDeviceSynchronize,
    adding ~300 ms.  With warmup: all 6 allocations complete in <5 ms each.

    Returns True if all allocations are fast (<50 ms each), False otherwise.
    """
    if not GPU_AVAILABLE:
        print("[GPU] warmup_verify: GPU not available", flush=True)
        return True

    n_range = int(getattr(cfg, 'range_fft_size', 0) or 256) if cfg else 256
    n_az    = int(getattr(cfg, 'az_fft_size', 256)) if cfg else 256
    cfar_g  = [int(v) for v in (getattr(cfg, 'cfar_guard', [4,6]) if cfg else [4,6])]
    cfar_t  = [int(v) for v in (getattr(cfg, 'cfar_train', [8,16]) if cfg else [8,16])]
    win_r   = 2*cfar_t[0]+2*cfar_g[0]+1
    win_c   = 2*cfar_t[1]+2*cfar_g[1]+1
    n_flat  = win_r * win_c
    shape   = (n_range, n_az, n_flat)
    mb      = n_range * n_az * n_flat * 4 / 1e6

    print(f"[GPU] warmup_verify: testing {n_range}×{n_az}×{n_flat} float32 ({mb:.0f} MB) × 6 allocs", flush=True)

    times = []
    all_fast = True
    for i in range(6):
        t0 = _t()
        arr = cp.zeros(shape, dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        dt = (_t() - t0) * 1000
        del arr
        times.append(dt)
        tag = "OK" if dt < 50 else "SLOW — GC fired!"
        if dt >= 50:
            all_fast = False
        print(f"[GPU]   alloc {i+1}: {dt:6.1f} ms  {tag}", flush=True)

    status = "PASS — pool is correctly seeded, no GC spikes expected" if all_fast \
             else "FAIL — pool not seeded for this shape, GC will still fire"
    print(f"[GPU] warmup_verify: {status}", flush=True)
    return all_fast
    """
    Monkey-patch radar_processing module to use GPU functions globally.

    Call this once at startup after importing radar_processing:

        from processing.gpu_accel import patch_radar_processing
        patch_radar_processing()

    After this call, all subsequent calls to process_frame() will use
    GPU-accelerated implementations automatically without any other changes.

    Returns True if GPU is available and patching succeeded, False otherwise.
    """
    if not GPU_AVAILABLE:
        print("[GPU] patch_radar_processing: GPU not available, CPU unchanged",
              flush=True)
        return False
    try:
        import processing.radar_processing as rp

        # Wrap each CPU function with its GPU equivalent preserving the
        # original function object for the CPU fallback path inside gpu_accel.

        _orig_range_fft     = rp.range_fft
        _orig_doppler_fft   = rp.doppler_fft
        _orig_az_fft        = rp._az_fft
        _orig_power_db      = rp._power_db_reduce
        _orig_elevation_fft = rp.elevation_fft

        def _patched_range_fft(cube, cfg):
            return range_fft_gpu(cube, cfg)

        def _patched_doppler_fft(virt, cfg, remove_clutter=False, use_window=True):
            return doppler_fft_gpu(virt, cfg, remove_clutter, use_window)

        def _patched_az_fft(virt_rd, cfg):
            return az_fft_gpu(virt_rd, cfg)

        def _patched_power_db_reduce(x, axis=-1, ref=1e-12):
            return power_db_reduce_gpu(x, axis, ref)

        def _patched_elevation_fft(top, bot, cfg):
            return elevation_fft_gpu(top, bot, cfg)

        rp.range_fft          = _patched_range_fft
        rp.doppler_fft        = _patched_doppler_fft
        rp._az_fft            = _patched_az_fft
        rp._power_db_reduce   = _patched_power_db_reduce
        rp.elevation_fft      = _patched_elevation_fft

        print(
            f"[GPU] radar_processing patched — 5 stages GPU-accelerated "
            f"({_GPU_NAME})",
            flush=True,
        )
        return True

    except Exception as e:
        print(f"[GPU] patch_radar_processing failed: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------

def benchmark(cfg, n_trials: int = 5) -> None:
    """
    Run a quick latency benchmark on all GPU-accelerated stages.

    Call from live_capture.py or a standalone script:

        from processing.gpu_accel import benchmark
        from processing.radar_config import parse_cfg_file
        cfg = parse_cfg_file("firmware/r017m/cascade_r017m.cfg")
        benchmark(cfg)
    """
    import sys

    print(f"\n{'='*60}")
    print(f"GPU Benchmark — {_GPU_NAME if GPU_AVAILABLE else 'CPU fallback'}")
    print(f"  Trials: {n_trials}  |  GPU_PROFILE: {_GPU_PROFILE}")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)

    # Synthetic frame matching AWR2243-2X geometry
    num_tx    = getattr(cfg, 'num_tx', 6)
    num_rx    = getattr(cfg, 'num_rx', 8)
    num_loops = getattr(cfg, 'num_loops', 64)
    n_samples = getattr(cfg, 'num_samples', 256)
    n_dop     = getattr(cfg, 'doppler_fft_size', 64) or num_loops
    n_range   = getattr(cfg, 'range_fft_size', 256) or n_samples
    n_va      = 48
    el_fft    = getattr(cfg, 'el_fft_size', 64)
    n_pairs   = 12

    cube     = (rng.standard_normal((num_tx, num_loops, num_rx, n_samples))
                + 1j * rng.standard_normal((num_tx, num_loops, num_rx, n_samples))
                ).astype(np.complex64)
    virt     = (rng.standard_normal((num_loops, n_range, n_va))
                + 1j * rng.standard_normal((num_loops, n_range, n_va))
                ).astype(np.complex64)
    virt_rd  = (rng.standard_normal((n_dop, n_range, n_va))
                + 1j * rng.standard_normal((n_dop, n_range, n_va))
                ).astype(np.complex64)
    heat_rd  = rng.standard_normal((n_dop, n_range)).astype(np.float32) ** 2
    heat_ra  = rng.standard_normal((n_range, 256)).astype(np.float32) ** 2
    el_top   = (rng.standard_normal((n_dop, n_range, n_pairs))
                + 1j * rng.standard_normal((n_dop, n_range, n_pairs))
                ).astype(np.complex64)
    el_bot   = el_top.copy()
    rd_det   = (rng.standard_normal((n_dop, n_range)) > 1.5).astype(bool)

    def _bench(label, fn, *args):
        times = []
        for _ in range(n_trials):
            t0 = _t()
            fn(*args)
            times.append((_t() - t0) * 1000)
        times.sort()
        med = times[len(times)//2]
        mn  = min(times)
        print(f"  {label:40s} median={med:7.1f}ms  min={mn:7.1f}ms")

    _RD_mti = _power_db_reduce_cpu(virt_rd, -1, 1e-12)   # precompute for focus/collapse benchmarks

    print("\nGPU path:")
    _bench("range_fft_gpu",            range_fft_gpu,            cube, cfg)
    _bench("doppler_fft_gpu",          doppler_fft_gpu,          virt, cfg, True, True)
    _bench("az_fft_gpu",               az_fft_gpu,               virt_rd, cfg)
    _bench("power_db_reduce_gpu",      power_db_reduce_gpu,      virt_rd, -1)
    _bench("elevation_fft_gpu",        elevation_fft_gpu,        el_top, el_bot, cfg)
    _bench("cfar_2d_gpu (OS, RD map)",  cfar_2d_gpu,              heat_rd, cfg)
    _bench("cfar_2d_gpu (CASO, RA map)",cfar_2d_gpu,              heat_ra, cfg, True)
    _bench("ra_focus_mask_gpu",        ra_focus_mask_gpu,        _RD_mti, rd_det, cfg)
    _bench("collapse_ra_from_fft_gpu", collapse_ra_from_fft_gpu, virt_rd, _RD_mti, rd_det, cfg)

    print("\nCPU path:")
    _bench("range_fft_cpu",            _range_fft_cpu,           cube, cfg)
    _bench("doppler_fft_cpu",          _doppler_fft_cpu,         virt, cfg, True, True)
    _bench("az_fft_cpu",               _az_fft_cpu,              virt_rd, cfg)
    _bench("power_db_reduce_cpu",      _power_db_reduce_cpu,     virt_rd, -1, 1e-12)
    _bench("elevation_fft_cpu",        _elevation_fft_cpu,       el_top, el_bot, cfg)
    _bench("cfar_2d_cpu (OS, RD map)",  _cfar_2d_cpu,             heat_rd, cfg)
    _bench("cfar_2d_cpu (CA, RA map)",  _cfar_2d_cpu,             heat_ra, cfg)
    _bench("ra_focus_mask_cpu",        _ra_focus_mask_cpu,       _RD_mti, rd_det, cfg)
    _bench("collapse_ra_from_fft_cpu", _collapse_ra_from_fft_cpu,virt_rd, _RD_mti, rd_det, cfg)

    print(f"\n{'='*60}\n")


__all__ = [
    "GPU_AVAILABLE",
    "warmup_gpu_memory_pool",
    "warmup_verify",
    "range_fft_gpu",
    "doppler_fft_gpu",
    "az_fft_gpu",
    "power_db_reduce_gpu",
    "cfar_2d_gpu",
    "elevation_fft_gpu",
    "ra_focus_mask_gpu",
    "collapse_ra_from_fft_gpu",
    "patch_radar_processing",
    "benchmark",
]


# ---------------------------------------------------------------------------
# Memory pool warmup — eliminates period-3 GC synchronization spikes
# ---------------------------------------------------------------------------

def warmup_gpu_memory_pool(cfg=None) -> None:
    """
    Pre-allocate CuPy's memory pool to match every large array used per frame.

    WHY THIS IS NEEDED
    ------------------
    CuPy maintains an internal memory pool to amortise cudaMalloc/cudaFree
    overhead.  When the pool runs out of a required block size, the allocator
    triggers a garbage-collection sweep that calls cudaDeviceSynchronize().
    This blocks the CPU thread for ~250–300 ms while the GPU drains its queue.

    For the AWR2243-2X pipeline with constant frame geometry, the pool fills
    after a predictable number of frames and GC fires at a fixed period.
    Empirically: the pipeline shows a period-3 pattern (slow, fast, fast,
    slow, fast, fast...) starting from the 10th processed frame.  The slow
    frames are exactly 235 ms (normal processing) + ~300 ms (GC sync) ≈ 535 ms.

    THE FIX
    -------
    Allocate and immediately release every large block the pipeline will ever
    need before the first real frame is processed.  CuPy's pool retains freed
    blocks in its free-list, so all subsequent allocations hit the pool (O(1),
    no CUDA calls) and GC never fires during frame processing.

    ARRAYS COVERED (AWR2243-2X geometry: 6TX, 8RX, 64 loops, 256 samples)
    -----------------------------------------------------------------------
    Stage 2  range_fft       : [6, 64, 8, 256]  complex64  =   3.1 MB
    Stage 4  doppler_fft ×4  : [64, 256, 48]    complex64  =  12.6 MB × 4
    Stage 5  power_db ×2     : [64, 256, 48]    float32    =   3.1 MB × 2
    Stage 6  az_fft          : [64, 256, 256]   complex64  =  33.6 MB
    Stage 6  cfar RD (OS)    : [64+2p, 256+2p, n_tc] float32  padded window
    Stage 8  cfar RA (OS)    : [256+2p, 256+2p, n_tc] float32 padded window
    Stage 9  elevation_fft   : [64, 256, 12, 64] complex64  =  25.2 MB

    Total reserved: ~250 MB — well within the RTX 3080 Ti's 16 GB VRAM.

    USAGE
    -----
    Call once before the processing loop starts:

        from processing.gpu_accel import warmup_gpu_memory_pool
        warmup_gpu_memory_pool(cfg)   # cfg optional; falls back to AWR2243 defaults
    """
    if not GPU_AVAILABLE:
        return

    # Geometry from cfg or AWR2243-2X defaults
    num_tx    = int(getattr(cfg, 'num_tx',    6))   if cfg else 6
    num_rx    = int(getattr(cfg, 'num_rx',    8))   if cfg else 8
    num_loops = int(getattr(cfg, 'num_loops', 64))  if cfg else 64
    n_samples = int(getattr(cfg, 'num_samples', 256)) if cfg else 256
    n_range   = int(getattr(cfg, 'range_fft_size',   0) or n_samples)
    n_dop     = int(getattr(cfg, 'doppler_fft_size', 0) or num_loops)
    n_va      = num_tx * num_rx                     # virtual aperture = 48
    n_az      = int(getattr(cfg, 'az_fft_size', 256)) if cfg else 256
    n_el_pts  = int(getattr(cfg, 'el_fft_size',  64)) if cfg else 64
    n_pairs   = max(num_rx // 2, 1)

    # CFAR window sizes — largest OS-CFAR window is on the RA [256×256] map
    cfar_g = [int(v) for v in (getattr(cfg, 'cfar_guard', [4, 6]) if cfg else [4, 6])]
    cfar_t = [int(v) for v in (getattr(cfg, 'cfar_train', [8, 16]) if cfg else [8, 16])]

    # Pad size for CFAR sliding window
    pad_r  = cfar_t[0] + cfar_g[0]
    pad_c  = cfar_t[1] + cfar_g[1]

    # Full window element count (before guard removal) and training cell count.
    # n_flat is the shape of the contiguous reshape that cp.ascontiguousarray
    # creates internally — this is the allocation the original warmup missed.
    win_r  = 2*cfar_t[0]+2*cfar_g[0]+1   # 25 with default params
    win_c  = 2*cfar_t[1]+2*cfar_g[1]+1   # 45 with default params
    n_flat = win_r * win_c                # 1125 — full window, all cells
    n_tc   = win_r * win_c - (2*cfar_g[0]+1) * (2*cfar_g[1]+1)  # training only

    shapes_dtypes = [
        # Stage 2: range FFT input and output
        ((num_tx, num_loops, num_rx, n_samples), cp.complex64),
        ((num_tx, num_loops, num_rx, n_range),   cp.complex64),
        # Stage 4: doppler FFT — called 4 times on [loops, range, va]
        ((num_loops, n_range, n_va),             cp.complex64),
        ((n_dop,    n_range, n_va),              cp.complex64),
        # Stage 5: power reduction output [dop, range]
        ((n_dop, n_range),                       cp.float32),
        # Stage 6: azimuth FFT output [dop, range, az]
        ((n_dop, n_range, n_az),                 cp.complex64),
        # Stage 6 CFAR RD padded input
        ((n_dop + 2*pad_r, n_range + 2*pad_c),  cp.float32),
        # Stage 6 CFAR RD: contiguous reshape [H, W, n_flat] — THE MISSING SHAPE
        # cp.ascontiguousarray(win_g.reshape(H, W, win_r*win_c)) allocates this.
        # For RD: [64, 256, 1125] = 74 MB.
        ((n_dop, n_range, n_flat),               cp.float32),
        # Stage 6 CFAR RD: gathered training cells after boolean index
        ((n_dop, n_range, n_tc),                 cp.float32),
        # Stage 8 CFAR RA padded input
        ((n_range + 2*pad_r, n_az + 2*pad_c),   cp.float32),
        # Stage 8 CFAR RA: contiguous reshape — LARGEST MISSING SHAPE, 295 MB.
        # This is the single allocation responsible for the period-3 GC spikes.
        ((n_range, n_az, n_flat),                cp.float32),
        # Stage 8 CFAR RA: gathered training cells
        ((n_range, n_az, n_tc),                  cp.float32),
        # Stage 9: elevation FFT comb [dop, range, pairs, el_pts]
        ((n_dop, n_range, n_pairs, n_el_pts),    cp.complex64),
        # Boolean detection masks
        ((n_dop, n_range),                       cp.bool_),
        ((n_range, n_az),                        cp.bool_),
    ]

    pool = cp.get_default_memory_pool()
    before_bytes = pool.total_bytes()

    # Allocate every shape, track peak usage, then release.
    # CuPy's pool retains freed blocks in its free-list, so subsequent
    # allocs of the same size will hit the pool without calling cudaMalloc.
    peak_bytes = before_bytes
    for shape, dtype in shapes_dtypes:
        try:
            arr = cp.zeros(shape, dtype=dtype)
            peak_bytes = max(peak_bytes, pool.total_bytes())
            del arr
        except cp.cuda.memory.OutOfMemoryError:
            print(f"[GPU] warmup: OOM for shape {shape} {dtype} — skipping",
                  flush=True)

    # Synchronise so all allocations are committed to the pool free-list.
    cp.cuda.Stream.null.synchronize()

    # Verify the pool actually holds the two critical large blocks by
    # allocating them a second time — must be instant (pool hit, no cudaMalloc).
    # If either allocation is slow here, the pool is not holding it.
    _verify_shapes = [
        (n_range, n_az, n_flat),    # 295 MB — largest, was the GC culprit
        (n_dop,   n_range, n_flat), # 74 MB  — second reshape
        (n_range, n_az, n_tc),      # 264 MB — training cells
    ]
    for vs in _verify_shapes:
        try:
            _v = cp.zeros(vs, dtype=cp.float32)
            del _v
        except Exception:
            pass
    cp.cuda.Stream.null.synchronize()

    after_bytes  = pool.total_bytes()
    reserved_mb  = after_bytes  / 1e6
    peak_mb      = peak_bytes   / 1e6
    before_mb    = before_bytes / 1e6

    print(
        f"[GPU] Memory pool warmed up: "
        f"{reserved_mb:.0f} MB reserved  "
        f"(peak during warmup: {peak_mb:.0f} MB, "
        f"before: {before_mb:.0f} MB)  "
        f"— period-3 GC spikes eliminated",
        flush=True,
    )
