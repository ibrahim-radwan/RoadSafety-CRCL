"""
radar_processing.py  —  AWR2243-2X cascade DSP pipeline
========================================================
Stages
  1.  Calibration (phase + amplitude)
  2.  Range FFT (windowed)
  3.  MTI clutter removal (slow-time mean subtraction)
  4.  Virtual-array formation
  5.  Doppler FFT (windowed, fftshifted) + zero-Doppler notch
  6.  Range-Doppler maps (dB, VA-averaged)
  7.  2-D K-Radar CA-CFAR + NMS on RD map
  8.  Azimuth beamforming (FFT / Capon / MUSIC / Bartlett)
  9.  2-D CFAR + NMS on RA map
  10. Elevation FFT
  11. arrDREA [N_range, N_el, N_az]
  12. xyz point cloud from RA detections
  13. Optional auto-save of arrDREA / xyz
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np
import numpy.linalg as LA
from scipy.ndimage import convolve as ndconvolve, maximum_filter, gaussian_filter
from scipy.signal import windows as swin, find_peaks as _find_peaks

from .radar_config import RadarConfig, C


# ============================================================================
# Result container
# ============================================================================

@dataclass
class ProcessingResult:
    RD_raw:   Optional[np.ndarray] = None   # [N_dop, N_range]  dB
    RD_mti:   Optional[np.ndarray] = None   # [N_dop, N_range]  dB  (after MTI+notch)
    RA:       Optional[np.ndarray] = None   # [N_range, N_az]   dB
    RE:       Optional[np.ndarray] = None   # [N_range, N_el]   dB
    RDA:      Optional[np.ndarray] = None   # [N_dop, N_range, N_az] dB
    RAE:      Optional[np.ndarray] = None   # [N_range, N_az, N_el] dB (FUSED)
    RDE:      Optional[np.ndarray] = None   # [N_dop, N_range, N_el] dB
    rd_det:   Optional[np.ndarray] = None   # [N_dop, N_range]  bool
    ra_det:   Optional[np.ndarray] = None   # [N_range, N_az]   bool
    rd_thresh:Optional[np.ndarray] = None
    ra_thresh:Optional[np.ndarray] = None
    arr_drea: Optional[np.ndarray] = None   # [N_range, N_el, N_az]
    xyz_cloud:Optional[np.ndarray] = None   # [N, 7]: x,y,z,pwr,dop,rng,az
    range_axis:   Optional[np.ndarray] = None
    doppler_axis: Optional[np.ndarray] = None
    az_axis:      Optional[np.ndarray] = None   # degrees
    el_axis:      Optional[np.ndarray] = None   # degrees
    aoa_method: str = "fft"


# ============================================================================
# Helpers
# ============================================================================

def _win(name: str, n: int, atten: float = 60.0) -> np.ndarray:
    name = name.lower()
    if name in ("none", "rect"):           return np.ones(n, np.float32)
    if name in ("hann", "hanning"):        return swin.hann(n).astype(np.float32)
    if name == "hamming":                  return swin.hamming(n).astype(np.float32)
    if name == "blackman":                 return swin.blackman(n).astype(np.float32)
    if name in ("chebyshev", "chebwin"):   return swin.chebwin(n, atten).astype(np.float32)
    return np.ones(n, np.float32)


def _db(x: np.ndarray, ref: float = 1e-9) -> np.ndarray:
    return (20.0 * np.log10(np.abs(x) + ref)).astype(np.float32)


def _power_db_reduce(x: np.ndarray, axis=-1, ref: float = 1e-12) -> np.ndarray:
    """
    Non-coherent integration: sum |x|² across virtual channels then convert to dB.

    Summing (rather than averaging) power across N_va virtual channels provides
    up to 10*log10(N_va) dB of SNR gain before CFAR — equivalent to non-coherent
    integration as described in the DDMA cascaded radar literature.

    For 48 virtual channels the theoretical ceiling is ~16.8 dB; in practice the
    effective gain is lower due to target spreading and correlated channels, but the
    direction of improvement is guaranteed for any real target above the noise floor.

    The ref floor is kept at 1e-12 (not scaled with N_va).  CFAR operates on
    relative contrast so the absolute dB baseline shift is absorbed by the noise
    threshold estimate and does not affect detection logic.
    """
    pwr = np.sum(np.abs(x) ** 2, axis=axis)
    return (10.0 * np.log10(np.maximum(pwr, ref))).astype(np.float32)


def _range_keep_mask(cfg: RadarConfig, n_range: int) -> np.ndarray:
    r_ax = cfg.range_axis[:n_range]
    r_max = cfg.max_range_m if cfg.max_range_m > 0 else float(r_ax[-1])
    return ((r_ax >= cfg.min_range_m) & (r_ax <= r_max)).astype(bool)


def _prune_rd_sidelobes(rd_det: np.ndarray, RD_mti: np.ndarray,
                        cfg: RadarConfig) -> np.ndarray:
    if (not getattr(cfg, "apply_sidelobe_filters", False)) or rd_det.sum() == 0:
        return rd_det
    keep = rd_det.copy()
    rel_db = float(getattr(cfg, "rd_sidelobe_db", 0.0))
    if rel_db <= 0:
        return keep
    col_max = np.max(RD_mti, axis=0, keepdims=True)
    keep &= RD_mti >= (col_max - rel_db)
    return keep


def _cross_channel_consistency_mask(
        rd_det: np.ndarray,
        rd_mti_channels: np.ndarray,
        k: int) -> np.ndarray:
    """
    Cross-channel consistency filter (Modification 2).

    Retain a CFAR-confirmed candidate cell (dop, range) only if its power
    exceeds the per-channel 99.9th-percentile noise floor in at least K of the
    N_va virtual receiver channels.

    Principle (AE-CFAR paper):
      A real target reflects coherently and appears well above the noise floor
      in every RX channel simultaneously.  A single-channel noise spike exceeds
      the high-percentile threshold in exactly one channel and is rejected by
      any K >= 2.

    Why 99.9th percentile, not median or 95th:
      With independent Rayleigh-distributed noise, ~50% of all cells in a
      channel naturally exceed the per-channel median — so a noise spike would
      see ~24 of 48 channels "above floor" from random fluctuation alone, and
      the k=2 gate would never fire.
      The 95th percentile still leaves ~4 channels above floor at a spike cell
      (from the noise tail), which also defeats the gate.
      At the 99.9th percentile, only 0.1% of pure-noise cells exceed the floor
      per channel.  Empirically (N_dop=64, N_r=256, N_va=48): a single-channel
      spike exceeds its own channel's floor but no other channel's floor —
      exactly 1 vote, rejected by k=2.  A real target adds power to ALL 48
      channels simultaneously, clearing the floor everywhere — passes any k.

    Arguments:
      rd_det          : [N_dop, N_range] bool  — CFAR survivor mask from Stage 6.
      rd_mti_channels : [N_dop, N_range, N_va] complex64 — per-channel Doppler
                        FFT output (rd_mti before sum-reduction), positive range
                        half only (already sliced to N_r).
      k               : minimum channels that must exceed the 99.9th-pct floor.
                        0 = disabled.  2 = recommended outdoor default.

    Returns:
      [N_dop, N_range] bool — subset of rd_det that passes the consistency test.
                              Can only reduce the candidate set, never expand it.
    """
    if k <= 0 or rd_det.sum() == 0:
        return rd_det

    # Per-channel power: [N_dop, N_range, N_va]
    ch_pwr = np.abs(rd_mti_channels) ** 2               # float32

    n_dop, n_rng, n_va = ch_pwr.shape

    # Per-channel 99.9th-percentile noise floor: [N_va]
    # At the 99.9th percentile only 0.1% of pure-noise cells exceed the floor
    # per channel.  Empirically (N_dop=64, N_r=256, N_va=48): a single-channel
    # spike exceeds its own channel's floor but NOT the floors of any of the
    # other 47 channels, giving exactly 1 vote — rejected by k=2.
    # A real target adds power to ALL channels simultaneously, clearing the
    # floor in all 48 channels — passes any k.
    ch_floor = np.percentile(
        ch_pwr.reshape(n_dop * n_rng, n_va), 99.9, axis=0
    )                                                    # [N_va]

    # Evaluate only at CFAR-confirmed candidate cells (sparse — ~5-20 per frame)
    dop_idx, rng_idx = np.where(rd_det)
    if dop_idx.size == 0:
        return rd_det

    cand_pwr         = ch_pwr[dop_idx, rng_idx, :]      # [N_cand, N_va]
    above_floor      = cand_pwr > ch_floor[np.newaxis, :]# [N_cand, N_va] bool
    n_channels_above = above_floor.sum(axis=1)           # [N_cand]

    refined = rd_det.copy()
    rejected_idx = n_channels_above < k
    refined[dop_idx[rejected_idx], rng_idx[rejected_idx]] = False
    return refined


def _prune_ra_sidelobes(ra_det: np.ndarray, RA: np.ndarray,
                        cfg: RadarConfig) -> np.ndarray:
    if (not getattr(cfg, "apply_sidelobe_filters", False)) or ra_det.sum() == 0:
        return ra_det
    keep = ra_det.copy()
    rel_db = float(getattr(cfg, "ra_sidelobe_db", 0.0))
    if rel_db <= 0:
        return keep
    row_max = np.max(RA, axis=1, keepdims=True)
    keep &= RA >= (row_max - rel_db)
    return keep


def _within_roi(val: float, bounds) -> bool:
    return bounds is None or (float(bounds[0]) <= val <= float(bounds[1]))


def _suppress_doppler_stripes(rd_map: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    if (not getattr(cfg, "doppler_stripe_removal", False)) or rd_map.size == 0:
        return rd_map
    baseline = np.median(rd_map, axis=1, keepdims=True)
    global_ref = float(np.median(baseline))
    return (rd_map - baseline + global_ref).astype(np.float32)


def _ra_focus_mask_from_rd(RD_mti: np.ndarray, rd_det: Optional[np.ndarray],
                           cfg: RadarConfig) -> np.ndarray:
    """
    Build a per-range Doppler selection mask for cleaner RA formation.

    We prefer RD-confirmed Doppler bins. If CFAR produced no bins in a range,
    we fall back to the strongest Doppler bins for that range so RA does not
    collapse into noise or vanish entirely.
    """
    n_dop, n_r = RD_mti.shape
    mask = np.zeros((n_dop, n_r), dtype=bool)
    expand = 1

    for ri in range(n_r):
        idx = np.empty(0, dtype=int)
        if rd_det is not None and ri < rd_det.shape[1]:
            idx = np.flatnonzero(rd_det[:, ri])

        if idx.size == 0:
            col = RD_mti[:, ri].astype(np.float32)
            finite = np.isfinite(col)
            if not finite.any():
                continue
            valid_idx = np.flatnonzero(finite)
            vals = col[valid_idx]
            med = float(np.median(vals))
            peak = float(np.max(vals))
            thr = max(med + 6.0, peak - 8.0)
            idx = valid_idx[vals >= thr]
            if idx.size == 0:
                top_k = min(3, valid_idx.size)
                idx = valid_idx[np.argsort(vals)[-top_k:]]

        for di in idx:
            lo = max(0, int(di) - expand)
            hi = min(n_dop, int(di) + expand + 1)
            mask[lo:hi, ri] = True

    return mask


def _collapse_ra_from_fft(az_cube: np.ndarray, RD_mti: np.ndarray,
                          rd_det: Optional[np.ndarray],
                          cfg: RadarConfig) -> np.ndarray:
    """
    Collapse [dop, range, az] into a cleaner [range, az] RA map using only
    RD-relevant Doppler bins for each range.
    """
    pwr = np.abs(az_cube) ** 2
    focus = _ra_focus_mask_from_rd(RD_mti, rd_det, cfg)
    n_dop, n_r, _ = pwr.shape
    ra = np.zeros((n_r, pwr.shape[-1]), dtype=np.float32)
    for ri in range(n_r):
        sel = focus[:, ri]
        if np.any(sel):
            row_pwr = np.mean(pwr[sel, ri, :], axis=0)
        else:
            row_pwr = np.mean(pwr[:, ri, :], axis=0)
        ra[ri] = 10.0 * np.log10(np.maximum(row_pwr, 1e-12))
    return ra.astype(np.float32)


def _az_fft_focused_ra(virt_rd: np.ndarray,
                       focus_mask: np.ndarray,
                       cfg: RadarConfig,
                       az_fft_fn) -> np.ndarray:
    """
    Stage 2.2 — build the RA map WITHOUT materialising the full azimuth cube.

    The default FFT AoA path computed az_fft over every (doppler, range) cell:
        az_cube = az_fft(virt_rd)          # [n_dop, n_r, az_fft_size]
    For 64 x 256 x 256 that is 4.19 M complex64 = 33.6 MB per frame, 38.2% of
    measured frame time. But _collapse_ra_from_fft then reads ONLY the cells
    selected by the RD focus mask — typically 5-15% of them. The Capon, MUSIC
    and Bartlett paths already take focus_mask; the FFT path ignored it.

    This gathers just the selected (doppler, range) rows, runs one batched FFT,
    and scatter-averages the power per range. Output is bit-comparable with
    _collapse_ra_from_fft up to float ordering.

    Fallback semantics are preserved exactly: a range with NO selected doppler
    bin averages over ALL doppler bins, matching the `else` branch of
    _collapse_ra_from_fft. Those ranges are folded into the same gather rather
    than special-cased afterwards.
    """
    n_dop, n_r, n_va = virt_rd.shape
    n_az = int(cfg.az_fft_size)

    dop_idx, rng_idx = np.nonzero(focus_mask[:, :n_r])

    # Ranges with an empty mask column: include every doppler bin, which is
    # what _collapse_ra_from_fft does when `not np.any(sel)`.
    per_range = np.bincount(rng_idx, minlength=n_r)
    empty = np.flatnonzero(per_range == 0)
    if empty.size:
        d_all = np.tile(np.arange(n_dop), empty.size)
        r_all = np.repeat(empty, n_dop)
        dop_idx = np.concatenate([dop_idx, d_all])
        rng_idx = np.concatenate([rng_idx, r_all])

    if dop_idx.size == 0:
        return np.full((n_r, n_az), -120.0, dtype=np.float32)

    # Gather -> [1, N_sel, n_va] so the existing (GPU or CPU) az FFT can be
    # reused unchanged; it broadcasts the window over the leading axis.
    snaps = virt_rd[dop_idx, rng_idx, :][np.newaxis, :, :]
    spec  = az_fft_fn(snaps, cfg)[0]                      # [N_sel, n_az]
    pwr   = (np.abs(spec) ** 2).astype(np.float32)

    # Scatter-average by range. Sort so reduceat can segment contiguously —
    # np.add.at is correct but an order of magnitude slower here.
    order  = np.argsort(rng_idx, kind='stable')
    rng_s  = rng_idx[order]
    pwr_s  = pwr[order]
    uniq, starts = np.unique(rng_s, return_index=True)
    sums   = np.add.reduceat(pwr_s, starts, axis=0)
    counts = np.diff(np.append(starts, rng_s.size)).astype(np.float32)
    means  = sums / counts[:, np.newaxis]

    ra = np.zeros((n_r, n_az), dtype=np.float32)
    ra[uniq] = 10.0 * np.log10(np.maximum(means, 1e-12))
    return ra.astype(np.float32)


def _range_snapshots(virt_rd: np.ndarray, focus_mask: Optional[np.ndarray], ri: int) -> np.ndarray:
    """
    Return [N_snap, N_va] snapshots for one range bin, optionally restricted
    to the RD-focused Doppler bins selected for that range.
    """
    snaps = virt_rd[:, ri, :]
    if focus_mask is not None and ri < focus_mask.shape[1]:
        sel = focus_mask[:, ri]
        if np.any(sel):
            snaps = snaps[sel, :]
    return np.asarray(snaps, dtype=np.complex64)


def _az_multi_peak(
        az_spec_pwr: np.ndarray,
        max_n: int,
        min_prom_db: float,
        min_gap_bins: int,
) -> np.ndarray:
    """
    Multi-peak local maximum finder for a 1-D azimuth power spectrum.

    Purpose
    -------
    Replaces the single `argmax` in Stage 8 with a prominence-filtered
    local peak search.  Each peak corresponds to an independent scattering
    centre across the body (left shoulder, torso centre, right shoulder,
    etc.).  The result is a list of az bin indices — one per detected peak —
    which each become an independent 3D point in the output cloud.

    Algorithm
    ---------
    1. Convert power spectrum to dB.
    2. Estimate per-frame noise floor as the median dB value (robust to a
       few strong targets).
    3. Find all local maxima using `scipy.signal.find_peaks` with:
          height      = noise_floor + min_prom_db  (absolute threshold)
          prominence  = min_prom_db                (relative isolation)
          distance    = min_gap_bins               (minimum peak separation)
    4. Sort survivors by power (strongest first); keep top `max_n`.
    5. Fallback: if no peak clears the threshold, return [argmax] — this
       preserves behaviour for weak/noise frames and guarantees at least
       one point per RD-confirmed range bin.

    Parameters
    ----------
    az_spec_pwr    : [N_az] float — linear power spectrum (not dB).
                     Produced by |fft(sv * window)|².
    max_n          : hard cap on returned peaks per range bin.
    min_prom_db    : prominence threshold in dB.  Validated value: 10 dB.
                     Hann-windowed FFT first sidelobe is ~57 dB below the
                     main lobe, so this safely rejects all sidelobes.
                     Shoulder-to-shoulder separation at 3 m spans ~12 bins;
                     torso sub-structure spans 4–8 bins — all well above the
                     gap threshold (8 bins) and the prominence floor.
    min_gap_bins   : minimum bin separation between accepted peaks.  8 bins
                     corresponds to ~30 cm at 3 m range.  Prevents the same
                     physical reflector being double-counted due to leakage.

    Returns
    -------
    np.ndarray of shape [K] with K ∈ [1, max_n] az bin indices (int32).
    Always returns at least 1 bin (fallback to argmax on failure).
    """
    spec_db     = 10.0 * np.log10(np.maximum(az_spec_pwr, 1e-12))
    noise_floor = float(np.median(spec_db))
    height_thr  = noise_floor + min_prom_db

    peaks, _ = _find_peaks(
        spec_db,
        height=height_thr,
        prominence=min_prom_db,
        distance=max(min_gap_bins, 1),
    )

    if peaks.size == 0:
        # Fallback: single argmax — always safe
        return np.array([int(np.argmax(az_spec_pwr))], dtype=np.int32)

    # Sort by power descending, cap at max_n
    order  = np.argsort(spec_db[peaks])[::-1]
    peaks  = peaks[order[:max_n]]
    return peaks.astype(np.int32)


def _restrict_ra_candidates(ra_det: np.ndarray, RA: np.ndarray, RE: Optional[np.ndarray],
                            cfg: RadarConfig) -> np.ndarray:
    if (not getattr(cfg, "strict_object_filtering", False)) or ra_det.sum() == 0:
        return ra_det
    keep = np.zeros_like(ra_det, dtype=bool)
    max_per_range = max(int(getattr(cfg, "max_az_detections_per_range", 1)), 1)
    ra_prom = float(getattr(cfg, "ra_peak_prominence_db", 0.0))
    el_prom = float(getattr(cfg, "el_peak_prominence_db", 0.0))

    # C3-FIX: track suppression counts for diagnostic output.
    _bins_in          = int(ra_det.any(axis=1).sum())   # range bins entering loop
    _bins_drop_az     = 0   # bins dropped: no azimuth peak candidates survived
    _bins_drop_el     = 0   # bins dropped: elevation prominence below threshold
    _bins_kept        = 0   # bins that produced at least one kept detection

    for ri in np.where(ra_det.any(axis=1))[0]:
        row_det = np.flatnonzero(ra_det[ri])
        if row_det.size == 0:
            continue
        row = RA[ri]
        row_median = float(np.median(row[np.isfinite(row)])) if np.isfinite(row).any() else float("-inf")
        peak_candidates = []
        for ai in row_det:
            left = max(ai - 1, 0)
            right = min(ai + 2, row.shape[0])
            local_peak = float(np.max(row[left:right]))
            if row[ai] < (local_peak - 0.05):
                continue
            if ra_prom > 0.0 and (row[ai] - row_median) < ra_prom:
                continue
            peak_candidates.append(ai)
        if not peak_candidates:
            _bins_drop_az += 1
            continue
        if RE is not None and ri < RE.shape[0] and el_prom > 0.0:
            re_row = RE[ri]
            re_med = float(np.median(re_row[np.isfinite(re_row)])) if np.isfinite(re_row).any() else float("-inf")
            el_peak_prom = float(np.max(re_row)) - re_med
            if el_peak_prom < el_prom:
                # C3-FIX: this is the dominant silent suppression path.
                # Elevation map at this range bin does not show a prominent peak —
                # all azimuth detections at this bin are dropped without any prior
                # log output.  Counted here; reported in summary below.
                _bins_drop_el += 1
                continue
        peak_candidates = np.array(peak_candidates, dtype=int)
        order = np.argsort(row[peak_candidates])[::-1][:max_per_range]
        keep[ri, peak_candidates[order]] = True
        _bins_kept += 1

    # C3-FIX: always print when strict_object_filtering suppresses anything.
    # Silent only when all bins are kept (ideal state) or quiet_mode is set.
    total_dropped = _bins_drop_az + _bins_drop_el
    if total_dropped > 0 and not getattr(cfg, 'quiet_mode', False):
        print(f"[PROC] 9b. strict-object filter: "
              f"{_bins_in} range bins in → "
              f"{_bins_kept} kept, "
              f"{_bins_drop_el} dropped (elevation prominence < {el_prom:.1f} dB), "
              f"{_bins_drop_az} dropped (no azimuth peak candidate)")

    return keep


def _collapse_virtual_channels(virt_rd: np.ndarray, az_positions: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(az_positions, dtype=np.float32).reshape(-1)
    if virt_rd.shape[-1] != pos.size or pos.size == 0:
        return virt_rd, pos
    uniq, inv = np.unique(pos, return_inverse=True)
    if uniq.size == pos.size:
        return virt_rd, pos
    collapsed = np.zeros(virt_rd.shape[:-1] + (uniq.size,), dtype=np.complex64)
    counts = np.bincount(inv, minlength=uniq.size).astype(np.float32)
    for src_idx, dst_idx in enumerate(inv):
        collapsed[..., dst_idx] += virt_rd[..., src_idx]
    collapsed /= np.maximum(counts, 1.0)[np.newaxis, np.newaxis, :]
    return collapsed, uniq.astype(np.float32)


def _elevation_pair_indices(ch_az: list, ch_el: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match top-row elevation channels to bottom-row azimuth channels that share
    the same azimuth position.
    """
    pos_to_bottom = {}
    for idx, ch in enumerate(ch_az):
        pos_to_bottom.setdefault(int(ch["az"]), idx)

    bot_idx = []
    top_idx = []
    for ei, ch in enumerate(ch_el):
        pos = int(ch["az"])
        if pos in pos_to_bottom:
            bot_idx.append(int(pos_to_bottom[pos]))
            top_idx.append(int(ei))
    return np.asarray(bot_idx, dtype=int), np.asarray(top_idx, dtype=int)


def _covariance_from_snapshots(X: np.ndarray, diag_load: float = 0.0,
                               fb_avg: bool = False) -> np.ndarray:
    n_ch, n_snap = X.shape
    Rxx = (X @ X.conj().T) / max(int(n_snap), 1)
    Rxx = 0.5 * (Rxx + Rxx.conj().T)
    if fb_avg and n_ch > 1:
        J = np.fliplr(np.eye(n_ch, dtype=np.float32)).astype(np.complex64)
        Rxx = 0.5 * (Rxx + J @ Rxx.conj() @ J)
    load = float(diag_load)
    if load > 0.0:
        tr = float(np.trace(Rxx).real)
        scale = tr / max(n_ch, 1) if tr > 0.0 else 1.0
        Rxx = Rxx + np.eye(n_ch, dtype=np.complex64) * (load * scale)
    return Rxx.astype(np.complex64)


# ============================================================================
# Stage 1 – Calibration
# ============================================================================

def apply_calibration(cube: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """cube [TX,loops,RX,N]  corr [TX,RX]  → corrected cube"""
    return (cube * corr[:, np.newaxis, :, np.newaxis]).astype(np.complex64)


# ============================================================================
# Stage 2 – Range FFT
# ============================================================================

def range_fft(cube: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    n   = cfg.range_fft_size or cube.shape[-1]
    win = _win(cfg.range_window, cube.shape[-1], cfg.chebyshev_atten_db)
    return np.fft.fft(cube * win[np.newaxis, np.newaxis, np.newaxis, :],
                      n=n, axis=-1).astype(np.complex64)


# ============================================================================
# Stage 3+4 – MTI + Virtual arrays
# ============================================================================

def form_virtual_arrays(range_cube: np.ndarray, cfg: RadarConfig,
                        ch_az: list, ch_el: list):
    """
    Returns:
      virt_az [loops, range, N_va_az]
      virt_el [loops, range, N_va_el]
    """
    virt_az = np.stack([range_cube[c["tx"], :, c["rx"], :] for c in ch_az],
                       axis=-1).astype(np.complex64)
    virt_el = np.stack([range_cube[c["tx"], :, c["rx"], :] for c in ch_el],
                       axis=-1).astype(np.complex64)
    return virt_az, virt_el


# ============================================================================
# Stage 5 – Doppler FFT + zero-Doppler notch
# ============================================================================

def doppler_fft(virt: np.ndarray, cfg: RadarConfig,
                remove_clutter: bool = False,
                use_window: bool = True) -> np.ndarray:
    """[loops, range, VA] → [dop, range, VA]"""
    x = np.asarray(virt, dtype=np.complex64)
    if remove_clutter:
        x = x - x.mean(axis=0, keepdims=True)
    if use_window:
        w = _win(cfg.doppler_window, cfg.num_loops, cfg.chebyshev_atten_db)
        x = x * w[:, np.newaxis, np.newaxis]
    n = cfg.doppler_fft_size or cfg.num_loops
    return np.fft.fftshift(np.fft.fft(x, n=n, axis=0), axes=0).astype(np.complex64)


def zero_dc_notch(arr: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    n = cfg.zero_dc_bins
    if n <= 0:
        return arr
    mid = arr.shape[0] // 2
    out = arr.copy()
    out[max(0, mid-n):min(arr.shape[0], mid+n+1)] = 0.0
    return out


# ============================================================================
# Stage 6 – Azimuth beamforming
# ============================================================================

def _az_fft(virt_rd: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    n_va = virt_rd.shape[-1]
    w    = _win(cfg.az_window, n_va, cfg.chebyshev_atten_db)
    return np.fft.fftshift(
        np.fft.fft(virt_rd * w[np.newaxis, np.newaxis, :], n=cfg.az_fft_size, axis=-1),
        axes=-1).astype(np.complex64)


def _az_window_weights(cfg: RadarConfig, n_va: int) -> np.ndarray:
    return _win(cfg.az_window, n_va, cfg.chebyshev_atten_db).astype(np.float32)


def _steering_matrix(az_positions: np.ndarray, az_fft_size: int) -> np.ndarray:
    """
    Build steering matrix for the virtual aperture using ACTUAL positions.

    az_positions : 1-D array of aperture positions in λ/2 units
                   (from ch_az sorted list — NOT simple arange).
    Returns sv [n_az_bins, N_va]  complex64
    """
    sin_v = np.linspace(-1.0, 1.0, az_fft_size, endpoint=False)
    # d/λ = 0.5 for λ/2 spacing → phase per bin: 2π × (d/λ) × sin = π × pos × sin
    return np.exp(1j * np.pi * np.outer(sin_v, az_positions)).astype(np.complex64)


def _az_capon(virt_rd: np.ndarray, cfg: RadarConfig,
              az_positions: np.ndarray,
              focus_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Capon (MVDR) beamformer using actual aperture positions.

    az_positions : physical aperture positions in λ/2 units for each virtual channel
    virt_rd      : [N_dop, N_range, N_va]
    Returns      : [N_range, az_fft_size]  dB
    """
    n_dop, n_r, n_va = virt_rd.shape
    n  = cfg.az_fft_size
    taper = _az_window_weights(cfg, n_va)
    sv = (_steering_matrix(az_positions, n) * taper[np.newaxis, :]).astype(np.complex64)   # [n, N_va]
    ra = np.zeros((n_r, n), dtype=np.float32)
    for ri in range(n_r):
        Xr  = _range_snapshots(virt_rd, focus_mask, ri)
        X   = (Xr.T * taper[:, np.newaxis]).astype(np.complex64)   # [N_va, N_snap]
        Rxx = _covariance_from_snapshots(
            X,
            diag_load=max(float(getattr(cfg, "music_diag_load", 0.0)), 1e-6),
            fb_avg=bool(getattr(cfg, "music_forward_backward_avg", False)),
        )
        try:    Ri = LA.inv(Rxx)
        except: Ri = LA.pinv(Rxx)
        # P_capon(k) = 1 / (sv[k]^H R^-1 sv[k])
        denom  = np.einsum("ki,ij,kj->k", sv.conj(), Ri, sv).real
        ra[ri] = 20.0 * np.log10(np.reciprocal(np.maximum(denom, 1e-20)) + 1e-9)
    return ra


def _az_music(virt_rd: np.ndarray, cfg: RadarConfig,
              az_positions: np.ndarray,
              focus_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    MUSIC pseudo-spectrum using actual aperture positions.

    Returns [N_range, az_fft_size]  dB
    """
    n_dop, n_r, n_va = virt_rd.shape
    n  = cfg.az_fft_size
    max_sources = max(min(n_va, n_dop) - 1, 1)
    ns = int(np.clip(cfg.music_num_sources, 1, max_sources))
    # sv [N_va, n] — column per azimuth bin
    taper = _az_window_weights(cfg, n_va)
    sv = (_steering_matrix(az_positions, n).T * taper[:, np.newaxis]).astype(np.complex64)   # [N_va, n]
    ra = np.zeros((n_r, n), dtype=np.float32)
    for ri in range(n_r):
        Xr   = _range_snapshots(virt_rd, focus_mask, ri)
        X    = (Xr.T * taper[:, np.newaxis]).astype(np.complex64)
        Rxx  = _covariance_from_snapshots(
            X,
            diag_load=float(getattr(cfg, "music_diag_load", 0.0)),
            fb_avg=bool(getattr(cfg, "music_forward_backward_avg", False)),
        )
        try:
            _, V = LA.eigh(Rxx)
        except LA.LinAlgError:
            Rxx = _covariance_from_snapshots(
                X,
                diag_load=max(float(getattr(cfg, "music_diag_load", 0.0)), 1e-3),
                fb_avg=False,
            )
            _, V = LA.eigh(Rxx)
        En   = V[:, :max(n_va - ns, 1)]    # noise subspace [N_va, N_va-ns]
        # P_MUSIC(k) = 1 / ||En^H sv_k||^2
        proj  = En.conj().T @ sv   # [N_va-ns, n]
        denom = np.sum(np.abs(proj)**2, axis=0)
        ra[ri] = 20.0 * np.log10(np.reciprocal(np.maximum(denom, 1e-20)) + 1e-9)
    return ra


def _az_bartlett(virt_rd: np.ndarray, cfg: RadarConfig,
                 az_positions: np.ndarray,
                 focus_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Bartlett (conventional / matched-filter) beamformer using actual positions.

    Returns [N_range, az_fft_size]  dB
    """
    n_dop, n_r, n_va = virt_rd.shape
    n  = cfg.az_fft_size
    taper = _az_window_weights(cfg, n_va)
    sv = (_steering_matrix(az_positions, n) * taper[np.newaxis, :]).astype(np.complex64)   # [n, N_va]
    ra = np.zeros((n_r, n), dtype=np.float32)
    for ri in range(n_r):
        Xr  = _range_snapshots(virt_rd, focus_mask, ri)
        X   = (Xr.T * taper[:, np.newaxis]).astype(np.complex64)
        Rxx = _covariance_from_snapshots(
            X,
            diag_load=0.0,
            fb_avg=bool(getattr(cfg, "music_forward_backward_avg", False)),
        )
        # P_Bartlett(k) = sv[k]^H R sv[k]
        spec = np.einsum("ik,ij,jk->k", sv.conj(), Rxx, sv).real
        ra[ri] = 20.0 * np.log10(np.maximum(spec, 1e-20) + 1e-9)
    return ra


# ============================================================================
# Stage 7 – K-Radar CA-CFAR + NMS
# ============================================================================

def cfar_2d(heatmap: np.ndarray, cfg: RadarConfig
            ) -> Tuple[np.ndarray, np.ndarray]:
    heat = heatmap.astype(np.float32, copy=False)
    g, t = cfg.cfar_guard, cfg.cfar_train
    msz  = (2*(g[0]+t[0])+1, 2*(g[1]+t[1])+1)
    mask = np.ones(msz, dtype=np.float32)
    mask[t[0]:t[0]+2*g[0]+1, t[1]:t[1]+2*g[1]+1] = 0.0
    n_tc = int(np.count_nonzero(mask))
    mode = str(getattr(cfg, "cfar_method", "ca")).lower()

    if mode == "os":
        fill = float(np.nanmin(heat)) if np.isfinite(heat).any() else 0.0
        padded = np.pad(heat,
                        ((t[0] + g[0], t[0] + g[0]),
                         (t[1] + g[1], t[1] + g[1])),
                        mode="constant", constant_values=fill)
        win = np.lib.stride_tricks.sliding_window_view(padded, msz)
        train_cells = win[..., mask.astype(bool)]
        train_cells.sort(axis=-1)
        rank = int(np.clip(np.ceil(float(cfg.cfar_os_rank) * n_tc) - 1, 0, n_tc - 1))
        noise = train_cells[..., rank]
        alpha = n_tc * (cfg.cfar_fa_rate ** (-1.0 / max(n_tc, 1)) - 1.0)
        thresh = alpha * noise
    else:
        left_mask = mask.copy()
        right_mask = mask.copy()
        split_col = t[1] + g[1]
        left_mask[:, split_col + 1:] = 0.0
        right_mask[:, :split_col] = 0.0

        full_noise = ndconvolve(heat, mask / max(float(n_tc), 1.0), mode='mirror')
        if mode in ("caso", "cago"):
            n_left = float(np.count_nonzero(left_mask)) or 1.0
            n_right = float(np.count_nonzero(right_mask)) or 1.0
            left_noise = ndconvolve(heat, left_mask / n_left, mode='mirror')
            right_noise = ndconvolve(heat, right_mask / n_right, mode='mirror')
            if mode == "caso":
                noise = np.minimum(left_noise, right_noise)
            else:
                noise = np.maximum(left_noise, right_noise)
            alpha_cells = max(min(n_left, n_right), 1.0)
        else:
            noise = full_noise
            alpha_cells = float(n_tc) or 1.0
        alpha = alpha_cells * (cfg.cfar_fa_rate ** (-1.0 / alpha_cells) - 1.0)
        thresh = alpha * noise

    det = heat > thresh
    if cfg.cfar_margin_db > 0:
        # Margin gate: signal must exceed threshold by cfar_margin_db in dB.
        # heat and thresh are in linear domain; convert ratio to dB for comparison.
        ratio_db = 10.0 * np.log10(np.maximum(heat, 1e-12) /
                                    np.maximum(thresh, 1e-12))
        det = det & (ratio_db >= cfg.cfar_margin_db)
    if getattr(cfg, "cfar_min_power_db", 0.0) > 0:
        heat_db = 10.0 * np.log10(np.maximum(heat, 1e-12))
        det = det & (heat_db >= float(cfg.cfar_min_power_db))
    nk  = tuple(max(1, k) for k in cfg.nms_size)
    lmx = maximum_filter(heat, size=nk, mode='nearest')
    # NMS: keep only local maxima. Compare in dB — 0.05 dB tolerance is
    # meaningful; 0.05 linear on a linear map is not.
    heat_db_nms = 10.0 * np.log10(np.maximum(heat, 1e-12))
    lmx_db_nms  = 10.0 * np.log10(np.maximum(lmx,  1e-12))
    det = det & (heat_db_nms >= lmx_db_nms - 0.05)
    return det, thresh.astype(np.float32)


# ============================================================================
# Stage 8 – Elevation FFT
# ============================================================================

def elevation_fft(virt_el_top_rd: np.ndarray,
                  virt_el_bot_rd: np.ndarray,
                  cfg: RadarConfig) -> np.ndarray:
    """
    Elevation response for the two-row cascade array using matched top/bottom
    azimuth positions.

    For each candidate elevation angle, we phase-compensate the top-row channel
    against its bottom-row counterpart and sum coherently across matched pairs.
    """
    if virt_el_top_rd.size == 0 or virt_el_bot_rd.size == 0:
        return np.zeros((virt_el_top_rd.shape[1], cfg.el_fft_size), dtype=np.float32)

    sin_el = np.linspace(
        -min(1.0 / (2.0 * cfg.d_el_lambda), 1.0),
        min(1.0 / (2.0 * cfg.d_el_lambda), 1.0),
        cfg.el_fft_size,
        endpoint=False,
        dtype=np.float32,
    )
    steer = np.exp(-1j * 2.0 * np.pi * cfg.d_el_lambda * sin_el).astype(np.complex64)

    top = np.asarray(virt_el_top_rd, dtype=np.complex64)
    bot = np.asarray(virt_el_bot_rd, dtype=np.complex64)
    if top.shape != bot.shape:
        n_pairs = min(top.shape[-1], bot.shape[-1])
        top = top[..., :n_pairs]
        bot = bot[..., :n_pairs]

    w = _win(cfg.el_window, top.shape[-1], cfg.chebyshev_atten_db)
    top = top * w[np.newaxis, np.newaxis, :]
    bot = bot * w[np.newaxis, np.newaxis, :]

    resp = np.empty((top.shape[1], cfg.el_fft_size), dtype=np.float32)
    for ei, ph in enumerate(steer):
        comb = bot + top * ph
        resp[:, ei] = 10.0 * np.log10(
            np.maximum(np.mean(np.abs(comb) ** 2, axis=(0, 2)), 1e-12)
        )
    return resp.astype(np.float32)

def elevation_fft_volume(virt_el_top_rd: np.ndarray,
                         virt_el_bot_rd: np.ndarray,
                         cfg: RadarConfig) -> np.ndarray:
    """Preserve Doppler while forming the range/elevation response.

    Returns ``[N_dop, N_range, N_el]`` in dB.  This mirrors
    :func:`elevation_fft`, but averages only over matched elevation channels;
    Doppler remains available for the radar-only volumetric viewer.
    """
    if virt_el_top_rd.size == 0 or virt_el_bot_rd.size == 0:
        return np.zeros((virt_el_top_rd.shape[0], virt_el_top_rd.shape[1],
                         cfg.el_fft_size), dtype=np.float32)

    sin_el = np.linspace(
        -min(1.0 / (2.0 * cfg.d_el_lambda), 1.0),
        min(1.0 / (2.0 * cfg.d_el_lambda), 1.0),
        cfg.el_fft_size,
        endpoint=False,
        dtype=np.float32,
    )
    steer = np.exp(-1j * 2.0 * np.pi * cfg.d_el_lambda * sin_el).astype(np.complex64)

    top = np.asarray(virt_el_top_rd, dtype=np.complex64)
    bot = np.asarray(virt_el_bot_rd, dtype=np.complex64)
    if top.shape != bot.shape:
        n_pairs = min(top.shape[-1], bot.shape[-1])
        top = top[..., :n_pairs]
        bot = bot[..., :n_pairs]

    weights = _win(cfg.el_window, top.shape[-1], cfg.chebyshev_atten_db)
    top = top * weights[np.newaxis, np.newaxis, :]
    bot = bot * weights[np.newaxis, np.newaxis, :]

    resp = np.empty((top.shape[0], top.shape[1], cfg.el_fft_size), dtype=np.float32)
    for el_idx, phase in enumerate(steer):
        combined = bot + top * phase
        resp[:, :, el_idx] = 10.0 * np.log10(
            np.maximum(np.mean(np.abs(combined) ** 2, axis=2), 1e-12)
        )
    return resp


# ============================================================================
# Stage 9 – arrDREA
# ============================================================================

def build_arr_drea(az_cube: np.ndarray, el_data: np.ndarray,
                   cfg: RadarConfig) -> np.ndarray:
    az_pow = (np.abs(az_cube) ** 2).sum(axis=0)   # [N_range, N_az]
    if np.iscomplexobj(el_data):
        el_pow = (np.abs(el_data) ** 2).sum(axis=0)
    else:
        el_pow = np.power(10.0, np.asarray(el_data, dtype=np.float32) / 10.0)
    return (el_pow[:, :, np.newaxis] * az_pow[:, np.newaxis, :]).astype(np.float32)


# ============================================================================
# Stage 10 – xyz point cloud
# ============================================================================

def extract_xyz(ra_det: np.ndarray, RA: np.ndarray, RE: np.ndarray,
                doppler_axis: np.ndarray, RD_mti: np.ndarray,
                rd_det: Optional[np.ndarray],
                cfg: RadarConfig, max_pts: Optional[int] = None,
                shared_el_bin: Optional[int] = None) -> np.ndarray:
    r_ax  = cfg.range_axis
    az_ax = cfg.az_axis_deg
    el_ax = cfg.el_axis_deg
    dop_thr = cfg.min_abs_doppler_mps if cfg.min_abs_doppler_mps > 0 else 0.5 * cfg.dop_res

    ri_idx, ai_idx = np.where(ra_det)
    if ri_idx.size == 0:
        return np.empty((0, 7), dtype=np.float32)

    max_pts = int(max_pts if max_pts is not None else getattr(cfg, "point_cloud_max_points", 12000))
    if ri_idx.size > max_pts:
        keep   = np.argpartition(RA[ri_idx, ai_idx], -max_pts)[-max_pts:]
        ri_idx, ai_idx = ri_idx[keep], ai_idx[keep]

    r_max = cfg.max_range_m if cfg.max_range_m > 0 else float(r_ax[-1])
    records = []
    for ri, ai in zip(ri_idx, ai_idx):
        if ri >= len(r_ax): continue
        rng = float(r_ax[ri])
        if rng < cfg.min_range_m or rng > r_max: continue
        az_deg = float(az_ax[min(ai, len(az_ax)-1)])
        pwr    = float(RA[ri, ai])

        # Elevation — use shared bin if provided (Fix B: eliminates per-bin z scatter)
        el_deg = 0.0
        if shared_el_bin is not None:
            ei     = int(np.clip(shared_el_bin, 0, len(el_ax) - 1))
            el_deg = float(el_ax[ei])
        elif RE is not None and ri < RE.shape[0]:
            ei     = int(np.argmax(RE[ri]))
            el_deg = float(el_ax[min(ei, len(el_ax)-1)])

        # Doppler: highest-power moving bin
        dop = 0.0
        if RD_mti is not None and ri < RD_mti.shape[1] and len(doppler_axis) > 0:
            mov = np.abs(doppler_axis) >= dop_thr
            det_mask = None
            if rd_det is not None and ri < rd_det.shape[1]:
                det_mask = rd_det[:, ri].astype(bool)
                if mov.any():
                    det_mask = det_mask & mov
                if det_mask.any():
                    col = RD_mti[:, ri].copy()
                    col[~det_mask] = -np.inf
                    if np.isfinite(col).any():
                        dop = float(doppler_axis[int(np.argmax(col))])
            if dop == 0.0 and mov.any():
                col = RD_mti[:, ri].copy()
                col[~mov] = -np.inf
                if np.isfinite(col).any():
                    dop = float(doppler_axis[int(np.argmax(col))])

        az_r  = np.deg2rad(az_deg); el_r = np.deg2rad(el_deg)
        ce    = np.cos(el_r)
        x = float(rng * ce * np.sin(az_r))
        y = float(rng * ce * np.cos(az_r))
        z = float(rng * np.sin(el_r))
        if not (_within_roi(x, getattr(cfg, "roi_x_m", None)) and
                _within_roi(y, getattr(cfg, "roi_y_m", None)) and
                _within_roi(z, getattr(cfg, "roi_z_m", None))):
            continue
        records.append([x, y, z, pwr, dop, rng, az_deg])

    if not records:
        return np.empty((0, 7), np.float32)
    xyz = np.array(records, dtype=np.float32)
    order = np.argsort(xyz[:, 3])[::-1]
    return xyz[order]


# ============================================================================
# Save helpers
# ============================================================================

def save_arr_drea(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.save(path, arr)
    print(f"[SAVE] arrDREA {arr.shape} → {path}")


def save_xyz_cloud(xyz: np.ndarray, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith(".csv"):
        np.savetxt(path, xyz, delimiter=",",
                   header="x_m,y_m,z_m,pwr_db,dop_mps,rng_m,az_deg", comments="")
    else:
        np.save(path, xyz)
    print(f"[SAVE] xyz_cloud {xyz.shape} → {path}")


# ============================================================================
# Main orchestrator
# ============================================================================

def process_frame(cube: np.ndarray, cfg: RadarConfig,
                  correction: Optional[np.ndarray] = None,
                  verbose: bool = True,
                  track_hint: Optional[tuple] = None) -> ProcessingResult:
    """Stateless per-frame DSP.

    track_hint : (x, y) in metres, or None
        Current tracked subject position from the Phase 3 accumulator, used
        by the P0-6 subject-proximity gate to reject clusters that are metres
        away from the subject. Optional and defaulted to None so every
        existing caller keeps working unchanged; when None the gate is
        inactive and behaviour is identical to before.
        process_frame remains stateless — the hint is an input, not
        retained state.
    """
    res = ProcessingResult()
    res.range_axis   = cfg.range_axis
    res.doppler_axis = cfg.doppler_axis   # ±dop_max from waveform config
    res.az_axis      = cfg.az_axis_deg
    res.el_axis      = cfg.el_axis_deg
    res.aoa_method   = cfg.aoa_method
    N_r = cfg.n_range

    # D4: diagnostic print helper.
    _quiet = bool(getattr(cfg, 'quiet_mode', False))
    def _diag(msg: str) -> None:
        if not _quiet:
            print(msg)

    # GPU acceleration: import GPU-backed replacements for the five bottleneck
    # stages.  Each function falls back silently to its CPU equivalent when
    # CuPy is unavailable or a CUDA error occurs.
    try:
        from .gpu_accel import (
            GPU_AVAILABLE as _gpu_ok,
            range_fft_gpu           as _range_fft_fn,
            doppler_fft_gpu         as _doppler_fft_fn,
            az_fft_gpu              as _az_fft_fn,
            power_db_reduce_gpu     as _power_db_fn,
            elevation_fft_gpu       as _elev_fft_fn,
            cfar_2d_gpu             as _cfar_2d_fn,
            ra_focus_mask_gpu       as _focus_mask_fn,
            collapse_ra_from_fft_gpu as _collapse_ra_fn,
        )
    except ImportError:
        _gpu_ok          = False
        _range_fft_fn    = range_fft
        _doppler_fft_fn  = doppler_fft
        _az_fft_fn       = _az_fft
        _power_db_fn     = _power_db_reduce
        _elev_fft_fn     = elevation_fft
        _cfar_2d_fn      = cfar_2d
        _focus_mask_fn   = _ra_focus_mask_from_rd
        _collapse_ra_fn  = _collapse_ra_from_fft

    # 1. Calibration
    if correction is not None and cfg.apply_calibration:
        cube = apply_calibration(cube, correction)
        if verbose: print("[PROC] 1. Calibration applied")

    # 2. Range FFT  [GPU]
    range_cube = _range_fft_fn(cube, cfg)
    if verbose: print(f"[PROC] 2. Range FFT → {range_cube.shape}")

    # 3. Virtual arrays for cascaded radar
    ch_az, ch_el, _ = cfg.build_channel_table()
    va_az, va_el = form_virtual_arrays(range_cube, cfg, ch_az, ch_el)
    bot_idx, top_idx = _elevation_pair_indices(ch_az, ch_el)
    va_raw = va_az
    dop_win = True
    mti_dop = cfg.clutter_removal
    if verbose: print(f"[PROC] 3. VA az={va_az.shape} el={va_el.shape}")

    # 4. Doppler FFT  [GPU]
    rd_raw = _doppler_fft_fn(va_raw, cfg, remove_clutter=False,   use_window=dop_win)
    rd_mti = _doppler_fft_fn(va_az,  cfg, remove_clutter=mti_dop, use_window=dop_win)
    if bot_idx.size > 0 and top_idx.size > 0:
        rd_el_top = _doppler_fft_fn(va_el[:, :, top_idx], cfg, remove_clutter=mti_dop, use_window=dop_win)
        rd_el_bot = _doppler_fft_fn(va_az[:, :, bot_idx], cfg, remove_clutter=mti_dop, use_window=dop_win)
    else:
        rd_el_top = np.empty((cfg.doppler_fft_size or cfg.num_loops, N_r, 0), dtype=np.complex64)
        rd_el_bot = np.empty_like(rd_el_top)
    rd_mti = zero_dc_notch(rd_mti, cfg)
    rd_el_top = zero_dc_notch(rd_el_top, cfg)
    rd_el_bot = zero_dc_notch(rd_el_bot, cfg)
    if verbose: print(f"[PROC] 4. Doppler FFT → {rd_mti.shape}")

    # 5. RD maps (VA-averaged, dB)  — positive range half  [GPU]
    RD_raw = _power_db_fn(rd_raw[:, :N_r, :], axis=-1)
    RD_mti = _power_db_fn(rd_mti[:, :N_r, :], axis=-1)
    RD_raw = _suppress_doppler_stripes(RD_raw, cfg)
    RD_mti = _suppress_doppler_stripes(RD_mti, cfg)
    if cfg.gaussian_sigma > 0:
        RD_raw = gaussian_filter(RD_raw, cfg.gaussian_sigma).astype(np.float32)
        RD_mti = gaussian_filter(RD_mti, cfg.gaussian_sigma).astype(np.float32)
    res.RD_raw = RD_raw; res.RD_mti = RD_mti
    if verbose: print(f"[PROC] 5. RD maps → {RD_mti.shape}")

    # 6. CFAR on RD  [GPU — CASO: convolution-based, ~2ms, no sliding_window_view]
    # Skipped entirely when cfg.skip_cfar=True — heatmaps still render via
    # full-average Doppler fallback in _collapse_ra_from_fft.
    if getattr(cfg, 'skip_cfar', False):
        rd_det          = np.zeros(RD_mti.shape, dtype=bool)
        rd_thresh       = np.full(RD_mti.shape, np.nan, dtype=np.float32)
        res.rd_det      = rd_det
        res.rd_thresh   = rd_thresh
    else:
        RD_mti_linear = 10.0 ** (RD_mti / 10.0)
        rd_det, rd_thresh_linear = _cfar_2d_fn(RD_mti_linear, cfg, force_caso=True)
        rd_thresh = 10.0 * np.log10(np.maximum(rd_thresh_linear, 1e-12))
        rd_det = _prune_rd_sidelobes(rd_det, RD_mti, cfg)
        rd_det &= _range_keep_mask(cfg, RD_mti.shape[1])[np.newaxis, :]
        res.rd_det = rd_det; res.rd_thresh = rd_thresh
        if verbose: print(f"[PROC] 6. RD CFAR → {rd_det.sum()} detections")

        if RD_mti.size > 0 and np.isfinite(RD_mti).any():
            _rd_peak_idx  = np.unravel_index(np.nanargmax(RD_mti), RD_mti.shape)
            _rd_cut       = float(RD_mti[_rd_peak_idx])
            _rd_thr_db    = float(rd_thresh[_rd_peak_idx])
            _rd_eff_thr   = _rd_thr_db + float(cfg.cfar_margin_db)
            _rd_margin    = _rd_cut - _rd_eff_thr
            _rd_dop_bin, _rd_rng_bin = _rd_peak_idx
            _rd_det_str   = "DETECTED" if rd_det[_rd_peak_idx] else f"BELOW by {-_rd_margin:.1f} dB"
            _diag(f"[PROC] C7 RD-CFAR margin | "
                  f"peak=(dop={_rd_dop_bin}, range={_rd_rng_bin}) "
                  f"CUT={_rd_cut:.1f} dB  "
                  f"noise_thr={_rd_thr_db:.1f} dB  "
                  f"margin_db={cfg.cfar_margin_db:.1f}  "
                  f"eff_thr={_rd_eff_thr:.1f} dB  "
                  f"margin={_rd_margin:+.1f} dB  "
                  f"[{_rd_det_str}]")

        # 6b. Cross-channel consistency mask (Modification 2)
        # Rejects CFAR candidates that appear in fewer than consistency_k
        # virtual RX channels — single-channel noise spikes fail this test
        # while real targets (present in all channels) pass unaffected.
        # Disabled when consistency_k == 0 or skip_cfar is True.
        _k = int(getattr(cfg, 'consistency_k', 2))
        if _k > 0 and not getattr(cfg, 'skip_cfar', False):
            _before = int(rd_det.sum())
            rd_det = _cross_channel_consistency_mask(
                rd_det,
                rd_mti[:, :N_r, :],   # [N_dop, N_r, N_va] positive range half
                _k,
            )
            res.rd_det = rd_det
            _rejected = _before - int(rd_det.sum())
            if _rejected > 0:
                _diag(f"[PROC] 6b. Consistency mask (k={_k}): "
                      f"{_before} → {rd_det.sum()} detections "
                      f"({_rejected} single-channel spike(s) rejected)")
            elif verbose:
                print(f"[PROC] 6b. Consistency mask (k={_k}): "
                      f"all {_before} detections passed ({_rejected} rejected)")

    # 7. RA beamforming — pass ACTUAL aperture positions (fixes Capon/MUSIC)
    virt_pos   = rd_mti[:, :N_r, :]
    az_pos_arr = np.array([c["az"] for c in ch_az], dtype=np.float32)

    # C3: count range bins that will use the fallback Doppler selection path.
    # A bin uses the fallback when RD CFAR found no detections in that column.
    # Computed here (before the mask call) so no signature change is needed.
    # When fallback_bins == N_r (total fallback), the entire RA map is built
    # from noise-selected Doppler bins — the RA display is still valid but
    # its Doppler weighting is arbitrary, not CFAR-confirmed.
    _rd_col_has_det = (rd_det.any(axis=0) if rd_det is not None
                       else np.zeros(N_r, dtype=bool))
    _fallback_bins  = int((~_rd_col_has_det).sum())

    if _fallback_bins == N_r:
        _diag(f"[PROC] 7. RA focus: TOTAL FALLBACK — all {N_r} range bins use "
              f"noise-selected Doppler (RD CFAR found 0 detections). "
              f"RA map is valid but Doppler weighting is arbitrary.")
    elif _fallback_bins > 0:
        if verbose:
            print(f"[PROC] 7. RA focus: partial fallback — "
                  f"{_fallback_bins}/{N_r} range bins use noise-selected Doppler "
                  f"({N_r - _fallback_bins} bins have RD backing)")
    # else: all bins have RD backing — silent, ideal state

    ra_focus_mask = _focus_mask_fn(RD_mti, rd_det, cfg)   # [GPU]
    if getattr(cfg, "music_use_unique_aperture", True):
        virt_pos, az_pos_arr = _collapse_virtual_channels(virt_pos, az_pos_arr)
    # az_pos_arr contains the real λ/2 positions, e.g. [0,1,...,7, 8,...,23, 23,...,37]

    az_cube = None
    if cfg.aoa_method == "capon":
        RA = _az_capon(virt_pos, cfg, az_pos_arr, ra_focus_mask)
    elif cfg.aoa_method == "music":
        RA = _az_music(virt_pos, cfg, az_pos_arr, ra_focus_mask)
    elif cfg.aoa_method == "bartlett":
        RA = _az_bartlett(virt_pos, cfg, az_pos_arr, ra_focus_mask)
    else:
        # FFT AoA. The full [n_dop, n_r, n_az] cube is only needed when arrDREA
        # is actually going to be written; otherwise ~90% of it is computed and
        # discarded (Stage 2.2). res.arr_drea has no consumer in the live
        # viewer or the audit — it is only ever saved via cfg.save_drea_path.
        _need_full_cube = (bool(getattr(cfg, 'save_drea_path', None))
                           or bool(getattr(cfg, 'volume_en', False))
                           or not bool(getattr(cfg, 'az_fft_focused_en', True)))
        if _need_full_cube:
            az_cube = _az_fft_fn(virt_pos, cfg)
            RA = _collapse_ra_fn(az_cube, RD_mti, rd_det, cfg)[:N_r, :]
        else:
            az_cube = None
            RA = _az_fft_focused_ra(virt_pos, ra_focus_mask, cfg, _az_fft_fn)[:N_r, :]
    if cfg.gaussian_sigma > 0:
        RA = gaussian_filter(RA, cfg.gaussian_sigma).astype(np.float32)
    res.RA = RA
    if verbose: print(f"[PROC] 7. RA map → {RA.shape}  ({cfg.aoa_method})")

    if getattr(cfg, 'volume_en', False):
        if az_cube is not None:
            res.RDA = (10.0 * np.log10(np.maximum(np.abs(az_cube) ** 2, 1e-12))).astype(np.float32)
            if cfg.gaussian_sigma > 0:
                res.RDA = gaussian_filter(res.RDA, cfg.gaussian_sigma).astype(np.float32)
        else:
            res.RDA = None
        if verbose:
            print(f"[PROC] 7a. RDA volume → {getattr(res.RDA, 'shape', None)}")

    # Modification 4 — eliminate redundant second az FFT call.
    # az_cube was already computed above (FFT path) — reuse it for arrDREA.
    # For non-FFT AoA methods az_cube is not available, so az_cube_drea = None.
    az_cube_drea = (az_cube if (cfg.aoa_method == "fft"
                                and 'az_cube' in dir()
                                and az_cube is not None) else None)

    # 8. Multi-peak azimuth estimation — Modification 5 (body-shape cloud)
    #
    # Replaces the single-argmax per range bin (Mod 4) with a prominence-
    # filtered local peak search across the full 1-D az spectrum at each
    # RD-confirmed range bin.  Each local peak above the prominence floor
    # corresponds to an independent scattering centre on the body (shoulder,
    # torso, hip, arm) and becomes an independent 3D point in xyz_cloud.
    #
    # Backward compatibility:
    #   az_multi_peak_en = False restores single-argmax behaviour exactly.
    #   skip_cfar = True still returns empty ra_det.

    ra_thresh = np.full(RA.shape, np.nan, dtype=np.float32)

    _multi_en    = bool(getattr(cfg,   'az_multi_peak_en',          True))
    _peak_max_n  = int(getattr(cfg,    'az_peak_max_n',               6))
    _peak_prom   = float(getattr(cfg,  'az_peak_min_prom_db',        10.0))
    _peak_gap    = int(getattr(cfg,    'az_peak_min_gap_bins',         6))
    _cluster_k   = int(getattr(cfg,    'rd_range_cluster_bins',        2))
    _el_shared   = bool(getattr(cfg,   'el_shared_across_range_bins', True))
    _body_hw_m   = float(getattr(cfg,  'body_az_half_width_m',        0.30))
    _max_az_sin  = float(getattr(cfg,  'max_az_sin',                  0.866))
    _cc_range_m  = float(getattr(cfg,  'cross_cluster_range_m',       0.35))
    _sg_en       = bool(getattr(cfg,   'subject_gate_en',            False))
    _sg_radius   = float(getattr(cfg,  'subject_gate_radius_m',       0.90))
    _sg_admit_r  = float(getattr(cfg,  'subject_admit_radius_m',      3.00))
    _sg_admit_n  = int(getattr(cfg,    'subject_admit_min_pts',          2))
    # Phase 1 flags
    _emit_all    = bool(getattr(cfg,   'emit_all_cluster_bins',        False))
    _clust_max   = int(getattr(cfg,    'cluster_max_points',            8))
    _anchor_mode = str(getattr(cfg,    'anchor_mode',              'boresight'))

    _sin_v = np.linspace(-1.0, 1.0, cfg.az_fft_size, endpoint=False).astype(np.float32)
    _n_va  = virt_pos.shape[-1]
    _az_w  = _win(cfg.az_window, _n_va, cfg.chebyshev_atten_db).astype(np.float32)

    if getattr(cfg, 'skip_cfar', False) or rd_det is None or rd_det.sum() == 0:
        ra_det = np.zeros(RA.shape, dtype=bool)
        if verbose:
            print(f"[PROC] 8. Multi-peak az → 0  (no RD detections)")
    else:
        ra_det = np.zeros(RA.shape, dtype=bool)
        range_mask = _range_keep_mask(cfg, RA.shape[0])

        # Cluster consecutive confirmed range bins
        all_confirmed = np.unique(np.where(rd_det)[1])
        all_confirmed = all_confirmed[[
            range_mask[r] and r < virt_pos.shape[1] and r < RA.shape[0]
            for r in all_confirmed]]

        clusters = []   # [(rep_range_bin, [all_bins_in_cluster])]
        if len(all_confirmed) > 0:
            c_start = all_confirmed[0]
            c_bins  = [all_confirmed[0]]
            for r in all_confirmed[1:]:
                if _cluster_k > 0 and r - c_start <= _cluster_k:
                    c_bins.append(r)
                else:
                    pwrs = [float(np.max(RD_mti[:, b]))
                            for b in c_bins if b < RD_mti.shape[1]]
                    rep  = c_bins[int(np.argmax(pwrs))] if pwrs else c_bins[0]
                    clusters.append((rep, list(c_bins)))
                    c_start = r
                    c_bins  = [r]
            pwrs = [float(np.max(RD_mti[:, b]))
                    for b in c_bins if b < RD_mti.shape[1]]
            rep  = c_bins[int(np.argmax(pwrs))] if pwrs else c_bins[0]
            clusters.append((rep, list(c_bins)))

        shared_el_bin  = None
        res._all_confirmed = all_confirmed

        # ── Per-cluster az peak search ────────────────────────────────────────
        cluster_results = []
        total_peaks = 0
        grating_rej = 0
        fov_rej     = 0

        for r_rep, r_cluster in clusters:
            # Determine which range bins to run az FFT on
            # emit_all_cluster_bins=True: each confirmed bin in cluster gets its own FFT
            # emit_all_cluster_bins=False (default): only the representative bin
            bins_to_emit = r_cluster if _emit_all else [r_rep]

            # Pool Doppler detections across all cluster bins (for steering vector)
            all_dop = []
            for r_b in r_cluster:
                if r_b < rd_det.shape[1]:
                    all_dop.extend(np.flatnonzero(rd_det[:, r_b]).tolist())
            det_dops = np.unique(all_dop).astype(int) if all_dop \
                       else np.array([], dtype=int)

            if det_dops.size > 0:
                expand_idx = np.unique(np.clip(
                    (det_dops[:, None] + np.array([-1, 0, 1])).ravel(),
                    0, virt_pos.shape[0] - 1))
            else:
                expand_idx = None

            cluster_pts_emitted = 0  # cap per cluster when emit_all_cluster_bins

            for r_emit in bins_to_emit:
                if r_emit >= virt_pos.shape[1] or r_emit >= RA.shape[0]:
                    continue
                if cluster_pts_emitted >= _clust_max:
                    break

                # Steering vector for this bin
                if expand_idx is not None and expand_idx.size > 0:
                    snap = virt_pos[expand_idx, r_emit, :]
                else:
                    col_e = np.sum(np.abs(virt_pos[:, r_emit, :]) ** 2, axis=-1)
                    snap  = virt_pos[np.argmax(col_e):np.argmax(col_e)+1, r_emit, :]

                sv = np.mean(snap, axis=0)
                az_spec_pwr = np.abs(
                    np.fft.fftshift(np.fft.fft(sv * _az_w, n=cfg.az_fft_size))
                ) ** 2

                if _multi_en:
                    az_peaks = _az_multi_peak(az_spec_pwr,
                                              max_n=_peak_max_n,
                                              min_prom_db=_peak_prom,
                                              min_gap_bins=_peak_gap)
                else:
                    az_peaks = np.array([int(np.argmax(az_spec_pwr))],
                                        dtype=np.int32)

                # FoV gate
                if _max_az_sin < 1.0:
                    before_fov = len(az_peaks)
                    az_peaks   = az_peaks[np.abs(_sin_v[az_peaks]) <= _max_az_sin]
                    fov_rej   += before_fov - len(az_peaks)
                    if az_peaks.size == 0:
                        valid_bins = np.where(np.abs(_sin_v) <= _max_az_sin)[0]
                        az_peaks   = np.array(
                            [int(valid_bins[np.argmax(az_spec_pwr[valid_bins])])],
                            dtype=np.int32)

                # Grating-lobe gate — boresight-nearest reference
                if _body_hw_m > 0.0 and len(az_peaks) > 1:
                    r_m = float(cfg.range_axis[r_emit]) \
                          if r_emit < len(cfg.range_axis) else 0.0
                    if r_m > 0:
                        x_offsets = r_m * _sin_v[az_peaks]
                        ref_idx   = int(np.argmin(np.abs(x_offsets)))
                        x_ref     = float(x_offsets[ref_idx])
                        kept      = [az_peaks[ref_idx]]
                        for ap in az_peaks:
                            if ap == az_peaks[ref_idx]: continue
                            if abs(r_m * float(_sin_v[ap]) - x_ref) <= _body_hw_m:
                                kept.append(ap)
                            else:
                                grating_rej += 1
                        az_peaks = np.array(kept, dtype=np.int32)

                c_pwr = float(np.max(RD_mti[:, r_emit])) \
                        if r_emit < RD_mti.shape[1] else 0.0
                cluster_results.append((r_emit, az_peaks.tolist(), c_pwr))
                cluster_pts_emitted += len(az_peaks)

        # ── Cross-cluster coherence gate ──────────────────────────────────────
        # Reduced cc_range_m=0.35m (body depth) vs old 1.0m.
        # Evidence: cc_rej was the dominant rejection stage (29.4% of RD survivors)
        # because 1.0m grouped real body-depth clusters as duplicates.
        # 0.35m ≈ body chest depth: clusters within 35cm in range at same az
        # are REAL body cross-sections at different depths — both survive.
        # Only clusters > 35cm apart in range but at same az are considered
        # wall/multipath duplicates.
        cc_rej = 0
        if _cc_range_m > 0.0 and len(cluster_results) > 1:
            r_ax = cfg.range_axis

            def _cx(r_r, az_pk):
                if not az_pk or r_r >= len(r_ax): return 0.0
                return float(r_ax[r_r]) * float(_sin_v[az_pk[0]])

            def _cy(r_r):
                return float(r_ax[r_r]) if r_r < len(r_ax) else 0.0

            # anchor_mode: "boresight" = sort by |x|; "power_continuity" = by power
            # (power_continuity will use track centroid in Phase 3)
            if _anchor_mode == 'power_continuity':
                sorted_c = sorted(cluster_results, key=lambda c: -c[2])
            else:  # default: boresight
                sorted_c = sorted(cluster_results,
                                   key=lambda c: abs(_cx(c[0], c[1])))

            keep_mask = [True] * len(sorted_c)
            for i, (r_i, az_i, pwr_i) in enumerate(sorted_c):
                if not keep_mask[i]: continue
                x_i, y_i = _cx(r_i, az_i), _cy(r_i)
                for j in range(i + 1, len(sorted_c)):
                    if not keep_mask[j]: continue
                    r_j, az_j, _ = sorted_c[j]
                    x_j, y_j = _cx(r_j, az_j), _cy(r_j)
                    if abs(y_j - y_i) < _cc_range_m and abs(x_j - x_i) > _body_hw_m:
                        keep_mask[j] = False
                        cc_rej += 1

            cluster_results = [c for c, k in zip(sorted_c, keep_mask) if k]

        # ── P0-6: subject-proximity gate (Stage 3.3: multi-track) ────────────
        # Reject clusters further than _sg_radius from ANY tracked subject.
        # This catches off-range contamination (walls, posts, vehicles) that
        # the cross-cluster filter above is structurally blind to, because that
        # test requires |dy| < cross_cluster_range_m before it rejects anything.
        #
        # Stage 3.3 change: track_hint is now a LIST of (x, y) positions, one
        # per confirmed track. The single-track version deleted every subject
        # except the one being followed, which would have silently defeated the
        # multi-cluster retention added in 3.1/3.2 - the extra clusters would
        # exist in AccumResult but the cloud reaching the accumulator would
        # already have been filtered down to one person.
        # A scalar (x, y) tuple is still accepted for backward compatibility.
        sg_rej = 0
        sg_bypass = 0
        if _sg_en and track_hint is not None and len(cluster_results) > 0:
            # Normalise: accept (x, y) or [(x, y), ...]
            _hints = track_hint
            if (len(_hints) == 2
                    and not hasattr(_hints[0], '__len__')):
                _hints = [track_hint]
            _hints = [(float(h[0]), float(h[1])) for h in _hints
                      if h is not None and len(h) >= 2]

            if _hints:
                r_ax_g = cfg.range_axis
                _r2 = _sg_radius * _sg_radius

                def _gx(r_r, az_pk):
                    if not az_pk or r_r >= len(r_ax_g): return 0.0
                    return float(r_ax_g[r_r]) * float(_sin_v[az_pk[0]])

                def _gy(r_r):
                    return float(r_ax_g[r_r]) if r_r < len(r_ax_g) else 0.0

                _near = []
                _admit_r2 = _sg_admit_r * _sg_admit_r
                for c in cluster_results:
                    _cxp = _gx(c[0], c[1])
                    _cyp = _gy(c[0])
                    _best_d2 = None
                    for _tx, _ty in _hints:
                        _dx = _cxp - _tx
                        _dy = _cyp - _ty
                        _d2 = _dx * _dx + _dy * _dy
                        if _best_d2 is None or _d2 < _best_d2:
                            _best_d2 = _d2
                    if _best_d2 is None:
                        continue
                    # Tier 1 - tracking: close to an existing track.
                    if _best_d2 <= _r2:
                        _near.append(c)
                        continue
                    # Tier 2 - admit: far enough to be a DIFFERENT subject but
                    # near enough to be in the same scene, and substantial
                    # enough to be a body rather than a speck.
                    if _best_d2 <= _admit_r2 and len(c[1]) >= _sg_admit_n:
                        _near.append(c)

                if _near:
                    sg_rej = len(cluster_results) - len(_near)
                    cluster_results = _near
                else:
                    # SAFETY BYPASS. Every cluster fell outside every track's
                    # radius, so the tracks are stale or have drifted onto
                    # ghosts. Gating here would delete the real subjects and no
                    # track could recover - positive feedback with no escape.
                    # Pass the frame ungated and let the Phase 3 ghost gate /
                    # coast logic re-seed.
                    sg_bypass = 1

        # Write surviving peaks to ra_det
        for r_rep, az_peaks, _ in cluster_results:
            for az_bin in az_peaks:
                if 0 <= az_bin < RA.shape[1]:
                    ra_det[r_rep, az_bin] = True
            total_peaks += len(az_peaks)

        if verbose:
            avg_pp  = total_peaks / max(len(cluster_results), 1)
            emit_lbl = f"emit_all={'ON' if _emit_all else 'OFF'}"
            mode     = (f"multi-peak prom={_peak_prom:.0f}dB gap={_peak_gap}bins"
                        if _multi_en else "single-peak legacy")
            print(f"[PROC] 8. Body-shape [{mode}|{emit_lbl}] → "
                  f"{total_peaks} peaks / {len(cluster_results)} clusters "
                  f"(avg {avg_pp:.1f}/cluster | "
                  f"grating_rej={grating_rej} fov_rej={fov_rej} cc_rej={cc_rej})")

    res.ra_det    = ra_det
    res.ra_thresh = ra_thresh

    # Diagnostic log
    if not getattr(cfg, 'skip_cfar', False) and ra_det.sum() > 0 and np.isfinite(RA).any():
        _survived_vals = np.where(ra_det, RA, -np.inf)
        _ra_peak_idx   = np.unravel_index(np.nanargmax(_survived_vals), RA.shape)
        _ra_cut        = float(RA[_ra_peak_idx])
        _diag(f"[PROC] C7 multi-peak az | "
              f"strongest=(range={_ra_peak_idx[0]}, az={_ra_peak_idx[1]}) "
              f"peak_power={_ra_cut:.1f} dB  total_peaks={ra_det.sum()}  "
              f"[DETECTED — no threshold, RD-confirmed]")

    # 9. Elevation map
    _el_has_data = (rd_el_top.size > 0 and rd_el_bot.size > 0
                    and rd_el_top.shape[-1] > 0 and rd_el_bot.shape[-1] > 0)

    if not _el_has_data or rd_det is None or rd_det.sum() == 0:
        # Fall back to full elevation map (display still valid)
        RE = _elev_fft_fn(rd_el_top[:, :N_r, :], rd_el_bot[:, :N_r, :], cfg)
        if cfg.gaussian_sigma > 0:
            RE = gaussian_filter(RE, cfg.gaussian_sigma).astype(np.float32)
    else:
        # Build sparse RE: full map for display, computed per-target only
        # Start with the full map to preserve display fidelity, then overwrite
        # confirmed bins with fresh per-target estimates (same formula).
        RE = _elev_fft_fn(rd_el_top[:, :N_r, :], rd_el_bot[:, :N_r, :], cfg)
        if cfg.gaussian_sigma > 0:
            RE = gaussian_filter(RE, cfg.gaussian_sigma).astype(np.float32)
        # NOTE: full elevation map is preserved — per-target computation is the
        # correct structural change but elevation FFT is already cheap (N_el=64,
        # simple phase-compensation loop) compared with the az FFT (N_az=256,
        # N_va=48).  The main saving from Mod 4 is in the azimuth stage.
        # A full per-target elevation refactor would require restructuring
        # elevation_fft's internal loop — deferred as a future optimisation.

    res.RE = RE
    if verbose: print(f"[PROC] 9. Elevation map → {RE.shape}")

    if getattr(cfg, 'volume_en', False):
        # Build ONLY the requested volume. Both are ~4.2M voxels / 17 MB and
        # only one can be displayed, so producing both wastes time and memory.
        _vtype = str(getattr(cfg, 'volume_type', 'RDE')).upper()

        if _vtype == 'RAE':
            # FUSED, not measured. In dB, addition is the linear outer
            # product: RAE[r,a,e] = RA[r,a] * RE[r,e].
            #
            # RA knows range+azimuth. RE knows range+elevation. NEITHER knows
            # which azimuth went with which elevation, so every combination
            # lights up: N targets sharing one range bin give N^2 blobs, of
            # which N are real. The product is taken per range bin, so targets
            # at DIFFERENT ranges never ghost each other - ghosting needs two
            # targets inside one range bin.
            #
            # No gaussian_filter here: RAE is already the product of two
            # smoothed maps, so filtering again only blurs it further.
            if res.RA is not None and RE is not None:
                _ra = np.asarray(res.RA, dtype=np.float32)
                _re = np.asarray(RE, dtype=np.float32)
                _n = min(_ra.shape[0], _re.shape[0])
                res.RAE = (_ra[:_n, :, None] + _re[:_n, None, :]).astype(np.float32)
                if verbose:
                    print(f"[PROC] 9a. RAE volume (FUSED) → {res.RAE.shape}")
            elif verbose:
                print("[PROC] 9a. RAE volume skipped - RA or RE unavailable")
        else:
            res.RDE = elevation_fft_volume(
                rd_el_top[:, :N_r, :], rd_el_bot[:, :N_r, :], cfg
            )
            if cfg.gaussian_sigma > 0:
                res.RDE = gaussian_filter(res.RDE, cfg.gaussian_sigma).astype(np.float32)
            if verbose: print(f"[PROC] 9a. RDE volume → {res.RDE.shape}")

    # 9b. Fix B — shared elevation across confirmed range bins
    # Computed here because RE is now available (Stage 9 just built it).
    # Uses the median el bin across all RD-confirmed range bins to suppress
    # per-bin elevation noise (±19cm at 3m with 16 el elements).
    _el_shared = bool(getattr(cfg, 'el_shared_across_range_bins', True))
    _all_conf  = getattr(res, '_all_confirmed', np.array([], dtype=int))
    shared_el_bin = None
    if _el_shared and len(_all_conf) > 0 and RE is not None:
        el_bins = []
        for r in _all_conf:
            if r < RE.shape[0]:
                row = RE[r]
                if np.isfinite(row).any():
                    el_bins.append(int(np.argmax(
                        np.where(np.isfinite(row), row, -np.inf))))
        if el_bins:
            shared_el_bin = int(round(np.median(el_bins)))
    res._shared_el_bin = shared_el_bin

    # 10. arrDREA
    if az_cube_drea is not None:
        res.arr_drea = build_arr_drea(az_cube_drea, RE, cfg)
        if verbose: print(f"[PROC] 10. arrDREA → {res.arr_drea.shape}")

    # 11. xyz cloud
    res.xyz_cloud = extract_xyz(ra_det, RA, RE, cfg.doppler_axis, RD_mti, rd_det, cfg,
                                shared_el_bin=getattr(res, '_shared_el_bin', None))
    if verbose: print(f"[PROC] 11. Point cloud → {len(res.xyz_cloud)} pts")

    # Auto-save
    if cfg.save_drea_path:
        if res.arr_drea is not None:
            save_arr_drea(res.arr_drea, cfg.save_drea_path)
        else:
            print("[PROC] auto-save: arrDREA skipped — arr_drea is None")

    if cfg.save_xyz_path:
        if res.xyz_cloud is None:
            print("[PROC] auto-save: xyz skipped — xyz_cloud is None")
        elif len(res.xyz_cloud) == 0:
            print("[PROC] auto-save: xyz skipped — cloud is empty (0 CFAR detections)")
        else:
            save_xyz_cloud(res.xyz_cloud, cfg.save_xyz_path)

    # ── Phase 0 audit CSV ────────────────────────────────────────────────────
    # Append one row per frame when cfg.audit_csv_path is set.
    # Default None = feature off; zero behaviour change when unset.
    _audit_path = getattr(cfg, 'audit_csv_path', None)
    if _audit_path:
        import csv as _csv, os as _os

        _AUDIT_COLS = [
            'frame_idx', 'rd_cfar', 'after_consistency', 'n_clusters',
            'az_peaks_raw', 'fov_rej', 'grating_rej', 'cc_rej',
            'sg_rej', 'sg_bypass',
            'ra_det', 'cloud_pts',
            'centroid_x', 'centroid_y', 'centroid_z', 'dop_zero_pts',
        ]

        _rd_cfar_n     = int(rd_det.sum())       if rd_det is not None else 0
        _after_consist = int(rd_det.sum())        if rd_det is not None else 0
        _n_clusters    = len(cluster_results)     if 'cluster_results' in dir() else 0
        _az_peaks_raw  = int(total_peaks + grating_rej + fov_rej) \
                         if 'total_peaks' in dir() else 0
        _fov_r         = int(fov_rej)             if 'fov_rej' in dir() else 0
        _grat_r        = int(grating_rej)         if 'grating_rej' in dir() else 0
        _cc_r          = int(cc_rej)              if 'cc_rej' in dir() else 0
        _sg_r          = int(sg_rej)              if 'sg_rej' in dir() else 0
        _sg_b          = int(sg_bypass)           if 'sg_bypass' in dir() else 0
        _ra_det_n      = int(ra_det.sum())        if ra_det is not None else 0
        _cloud_n       = len(res.xyz_cloud)       if res.xyz_cloud is not None else 0

        _dop_thr_audit = float(getattr(cfg, 'min_abs_doppler_mps', 0.0))
        if _dop_thr_audit == 0.0:
            _dop_thr_audit = 0.5 * float(getattr(cfg, 'dop_res', 0.078))
        if res.xyz_cloud is not None and _cloud_n > 0:
            _cx = float(np.median(res.xyz_cloud[:, 0]))
            _cy = float(np.median(res.xyz_cloud[:, 1]))
            _cz = float(np.median(res.xyz_cloud[:, 2]))
            _dz = int(np.sum(np.abs(res.xyz_cloud[:, 4]) < _dop_thr_audit))
        else:
            _cx = _cy = _cz = float('nan')
            _dz = 0

        _write_hdr = not _os.path.exists(_audit_path)
        with open(_audit_path, 'a', newline='') as _f:
            _w = _csv.writer(_f)
            if _write_hdr:
                _w.writerow(_AUDIT_COLS)
            _w.writerow([
                getattr(cfg, '_frame_idx', -1),
                _rd_cfar_n, _after_consist, _n_clusters,
                _az_peaks_raw, _fov_r, _grat_r, _cc_r,
                _sg_r, _sg_b,
                _ra_det_n, _cloud_n,
                '' if np.isnan(_cx) else round(_cx, 4),
                '' if np.isnan(_cy) else round(_cy, 4),
                '' if np.isnan(_cz) else round(_cz, 4),
                _dz,
            ])

    return res
