"""
Diagnostic: per-frame stationarity report.

Reads each frame_*.bin in a folder, computes the range-FFT, and reports:
  - static_dB    : 10*log10(sum |mean_over_loops(R_fft)|^2)
                    -> energy of the part of the scene that is coherent across loops
                       (true static reflectors + DC offsets + persistent firmware state)
  - slow_var_dB  : 10*log10(sum var_over_loops(|R_fft|^2))
                    -> energy of the part that varies frame-to-frame within a frame
                       (thermal noise + moving targets + intra-frame calibration jitter)
  - raw_mean     : mean |IQ| in ADC LSBs (sanity vs capture --frame-stats)
  - file_bytes   : actual on-disk frame size (capture fidelity check)
  - sat_frac     : fraction of int16 samples at or near ADC rail (|s| > 32000)

Interpretation guide:
  * Stable static_dB, stable slow_var_dB
        -> input is stationary; CFAR=0 in static scene is meaningful.
  * Varying static_dB, stable slow_var_dB
        -> RF/TX gain is wandering frame-to-frame. Firmware-side calibration.
  * Stable static_dB, varying slow_var_dB
        -> noise figure / AGC wandering inside the frame.
  * Both varying together
        -> input-power scaling drift (clock/data-path scaling) or compressionCfg
           reaching LVDS.

WARM-UP EXCLUSION:
  Two distinct warm-up concepts must not be conflated:

    Firmware warm-up  : PLL/ADC settling frames discarded by the CAPTURE
                        SOFTWARE before writing any frame_*.bin to disk.
                        If your capture tool handled this (e.g. a 60-frame
                        pre-roll that was never saved), those frames are gone
                        and --warmup must NOT re-account for them.

    Script-side skip  : Additional leading frames in the saved set to exclude
                        from summary statistics.  Set --warmup N only if the
                        capture tool saved its own pre-roll alongside the
                        measurement frames.

  Default is --warmup 0: all saved frames are measurement frames.
  Excluded frames are still printed with a "W" prefix.
  The boundary is set by --warmup, never inferred from data.

PERIODICITY DETECTION:
  Autocorrelation of raw_mean across measurement frames is computed and the lag
  of the first significant peak (r >= --autocorr-threshold, default 0.5) is
  printed.  This distinguishes H9 (compression) from H10 (firmware cal cycle)
  after a clean-compression capture.

ADC SATURATION CHECK:
  Any sat_frac > 0 in a noise-floor capture (empty space, compression off)
  indicates the ADC input is not in the linear range.  This invalidates the
  stationarity metrics regardless of compressionCfg state.

TIMING CHECK (ADD-D):
  If frame filenames embed a unix timestamp (frame_<idx>_<ts>.bin) or a
  sidecar file named timestamps.txt exists in the folder (one float per line,
  seconds), inter-frame intervals are computed and printed.  Constant intervals
  support H14 falsification; jittered intervals support H14.

This script does NOT use the DSP pipeline (no MTI, no Doppler FFT, no CFAR,
no AoA, no calibration, no fallbacks). It reads bytes, reshapes, range-FFTs,
and reduces. That is all.

LVDS INTERLEAVE ASSUMPTION (Issue 2):
  _read_frame_int16 reshapes the flat int16 stream as:
      [chirps_per_frame, num_devices, num_rx_per_dev, num_samples, 2]
  This assumes the DCA1000EVM cascade capture packs samples in device-major,
  rx-within-device-minor order, consistent with TI MMWAVE-SDK cascade
  capture format (SWRU546, section 3.2).  The assertion
      num_devices * num_rx_per_dev == num_rx
  is checked at startup.  If your capture uses a different interleave order
  (e.g. rx-interleaved across devices) this reshape produces wrong channel
  assignments WITHOUT raising an error -- verify against your DCA1000 config.

Usage:
    python diag_frame_stationarity.py --folder capture/frames
    python diag_frame_stationarity.py --folder capture/frames \\
        --cfg firmware/r017m/cascade_r017m.cfg \\
        --warmup 60 \\
        --stable-threshold-db 2.0 \\
        --autocorr-threshold 0.5
"""
from __future__ import annotations

import argparse
import os
import sys
import importlib.util
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load processing/radar_config.py directly without triggering __init__.py
# (which eagerly imports matplotlib via radar_plots).  This script is a
# pure-numpy diagnostic and must not depend on the plotting stack.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_RC_PATH = _HERE / "radar_config.py"
_spec = importlib.util.spec_from_file_location("_diag_radar_config", _RC_PATH)
_rc = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None, "Cannot load radar_config.py -- check _HERE path"
sys.modules["_diag_radar_config"] = _rc
_spec.loader.exec_module(_rc)
RadarConfig = _rc.RadarConfig
parse_cfg_file = _rc.parse_cfg_file


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_geom(cfg_path: str | None) -> RadarConfig:
    if cfg_path and os.path.isfile(cfg_path):
        return parse_cfg_file(cfg_path)
    return RadarConfig()


def _validate_geom(cfg: RadarConfig) -> None:
    """
    Assert the LVDS interleave geometry is self-consistent.

    FIX (Issue 2): The reshape in _read_frame_int16 assumes
        num_devices * num_rx_per_dev == num_rx
    If this is false, the per-device block layout is wrong and channel
    assignments will be silently corrupted.  Fail loudly here rather than
    silently produce garbage metrics.
    """
    expected_rx = cfg.num_devices * cfg.num_rx_per_dev
    if expected_rx != cfg.num_rx:
        raise RuntimeError(
            f"Geometry inconsistency: num_devices({cfg.num_devices}) * "
            f"num_rx_per_dev({cfg.num_rx_per_dev}) = {expected_rx} "
            f"!= num_rx({cfg.num_rx}).  "
            f"Check RadarConfig or --cfg for this capture."
        )


# ---------------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------------

def _read_frame_int16(path: Path, cfg: RadarConfig) -> np.ndarray:
    """
    Return raw int16 array shaped [chirps, num_devices, num_rx_per_dev, num_samples, 2].

    FIX (Issue 5): Explicit file-size check before fromfile.
    A file larger than expected means a capture trailer is being silently
    truncated, which would misalign IQ pairs.  A file smaller than expected
    means a dropped-write partial frame.  Both are fatal for this diagnostic.
    """
    num_rx       = cfg.num_rx
    num_samples  = cfg.num_samples
    chirps       = cfg.chirps_per_frame
    n_int16      = chirps * num_rx * num_samples * 2
    expected_bytes = n_int16 * 2  # int16 = 2 bytes

    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"{path.name}: file size mismatch: {actual_bytes:,} bytes on disk, "
            f"expected {expected_bytes:,} ({n_int16} int16s = "
            f"chirps={chirps} * rx={num_rx} * samples={num_samples} * 2 IQ * 2 bytes). "
            f"Possible capture trailer or partial write."
        )

    raw = np.fromfile(path, dtype=np.int16, count=n_int16)
    # Redundant after size check, but kept as belt-and-braces.
    if raw.size != n_int16:
        raise RuntimeError(
            f"{path.name}: short read ({raw.size} vs {n_int16}) "
            f"despite correct file size -- filesystem issue?"
        )

    # Reshape: device-major, rx-within-device-minor (TI cascade LVDS order).
    # See module docstring for assumption and verification requirement.
    return raw.reshape(chirps, cfg.num_devices, cfg.num_rx_per_dev,
                       num_samples, 2)


def _frame_complex(path: Path, cfg: RadarConfig) -> tuple[np.ndarray, int]:
    """
    Return (cube, file_bytes).

    cube : complex64  [num_tx, num_loops, num_rx, num_samples]  (no calib)
    file_bytes : actual on-disk size of this frame file (ADD-B)
    """
    file_bytes = path.stat().st_size   # ADD-B: capture fidelity check
    raw = _read_frame_int16(path, cfg)

    if cfg.iq_swap:
        i_ch = raw[..., 1].astype(np.float32)
        q_ch = raw[..., 0].astype(np.float32)
    else:
        i_ch = raw[..., 0].astype(np.float32)
        q_ch = raw[..., 1].astype(np.float32)

    cdata = (i_ch + 1j * q_ch).astype(np.complex64)
    cdata = cdata.reshape(cfg.chirps_per_frame, cfg.num_rx, cfg.num_samples)

    # Separate TX virtual channels by TDM slot.
    cube = np.stack([cdata[t::cfg.num_tx] for t in range(cfg.num_tx)], axis=0)
    return cube, file_bytes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _stationarity_metrics(
    cube: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """
    cube: complex64 [num_tx, num_loops, num_rx, num_samples]

    Returns:
        raw_mean         : mean |IQ| in ADC LSBs
        static_dB        : 10*log10(sum |mean_over_loops(R_fft)|^2)
        slow_var_dB      : 10*log10(sum var_over_loops(|R_fft|^2))
        static_minus_slow: static_dB - slow_var_dB
        dc_frac          : fraction of static energy in range bin 0
        sat_frac         : fraction of raw int16-equivalent samples near rail
                           (ADD-C: ADC linearity check)

    ADD-C NOTE:
        sat_frac is computed from the float32 magnitude before complex
        combination, using the threshold |sample_component| > 32000
        (out of 32767 max for int16).  With compression off and empty space,
        sat_frac must be 0.0.  Any non-zero value invalidates the metrics.
    """
    # ADC saturation check (ADD-C): inspect float32 IQ components.
    # cube is complex64; view as float32 pairs [Re, Im] to recover ADC values.
    iq_flat = cube.view(np.float32)                     # shape: [..., 2*num_samples]
    sat_frac = float(np.mean(np.abs(iq_flat) > 32000.0))

    raw_mean = float(np.mean(np.abs(cube)))

    # Range FFT along ADC sample axis (no window, no FFT shift).
    # No window: intentional -- we are comparing absolute energies across
    # frames, not resolving sidelobes.  Windowing would rescale the total
    # power and make frame-to-frame comparisons noisier.
    R = np.fft.fft(cube, axis=-1).astype(np.complex64)

    # Static energy: coherent average across slow time (loops axis=1), then power.
    static_complex = R.mean(axis=1)                     # [tx, rx, range]
    static_pwr     = np.abs(static_complex) ** 2        # [tx, rx, range]
    static_total   = float(static_pwr.sum())

    # Slow-time variance: variance of |R|^2 across loops, then sum.
    pwr_per_loop = np.abs(R) ** 2                       # [tx, loops, rx, range]
    slow_var     = pwr_per_loop.var(axis=1)             # [tx, rx, range]
    slow_var_total = float(slow_var.sum())

    static_db = 10.0 * np.log10(max(static_total,   1e-12))
    slow_db   = 10.0 * np.log10(max(slow_var_total, 1e-12))

    # Fraction of static energy in the DC range bin (bin 0).
    # High dc_frac -> scene energy dominated by IQ DC offset / TX coupling,
    # not physical reflectors.
    dc_frac = float(static_pwr[..., 0].sum() / max(static_total, 1e-12))

    return raw_mean, static_db, slow_db, static_db - slow_db, dc_frac, sat_frac


# ---------------------------------------------------------------------------
# ADD-A: Periodicity detection via normalised autocorrelation
# ---------------------------------------------------------------------------

def _periodicity(
    signal: np.ndarray,
    threshold: float,
    label: str,
) -> None:
    """
    Print the dominant lag (in frames) of the first autocorrelation peak
    above `threshold` in the 1-D signal `signal`.

    Uses unbiased normalised autocorrelation (numpy full mode, divided by
    variance * N).  Lag-0 is excluded (always 1.0 by definition).

    ADD-A: This distinguishes H10 (firmware calibration cycle, periodicity
    survives compression-off) from H9 (compression was the sole source,
    no periodicity survives).
    """
    n = len(signal)
    if n < 4:
        print(f"  {label}: too few frames ({n}) for autocorrelation -- skip")
        return

    s = signal - signal.mean()
    var = float(np.var(s))
    if var < 1e-18:
        print(f"  {label}: signal is constant -- no periodicity (var ~ 0)")
        return

    # Full autocorrelation, then keep positive lags only (lag 1 .. n-1).
    acf_full = np.correlate(s, s, mode="full")          # length 2n-1
    acf = acf_full[n:]                                  # lags 1 .. n-1
    acf /= (var * n)                                    # normalise to [-1, 1]

    peaks_above = np.where(acf >= threshold)[0]         # 0-indexed -> lag = idx+1
    if len(peaks_above) == 0:
        print(f"  {label}: no autocorrelation peak >= {threshold:.2f} "
              f"(max r={acf.max():.3f} at lag {int(acf.argmax()) + 1}) "
              f"-> no significant periodicity detected")
    else:
        first_lag = int(peaks_above[0]) + 1
        r_val     = float(acf[peaks_above[0]])
        print(f"  {label}: first significant peak at lag={first_lag} frames "
              f"(r={r_val:.3f}, threshold={threshold:.2f})")
        print(f"           -> PERIODICITY DETECTED at ~{first_lag}-frame cycle")
        print(f"              (H10 firmware cal cycle candidate if lag matches "
              f"known TI recalibration cadence)")


# ---------------------------------------------------------------------------
# ADD-D: Timestamp / inter-frame interval reader
# ---------------------------------------------------------------------------

def _load_timestamps(folder: Path, frames: list[Path]) -> np.ndarray | None:
    """
    Attempt to load per-frame capture timestamps (seconds).

    Strategy 1: sidecar file timestamps.txt in folder, one float per line,
                 in the same order as sorted frame filenames.
    Strategy 2: frame filenames embed a unix timestamp as the second
                 underscore-delimited token: frame_<idx>_<ts>.bin

    Returns a float64 array of length len(frames), or None if no timing
    information is available.
    """
    # Strategy 1: sidecar
    sidecar = folder / "timestamps.txt"
    if sidecar.is_file():
        try:
            ts = np.loadtxt(sidecar, dtype=np.float64)
            if ts.ndim == 1 and len(ts) == len(frames):
                return ts
            print(f"[diag][timing] timestamps.txt found but length mismatch "
                  f"({len(ts)} vs {len(frames)} frames) -- skipping", file=sys.stderr)
        except Exception as exc:
            print(f"[diag][timing] timestamps.txt parse error: {exc} -- skipping",
                  file=sys.stderr)

    # Strategy 2: filename embedding (frame_<idx>_<ts>.bin)
    ts_list: list[float] = []
    for p in frames:
        parts = p.stem.split("_")
        if len(parts) >= 3:
            try:
                ts_list.append(float(parts[2]))
            except ValueError:
                break
        else:
            break
    if len(ts_list) == len(frames):
        return np.asarray(ts_list, dtype=np.float64)

    return None


def _print_timing(timestamps: np.ndarray, warmup: int) -> None:
    """
    Print inter-frame interval statistics for measurement frames only.

    ADD-D: Constant inter-frame intervals with correct cadence support
    H14 falsification.  Jittered or wrong-rate intervals support H14.
    """
    meas_ts = timestamps[warmup:]
    if len(meas_ts) < 2:
        print("  [timing] fewer than 2 measurement timestamps -- cannot compute intervals")
        return

    diffs = np.diff(meas_ts)
    mean_ifi  = float(diffs.mean())
    std_ifi   = float(diffs.std())
    min_ifi   = float(diffs.min())
    max_ifi   = float(diffs.max())
    fps_actual = 1.0 / mean_ifi if mean_ifi > 0 else float("nan")

    print(f"  inter-frame interval (measurement frames only, n={len(diffs)}):")
    print(f"    mean = {mean_ifi*1000:.2f} ms  "
          f"std = {std_ifi*1000:.2f} ms  "
          f"min = {min_ifi*1000:.2f} ms  "
          f"max = {max_ifi*1000:.2f} ms")
    print(f"    actual capture rate = {fps_actual:.2f} fps")
    jitter_pct = 100.0 * std_ifi / mean_ifi if mean_ifi > 0 else float("nan")
    print(f"    jitter (std/mean)   = {jitter_pct:.1f}%")
    if jitter_pct > 10.0:
        print("    WARNING: jitter > 10% -- timing is NOT consistent; "
              "supports H14 (capture/firmware phase incoherence)")
    else:
        print("    jitter within 10% -- timing is consistent; "
              "H14 (timing incoherence) not supported by this metric alone")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--folder", required=True, type=Path,
                        help="Folder containing frame_*.bin files.")
    parser.add_argument("--cfg", default=None,
                        help="Optional cascade .cfg; defaults to shortrange RadarConfig().")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only N frames after --start (0 = all).")

    # FIX (Issue 1 + Issue 3): explicit warm-up boundary and start offset.
    #
    # IMPORTANT -- two distinct warm-up concepts must not be conflated:
    #
    #   Firmware warm-up  : PLL/ADC settling frames discarded by the CAPTURE
    #                       SOFTWARE before writing any frame_*.bin to disk.
    #                       If your capture tool already handled this (e.g. a
    #                       60-frame pre-roll that was never saved), those frames
    #                       are GONE and --warmup must NOT re-account for them.
    #                       Default is therefore 0.
    #
    #   Script-side skip  : Additional leading frames in the saved set that you
    #                       want excluded from summary statistics (e.g. if the
    #                       capture tool saved everything including its own
    #                       pre-roll).  Set --warmup N only in that case.
    #
    # For this experiment the capture software performed the 60-frame warm-up
    # and saved ONLY measurement frames.  Run without --warmup (default 0) or
    # pass --warmup 0 explicitly.
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of SAVED leading frames to exclude from summary "
                             "statistics (default: 0). Set > 0 only if the capture "
                             "software saved its own pre-roll frames alongside the "
                             "measurement frames.  Do NOT set this to 60 if the "
                             "capture tool already discarded the warm-up before saving. "
                             "Excluded frames are still printed with a 'W' prefix.")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip this many frames from the sorted list before applying "
                             "--warmup and --limit (default: 0).  Use to resume from a "
                             "specific frame index.")

    # FIX (Issue 4): documented, CLI-configurable threshold.
    parser.add_argument("--stable-threshold-db", type=float, default=2.0,
                        dest="stable_db",
                        help="Span threshold (dB) below which a metric is considered "
                             "stable (default: 2.0).  For a 16-bit ADC at noise floor, "
                             "thermal variation over 50 frames is typically < 1 dB; "
                             "2.0 dB is conservative.")

    # ADD-A: autocorrelation threshold.
    parser.add_argument("--autocorr-threshold", type=float, default=0.5,
                        dest="autocorr_thresh",
                        help="Normalised autocorrelation threshold for periodicity "
                             "detection (default: 0.5).  Peaks above this level are "
                             "reported as significant.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate folder
    # ------------------------------------------------------------------
    if not args.folder.is_dir():
        print(f"ERROR: folder not found: {args.folder}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # Load and validate geometry
    # ------------------------------------------------------------------
    cfg = _load_geom(args.cfg)

    try:
        _validate_geom(cfg)          # FIX (Issue 2): LVDS interleave assertion
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    expected_bytes = cfg.chirps_per_frame * cfg.num_rx * cfg.num_samples * 2 * 2

    print("[diag] geometry:")
    print(f"  num_samples    = {cfg.num_samples}")
    print(f"  num_loops      = {cfg.num_loops}")
    print(f"  num_tx         = {cfg.num_tx}")
    print(f"  num_rx         = {cfg.num_rx}")
    print(f"  num_devices    = {cfg.num_devices}")
    print(f"  num_rx_per_dev = {cfg.num_rx_per_dev}")
    print(f"  iq_swap        = {cfg.iq_swap}")
    print(f"  chirps_per_frame = {cfg.chirps_per_frame}")
    print(f"  expected bytes/frame = {expected_bytes:,}  "
          f"(cfg.bytes_per_frame = {cfg.bytes_per_frame:,})")
    print(f"  LVDS interleave: device-major, rx-within-device-minor "
          f"(TI SWRU546 §3.2 -- verify against your DCA1000 config)")
    print()
    print("[diag] run parameters:")
    print(f"  --warmup              = {args.warmup} frames (excluded from summary)")
    print(f"  --start               = {args.start}")
    print(f"  --limit               = {args.limit} (0 = all)")
    print(f"  --stable-threshold-db = {args.stable_db} dB")
    print(f"  --autocorr-threshold  = {args.autocorr_thresh}")
    print()

    # ------------------------------------------------------------------
    # Collect and slice frame list
    # FIX (Issue 1 + Issue 3): apply --start, then --warmup boundary,
    # then --limit.  Warm-up boundary is not inferred from data.
    # ------------------------------------------------------------------
    all_frames = sorted(args.folder.glob("frame_*.bin"))
    if args.start > 0:
        all_frames = all_frames[args.start:]
    if args.limit > 0:
        all_frames = all_frames[: args.limit]

    if not all_frames:
        print(f"ERROR: no frame_*.bin in {args.folder} "
              f"(after --start={args.start}, --limit={args.limit})",
              file=sys.stderr)
        return 2

    total_frames = len(all_frames)
    warmup_count = min(args.warmup, total_frames)
    meas_count   = total_frames - warmup_count
    print(f"[diag] frames found   : {total_frames}")
    print(f"[diag] warm-up frames : {warmup_count}  (printed with 'W' prefix, excluded from summary)")
    print(f"[diag] meas frames    : {meas_count}  (used for summary and verdict)")
    if meas_count == 0:
        print("WARNING: all frames are warm-up -- no measurement frames for summary",
              file=sys.stderr)
    print()

    # ADD-D: load timestamps if available.
    timestamps = _load_timestamps(args.folder, all_frames)
    if timestamps is not None:
        print(f"[diag] timing: timestamps loaded ({len(timestamps)} entries)")
    else:
        print("[diag] timing: no timestamps available "
              "(provide timestamps.txt or embed ts in filename as frame_<idx>_<ts>.bin)")
    print()

    # ------------------------------------------------------------------
    # Per-frame processing
    # FIX (Issue 6): track n_ok, n_err separately for warm-up and meas.
    # ------------------------------------------------------------------
    hdr = (f"{'frame':>6} {'type':>5} {'bytes':>10} {'raw_mean':>9} "
           f"{'static_dB':>10} {'slow_dB':>9} {'s-sl_dB':>8} "
           f"{'dc_frac':>8} {'sat_frac':>9}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    static_vals:  list[float] = []
    slow_vals:    list[float] = []
    raw_vals:     list[float] = []
    file_sizes:   list[int]   = []
    sat_fracs:    list[float] = []

    n_ok_warmup = n_err_warmup = 0
    n_ok_meas   = n_err_meas   = 0

    for frame_idx, path in enumerate(all_frames):
        is_warmup = frame_idx < warmup_count
        tag = "W" if is_warmup else "M"
        idx_str = path.stem.split("_")[-1]

        try:
            cube, file_bytes = _frame_complex(path, cfg)
        except Exception as exc:
            label = f"{'W-ERR' if is_warmup else 'M-ERR'}"
            print(f"{idx_str:>6} {label:>5}  READ ERROR: {exc}", file=sys.stderr)
            if is_warmup:
                n_err_warmup += 1
            else:
                n_err_meas += 1
            continue

        (raw_mean, static_db, slow_db,
         diff_db, dc_frac, sat_frac) = _stationarity_metrics(cube)

        print(f"{idx_str:>6} {tag:>5} {file_bytes:>10,} {raw_mean:9.2f} "
              f"{static_db:10.2f} {slow_db:9.2f} {diff_db:+8.2f} "
              f"{dc_frac:8.4f} {sat_frac:9.6f}"
              + (" <<< SAT" if sat_frac > 0 else ""))

        if is_warmup:
            n_ok_warmup += 1
        else:
            n_ok_meas += 1
            static_vals.append(static_db)
            slow_vals.append(slow_db)
            raw_vals.append(raw_mean)
            file_sizes.append(file_bytes)
            sat_fracs.append(sat_frac)

    print(sep)
    print()

    # ------------------------------------------------------------------
    # Frame count report (FIX Issue 6)
    # ------------------------------------------------------------------
    print("[diag] frame accounting:")
    print(f"  warm-up : {n_ok_warmup} ok,  {n_err_warmup} errors  "
          f"(excluded from all summary statistics)")
    print(f"  meas    : {n_ok_meas} ok,  {n_err_meas} errors  "
          f"(used for summary and verdict)")
    if n_err_meas > 0:
        print(f"  WARNING: {n_err_meas} measurement frames failed -- "
              f"summary covers only {n_ok_meas} frames; "
              f"investigate read errors before trusting verdict")
    print()

    if not static_vals:
        print("[diag] no measurement frames processed -- cannot produce summary",
              file=sys.stderr)
        return 1

    sv = np.asarray(static_vals)
    ll = np.asarray(slow_vals)
    rv = np.asarray(raw_vals)
    fz = np.asarray(file_sizes, dtype=np.int64)
    sf = np.asarray(sat_fracs)

    # ------------------------------------------------------------------
    # Summary statistics (measurement frames only)
    # ------------------------------------------------------------------
    print(f"[diag] summary  (measurement frames: n={n_ok_meas}, "
          f"stable_threshold={args.stable_db} dB)")
    print()

    # raw_mean
    raw_ratio = rv.max() / max(float(rv.min()), 1e-6)
    print(f"  raw_mean    : min={rv.min():8.2f}  max={rv.max():8.2f}  "
          f"mean={rv.mean():8.2f}  max/min={raw_ratio:5.2f}")

    # static_dB
    static_span = float(sv.max() - sv.min())
    print(f"  static_dB   : min={sv.min():8.2f}  max={sv.max():8.2f}  "
          f"span={static_span:5.2f} dB  "
          f"{'STABLE' if static_span < args.stable_db else 'VARIES  <-- !'}")

    # slow_var_dB
    slow_span = float(ll.max() - ll.min())
    print(f"  slow_var_dB : min={ll.min():8.2f}  max={ll.max():8.2f}  "
          f"span={slow_span:5.2f} dB  "
          f"{'STABLE' if slow_span < args.stable_db else 'VARIES  <-- !'}")

    # ADD-B: file sizes
    fz_unique = np.unique(fz)
    if len(fz_unique) == 1:
        print(f"  file_bytes  : consistent = {int(fz_unique[0]):,}  "
              f"(expected {expected_bytes:,})  "
              f"{'OK' if fz_unique[0] == expected_bytes else 'MISMATCH <-- !'}")
    else:
        print(f"  file_bytes  : NOT consistent -- "
              f"{len(fz_unique)} distinct sizes: {fz_unique.tolist()}  <-- !")
        print(f"                CAPTURE INTEGRITY FAILURE: "
              f"frame sizes vary; IQ alignment cannot be trusted")

    # ADD-C: saturation
    sat_any = float(sf.max())
    sat_mean = float(sf.mean())
    if sat_any == 0.0:
        print(f"  sat_frac    : 0.000000 across all measurement frames  (OK -- ADC not saturated)")
    else:
        print(f"  sat_frac    : max={sat_any:.6f}  mean={sat_mean:.6f}  "
              f"<-- ADC SATURATION DETECTED -- stationarity metrics are INVALID")

    print()

    # ------------------------------------------------------------------
    # ADD-A: periodicity detection on raw_mean trace
    # ------------------------------------------------------------------
    print("[diag] periodicity analysis (measurement frames):")
    _periodicity(rv, args.autocorr_thresh, "raw_mean")
    _periodicity(sv, args.autocorr_thresh, "static_dB")
    _periodicity(ll, args.autocorr_thresh, "slow_var_dB")
    print()

    # ------------------------------------------------------------------
    # ADD-D: inter-frame timing
    # ------------------------------------------------------------------
    print("[diag] inter-frame timing:")
    if timestamps is not None:
        _print_timing(timestamps, warmup_count)
    else:
        print("  no timestamps available -- H14 (timing incoherence) "
              "cannot be evaluated from this run")
    print()

    # ------------------------------------------------------------------
    # Verdict (FIX Issue 4: threshold is --stable-threshold-db, documented)
    # ------------------------------------------------------------------
    print(f"[diag] verdict heuristic  (threshold = {args.stable_db} dB):")

    if sat_any > 0.0:
        print("  INVALID -- ADC saturation detected; metrics cannot be interpreted.")
        print("             Ensure ADC input is in linear range before re-running.")
        return 1

    if len(fz_unique) > 1:
        print("  INVALID -- frame sizes are inconsistent; IQ alignment is corrupt.")
        return 1

    if static_span < args.stable_db and slow_span < args.stable_db:
        print("  STABLE")
        print("  -> Input is stationary.  H9 (compressionCfg) is FALSIFIED for this capture.")
        print("  -> CFAR=0 in a static scene with this data IS meaningful.")
        print("  -> If periodicity was also absent above, H10 is unsupported by this data.")
    elif static_span >= args.stable_db and slow_span < args.stable_db:
        print("  STATIC VARIES, NOISE STABLE")
        print("  -> RF/TX gain or firmware state is wandering between frames.")
        print("  -> H10 (firmware inter-frame calibration cycle) is primary candidate.")
        print("  -> Check periodicity lag above against TI firmware recalibration cadence.")
    elif static_span < args.stable_db and slow_span >= args.stable_db:
        print("  STATIC STABLE, NOISE VARIES")
        print("  -> Noise figure or intra-frame AGC is wandering within frames.")
        print("  -> Not consistent with H9 or H10; investigate ADC thermal noise or")
        print("     intra-frame firmware state changes.")
    else:
        print("  BOTH VARY")
        print("  -> Input-power scaling drift persists with compressionCfg disabled.")
        print("  -> H9 (compression) is NOT the sole cause.")
        print("  -> H10 (firmware cal cycle) and H14 (LO/timing incoherence) remain open.")
        print("  -> Check periodicity lag and timing jitter above for discrimination.")

    print()
    print("[diag] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())