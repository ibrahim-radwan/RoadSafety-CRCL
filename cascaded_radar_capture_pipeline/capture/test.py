"""
Main entrypoint for the fixed shortrange cascade project.

This module owns:
- CLI argument parsing
- offline frame extraction
- cfg validation
- the top-level run/stop flow
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capture.config import (
    CASCADE_CFG_PATH,
    FRAMES_DIR,
    CascadeFrameGeometry,
    apply_raw_iq_order_override,
    create_capture_settings,
    expected_frame_shape,
    get_profile_cfg_tokens,
    nyquist_beat_range_m,
    parse_cascade_cfg,
    replace_capture_settings,
    unambiguous_velocity_halfspan_mps,
    velocity_axis_m_per_s,
)
from capture.dca import run_capture, stop_capture


def extract_frame(
    raw_int16: np.ndarray,
    frame_idx: int,
    geom: CascadeFrameGeometry,
    *,
    time_order: str = "loop_chirp",
) -> np.ndarray:
    # Purpose: extract one frame cube from raw int16 data. Input: raw vector, frame index, geometry, time order. Output: complex frame cube.
    n_samples = geom.num_adc_samples
    num_rx = geom.num_rx_total
    num_chirps = geom.num_chirps_per_loop
    num_loops = geom.num_loops
    num_devices = geom.num_devices
    rx_per_device = geom.num_rx_per_device

    frame_int16 = geom.bytes_per_frame // 2
    offset = frame_idx * frame_int16
    expected_int16 = num_loops * num_chirps * num_rx * n_samples * 2
    if frame_int16 != expected_int16:
        raise ValueError(
            "CascadeFrameGeometry is inconsistent "
            f"(bytes_per_frame/2={frame_int16} != expected {expected_int16})."
        )
    if offset + frame_int16 > len(raw_int16):
        raise ValueError(
            f"Frame {frame_idx} out of range. File has {len(raw_int16) / frame_int16:.2f} complete frames."
        )

    chunk = raw_int16[offset : offset + frame_int16]
    normalized_time_order = str(time_order).lower().replace("-", "_")
    if normalized_time_order not in ("loop_chirp", "chirp_loop"):
        raise ValueError(f"time_order must be 'loop_chirp' or 'chirp_loop', got {time_order!r}")

    if not geom.chan_interleaved:
        cell_int16 = num_devices * rx_per_device * n_samples * 2
        if normalized_time_order == "loop_chirp":
            time_flat = chunk.reshape(num_loops, num_chirps, cell_int16)
        else:
            time_flat = chunk.reshape(num_chirps, num_loops, cell_int16).swapaxes(0, 1)
        raw4d = time_flat.reshape(num_loops * num_chirps, num_devices, rx_per_device, n_samples, 2)
        raw4d = raw4d.reshape(num_loops * num_chirps, num_rx, n_samples, 2).astype(np.float32)
    else:
        cell_int16 = n_samples * num_rx * 2
        if normalized_time_order == "loop_chirp":
            time_flat = chunk.reshape(num_loops, num_chirps, cell_int16)
        else:
            time_flat = chunk.reshape(num_chirps, num_loops, cell_int16).swapaxes(0, 1)
        raw4d = time_flat.reshape(num_loops * num_chirps, n_samples, num_rx, 2).astype(np.float32)
        raw4d = raw4d.transpose(0, 2, 1, 3)

    if geom.iq_swap:
        i_channel = raw4d[..., 1]
        q_channel = raw4d[..., 0]
    else:
        i_channel = raw4d[..., 0]
        q_channel = raw4d[..., 1]
    complex_data = (i_channel + 1j * q_channel).astype(np.complex64)
    if not np.isfinite(complex_data).all():
        raise ValueError(f"Frame {frame_idx}: non-finite IQ values.")

    cdata = complex_data.reshape(num_loops * num_chirps, num_rx, n_samples)
    return np.stack([cdata[t::num_chirps] for t in range(num_chirps)], axis=0)


def extract_all_frames(
    bin_path: Path,
    geom: CascadeFrameGeometry,
    max_frames: int = 0,
    *,
    time_order: str = "loop_chirp",
) -> list[np.ndarray]:
    # Purpose: extract all complete frames from a raw bin file. Input: bin path, geometry, limits. Output: list of frame cubes.
    raw = np.fromfile(bin_path, dtype=np.int16)
    frame_int16 = geom.bytes_per_frame // 2
    total_frames = len(raw) // frame_int16
    frames_to_extract = total_frames if max_frames <= 0 else min(max_frames, total_frames)
    leftover_bytes = (len(raw) % frame_int16) * 2
    if leftover_bytes:
        print(f"  Warning: {leftover_bytes} trailing bytes ignored.")
    print(f"  {bin_path.name}: {len(raw) * 2 / 1e6:.1f} MB -> {total_frames} frames, extracting {frames_to_extract}.")
    return [extract_frame(raw, index, geom, time_order=time_order) for index in range(frames_to_extract)]


def save_frame_npz(frame: np.ndarray, out_path: Path, geom: CascadeFrameGeometry, frame_idx: int) -> None:
    # Purpose: save one extracted frame with metadata. Input: frame cube, output path, geometry, frame index. Output: compressed npz file.
    np.savez_compressed(
        out_path,
        data=frame,
        chirp_tx_map=np.array(geom.chirp_tx_map, dtype=np.int32),
        frame_idx=frame_idx,
        num_loops=geom.num_loops,
        num_chirps_per_loop=geom.num_chirps_per_loop,
        num_rx_per_device=geom.num_rx_per_device,
        num_tx_per_device=geom.num_tx_per_device,
        num_devices=geom.num_devices,
        num_rx_total=geom.num_rx_total,
        num_tx_total=geom.num_tx_total,
        num_adc_samples=geom.num_adc_samples,
        iq_swap=geom.iq_swap,
        chan_interleaved=geom.chan_interleaved,
    )


def validate_cfg_and_extraction(geom: CascadeFrameGeometry, cfg_path: Path = CASCADE_CFG_PATH) -> int:
    # Purpose: smoke-test cfg parsing and extraction layout. Input: geometry and cfg path. Output: process exit code.
    try:
        profile_cfg = get_profile_cfg_tokens(cfg_path)
        nyquist_range = nyquist_beat_range_m(profile_cfg)
    except Exception as exc:
        print(f"validate-cfg FAIL (profile): {exc}", file=sys.stderr)
        return 1

    try:
        raw_zero = np.zeros(geom.bytes_per_frame // 2, dtype=np.int16)
        frame = extract_frame(raw_zero, 0, geom)
    except Exception as exc:
        print(f"validate-cfg FAIL (extract_frame): {exc}", file=sys.stderr)
        return 1

    expected_shape = expected_frame_shape(geom)
    if tuple(frame.shape) != expected_shape:
        print(f"validate-cfg FAIL: shape {frame.shape} != {expected_shape}", file=sys.stderr)
        return 1
    if int(profile_cfg[9]) != geom.num_adc_samples:
        print("validate-cfg FAIL: profileCfg numAdcSamples mismatch", file=sys.stderr)
        return 1

    print(f"validate-cfg OK: {cfg_path.name}")
    print(f"  bytes_per_frame     = {geom.bytes_per_frame:,}")
    print(f"  iq_swap (effective) = {geom.iq_swap}  ({'Q,I' if geom.iq_swap else 'I,Q'})")
    print(f"  shape [NTX,NL,NRX,N] = {frame.shape}")
    print(f"  Nyquist alias range = {nyquist_range:.3f} m")
    try:
        velocity_halfspan = unambiguous_velocity_halfspan_mps(geom, cfg_path)
        velocity_axis = velocity_axis_m_per_s(geom.num_loops, geom, cfg_path)
        print(f"  TDM |v_max|         = {velocity_halfspan:.4f} m/s  (lambda/(4*T_loop))")
        print(f"  Doppler axis ends   = {velocity_axis[0]:.4f} .. {velocity_axis[-1]:.4f} m/s")
    except Exception as exc:
        print(f"  (velocity: {exc})", file=sys.stderr)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Purpose: parse CLI arguments for capture, extract, and stop flows. Input: optional argv list. Output: argparse namespace.
    parser = argparse.ArgumentParser(
        description="Capture or extract frames using the fixed shortrange cascade project config.",
    )
    parser.add_argument(
        "--raw-iq-order",
        choices=("cfg", "i-first", "q-first"),
        default="i-first",
        metavar="MODE",
        help="'cfg'=use adcbufCfg; 'i-first'=[I,Q]; 'q-first'=[Q,I].",
    )
    parser.add_argument("--uart", type=str, default="COM20", metavar="COM_PORT", help="UART port used for stop.")
    parser.add_argument("--frames", type=int, default=0, help="0 means capture or extract all available frames.")
    parser.add_argument("--out-dir", type=Path, default=FRAMES_DIR, help="Directory for saved frames.")
    parser.add_argument("--skip-load", action="store_true", help="Skip firmware load because it is already running.")
    parser.add_argument("--stop", action="store_true", help="Stop a running capture.")
    parser.add_argument("--sync-wait", type=float, default=60.0, help="Seconds to wait for firmware init.")
    parser.add_argument("--diag", action="store_true", help="Keep the debugger attached longer during firmware load.")
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        metavar="N",
        help="Discard the first N frames after sensorStart (firmware front-end calibration).",
    )
    parser.add_argument(
        "--frame-stats",
        action="store_true",
        help="Log per-frame ADC magnitude statistics (mean|x|, std, max, I_dc, Q_dc).",
    )
    parser.add_argument("--extract", action="store_true", help="Extract frames from a raw bin file into .npz files.")
    parser.add_argument("--bin-file", type=Path, default=None, metavar="CAPTURE.bin", help="Raw DCA1000 .bin file.")
    parser.add_argument(
        "--extract-time-order",
        choices=("loop_chirp", "chirp_loop"),
        default="loop_chirp",
        help="Raw buffer order: loop_chirp (default) or chirp_loop.",
    )
    parser.add_argument("--print-geometry", action="store_true", help="Print the fixed config geometry and exit.")
    parser.add_argument("--validate-cfg", action="store_true", help="Validate the fixed cfg and extraction layout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Purpose: run the top-level CLI flow. Input: optional argv list. Output: process exit code.
    args = parse_args(argv)
    cfg_path = CASCADE_CFG_PATH.resolve()
    if not cfg_path.exists():
        print(f"Error: fixed cfg not found: {cfg_path}", file=sys.stderr)
        return 1

    geom = apply_raw_iq_order_override(parse_cascade_cfg(cfg_path), args.raw_iq_order)
    if args.raw_iq_order != "cfg":
        print(
            f"Note: --raw-iq-order={args.raw_iq_order!r}  "
            f"(iq_swap={geom.iq_swap}; adcbufCfg SampleSwap overridden).",
            file=sys.stderr,
        )

    if args.validate_cfg:
        return validate_cfg_and_extraction(geom, cfg_path)

    if args.print_geometry:
        print(geom.describe())
        return 0

    print(geom.describe())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.extract:
        if args.bin_file is None:
            print("Error: --extract requires --bin-file", file=sys.stderr)
            return 1
        bin_path = args.bin_file.resolve()
        if not bin_path.exists():
            print(f"Error: .bin not found: {bin_path}", file=sys.stderr)
            return 1
        frames = extract_all_frames(bin_path, geom, max_frames=args.frames, time_order=str(args.extract_time_order))
        for index, frame in enumerate(frames):
            save_frame_npz(frame, out_dir / f"frame_{index:05d}.npz", geom, index)
        print(f"  Saved {len(frames)} frames to {out_dir}")
        return 0

    settings = create_capture_settings(
        out_dir=out_dir,
        uart_port=args.uart,
        max_frames=args.frames,
        sync_wait=args.sync_wait,
        diag=args.diag,
        skip_load=args.skip_load,
    )
    settings = replace_capture_settings(settings, out_dir=out_dir)

    if args.stop:
        stop_capture(settings, out_dir=out_dir, frame_bytes=geom.bytes_per_frame, uart_port=args.uart)
        return 0

    run_capture(
        settings,
        out_dir=out_dir,
        max_frames=args.frames,
        skip_load=args.skip_load,
        sync_wait=args.sync_wait,
        uart_port=args.uart,
        frame_bytes_override=geom.bytes_per_frame,
        diag=args.diag,
        warmup_frames=args.warmup_frames,
        frame_stats=args.frame_stats,
    )
    return 0


__all__ = [
    "CASCADE_CFG_PATH",
    "CascadeFrameGeometry",
    "FRAMES_DIR",
    "apply_raw_iq_order_override",
    "expected_frame_shape",
    "extract_all_frames",
    "extract_frame",
    "main",
    "parse_args",
    "parse_cascade_cfg",
    "save_frame_npz",
    "validate_cfg_and_extraction",
]


if __name__ == "__main__":
    raise SystemExit(main())
