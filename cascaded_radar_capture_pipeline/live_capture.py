"""
live_capture.py  —  AWR2243-2X live hardware capture with real-time display
============================================================================

Stage 2 entry point.  Runs radar capture and live DSP display simultaneously.
Replaces live_streaming_test.py for hardware operation.

Usage
-----
# Infinite capture (stop by closing window or Ctrl-C):
python live_capture.py

# Capture N frames then stop automatically:
python live_capture.py --frames 200

# Skip firmware load (already running):
python live_capture.py --skip-load

# Suppress per-frame terminal logs (clean operational display):
python live_capture.py --quiet

# Discard first 2 warm-up frames:
python live_capture.py --warmup 2

# Specify UART port for firmware stop command:
python live_capture.py --uart COM20

Architecture
------------
Main thread  : Qt event loop — required by Qt on Windows
CaptureThread: run_capture → FileMonitor → on_frame_saved → FrameQueue
StreamingProcessor: FrameQueue → process_frame → ResultQueue
QTimer 50ms  : ResultQueue → LiveViewer (pyqtgraph, GPU-accelerated)

The display shows three heatmaps (Range-Doppler, Range-Elevation,
Range-Azimuth) updated at ~20 fps regardless of the radar frame rate.
CFAR detection markers appear as white + symbols on the RA map.
The status bar shows frame count, detection counts, and drop rate.

Stopping
--------
- Close the display window  → capture stops cleanly (UART stop → DCA stop)
- Ctrl-C in terminal        → same clean shutdown
- Place a .stop_capture file in the project root → FileMonitor stops

Performance note
----------------
process_frame() currently takes ~2600 ms on CPU-only machines.
At 4 fps (256 ms frame period) this means ~10 of every 11 live frames
are dropped and the display updates every ~2.6 seconds.
To achieve real-time display at 4 fps, process_frame() must run < 250 ms.
GPU acceleration (CuPy batch FFT or PyTorch Capon beamforming) is the
Stage 3 path to reach this target.  The display remains correct regardless
of drop rate — it always shows the most recently processed frame.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from processing.radar_config import parse_cfg_file, parse_calibration
from processing.streaming   import run_live_hardware, run_live
from capture.config import (
    CASCADE_CFG_PATH,
    FRAMES_DIR,
    create_capture_settings,
    parse_cascade_cfg,
    apply_raw_iq_order_override,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AWR2243-2X live capture with real-time streaming display.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cfg",
                   type=Path,
                   default=CASCADE_CFG_PATH,
                   help="Path to cascade_r017m.cfg")
    p.add_argument("--out-dir",
                   type=Path,
                   default=FRAMES_DIR,
                   help="Directory for saved frame_*.bin files (only used with --record)")
    p.add_argument("--record",
                   action="store_true",
                   help="Write frame_*.bin files to disk (OFF by default for true live streaming)")
    p.add_argument("--frames",
                   type=int,
                   default=0,
                   help="Number of frames to capture (0 = infinite)")
    p.add_argument("--uart",
                   type=str,
                   default="COM20",
                   metavar="COM_PORT",
                   help="UART port for firmware stop command")
    p.add_argument("--replay",
                   metavar="FOLDER",
                   default=None,
                   help="Replay recorded frames from FOLDER through the streaming "
                        "pipeline instead of connecting to hardware. "
                        "Combine with --display pointcloud to view the 3D body "
                        "cloud from the 400 recorded frames. "
                        "Example: --replay capture\\frames --display pointcloud")
    p.add_argument("--loop",
                   action="store_true",
                   help="Loop the replay folder continuously until the window is "
                        "closed (only used with --replay).")
    p.add_argument("--replay-fps",
                   type=float,
                   default=0.0,
                   metavar="FPS",
                   dest="replay_fps",
                   help="Frame feed rate for --replay mode. "
                        "Default 0 = auto (matches processing speed). "
                        "Set to 4.0 to match real-time hardware rate. "
                        "Set lower (e.g. 1.0) to step through frames slowly.")
    p.add_argument("--skip-load",
                   action="store_true",
                   help="Skip firmware load (firmware already running)")
    p.add_argument("--sync-wait",
                   type=float,
                   default=60.0,
                   help="Seconds to wait for firmware init after load")
    p.add_argument("--warmup",
                   type=int,
                   default=0,
                   metavar="N",
                   help="Discard first N frames after sensorStart "
                        "(firmware front-end calibration settling)")
    p.add_argument("--frame-stats",
                   action="store_true",
                   help="Log per-frame ADC magnitude statistics")
    p.add_argument("--diag",
                   action="store_true",
                   help="Keep CCS debugger attached longer during firmware load")
    p.add_argument("--quiet",
                   action="store_true",
                   help="Suppress per-frame StreamingProcessor log lines "
                        "(frame summary still printed)")
    p.add_argument("--raw-iq-order",
                   choices=("i-first", "q-first", "cfg"),
                   default="i-first",
                   help="IQ byte order override. 'i-first' matches DCA1000EVM "
                        "hardware behaviour regardless of adcbufCfg SampleSwap.")
    p.add_argument("--pause-ms",
                   type=int,
                   default=50,
                   help="Display refresh interval in ms (~20 fps at 50 ms)")
    p.add_argument("--display",
                   choices=["heatmap", "pointcloud", "focused", "volume"],
                   default="heatmap",
                   help="Display mode. 'heatmap' (default) shows RD/RE/RA "
                        "plots. 'pointcloud' shows the 5-panel diagnostic view. "
                        "'volume' shows one software-rendered 3D "
                        "range-doppler-azimuth voxel volume that avoids "
                        "OpenGL driver crashes. "
                        "'focused' shows a single operator panel: subject "
                        "position, direction, speed and activity only - the "
                        "quantities measured reliable on the recorded sessions. "
                        "pointcloud and focused require CFAR "
                        "(do not combine with --no-cfar).")
    p.add_argument("--no-cfar",
                   action="store_true",
                   help="Skip CFAR detection stages for maximum display throughput. "
                        "Heatmaps still update. No detections or point cloud.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    cfg_path = args.cfg.resolve()
    if not cfg_path.exists():
        print(f"[ERROR] cfg not found: {cfg_path}", file=sys.stderr)
        return 1

    # ── Parse radar config ────────────────────────────────────────────────────
    print(f"\n=== Loading radar configuration: {cfg_path.name} ===")
    cfg = parse_cfg_file(str(cfg_path))

    # ── Apply CLI overrides to cfg ────────────────────────────────────────────
    if args.quiet:
        cfg.quiet_mode = True
    if args.no_cfar:
        cfg.skip_cfar = True
    cfg.volume_en = args.display == "volume"
    cfg.cfar_margin_db = getattr(args, 'cfar_margin', cfg.cfar_margin_db)

    # ── Load calibration ──────────────────────────────────────────────────────
    corr = parse_calibration(str(cfg_path), cfg.num_tx, cfg.num_rx)

    # Resolve display mode (guard: pointcloud needs CFAR)
    if args.display == "pointcloud" and args.no_cfar:
        print("[WARNING] --display pointcloud requires CFAR detections. "
              "--no-cfar overrides to heatmap mode.", flush=True)
        display_mode = "heatmap"
    else:
        display_mode = args.display

    # ── REPLAY MODE — drive existing 3D viewer from recorded frames ───────────
    if args.replay is not None:
        replay_folder = Path(args.replay).resolve()
        if not replay_folder.is_dir():
            print(f"[ERROR] --replay folder not found: {replay_folder}",
                  file=sys.stderr)
            return 1

        raw_iq = getattr(args, 'raw_iq_order', 'i-first')
        fps    = getattr(args, 'replay_fps', 0.0)
        loop   = getattr(args, 'loop', False)

        print(f"\n=== Replay mode (no hardware) ===")
        print(f"  folder     : {replay_folder}")
        print(f"  cfg        : {cfg_path.name}")
        print(f"  display    : {display_mode}")
        print(f"  fps        : {'auto' if fps <= 0 else fps}")
        print(f"  loop       : {loop}")
        print(f"  iq_order   : {raw_iq}")

        run_live(
            cfg,
            corr,
            frame_folder=replay_folder,
            frame_rate_hz=fps if fps > 0 else 4.0,
            auto_rate=(fps <= 0),          # auto-pace when fps not specified
            raw_iq_order=raw_iq,
            pause_ms=getattr(args, 'pause_ms', 50),
            quiet=getattr(args, 'quiet', True),
            title=f"AWR2243-2X Replay — {replay_folder.name}",
            loop=loop,
            display_mode=display_mode,
        )
        return 0

    # ── LIVE HARDWARE MODE ────────────────────────────────────────────────────
    geom = parse_cascade_cfg(cfg_path)
    geom = apply_raw_iq_order_override(geom, args.raw_iq_order)

    out_dir = args.out_dir.resolve() if args.record else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    settings = create_capture_settings(
        out_dir=args.out_dir.resolve(),
        uart_port=args.uart,
        max_frames=args.frames,
        sync_wait=args.sync_wait,
        diag=args.diag,
        skip_load=args.skip_load,
    )

    print(f"\n=== Starting live capture + display ===")
    print(f"  cfg        : {cfg_path.name}")
    print(f"  out_dir    : {'disabled (streaming mode)' if out_dir is None else out_dir}")
    print(f"  record     : {args.record}")
    print(f"  frames     : {'infinite' if args.frames == 0 else args.frames}")
    print(f"  uart       : {args.uart}")
    print(f"  skip_load  : {args.skip_load}")
    print(f"  warmup     : {args.warmup} frame(s)")
    print(f"  iq_order   : {args.raw_iq_order}")
    print(f"  quiet      : {args.quiet}")
    print(f"  no_cfar    : {args.no_cfar}  {'(heatmaps only, no detections)' if args.no_cfar else ''}")
    print(f"  display    : {display_mode}")

    result = run_live_hardware(
        cfg,
        corr,
        settings,
        raw_iq_order=args.raw_iq_order,
        out_dir=out_dir,
        max_frames=args.frames,
        skip_load=args.skip_load,
        sync_wait=args.sync_wait,
        uart_port=args.uart,
        frame_bytes_override=geom.bytes_per_frame,
        warmup_frames=args.warmup,
        frame_stats=args.frame_stats,
        diag=args.diag,
        pause_ms=args.pause_ms,
        quiet=args.quiet,
        display_mode=display_mode,
    )

    if result is not None:
        print(f"\n=== Capture complete: {result.frame_idx} frames saved to {result.out_dir} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
