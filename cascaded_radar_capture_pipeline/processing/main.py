"""
main.py  —  AWR2243-2X cascade radar processing pipeline CLI

Usage
-----
    python -m processing.main --bin data.bin --cfg firmware/r017m/cascade_r017m.cfg

  # Long-range config
    python -m processing.main --bin data.bin --cfg firmware/r168m/cascade_r168m.cfg

  # Tune CFAR, AoA, display
    python -m processing.main --bin data.bin --cfg firmware/r017m/cascade_r017m.cfg \\
      --cfar os --cfar-fa-rate 1e-4 --aoa capon \\
      --fixed-db --db-vmin 60 --db-vmax 95 \\
      --save output.png --no-show

  # All frames → folder
    python -m processing.main --bin data.bin --cfg firmware/r017m/cascade_r017m.cfg \\
      --all-frames --save-dir frames/ --no-show

  # Sequential folder viewer
    python -m processing.main --folder data_dir/ --cfg firmware/r017m/cascade_r017m.cfg

  # Save arrDREA and xyz cloud
    python -m processing.main --bin data.bin --cfg firmware/r017m/cascade_r017m.cfg \\
      --save-drea out/drea.npy --save-xyz out/cloud.csv

  # Print config summary only
    python -m processing.main --bin data.bin --cfg firmware/r017m/cascade_r017m.cfg --summary
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from processing.radar_config     import (RadarConfig, parse_cfg_file,
                                          apply_raw_iq_order_override,
                                          parse_calibration, shortrange_config)
from processing.radar_reader     import read_frame, count_frames
from processing.radar_processing import (process_frame,
                                          save_arr_drea, save_xyz_cloud)
from processing.radar_plots      import plot_all, run_folder_viewer, _det_counts, _det_label


def _build_parser():
    p = argparse.ArgumentParser(
        description="AWR2243-2X cascade radar processing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O
    src = p.add_mutually_exclusive_group()
    src.add_argument("--bin",    metavar="FILE")
    src.add_argument("--folder", default=r"frames", metavar="DIR")
    p.add_argument("--cfg",         metavar="FILE",
                   default=os.path.join(_PARENT, "firmware", "r017m", "cascade_r017m.cfg"))
    p.add_argument("--raw-iq-order",
                   choices=("cfg", "i-first", "q-first"),
                   default="i-first",
                   help="Match capture/test.py raw IQ override: cfg=SampleSwap from .cfg, i-first=[I,Q], q-first=[Q,I]")
    p.add_argument("--frame",       type=int, default=0)
    p.add_argument("--all-frames",  action="store_true")
    p.add_argument("--no-show",     action="store_true")
    p.add_argument("--save",        metavar="FILE", default=None)
    p.add_argument("--save-dir",    metavar="DIR",  default=None)
    p.add_argument("--summary",     action="store_true")
    p.add_argument("--pause",       type=float, default=0.15)
    p.add_argument("--quiet",       action="store_true",
                   help="Suppress per-frame diagnostic prints (CFAR margin, "
                        "fallback, suppression). Use for clean operational "
                        "display. Terminal still shows frame summary lines.")

    # Calibration
    p.add_argument("--no-calib",    action="store_true")

    # AoA
    p.add_argument("--aoa", default=None,
                   choices=["fft","capon","music","bartlett"])
    p.add_argument("--music-sources", type=int, default=None)
    p.add_argument("--music-diag-load", type=float, default=None,
                   help="Relative diagonal loading used by MUSIC covariance estimation")
    p.add_argument("--no-music-fb", action="store_true",
                   help="Disable forward-backward averaging in MUSIC")
    p.add_argument("--no-music-unique-aperture", action="store_true",
                   help="Disable duplicate-aperture collapsing before MUSIC/Capon/Bartlett")

    # CFAR
    p.add_argument("--cfar", default=None,
                   choices=["ca","os","caso","cago"])
    p.add_argument("--cfar-guard",   nargs=2, type=int, default=None, metavar=("R","S"))
    p.add_argument("--cfar-train",   nargs=2, type=int, default=None, metavar=("R","S"))
    p.add_argument("--cfar-fa-rate", type=float, default=None)
    p.add_argument("--cfar-os-rank", type=float, default=None)
    p.add_argument("--cfar-margin",  type=float, default=None)
    p.add_argument("--cfar-min-power", type=float, default=None,
                   help="Absolute minimum dB level before CFAR/NMS can keep a cell")
    p.add_argument("--nms-size", nargs=2, type=int, default=None, metavar=("R","A"))

    # Display
    p.add_argument("--fixed-db",   action="store_true",
                   help="Use fixed dB limits instead of adaptive")
    p.add_argument("--db-vmin",    type=float, default=None)
    p.add_argument("--db-vmax",    type=float, default=None)
    p.add_argument("--db-span",    type=float, default=None,
                   help="Adaptive window (dB) — default 35")
    p.add_argument("--pc2d-fixed-db", action="store_true",
                   help="Use fixed dB limits for the PC2D cloud panel")
    p.add_argument("--pc2d-db-vmin", type=float, default=None)
    p.add_argument("--pc2d-db-vmax", type=float, default=None)
    p.add_argument("--pc2d-db-span", type=float, default=None,
                   help="Adaptive dB window width for the PC2D cloud panel")
    p.add_argument("--pc2d-pct-low", type=float, default=None,
                   help="Lower percentile for adaptive PC2D color scaling")
    p.add_argument("--pc2d-pct-high", type=float, default=None,
                   help="Upper percentile for adaptive PC2D color scaling")

    # Filter
    p.add_argument("--moving-only",  action="store_true")
    p.add_argument("--no-clutter",   action="store_true")
    p.add_argument("--gaussian",     type=float, default=None)
    p.add_argument("--dsp-mode",     default=None, choices=["cascade"],
                   help="Cascaded radar processing path")
    p.add_argument("--zero-dc-bins", type=int, default=None)
    p.add_argument("--min-doppler",  type=float, default=None,
                   help="Minimum |doppler| in m/s for moving-object filters")
    p.add_argument("--no-rd-confirmation", action="store_true")
    p.add_argument("--high-moving", action="store_true",
                   help="Keep only strong moving detections above local RA noise floor")
    p.add_argument("--high-moving-min-dop", type=float, default=None)
    p.add_argument("--high-moving-db", type=float, default=None)
    p.add_argument("--no-sidelobe-filter", action="store_true")
    p.add_argument("--no-rd-stripe-removal", action="store_true",
                   help="Disable Doppler-bin background suppression in the RD map")
    p.add_argument("--rd-sidelobe-db", type=float, default=None,
                   help="Keep RD detections within this dB of the strongest Doppler peak per range bin")
    p.add_argument("--ra-sidelobe-db", type=float, default=None,
                   help="Keep RA detections within this dB of the strongest azimuth peak per range bin")
    p.add_argument("--range-window", default=None,
                   choices=["none","rect","hann","hanning","hamming","blackman","chebyshev","chebwin"])
    p.add_argument("--doppler-window", default=None,
                   choices=["none","rect","hann","hanning","hamming","blackman","chebyshev","chebwin"])
    p.add_argument("--az-window", default=None,
                   choices=["none","rect","hann","hanning","hamming","blackman","chebyshev","chebwin"])
    p.add_argument("--el-window", default=None,
                   choices=["none","rect","hann","hanning","hamming","blackman","chebyshev","chebwin"])
    p.add_argument("--cheb-atten-db", type=float, default=None)

    # Save outputs
    p.add_argument("--save-drea", metavar="FILE", default=None,
                   help="Save arrDREA array to this path (.npy). "
                        "Auto-save is DISABLED by default; provide this flag to enable.")
    p.add_argument("--save-xyz",  metavar="FILE", default=None,
                   help="Save xyz point cloud to this path (.npy or .csv). "
                        "Auto-save is DISABLED by default; provide this flag to enable.")

    # Range
    p.add_argument("--max-range", type=float, default=None)
    p.add_argument("--roi-x", nargs=2, type=float, default=None, metavar=("MIN","MAX"))
    p.add_argument("--roi-y", nargs=2, type=float, default=None, metavar=("MIN","MAX"))
    p.add_argument("--roi-z", nargs=2, type=float, default=None, metavar=("MIN","MAX"))

    return p


def _apply(cfg: RadarConfig, args: argparse.Namespace) -> RadarConfig:
    if args.no_calib:                  cfg.apply_calibration = False
    if args.aoa:                       cfg.aoa_method        = args.aoa
    if args.music_sources:             cfg.music_num_sources = args.music_sources
    if args.music_diag_load is not None: cfg.music_diag_load = args.music_diag_load
    if args.no_music_fb:               cfg.music_forward_backward_avg = False
    if args.no_music_unique_aperture:  cfg.music_use_unique_aperture = False
    if args.cfar:                      cfg.cfar_method       = args.cfar
    if args.cfar_guard:                cfg.cfar_guard        = list(args.cfar_guard)
    if args.cfar_train:                cfg.cfar_train        = list(args.cfar_train)
    if args.cfar_fa_rate is not None:  cfg.cfar_fa_rate      = args.cfar_fa_rate
    if args.cfar_os_rank is not None:  cfg.cfar_os_rank      = args.cfar_os_rank
    if args.cfar_margin  is not None:  cfg.cfar_margin_db    = args.cfar_margin
    if args.cfar_min_power is not None: cfg.cfar_min_power_db = args.cfar_min_power
    if args.nms_size:                  cfg.nms_size          = list(args.nms_size)
    if args.fixed_db:                  cfg.use_fixed_db      = True
    if args.db_vmin is not None:       cfg.db_vmin           = args.db_vmin
    if args.db_vmax is not None:       cfg.db_vmax           = args.db_vmax
    if args.db_span is not None:       cfg.db_span           = args.db_span
    if args.pc2d_fixed_db:             cfg.pc2d_dynamic_db   = False
    if args.pc2d_db_vmin is not None:  cfg.pc2d_db_vmin      = args.pc2d_db_vmin
    if args.pc2d_db_vmax is not None:  cfg.pc2d_db_vmax      = args.pc2d_db_vmax
    if args.pc2d_db_span is not None:  cfg.pc2d_db_span      = args.pc2d_db_span
    if args.pc2d_pct_low is not None:  cfg.pc2d_cmap_pct_low = args.pc2d_pct_low
    if args.pc2d_pct_high is not None: cfg.pc2d_cmap_pct_high = args.pc2d_pct_high
    if args.moving_only:               cfg.moving_objects_only = True
    if args.no_clutter:                cfg.clutter_removal   = False
    if args.gaussian is not None:      cfg.gaussian_sigma    = args.gaussian
    if args.dsp_mode:                  cfg.dsp_mode          = args.dsp_mode
    if args.zero_dc_bins is not None:  cfg.zero_dc_bins      = args.zero_dc_bins
    if args.min_doppler is not None:   cfg.min_abs_doppler_mps = args.min_doppler
    if args.no_rd_confirmation:        cfg.require_rd_confirmation = False
    if args.high_moving:               cfg.filter_high_moving = True
    if args.high_moving_min_dop is not None: cfg.high_moving_min_dop = args.high_moving_min_dop
    if args.high_moving_db is not None: cfg.high_moving_db_above = args.high_moving_db
    if args.no_sidelobe_filter:        cfg.apply_sidelobe_filters = False
    if args.no_rd_stripe_removal:      cfg.doppler_stripe_removal = False
    if args.rd_sidelobe_db is not None: cfg.rd_sidelobe_db  = args.rd_sidelobe_db
    if args.ra_sidelobe_db is not None: cfg.ra_sidelobe_db  = args.ra_sidelobe_db
    if args.range_window:              cfg.range_window      = args.range_window
    if args.doppler_window:            cfg.doppler_window    = args.doppler_window
    if args.az_window:                 cfg.az_window         = args.az_window
    if args.el_window:                 cfg.el_window         = args.el_window
    if args.cheb_atten_db is not None: cfg.chebyshev_atten_db = args.cheb_atten_db
    if args.save_drea:                 cfg.save_drea_path    = args.save_drea
    if args.save_xyz:                  cfg.save_xyz_path     = args.save_xyz
    if args.max_range is not None:     cfg.max_range_m       = args.max_range
    if args.roi_x:                     cfg.roi_x_m           = tuple(args.roi_x)
    if args.roi_y:                     cfg.roi_y_m           = tuple(args.roi_y)
    if args.roi_z:                     cfg.roi_z_m           = tuple(args.roi_z)
    if args.quiet:                     cfg.quiet_mode        = True
    return cfg


def run_frame(bin_path, cfg, correction, fi, args, save_path=None):
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n[MAIN] {os.path.basename(bin_path)}  frame {fi} …")
    cube   = read_frame(bin_path, cfg, fi)
    result = process_frame(cube, cfg, correction, verbose=True)

    rd, ra, cloud = _det_counts(result)
    print(f"\n[MAIN] {_det_label(rd, ra, cloud)}")
    if cloud:
        ranges = np.unique(np.round(result.xyz_cloud[:, 5].astype(float), 2))
        print("  Ranges (m): " + ", ".join(f"{r:.2f}" for r in ranges[:20]))
        if len(ranges) > 20:
            print(f"  ... and {len(ranges)-20} more ranges")
        print("  x(m)   y(m)   z(m)  pwr(dB)  dop(m/s) rng(m) az(°)")
        for row in result.xyz_cloud[:min(cloud, 15)]:
            print(f"  {row[0]:5.2f}  {row[1]:5.2f}  {row[2]:5.2f}  "
                  f"{row[3]:6.1f}  {row[4]:+6.3f}   {row[5]:5.2f}  {row[6]:6.1f}")
        if cloud > 15: print(f"  … and {cloud-15} more")

    title = (f"Frame {fi}  |  {os.path.basename(bin_path)}  |  "
             f"AoA:{cfg.aoa_method.upper()}  CFAR:{cfg.cfar_method.upper()}  |  "
             f"{_det_label(rd, ra, cloud)}  |  "
             f"dop_max=±{cfg.dop_max:.2f} m/s")
    fig   = plot_all(result, cfg, title=title, save_path=save_path)

    if not args.no_show:
        plt.tight_layout()
        plt.show(block=True)
    plt.close(fig)
    return result


def main():
    args = _build_parser().parse_args()
    if args.bin is None and args.folder is None:
        sys.exit("Provide --bin FILE  or  --folder DIR")

    if args.cfg:
        cfg        = parse_cfg_file(args.cfg)
        correction = parse_calibration(args.cfg, cfg.num_tx, cfg.num_rx,
                                        enabled=not args.no_calib)
    else:
        print("[MAIN] No --cfg — using shortrange defaults")
        cfg        = shortrange_config()
        correction = np.ones((cfg.num_tx, cfg.num_rx), dtype=np.complex64)

    cfg = apply_raw_iq_order_override(cfg, args.raw_iq_order)
    if args.raw_iq_order != "cfg":
        print(f"[MAIN] raw IQ override -> {'Q,I' if cfg.iq_swap else 'I,Q'}")
    cfg = _apply(cfg, args)
    cfg.print_summary()

    # C6: print auto-save state once at startup so it is visible in both
    # --folder (run_folder_viewer) and --bin (run_frame) modes.
    # Printed here rather than inside process_frame to avoid repeating 70x.
    _drea = cfg.save_drea_path if cfg.save_drea_path else "disabled"
    _xyz  = cfg.save_xyz_path  if cfg.save_xyz_path  else "disabled"
    print(f"[MAIN] auto-save: arrDREA={_drea}  xyz={_xyz}")

    if args.summary:
        return

    if args.folder:
        run_folder_viewer(args.folder, cfg, correction,
                          pause_s=args.pause, save_dir=args.save_dir)
        return

    bin_path = args.bin
    if not os.path.isfile(bin_path):
        sys.exit(f"[ERROR] {bin_path} not found")

    n_frames = count_frames(bin_path, cfg)
    print(f"[MAIN] {bin_path}  →  {n_frames} frame(s)")
    if n_frames == 0:
        sys.exit("[ERROR] No complete frames — check config/file size")

    if args.all_frames:
        sdir = args.save_dir
        if sdir: os.makedirs(sdir, exist_ok=True)
        for i in range(n_frames):
            sp = os.path.join(sdir, f"frame_{i:04d}.png") if sdir else None
            run_frame(bin_path, cfg, correction, i, args, save_path=sp)
    else:
        fi = args.frame
        if fi >= n_frames:
            sys.exit(f"[ERROR] --frame {fi} but file has {n_frames} frames")
        run_frame(bin_path, cfg, correction, fi, args, save_path=args.save)


if __name__ == "__main__":
    main()
