"""
AWR2243-2X-CAS-EVM Radar Processing Pipeline
=============================================
  radar_config     – RadarConfig dataclass + .cfg parser + calibration
  radar_reader     – DCA1000 .bin frame reader
  radar_processing – Full DSP pipeline (range/Doppler/AoA/CFAR/arrDREA/xyz)
  radar_plots      – All 6 visualisation plots + sequential folder viewer
  main             – CLI entry point

Quick start
-----------
  from processing import parse_cfg_file, parse_calibration
  from processing import read_frame, process_frame, plot_all

  cfg    = parse_cfg_file("firmware/r017m/cascade_r017m.cfg")
  corr   = parse_calibration("firmware/r017m/cascade_r017m.cfg")
  cube   = read_frame("data.bin", cfg, frame_idx=0)
  result = process_frame(cube, cfg, corr)
  plot_all(result, cfg)

  # Or sequential folder viewer:
  from processing.radar_plots import run_folder_viewer
  run_folder_viewer("data_dir/", cfg, corr)
"""

from .radar_config     import (RadarConfig, parse_cfg_file, parse_calibration,
                                shortrange_config)
from .radar_reader     import read_frame, count_frames, read_all_frames
from .radar_processing import process_frame, ProcessingResult
from .radar_plots      import (plot_all, plot_range_doppler,
                                plot_range_azimuth, plot_range_elevation,
                                plot_range_azimuth_polar, plot_ra_sector,
                                plot_pc3d, run_folder_viewer,
                                compute_global_scale)

__all__ = [
    "RadarConfig", "parse_cfg_file", "parse_calibration", "shortrange_config",
    "read_frame", "count_frames", "read_all_frames",
    "process_frame", "ProcessingResult",
    "plot_all", "plot_range_doppler", "plot_range_azimuth",
    "plot_range_elevation", "plot_range_azimuth_polar",
    "plot_ra_sector", "plot_pc3d",
    "run_folder_viewer", "compute_global_scale",
]
