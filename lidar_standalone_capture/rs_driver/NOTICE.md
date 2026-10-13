# NOTICE: modified copy of RoboSense rs_driver

This directory is **based on RoboSense's open-source `rs_driver`**:

- **Upstream:** https://github.com/RoboSense-LiDAR/rs_driver
- **Base commit:** `897b14d3bdb6186a75df27ba51b65b5bd5557723` (2026-07-10,
  "[feat]:support RSAIRYLITE_ETH (#93)")
- **License:** 3-clause BSD, Copyright (c) 2020 RoboSense. See
  [LICENSE](LICENSE), which must be kept with this code.

It is redistributed in this repository **with modifications** made for the
RoadSafety-CRCL project. RoboSense has not reviewed or endorsed these changes,
and they are not part of the upstream project.

## Changes relative to upstream

**Modified upstream files:**
- `CMakeLists.txt`: a build message for the PCL option; adds the
  `python_wrapper` subdirectory when demos are built.
- `demo/CMakeLists.txt`: builds the project's Python-binding demo setup.
- `demo/demo_online.cpp`: reads the driver parameters (LiDAR type, ports,
  IPs, distance and angle limits, timestamp options, calibration offsets) from
  a YAML config instead of hard-coded values; uses PCL point types.

**Added by this project:**
- `python_wrapper/`, `python_wrapper_2/`: pybind11 Python bindings
  (`robosense_lidar.cpp`, `lidar.py`). `python_wrapper_2` is the one in use.
  Its changes:
  - double-precision timestamps;
  - GIL-safe `start()`/`stop()`;
  - pcap playback (`pcap_path`, `pcap_rate`, `input_finished()`);
  - per-point time offsets (PointXYZIRT, 6th column);
  - optional `split_angle`.
- `config/config.yaml`: project LiDAR/host addresses (192.168.34.x) and
  `split_angle: 180`.
- `demo/record_demo_online.cpp`: recording demo.

**Not changed:** every other upstream file. Upstream file-permission bits
were also normalised; that is not a content change.

The original upstream documentation (`README.md`, `README_CN.md`, `doc/`) is
kept unchanged and describes the upstream driver.
