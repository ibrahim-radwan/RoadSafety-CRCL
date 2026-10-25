# AGENTS.md: start here

This is the entry point for any AI agent (or new developer) working in this
repository. Read it fully before doing any task, then read the documents it
points to for that task.

## What this project is

A **three-sensor data-collection rig for two-wheeler road safety**, running on
an NVIDIA **Jetson AGX Orin**. The sensors:

- a **cascaded TI mmWave radar**: 2 × AWR2243 + AM273x, raw ADC through a
  DCA1000, 8 frames/s;
- a **RoboSense RS-Fairy LiDAR**: 10 Hz, raw packets to pcap;
- a **Stereolabs ZED 2i stereo camera**: 30 fps, lossless SVO2.

The system records raw data from all three, aligns them offline on one clock,
validates and reports each session, and replays them. The data feeds an
end-user research team (annotation and ML). Firmware for the radar is built on
a separate Windows machine with its own repository.

## Read in this order

1. **[docs/knowledge/01_status_and_roadmap.md](docs/knowledge/01_status_and_roadmap.md)**:
   current state, reference recordings, ownership, open items.
2. **[docs/knowledge/09_working_practices.md](docs/knowledge/09_working_practices.md)**:
   the rules agreed with the owner. **Mandatory.**
3. **[docs/knowledge/08_pitfalls_and_lessons.md](docs/knowledge/08_pitfalls_and_lessons.md)**:
   what has already gone wrong and why.
4. The topic page for your task:

   | Task area | Read |
   |---|---|
   | Hardware, network, OS settings | [02_hardware_and_network.md](docs/knowledge/02_hardware_and_network.md), [docs/NEW_BOARD_SETUP.md](docs/NEW_BOARD_SETUP.md) |
   | Radar data, firmware, DCA1000 | [03_radar_firmware_and_dca.md](docs/knowledge/03_radar_firmware_and_dca.md), `cascaded_radar_capture_pipeline/firmware/FIRMWARE_STATUS.md` |
   | Storage, recording | [04_storage_and_recording.md](docs/knowledge/04_storage_and_recording.md) |
   | LiDAR | [05_lidar.md](docs/knowledge/05_lidar.md) |
   | Camera | [06_camera.md](docs/knowledge/06_camera.md) |
   | Time sync, Stage 3 calibration | [07_synchronisation_and_stage3.md](docs/knowledge/07_synchronisation_and_stage3.md), `sync_capture_pipeline/SYNC_DESIGN_3_SENSORS.md` |
   | Operating the rig | `sync_capture_pipeline/DATA_COLLECTION_GUIDE.md` |
   | Replay tool | `sync_capture_pipeline/replay/README.md` |
| Detection (ours), processing speed | `sync_capture_pipeline/perception/README.md`, [03](docs/knowledge/03_radar_firmware_and_dca.md), [05](docs/knowledge/05_lidar.md) |
   | Facts for the research team | `sync_capture_pipeline/NOTES_FOR_TEAM.md` |

## Repository map

```
sync_capture_pipeline/          <- MAIN CODE: capture, sync, validation, report, replay, web UI
  capture_session.py            three-sensor launcher (preflight, watchdogs, session.json)
  dca_capture_radar_sync.py     radar receiver (DCA1000 control + UDP, frames.csv), stdlib only
  align_session.py              offline alignment -> sync/*_timeline.csv, pairs.csv, alignment.json
  check_session.py              completeness check, PASS/FAIL (stdlib only)
  measure_sync_offset.py        Stage 3 offset measurement (plate routine)
  make_report.py, report/       descriptive HTML session report (no verdicts) + data collection
  radar_cloud.py                filtered radar point cloud via the team's process_frame (pre-computed; CuPy GPU path)
  perception/                   OUR GPU detection: radar.py (CFAR, TDM comp, multipath flags), lidar.py
                                (local ground, objects), tracking.py, estimate_rx_phase.py; README.md
  run_perception.py             our detection + tracking over a session -> sync/perception_*.npz, scores
  timing_consistency.py         range-based timing check per window (geometry-biased; see 07)
  radar_firmware.py             which firmware recorded a session -> which .cfg
  board_health.py               Jetson temperatures
  check_radar_frames.py         radar stale-word / range-time check
  calibration/stage3_offsets.json   the offsets in use (applied by align_session by default)
  calibration/radar_rx_phase.json   in-situ radar phase correction (provisional, until the team's recalibration)
  web_control/                  Flask iPad control (systemd service, port 8080)
  replay/                       PyQt/pyqtgraph replay: one window, 6 panels, live detection (camera in its own process)
  tests/                        pytest (needs the reference sessions; skips when absent)
  handover/                     briefs sent to the Windows firmware-build agent
  legacy/                       superseded originals, kept for reference
  sessions/                     OLD eMMC recordings (git-ignored)
camera_standalone_capture/ZED_2i_Sensor/   ZED recorder (recorder.py, config.yaml)
lidar_standalone_capture/                  LiDAR raw recorder + checker
  rs_driver/                    RoboSense rs_driver (third-party, modified; see its NOTICE.md)
    python_wrapper_2/           our pybind11 wrapper (robosense_lidar.cpp, lidar.py)
cascaded_radar_capture_pipeline/           the research team's radar code
  processing/                   their processing chain - DO NOT modify without their approval
  firmware/                     20 radar firmware images + .cfg + flash configs + firmware_map.json
flashing/boot/                  TI UART flashing tools + flash_variant.sh (third-party, see NOTICE)
utils/                          earlier investigation and benchmarking scripts
environment/rel_1_py_3_8.yml    conda environment (vendor wheels installed separately)
docs/                           knowledge base + new-board setup
```

## Hard rules (summary; full list in 09_working_practices.md)

- **Ask the owner first** before:
  - deleting recordings;
  - flashing firmware;
  - changing network/fstab/sysctl/services;
  - installing into the shared conda environment;
  - pushing to GitHub.
- **Don't modify** `cascaded_radar_capture_pipeline/processing/` (the research
  team's chain) or the radar's network/DCA settings unless asked.
- **Never commit** recordings (`sessions/`, `*.svo2`, `*.pcap`, radar
  `frame_*.bin`) or vendor wheels.
- **Commits carry no AI-assistant attribution** (no "Co-Authored-By" lines, no
  tool references) and are authored by the human committer.
- **Verify on real data** (render it, run it, test it) before claiming
  something works, and say what wasn't verified.
- **Update the docs** (topic page, README, design doc) in the same piece of
  work as a behaviour change.

## Conventions

- **Time:** CLOCK_MONOTONIC seconds. `t_ref` = aligned time with the Stage 3
  offsets applied. Report and replay plot time from the first radar frame.
- **Spatial frame:** x forward, y **left**, z up (the LiDAR frame; the radar is
  assumed co-located until the team's extrinsic calibration). Bird's-eye views
  draw forward up and vehicle-left on the left, like the camera image.
- **Radar azimuth:** the team's processing azimuth is positive to the
  **right** (their convention; it is not a mirror error). World azimuth
  (+left) = − processing azimuth. Report (`RadarProcessor.MIRROR_AZIMUTH`),
  `radar_cloud.py` and `perception/radar.py` all convert. Radar v_r > 0 =
  receding.
- **Radar frame layout:** int16 LE, I first,
  `[loop 64][tx 6][dev 2][rx 4][sample 256][I,Q]`.
- **Sensor colours** in figures and UI: radar `#2a78d6`, LiDAR `#eb6834`,
  camera `#1baf7a`.
- **Python:** 3.8 for everything that runs on the rig. Use
  `from __future__ import annotations`; no 3.9+ APIs.

## How to run things

```bash
conda activate rel_1_py_3_8                       # most tools
cd sync_capture_pipeline
python capture_session.py --out /mnt/sdcard/sessions/<name> --secs 60 --camera-stage ~/camera_stage   # (normally via the iPad page)
python make_report.py --session /mnt/sdcard/sessions/<name> [--radar-cloud]   # align + pairs.csv + HTML report
python replay/replay_session.py --session /mnt/sdcard/sessions/<name> [--maximise lidar3d] [--radar-points detections|raw|filtered]
python run_perception.py --session /mnt/sdcard/sessions/<name>               # our detections + scores
python -m pytest -q tests
```

A new machine needs [docs/NEW_BOARD_SETUP.md](docs/NEW_BOARD_SETUP.md) first.
The reference recordings are **not in git**. They are on the project's 1 TB SD
card and an external drive (`freewalk_ref_02` is the main one). Tests that
need them skip.

## Keeping this file useful

When you finish a piece of work, update
`docs/knowledge/01_status_and_roadmap.md` and any topic page you touched.
Change this file only if the map, rules or conventions change.
