# RoadSafety-CRCL
A dedicated project for recording and processing data from Cascaded Radar, Camera, and Lidar sensors mounted on two-wheelers to check the road safety factors.

Data-collection system for a two-wheeler road-safety research project. It
records three sensors on an NVIDIA Jetson AGX Orin, synchronises them on one
clock, and produces validated, paired data for annotation and machine
learning:

| Sensor | Model | Recorded as |
|---|---|---|
| Radar | Cascaded TI mmWave: 2 × AWR2243 + AM273x, DCA1000 capture | raw ADC frames, 8 fps (`frame_*.bin`) |
| LiDAR | RoboSense RS-Fairy | raw UDP packets (`lidar.pcap`), 10 Hz |
| Camera | Stereolabs ZED 2i | lossless stereo video (`camera.svo2`), 30 fps |

## What's in here

- **Capture:**
  - a three-sensor launcher with preflight checks and watchdogs;
  - an iPad web control;
  - per-sensor recorders;
  - radar firmware images and flashing tools.
- **Synchronisation:** offline alignment on CLOCK_MONOTONIC, with measured
  sensor time offsets ("Stage 3"), producing `pairs.csv`: one row per radar
  frame with its camera frame and LiDAR sweep.
- **Validation:** an automatic completeness check after each session, plus a
  self-contained **HTML report** per session.
- **Replay:** GPU playback of all sensors in one window, with bird's-eye
  views, a 3D LiDAR view, and live radar and LiDAR detection and tracking.
- **Processing:** one PyTorch library (`sync_capture_pipeline/perception/`)
  for radar signal processing, detection and tracking on the GPU.

## Start here

| You want to… | Read |
|---|---|
| Understand the project (people **and AI agents**) | [AGENTS.md](AGENTS.md) |
| Set up a new Jetson | [docs/NEW_BOARD_SETUP.md](docs/NEW_BOARD_SETUP.md) |
| Record data in the field | [sync_capture_pipeline/DATA_COLLECTION_GUIDE.md](sync_capture_pipeline/DATA_COLLECTION_GUIDE.md) |
| Process the data (research team) | [sync_capture_pipeline/NOTES_FOR_TEAM.md](sync_capture_pipeline/NOTES_FOR_TEAM.md) |
| Know the design and its decisions | [sync_capture_pipeline/SYNC_DESIGN_3_SENSORS.md](sync_capture_pipeline/SYNC_DESIGN_3_SENSORS.md), [docs/knowledge/](docs/knowledge/) |

## Typical workflow

```bash
# 1. record: from the iPad page (http://<jetson>.local:8080) - writes /mnt/sdcard/sessions/<name>/
# 2. align + pair + report (one command):
python sync_capture_pipeline/make_report.py --session /mnt/sdcard/sessions/<name>
#    -> <session>/sync/pairs.csv, <session>/session_report.html
# 3. look at it:
python sync_capture_pipeline/replay/replay_session.py --session /mnt/sdcard/sessions/<name>
```

## Data

Recordings are **not** stored in this repository (~112 GB per hour of
recording). The reference sessions, including the 10-minute `freewalk_ref_02`,
are kept on separate storage by the project owner.

## License

This project's own code is licensed under the **GNU General Public License
v3.0**; see [LICENSE](LICENSE).

Third-party software included in this repository keeps its own license:
- RoboSense **rs_driver** (BSD-3-Clause, modified);
- Texas Instruments flashing tools (BSD-3-Clause);
- the radar firmware images (TI components, including the AWR2243 patch under
  the TI Text File License). These are separate programs distributed
  alongside the code, not covered by the GPL, and may only be used with TI
  devices.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
