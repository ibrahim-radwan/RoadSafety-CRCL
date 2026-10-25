# Project status and roadmap (as of 2026-10-24)

## Where the project stands

A three-sensor data-collection rig for two-wheeler road safety, running on an
NVIDIA Jetson AGX Orin. All of the following are built and verified on
hardware:

| Area | State |
|---|---|
| Radar firmware | `r075m_v2_8fps` flashed (8 fps, 2.62–75 m). All 20 variants rebuilt with the DSS-L3 memory-init fix. See [03](03_radar_firmware_and_dca.md). |
| Capture | Three-sensor launcher with preflight checks, board-temperature monitoring and a radar-rate watchdog; iPad web control. See `sync_capture_pipeline/DATA_COLLECTION_GUIDE.md`. |
| Storage | 1 TB SD card (ext4), camera staged on the eMMC; 0.2% camera drops. See [04](04_storage_and_recording.md). |
| Synchronisation | Offline alignment on CLOCK_MONOTONIC. **Stage 3 offsets measured**: radar +12.65 ± 1.45 ms, camera +23.34 ± 3.28 ms (LiDAR reference). Applied automatically. See [07](07_synchronisation_and_stage3.md). |
| Validation | `check_session.py` (completeness, pass/fail) runs after every session. `make_report.py` produces a descriptive HTML report per session, with no verdict. |
| Replay | One window, six panels (camera, LiDAR, LiDAR 3D, radar BEV, range-Doppler, objects), each can be maximised; bird's-eye views fill their panels; live GPU detection and tracking on radar and LiDAR; event strip; radar point cloud (detections, raw, team-filtered). See `sync_capture_pipeline/replay/README.md`. |
| Detection (ours) | `sync_capture_pipeline/perception/` + `run_perception.py`: GPU radar CFAR (TDM Doppler compensation, in-situ phase correction, multipath-ghost flags), GPU LiDAR objects (local ground, overhead structures removed), Doppler-aware tracking. Scored against the LiDAR walker: see `perception/README.md`. The team's early chain is unchanged so far; it will be merged into this one (see [09](09_working_practices.md)). |
| Tests | `sync_capture_pipeline/tests/`: 96 report tests, 6 radar-cloud tests and 8 perception tests. Session tests need the reference sessions and skip if absent. |

## Reference recordings (not in git)

Kept on the 1 TB SD card (`/mnt/sdcard/sessions/`) and to be copied to an
external drive by the project owner.

| Session | Content |
|---|---|
| `freewalk_ref_02` | **The main reference.** A 10-minute scripted walk with these segments: empty scene, left/right check, free walk, close range (~2 m), far range (~20 m), standing, two people, plate check, empty scene. |
| `freewalk_ref_01` | DCA1000 throttle fault (only the first 103 s of radar is valid; nominal frame times). The fault test case. |
| `stage3_01..03` | Daylight Stage 3 calibration sessions (90 s each). |

Older eMMC sessions (`sync_capture_pipeline/sessions/`) predate the radar
firmware fix; their radar data is about 50% corrupted. They are kept only as
history.

## Ownership

- **End-user research team:** all sensor calibration (radar antenna/intrinsic
  and the three-sensor extrinsics), data collection, and the use of the data
  (annotation, ML). They pull this repository and read
  `sync_capture_pipeline/NOTES_FOR_TEAM.md`.
- **This repository:** capture, synchronisation, validation, replay,
  firmware packaging, and **all processing, filtering and detection** (since
  2026-10-25, at the team's request). That includes the radar processing
  chain that came from the team's initial Windows work
  (`cascaded_radar_capture_pipeline/processing`), which is to be merged into
  `sync_capture_pipeline/perception/`. The team no longer runs the tools in
  that folder.
- **The Windows firmware-build machine** (TI SDK, firmware source) has its own
  repository. Briefs to it go through a USB handover folder (see
  [09](09_working_practices.md)).

## Second machine: `ibrahim-desktop` (2026-10-24)

A second Orin (JetPack R35.4.1, same as the rig) set up for replay and
processing work, user `ibrahim`.

- `rel_1_py_3_8` was created per `docs/NEW_BOARD_SETUP.md`, with these
  adjustments, which are now in that file:
  - no `fpga-udp`;
  - no `ultralytics`, which would replace the CUDA torch;
  - `fastrlock` 0.8.2 and CuPy 12.3 added.
- Pyzed 4.1.4 is installed and the LiDAR wrapper is built.
- The reference sessions are on the external drive at
  `/media/ibrahim/Elements/HASSAN_JETSON_SYNC_CAPTURE_SESSIONS/`. It is NTFS
  through FUSE, which is slow (see [08](08_pitfalls_and_lessons.md)).
- ROS Foxy sets `PYTHONPATH` in this user's shell, so run the tools with it
  unset.

## Open items and ideas

- **On hold (owner's decision):** recommending a 90 s Stage 3 check on the
  first field day after mounting. Do not add it to the guide unless asked.
- **The azimuth "mirror" is closed:** it is a frame convention (the team's
  azimuth is right-positive), not an error in `process_frame`. See
  [03](03_radar_firmware_and_dca.md).
- **Radar calibration findings for the team** (in `NOTES_FOR_TEAM.md`):
  - an inter-device TX phase error (TX3/TX4 ~100–125° off);
  - beam sidelobes only 5–10 dB down;
  - the radar–LiDAR yaw differs per mounting;
  - the elevation bias.
  
  Our in-situ correction (`calibration/radar_rx_phase.json`) is provisional
  until their recalibration.
- **The team's CuPy and CPU paths differ** (CASO vs OS CFAR on
  range-Doppler). Reported to them. `radar_cloud.py` now uses the GPU path.
- Next for detection:
  - a learned LiDAR classifier, or at least a static/moving split from
    tracks;
  - camera detection (needs a model and the team's camera extrinsics for
    overlays);
  - radar static-scene detection (zero Doppler, for parked vehicles on the
    road);
  - a per-session radar yaw estimate, if the team wants one before their
    extrinsics.
- The DCA1000 throttle fault has an unknown cause (power cycle clears it).
- The Windows capture config `AM273X_Capture_fixed.json` still uses a 50 µs
  packet delay, which loses data at 8 fps.
- Only `r075m_v2` and `r075m_v2_8fps` are hardware-proven. Run the acceptance
  check before using any other variant.
- Done 2026-10-24: CuPy (12.3, CUDA 11) is in `rel_1_py_3_8`. The team's
  `process_frame` runs its GPU path at ~0.1 s per frame, and `radar_cloud.py`
  uses it (38 ms per frame with 3 workers).
- Possible later work: a CI subset that runs without hardware or data. The
  synthetic perception tests already do.
- Possible later work: GPU LiDAR rendering in the replay. The numpy render
  takes ~10 ms per sweep and holds the GIL.
