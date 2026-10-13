# Project status and roadmap (as of 2026-10-13)

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
| Replay | GPU replay of all sensors: bird's-eye views, LiDAR target overlay, event strip, 3D LiDAR, radar point cloud (raw live, filtered pre-computed). |
| Tests | `sync_capture_pipeline/tests/`: 96 report tests and 6 radar-cloud tests. They need the reference sessions; each test skips if its session is absent. |

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
  and the three-sensor extrinsics), and their own radar processing chain
  (`cascaded_radar_capture_pipeline/processing`, see
  `sync_capture_pipeline/NOTES_FOR_TEAM.md`).
- **This repository:** capture, synchronisation, validation, replay and
  firmware packaging.
- **The Windows firmware-build machine** (TI SDK, firmware source) has its own
  repository. Briefs to it go through a USB handover folder (see
  [09](09_working_practices.md)).

## Open items and ideas

- **On hold (owner's decision):** recommending a 90 s Stage 3 check on the
  first field day after mounting. Do not add it to the guide unless asked.
- The radar azimuth sign is mirrored in the team's `process_frame`, and its
  x/y convention differs. It is corrected for display only; changing it is the
  team's call.
- Radar angular focusing is weak and elevation is unreliable until the team's
  antenna calibration.
- The DCA1000 throttle fault has an unknown cause (power cycle clears it).
- The Windows capture config `AM273X_Capture_fixed.json` still uses a 50 µs
  packet delay, which loses data at 8 fps.
- Only `r075m_v2` and `r075m_v2_8fps` are hardware-proven. Run the acceptance
  check before using any other variant.
- Untested idea: run `radar_cloud.py` under the `cascaded-radar` conda environment, which has CuPy. The team's `process_frame` would then use its GPU path and might run much faster than the ~1 s/frame CPU fallback.
- Possible later work: a GPU CFAR so the filtered radar cloud can run live
  (needs the team's agreement, since it changes their code); a CI subset that
  runs without hardware or data.
