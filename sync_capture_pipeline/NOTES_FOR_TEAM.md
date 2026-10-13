# Notes for the research team: things to know before processing the data

Findings from building and validating the capture pipeline, 2026-10-09 to
2026-10-11. Each item says what was seen, where, and what it means for
processing. The reference session for all of them is
`/mnt/sdcard/sessions/freewalk_ref_02`: a 10-minute scripted walk whose
segments are listed in `SYNC_DESIGN_3_SENSORS.md` §11.4.

## Radar

1. **Azimuth sign is mirrored in the current processing.**
   - **Seen:** a walker 3 m to the right at 8 m (camera and LiDAR agree)
     comes out on the left, and vice versa.
   - **Where:** both in the replay's processor and in
     `processing/radar_processing.process_frame`. A test frame with the LiDAR
     at −7.3° gave `xyz_cloud` azimuth **+6.3°**: right magnitude, opposite
     sign.
   - **Status:** the replay and the session report correct it
     (`RadarProcessor.MIRROR_AZIMUTH`). **`process_frame` has not been
     changed.** It is your research chain, so the fix is your decision. The
     likely source is the TX/RX → azimuth table in `radar_config`, which is
     unverified.
2. **Angular focusing is weak.**
   - **Seen:** a single person appears in range–azimuth maps as an arc across
     much of the field of view. The arc's peak is at the right angle; range
     is sharp.
   - **Cause:** the antenna calibration in the `.cfg` files was probably
     measured with the old HPF setting (2/2). The v2 firmware uses 0/0, which
     changes the RX phase response. Recalibrating with the v2 images should
     sharpen it.
3. **Elevation is unreliable.**
   - **Seen:** `process_frame` placed a walking person 1.8–2.1 m above the
     radar.
   - Elevation comes from a single 8-channel row (TX5) and has not been
     verified.
4. **Velocity wraps above ±4.7 m/s** with the r075m waveform (64 loops × 6
   TX TDM). Targets faster than ~17 km/h, such as passing vehicles, show an
   aliased radial velocity. This is a waveform property, not fixable in
   processing.
5. **Data from images before `_v2` is ~50% corrupted** when booted from QSPI
   (stale buffer words). Only `r075m_v2_8fps` has been used for the sessions
   on the SD card. See `cascaded_radar_capture_pipeline/firmware/FIRMWARE_STATUS.md`.
6. **I/Q order is I first.** The old `.cfg` files implied Q first (mirrored
   range spectrum); they are corrected.
7. **Frame times.** Use `sync/radar_timeline.csv` `t_ref`. It is de-jittered,
   has the Stage 3 offset applied, and falls back to nominal times if the
   DCA1000 backlogged (`alignment.json` → `radar.timing_source`).

## LiDAR (RoboSense RS-Fairy)

1. **Fixed 138,240-point grid.** Directions with no return come back as
   **zero-range points**. Filter `range > 0` before any statistics or
   learning; about 92k points per sweep are valid returns on this rig.
2. **About 44% of valid returns are within 1 m** on the bench rig (the mount
   and its surroundings). The replay hides returns nearer than 0.5 m by
   default.
3. **Per-point times are available.** The wrapper's 6th column is the time
   offset from the sweep timestamp. Sweeps are split at 180° (behind), so the
   forward field of view is never cut in two. `sync/lidar_timeline.csv` gives
   the time the radar-boresight points were scanned.
4. **Mounting must stay fixed.** The LiDAR was once found turned 57° after
   reconnecting. Re-record your extrinsic calibration if any sensor moves.

## Camera (ZED 2i)

1. **Recorded as lossless H.264 SVO2**; images are extracted offline. Decoding
   runs at only ~40–50 frames/s on the Orin, so extraction is slower than real
   time when other processing runs.
2. **Dropped frames:** typically 0.2–0.3%, single frames, recorded in
   `capture/timing/camera_gaps.csv` and flagged in `pairs.csv`.

## Synchronisation

- **Offsets** (Stage 3, LiDAR reference, added to each sensor's time):
  radar **+12.65 ± 1.45 ms**, camera **+23.34 ± 3.28 ms**. They are applied
  automatically by `align_session.py` / `make_report.py`.
- **Re-checked** on the free-walk session with a geometry-free velocity
  method: no significant shift.
- **Pairing** (`sync/pairs.csv`): one row per radar frame, with the nearest
  camera frame and LiDAR sweep and their time errors.
  - Camera errors are bounded at ±16.7 ms.
  - LiDAR errors fall on four levels (beat between 125 ms and 100 ms) within
    ±50 ms.
  - Rows in a camera or LiDAR gap are flagged.
- **One command per session:** `make_report.py` aligns the session, writes
  `pairs.csv`, and produces `session_report.html`.

## Radar point cloud (delivered 2026-10-13)

- **Raw:** every range–azimuth cell above a display threshold, live, z = 0.
- **Filtered:** your `process_frame` output, unchanged, pre-computed per
  session into `sync/radar_cloud_filtered.npz`. It holds your original
  `xyz_cloud` columns (`team_xyz`) and the same points in the pipeline frame
  (`xyz`: x forward, y left, mirror-corrected), indexed per radar frame
  (`start`). See `replay/README.md`.
- `process_frame` itself was not modified. Its azimuth sign and x/y
  convention are converted at display time only.
