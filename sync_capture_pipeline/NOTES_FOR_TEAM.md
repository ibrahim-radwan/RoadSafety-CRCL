# Notes for the research team: things to know before processing the data

Findings from building and validating the capture pipeline, 2026-10-09 to
2026-10-11. Each item says what was seen, where, and what it means for
processing. The reference session for all of them is
`/mnt/sdcard/sessions/freewalk_ref_02`: a 10-minute scripted walk whose
segments are listed in `SYNC_DESIGN_3_SENSORS.md` §11.4.

> **2026-10-25: one processing pipeline.** As you asked, your early processing
> chain (`cascaded_radar_capture_pipeline/processing`, `process_frame`) has been
> merged into this repository's PyTorch library,
> `sync_capture_pipeline/perception/`, and deleted (it is in git history).
>
> - **Your general stages were ported and tested against your code.** They
>   reproduce your chain's outputs on both of its paths: range/Doppler
>   processing, all CFAR variants, the sidelobe and consistency filters, the
>   focused range-azimuth map, Capon/MUSIC/Bartlett, and elevation.
> - **The single-person "body-shape" stage was retired.** It removed a second
>   road user at the same range. That covers the angle, cross-cluster and
>   subject gates, the xyz step, and the single-subject accumulation and
>   micro-Doppler.
> - **The radar point cloud is now the multi-object detections** described
>   at the end of this file. The "filtered" cloud (`radar_cloud_filtered.npz`)
>   is no longer produced.
> - **Accumulation and micro-Doppler are to be rebuilt per tracked object.**
>
> The findings below that mention `process_frame` describe your former code.
> They were true of it, and they explain choices in the new pipeline.

## Radar

1. **Azimuth: your processing is right; it is a convention difference
   (updated 2026-10-24).**
   - Your azimuth is positive to the **right**, as your README says ("x =
     lateral, positive = right of boresight").
   - Our pipeline frame is y **left**, so we negate it.
   - Checked on `freewalk_ref_02` against the LiDAR walker: 90% sign agreement
     beyond ±8°.
   - The earlier note called this a mirror in `process_frame`. That was wrong:
     our replay had read your azimuth as left-positive. Nothing in your code
     needs to change.
2. **Weak angular focusing: phase errors, now measured.** A single person
   shows as an arc, with a second peak ~13° away only ~4 dB down.
   - Using the LiDAR as the angle reference, we measured the residual phase of
     each virtual channel on the plate sessions (`stage3_02`, `stage3_03`).
   - It is stable: the two halves of the data agree within 2.7° median.
   - **TX3 and TX4 are ~100–125° off TX0–TX2.** Your chirp table puts TX0–2 on
     the master AWR2243 and TX3–5 on the slave, so this looks like an
     **inter-device TX phase error** in the cascade calibration. There is also
     a smaller per-RX pattern, consistent with the HPF change (2/2 → 0/0) in
     the v2 images.
   - The values are in `calibration/radar_rx_phase.json` (deg, [tx][rx]).
   - Applied on top of the `.cfg` calibration, they cut the azimuth spread on
     `freewalk_ref_02` (not used to measure them) from 5.95° to 2.43°.
   - Sidelobes still stay only ~5–10 dB below the peak. A proper
     recalibration with a reflector over a wide angle span should do much
     better. Ours is a stopgap until yours arrives.
3. **Elevation is unreliable.**
   - `process_frame` placed a walking person 1.8–2.1 m above the radar, and
     our detector reads ~+13° on the walker.
   - Elevation comes from the single 8-channel row on TX5, which is on the
     **slave device**. The inter-device phase error above would bias it in
     exactly this way. Worth checking in your recalibration.
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

8. **TDM-MIMO Doppler phase is not compensated in `process_frame`.** A moving
   target's phase advances between the 6 TX slots of a loop (2π f_d × 35 µs
   per slot). Our detector rotates each channel back by its TX slot, per
   Doppler bin. It is a small gain at walking speed (median azimuth error
   3.65° → 3.42°), but it grows with speed.
9. **Your GPU and CPU paths give different detections.** `process_frame`
   calls `cfar_2d_gpu(RD_mti_linear, cfg, force_caso=True)`.
   - Without CuPy, `cfar_2d_gpu` returns `_cfar_2d_cpu(heat, cfg)`, which
     drops `force_caso` and runs `cfg.cfar_method` (`os`). The CPU path
     therefore runs **OS**-CFAR on range-Doppler; the GPU path runs **CASO**.
   - The thresholds differ by up to 34 dB.
   - On `freewalk_ref_02` frames 2000–2029, half the frames had a different
     point count; the GPU path usually keeps fewer points.
   - Your environment has CuPy, so you see the CASO results. Our filtered
     clouds made before 2026-10-24 used the CPU path; `radar_cloud.py` now
     uses the GPU one and records which (`radar_cloud_filtered.json`).
   - Also, `gpu_accel.py` defines `cfar_2d_gpu` twice; the second definition
     wins.
10. **Multipath.** A walker produces radar images:
    - at **2× its range with 1.9× its radial velocity** (radar → person → rig
       → person → radar);
    - behind it at its own radial velocity (via walls and fences).

    Our detector flags both (`ghost` column). They were most of the radar
    detections the LiDAR did not support on `stage3_02`.
11. **The radar–LiDAR yaw differs between mountings by several degrees**
    (about −5° to +3° on the reference sessions). This is extrinsic
    calibration; ours assumes them co-located.

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

## Radar point cloud (delivered 2026-10-13, changed 2026-10-25)

- **Raw:** every range–azimuth cell above a display threshold, live in the
  replay, z = 0.
- **Filtered:** this was your `process_frame` output, pre-computed into
  `sync/radar_cloud_filtered.npz`. It was retired with the old chain on
  2026-10-25: the radar point cloud is now the detections below. Files
  already on disk are kept, and their scores are in
  `perception/baseline/scores.json`.

## The pipeline's detections (delivered 2026-10-24)

- `run_perception.py --session <s>` writes, in `sync/`:
  - `perception_radar.npz`: per frame, detections with a multipath flag,
    objects and tracks. Since 2026-10-26, each track row also carries its
    micro-Doppler features: torso velocity, Doppler spread, and cadence. The
    cadence is experimental.
  - `perception_lidar.npz`: per sweep, objects with boxes and size classes,
    and tracks. Since 2026-10-26, each track row also carries a moving /
    static flag. It is computed in the sensor frame, so it is valid for a
    stationary rig; a moving platform needs ego-motion compensation.
  - `perception.json`: scores against the LiDAR walker track.
- The column layouts are in the script's docstring. The frame is x forward,
  y left; times are `t_ref`.
- See `perception/README.md` for the method and the measured scores. On
  `freewalk_ref_02`, compared with your former `process_frame` on the same
  frames:
  - the walker is detected in 72% of radar frames against 50%;
  - the azimuth spread against the LiDAR is 2.4° against 4.9°.
