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
   - Sidelobes still stay only ~5–10 dB below the peak. Ours is a stopgap.
   - **Resolved 2026-11-07:** the error depended on the target's angle,
     which no calibration can fix. It was our processing's antenna table;
     the TX3/TX4 "offset" was part of it. The measured layout replaces this
     correction (see "Your follow-up questions", item 1).
3. **Elevation is unreliable.**
   - `process_frame` placed a walking person 1.8–2.1 m above the radar, and
     our detector reads ~+13° on the walker.
   - Elevation comes from the single 8-channel row on TX5, which is on the
     **slave device**. **2026-11-07:** chirp 5 is not the elevation
     transmitter; chirps 3 and 4 are. Elevation is off (z = 0) until their
     positions are measured (follow-up item 1).
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
    calibration. The reference sessions are processed as co-located; from
    2026-11-03 the LiDAR's mounting is applied (see "Your corner-reflector
    sessions", item 6).

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
    cadence is experimental. `static_det` / `static_start` hold stationary
    reflectors (zero Doppler; experimental). They are reliable within about
    10 m; farther out many are sidelobe or multipath images.
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
  - Since 2026-11-07 (measured array layout, CFAR threshold 16 dB): 83%
    and 2.0°, with ~40% fewer radar tracks that no LiDAR object supports.
    Re-run `run_perception.py` on sessions processed before.

## Live view for the extrinsic alignment (2026-10-31)

The iPad page has a **Live view** card, built for your radar–LiDAR–camera
alignment.

- **What it shows:**
  - the camera's left image;
  - the LiDAR top view, with the radar detections drawn on it if you want;
  - the radar range–azimuth map: the stationary map shows a corner
    reflector, and a moving-target map can be chosen instead;
  - per sensor, the strongest reflector (range, azimuth + left) inside a
    range window you set, and the radar − LiDAR difference.
- **The LiDAR is drawn in the vehicle (radar) frame** with the current
  mounting (`calibration/lidar_extrinsic.json`, since 2026-11-06). The
  difference you see is what that transform leaves, plus the radar's
  azimuth error (see "Radar" above). *Take background* picks the reflector
  as what changed (see item 7 of "Your corner-reflector sessions").
- **Nothing is saved, except calibration captures on request** (*Save
  background* / *Save position*, 2026-11-07; see "Your follow-up
  questions", item 4). **It never runs during a recording.**
- **It switches itself off after 10 minutes.**
- LiDAR points over the camera image will follow once your camera
  calibration is available.
- Details: `sync_capture_pipeline/live/README.md`.
- **Status:** tested on a recorded session only. The first use on the rig
  is also its first test with the sensors live.

## Your corner-reflector sessions of 2026-11-03: what the data shows

> **What to do next (updated 2026-11-07; see also "Your follow-up questions
> of 2026-11-06" below)**
>
> 1. **Pull the repository** (2026-11-07 or later) on the rig and on
>    the processing machines.
> 2. **Re-align every session recorded since 2026-11-03 again**: the LiDAR
>    yaw is now 37.2° (camera-referenced), not 39.76° (item 6 and follow-up
>    item 3). Then re-run the report and detection:
>    `python align_session.py --session <s>`, then
>    `python make_report.py --session <s>` and
>    `python run_perception.py --session <s>`.
> 3. **Name calibration positions from the rig's point of view:** "left" is
>    the vehicle's left, seen from the rig looking forward (item 1).
> 4. **For calibration, use the live view's *Save background* and *Save
>    position*** (follow-up item 4), with the reflector and people out of the
>    scene for the background.
> 5. **Review our LiDAR transform** (`calibration/lidar_extrinsic.json`:
>    +37.2°, from the camera; yours gave 38.6° from the radar). Replace it
>    with your extrinsic calibration when you accept it. Add a new entry
>    whenever the LiDAR mounting changes. **Mark the LiDAR's housing and
>    its collar,** so a turn shows (follow-up item 3).
> 6. **Radar azimuth off boresight: fixed (2026-11-07).** The 5–16°
>    errors came from our processing's antenna table, not the calibration.
>    After pulling, re-run `run_perception.py` (and the reports) on your
>    sessions. Radar elevation is off until measured. Nothing to record
>    (follow-up item 1).
> 7. **On the rig, please report:**
>    - the radar packet rate (pkt/s on the radar panel) during live view;
>    - that a recording made straight after a live view passes
>      `check_session`;
>    - that *Take background* works with the live sensors.

Analysed 2026-11-05: `calibration_dataset/` (an empty `background`, 14
reflector positions at 2, 3, 5 and 8 m, `validation_test`) and
`run_20261103_1600-Lidar-AR1`. Every session passes `check_session`. Each
has `sync/`, so it opens in the replay. The reflector was found in each
sensor by comparing with `background`, near its nominal position.

**1. Left and right: both sensors use + left; two things made it look
otherwise.**

- **The scene names are mirrored.** In every `*_left` scene the camera
  shows the tripod on the vehicle's right, and in every `*_right` scene on
  its left. The names were given from the reflector's side, facing the rig.
  Your `manual_calibration_result.yaml` agrees: its radar point for
  `5m_150_left` has y = −1.36 m.
- **The LiDAR is turned about 40° to the left** of the radar and camera on
  this mounting.
  - A reflector straight ahead appears at LiDAR azimuth −39°, at every
    distance from 2 to 8 m.
  - In the LiDAR's own frame, the one the replay and live view draw, a
    reflector 5 m ahead and 1.5 m to the left (+17° from the rig) is at
    17° − 40° ≈ −23°. That is why L813 had a negative azimuth.
  - The replay's table, 2D view and 3D view agree with each other: all three
    are in the LiDAR frame, + left.
  - The radar processing's own angle is positive to the right (your
    convention). The pipeline converts it once, so everything shown is + left.
    It is not mirrored.

**2. The LiDAR is the accurate sensor here.**

- A 2D rigid fit of the 13 LiDAR reflector positions to the nominal
  positions gives:
  - yaw +39.8°;
  - offset (−0.03, −0.01) m;
  - residual 0.11 m mean, 0.33 m max (8m_150_left).
- Your manual fit gave 38.6°.
- **Revised 2026-11-07:** against the camera the yaw is **+37.2° ± 0.4°**.
  The rig stood ~2.5° turned against the floor marks, and the fit to the
  nominal positions absorbed that (follow-up item 3).
- The reflector height is z ≈ −0.3 m in the LiDAR frame.
- The reflector is **not** the brightest LiDAR return. Its intensity is
  10–120, while people, hi-vis vests and anything within ~2 m saturate at
  255.

**3. The radar: range is accurate, azimuth only near boresight.**

- **Range:** +0.01 to +0.14 m from nominal on every scene.
- **Azimuth on boresight (centre scenes):** within ±3°. The reflector stands
  10–25 dB above the rest of its range row and is detected in every frame.
  Its absolute level is ~150 dB at 3–5 m.
- **Azimuth off-axis (0.5–1.5 m to the side):**
  - The reflector shows as a broad or split lobe only ~4–12 dB above the
    clutter.
  - The picked peak is 5–13° from the true angle.
  - This was our processing's antenna table (follow-up item 1). **Fixed
    2026-11-07:** all 9 positions at 2–5 m now read within 1° of the
    camera, after the mounting offset.
- **The radar has no elevation for stationary reflectors** (z = 0). Only the
  LiDAR gives the reflector's height.
- **validation_test:** the LiDAR (with the fit above) and the radar agree
  to 0.39 m, mostly in range.

**4. `run_20261103_1600-Lidar-AR1` is not usable for calibration.**

- The LiDAR sees the reflector at the stated position: LiDAR (4.35, −1.78),
  which is about (4.5, +1.4) m in the rig frame.
- The radar does not. There it is only ~7 dB above its range row, and a
  stationary detection is within 0.6 m in 29 of 570 frames. Most likely the
  reflector did not face the radar.
- The office around it is full of strong reflectors and people.

**5. Why the reflector was hard to pick out.**

- **Replay:** *moving only* (the default) removes the stationary scene, and
  the reflector with it. Use `--radar-all --radar-static --range-filter <a>
  <b>`. The objects table then lists the strongest stationary reflectors
  (`S1..S3`) with range, azimuth, x / y and dB.
- **Live view:** its "strongest reflector" is the strongest return in the
  range window, and that is often a wall or desk, not the reflector. In
  `5m_center` it picked an object at (2.8, +2.1) m. Narrow the window to the
  reflector's distance.
- **dB:** the live view's ~120–150 dB is the absolute power (uncalibrated,
  0 dB = 10⁻¹² FFT units). The replay's *raw >* threshold (now 3–80 dB) is
  in dB above the frame's median, a different quantity. The replay's table
  shows the absolute value too.

**6. The LiDAR transform is now applied (2026-11-06).**

- `calibration/lidar_extrinsic.json` holds the fit above: t = (−0.03,
  −0.01) m, valid from 2026-11-03. The yaw was +39.76° until 2026-11-07, and
  is now **+37.2°**, referenced to the camera (follow-up item 3).
- Every tool turns the LiDAR points into the vehicle (radar) frame right
  after decoding: alignment (the sweep's boresight time), the report, the
  replay, the live view, `run_perception.py` and Stage 3. The LiDAR views now
  show forward as forward, and radar marks line up with the LiDAR.
- Each session records the transform it used in `sync/alignment.json`
  (`lidar_extrinsic`). Sessions before 2026-11-03 use identity and are
  unchanged.
- `align_session.py --lidar-extrinsic none` turns it off for a session.
- It is provisional: replace it with your extrinsic calibration when you
  accept it. Add a new entry with a later `valid_from` whenever the mounting
  changes.
- z is not shifted.
- Re-run `align_session.py` (then the report and `run_perception.py`) on
  sessions aligned before 2026-11-06.

**7. "Take background" in the live view (2026-11-06).**

1. Start live view with the reflector and people out of the scene, and the
   rig still.
2. Press *Take background*. The radar collects 16 frames and the LiDAR 20
   sweeps, a few seconds.
3. Place the reflector. The reflector table then shows the **largest
   change from the background** inside the search range, instead of the
   strongest return.

- **Take background saves nothing:** the background lives in the
  live-view workers' memory and is gone when live view stops. Recordings
  and their replay are not affected (live view never runs during a
  recording). To keep it, use *Save background* (follow-up item 4).
- **Tested on these sessions:**
  - LiDAR: the reflector at 3 and 5 m is picked within 0.1 m.
  - LiDAR at 8 m: the reflector gives too few returns per sweep.
  - Radar: right on boresight; off-axis it is limited by the azimuth
    calibration.
- **Anything new counts as a change.** People who walk in show up too:
  keep them out of the search range.
- Re-take the background after moving the rig.

## Your follow-up questions of 2026-11-06 (answered 2026-11-07)

Checked on `calibration_dataset/` and `run_20261103_1600-Lidar-AR1`, with
the **camera as an independent angle reference**:

- the reflector is found in the left image as the top of the tripod, by
  difference from the `background` image;
- its azimuth is atan((cx − u) / fx), from the rectified intrinsics. No
  depth, no radar and no nominal position are needed.

All numbers below come from `calib_reflector.py` (new, item 4) on these
sessions.

**1. Radar antenna calibration: not needed. The off-axis error was our
processing's antenna table, now fixed (2026-11-07).**

Our earlier notes said your antenna calibration would fix the radar's
off-axis error. It was our processing: its table of where each antenna
sits did not match the board.

- **What the 2026-11-03 sessions showed** (against the camera, 9 positions
  at 2–5 m): near boresight within ~1°, but at +9° the peak 15° off, on
  another lobe.
- **Why it was not the calibration** (October plate sessions, LiDAR as the
  reference, raw channel data):
  - the `.cfg` calibration values barely change anything (as you saw with
    your intrinsic calibration);
  - the error depended on the target's angle, which no calibration can
    cause or fix.
- **The layout, measured from the data**
  (`perception/estimate_array_layout.py`): each channel's phase against the
  target's angle gives its position. Both radar chips agree, on two kinds
  of session.
  - Three of the six transmitters sat at the wrong places in the table.
  - Two transmitters (chirps 3 and 4) are raised, for elevation. The table
    had a different one (chirp 5) as the raised one.
  - The receivers were right.
  - The two chips have a fixed phase offset that grows with range (the
    second chip samples ~0.4 µs late). It is now removed.
- **New processing** (the default everywhere: replay, live view, reports,
  `run_perception.py`, `calib_reflector.py`):
  - the measured layout, in `calibration/radar_array.json`;
  - no `.cfg` calibration values and no in-situ correction (both made it
    worse);
  - each chip's antennas are beamformed and the two added: the map's
    sidelobes are ~20 dB down (before: 5–10 dB, and split lobes);
  - each detection's angle is then refined across both chips.
- **Results** (nothing fitted on these sessions):
  - your corner reflector, 2–5 m: radar − camera = −2.4° ± 0.4° at all 9
    positions, none more than 1° off. Before: outliers of +9.8°, −16.3°
    and +3.5°. The −2.4° is the radar's mounting angle against the camera;
  - a walker against the LiDAR (freewalk_ref_02): spread 1.8° instead of
    2.7°, and 12% of frames more than 5° off instead of 36%;
  - at 8 m the camera detection fails, but the radar matches the nominal
    positions within ~2° after the same offset.
- **What it means for you:**
  - **No radar recording or calibration is needed for azimuth.**
  - **Re-run `run_perception.py` and the reports** on your sessions after
    pulling, to get the new radar output.
  - **Radar elevation is off (z = 0)** until the two raised transmitters'
    positions are measured. The old elevation used the wrong transmitter.
  - The old processing is still there for comparison: the replay's
    `--radar-layout team`.
  - The data and results of your earlier intrinsic calibration would still
    be a useful cross-check, as would the board's antenna drawing, if you
    have it.
- **Also seen:** the receivers of the two chips differ by ~4 dB, stable
  across scenes. It does not affect the angle; not corrected.

**2. Your calibration results in the folder: not used.**

- Nothing in the repository reads `manual_calibration_result.yaml`,
  `calibration_output/` or `calibration_results.json` (checked by search).
- Our LiDAR yaw is our own measurement: first the nominal positions, now
  the camera (item 3).
- We only quoted from your file:
  - its 38.6° yaw, for comparison;
  - the sign of its radar point for `5m_150_left` (y = −1.36 m), as a
    check of the scene naming.
- Your file's `initial_transforms` already turns the LiDAR by 40°
  (quaternion z 0.342, w 0.940). Its LiDAR points show the same thing:
  2m_center at (1.54, −1.24) m is −38.7° in the LiDAR's own frame.

**3. The LiDAR's ~40°: how it was measured, and double-checked.**

- **Method:**
  1. The reflector in the LiDAR's own frame: voxels new compared with
     `background`.
  2. First, a 2D fit to the 13 nominal positions: 39.76°.
  3. Double-checked against the camera: camera azimuth minus LiDAR
     azimuth over the 9 positions at 2–5 m gives **37.2° ± 0.4°**. At
     8 m the tripod is too small in the image.
  - Example, 5m_center: the camera sees the reflector at −1.6°, the
    LiDAR at −39.1°.
- **Why the two estimates differ:** the camera also sees each "centre"
  reflector 1.6–1.9° to the right. The rig stood ~2.5° turned against the
  floor marks, and the fit to the nominal positions absorbed that.
  - The radar on boresight gives ~35°: it reads 2.7° right of the camera
    (item 1).
  - **We now use 37.2°** (`calibration/lidar_extrinsic.json`). All 17
    sessions on your drive were re-aligned with it; AR1's report and
    detection were rebuilt.
- **Centred: confirmed.** The fitted offset is (−0.03, −0.01) m.
- **"Facing forward" cannot be seen on this sensor.**
  - The RS-Fairy's 0° is set by its internal encoder, not by the round
    housing.
  - It is not our software: the azimuth comes straight from the LiDAR's
    packets (RoboSense's driver, with the per-channel angle corrections
    from the sensor), and we add no rotation of our own.
  - In your photos the cable appears to leave the housing at the back, on
    the vehicle's right. Please compare with the coordinate figure in the
    RS-Fairy user manual, which shows where 0° is relative to the cable or
    connector.
  - The housing sits in a round collar and can turn. It was found turned
    57° once (October).
  - **Please mark the housing and the collar.** If you rotate it to point
    0° forward, record a new background and positions; we then add a new
    mounting entry.

**4. Calibration capture in the live view (new).**

On the iPad page, *Live view* → **Calibration capture**:

1. Enter a set name (default `calib_<date>`).
2. With the reflector and people out of the scene, press **Save
   background**.
3. Place the reflector, type a position name, and press **Save position**.
   Name it from the rig's point of view (left = vehicle left). Repeat for
   each position.

What it does:

- **Each press saves ~2 s from every sensor** into
  `calibration/<set>/<name>/` on the SD card (~85 MB per position):
  - 16 raw radar frames (as the recorder writes them);
  - 20 LiDAR sweeps, in the LiDAR's own frame, so a new extrinsic can be
    applied later;
  - 3 full-size camera images, with the intrinsics.
- **Synchronised:** all sensors start on the same press, and every frame
  has its CLOCK_MONOTONIC time. For a still reflector the exact pairing
  does not matter.
- **Reflector only:**
  - each position's reflector is found as what changed from the set's
    background, inside the reflector search range, and saved
    (`reflector.json`, `reflector_points.npz`);
  - the table shows it per position: LiDAR x / y / z and its own-frame
    azimuth, radar range / azimuth, and the difference;
  - the **reflector only** tick draws only what changed from the
    background, in the LiDAR view and in the radar map. The radar
    subtracts the background coherently, channel by channel, so walls and
    their sidelobes cancel.
  - Anything new counts as a change, so keep people out of the search
    range.
- **Afterwards:** `python calib_reflector.py --set <set folder>` gives, per
  position:
  - the reflector points of every radar frame and its paired LiDAR sweep;
  - the camera azimuth;
  - a summary with the LiDAR yaw from the camera (and from the radar near
    its boresight);
  - a figure with only the reflector points.
- The same tool reads recorded sessions:
  `--background <session> --positions <s1> <s2> ...`.
- **Tested by playing your recordings through the live view:**
  - `5m_center`: LiDAR (4.94, +0.04) m, z −0.37 m; radar 5.08 m at
    −3.5°; camera −1.6°.
  - Two positions give a yaw of 37.5°.
- **Not yet tested with the live sensors.** Please try it on the rig.

**5. The wall that comes and goes in `5m_150_right`: the LiDAR skips a
sector once a second.**

- **What happens:** every second (every 10th rotation), for 4.6 ms, the
  RS-Fairy stops measuring between 0° and 16.5° of its own azimuth.
  - That sweep has 132,288 points instead of 138,240.
  - No data is lost on the way: the packet counter runs on without a gap.
  - It is in every session we checked, including 2026-10-11. It is the
    sensor's behaviour.
- **Where it shows:**
  - With the LiDAR turned 37°, the gap lies at +14° to +40° in the vehicle
    frame: the corridor's left wall 3–8 m ahead.
  - In the LiDAR's own frame, as drawn before re-aligning, it is just
    right of straight ahead.
  - There, that wall drops from ~6,700 returns to ~200, one sweep in ten.
- **Replay:** the replay now fills that sector for display from the
  previous sweep, and the LiDAR panel says so. Detection and all saved
  outputs use the sweep as recorded.
- **For annotation:** every 10th sweep lacks that sector.
- **Worth asking RoboSense** whether it can be turned off.
- The glass office fronts on the vehicle's right give the LiDAR returns
  from behind the glass too.

**6. How the stationary reflector's position was calculated, and why it
blinked.**

- **Until now** each `S1..S3` row was **one cell of one radar frame**:
  - the strongest stationary detections inside the range filter, on the
    zero-Doppler map without clutter removal;
  - each detection: 2D CFAR at 15 dB, a 3×3 local maximum, within 10 dB
    of the strongest cell at its range, not a multipath image;
  - the position is the cell's centre (0.29 m in range, on the azimuth
    grid), with no average over points or frames.
- **Why it came and went:** a reflector close to those thresholds passes
  in some frames and not in others.
  - AR1: in 14% of frames, in 37 separate runs.
  - `validation_test`: 19%.
  - On 2–5 m centre scenes it is the same cell in 100% of frames.
- **Now:** the rows come from the stationary map **averaged over the last
  2 s** (16 frames, mean power; restarted at a seek), and the position is
  interpolated between cells. The class column shows how many frames went
  in. The grey squares stay per frame.
- **Result:** on AR1 the blinking stops (1 run instead of 37). But there
  the reflector also rarely wins, because the radar barely sees it in that
  session (see item 4 of "Your corner-reflector sessions"): it needs to
  face the radar.
- **LiDAR:** in the live view and the calibration tool, the reflector is
  the median of its points, pooled over all sweeps of the capture.

- **Background in the replay (2026-11-07):** `--background` subtracts an
  empty-scene recording of the same rig position from a calibration
  session, in the radar (coherently, per channel) and the LiDAR (only new
  returns are drawn). The reflector then stands alone and is `S1`, its
  azimuth refined across the radar chips: on `5m_150_left` −20.9°, the
  same as `calib_reflector.py`. Key **b** switches it:

  ```bash
  python replay/replay_session.py --session calibration_dataset/5m_150_left \
      --background calibration_dataset/background --radar-all --radar-static \
      --range-filter 3 7 --lidar-range 10
  ```

  Shorter, with the reflector read out in both sensors (`--calibration`
  finds the sibling `background` session): the objects table's `REF` rows
  give the LiDAR's and the radar's reflector and radar − LiDAR, and both
  are marked in both bird's-eye views. On `5m_150_left`: +0.14 m, −1.9°;
  on `8m_1m_left`: +0.25 m, −1.9°.

  ```bash
  python replay/replay_session.py --session calibration_dataset/5m_150_left \
      --calibration --range-filter 3 7 --lidar-range 10
  ```

**7. The mounting photos.**

- **Alignment:** camera, LiDAR and radar are stacked on the stem's centre
  line, which matches the fitted offset of a few cm.
- **The LiDAR's direction:** see item 3. The round collar lets it turn,
  and nothing shows where 0° points. Mark both.
- **What the LiDAR sees:**
  - Its returns span −15° to +88° in elevation: it mostly looks up.
  - The nearest floor ahead is at 3.7 m (the floor is 1.07 m below it).
  - 32% of its returns come from within 0.3 m: the camera mount above it.
    That is harmless (the replay hides returns nearer than 0.5 m), but
    those directions see nothing else.
- **The orange cable** runs at the side, clear of every sensor's view
  (checked on the rig).
- **The radar board** is uncovered and faces forward: fine.

**8. FusionApp's 2D and 3D views.**

- **Reviewed:** `RoadSafetyOnTwoWheelers-main/FusionApp`. Its PC-2D and
  PC-3D modes draw CFAR detections as points:
  1. static clutter removal (the mean over chirps);
  2. Capon (MVDR) range–azimuth beamforming;
  3. a two-pass CA-CFAR on the log heatmap;
  4. one dot per detection, coloured by intensity.
- **Our replay** draws the same kind of points: CFAR detections, and
  stationary reflectors as grey squares. But it draws them over the
  range–azimuth power map, which is the "complete wave" you saw.
- **New:**
  - the **map** tick in the replay's Radar group: untick it for points
    only (`--radar-no-map`);
  - with *static* ticked, the stationary reflectors also appear in the 3D
    view.
- **Two differences remain:**
  - Capon gives sharper peaks than our FFT beamformer. Since the measured
    layout (item 1) our map's sidelobes are ~20 dB down. A Capon option for the
    stationary map is possible as a next step.
  - FusionApp's clutter removal takes out everything still while the
    scooter stands still. Its "static" points on a moving scooter are
    Doppler-shifted, so they show up as moving.
