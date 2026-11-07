# Notes for the research team

The current state of the system, and what to know when recording and
processing. Updated 2026-11-07. Earlier versions of this file are in git
history.

## What to do now

1. **Pull the latest `main`** on the rig and on the processing machines.
   Then re-run, on every session you have:
   `python make_report.py --session <s>` and
   `python run_perception.py --session <s>`.
   The radar and LiDAR outputs have changed (see "New since your
   questions").
2. **Mark the LiDAR's housing and its collar** so that any turn shows. The
   housing can turn in its collar.
3. **Record a calibration set on the wheeler, as mounted.** Record the empty
   scene first, as a session named `background`. Then record the reflector
   positions in the same folder, without moving the vehicle. If anything is
   remounted later, record a new set.
4. **Record a 10–15 minute pilot and share it with us before collecting at
   scale.** It should contain:
   - the calibration set;
   - about a minute of someone walking in front of the stopped wheeler (a
     timing check);
   - a ride at mixed speeds;
   - a few stops.
5. **We check the pilot and send feedback or corrections:** data health,
   sync and calibration on the vehicle, and detection on real roads. Then
   you record more and share it, and we keep feeding back.

Meanwhile, please try the live view's calibration capture on the rig. So
far it is tested only on playback.

## Your questions of 2026-11-06

**1. Antenna calibration: not needed.**

- The off-axis angle error came from our processing: its table of where
  each antenna sits did not match the board.
- The real layout was measured from the data, and both radar chips agree.
  It is in `calibration/radar_array.json` and used everywhere: replay,
  live view, reports, `run_perception.py`, `calib_reflector.py`.
- The antenna calibration values in the `.cfg` are not used by the new
  processing. They were made for the old table.
- **Results:**
  - your corner reflector at 2–5 m: radar − camera = −2.4° ± 0.4° at all 9
    positions, none more than 1° off. Before, three positions were off by up
    to 16°. The −2.4° is the radar's mounting angle;
  - a walker against the LiDAR: 12% of frames more than 5° off, against 36%
    before.

**2. Your calibration files.**

The replay and the processing do not read them. We only quoted your 38.6°
yaw for comparison.

**3. The LiDAR's ~40°.**

- Measured against the camera (the reflector's angle in the left image
  against its angle in the LiDAR): **37.2° ± 0.4°**, centred within 3 cm.
- It is in `calibration/lidar_extrinsic.json`, and every session on your
  drive was re-aligned with it.
- Where the LiDAR's 0° points cannot be seen from outside. Please check the
  figure in the RS-Fairy manual.
- To use your own extrinsic calibration instead, add it as a new entry in
  that file (with its `valid_from` date). Add a new entry whenever the
  mounting changes.

**4. Calibration capture in the live view.**

On the iPad page, *Live view* → **Calibration capture**:

1. Enter a set name.
2. With the reflector and people out of the scene, press **Save
   background**.
3. Place the reflector, type the position name, and press **Save
   position**. Wait until each sensor says *saved* (about 3 s). Repeat for
   each position.

- Each press saves about 2 s from every sensor (radar frames, LiDAR sweeps,
  camera images) into `calibration/<set>/<name>/` on the SD card.
- The table shows each position's reflector in the LiDAR and the radar.
  Tick **reflector only** to see only what changed from the background.
- Name positions from the rig's point of view: "left" is the vehicle's left.
- Afterwards, on a processing machine:
  `python calib_reflector.py --set <set folder>` gives each position's
  reflector in every sensor, the LiDAR yaw from the camera, and a figure.
  The same tool reads recorded sessions:
  `--background <session> --positions <s1> <s2> ...`.

**5. The wall that comes and goes.**

Once a second (every 10th sweep), the RS-Fairy stops measuring a 16.5°
sector of its scan for 4.6 ms. No data is lost on the way. It is the
sensor's behaviour, worth asking RoboSense about. The replay fills that
sector from the previous sweep for display only; the saved data is as
recorded.

**6. The blinking stationary reflector.**

The replay's stationary-reflector rows (`S1..S3`) now come from the
stationary map averaged over the last 2 s (16 frames), not from single
frames.

**7. FusionApp-style points.**

Untick **map** in the replay's Radar group, or start with `--radar-no-map`,
to see the radar points without the power map behind them.

## New since your questions

**Replay for calibration sessions** (`replay/README.md`):

```bash
python replay/replay_session.py --session calibration_dataset/5m_150_left --calibration --range-filter 3 7 --lidar-range 10
```

- `--calibration` uses the sibling session named `background` and
  subtracts it, so only what changed is shown.
- It finds the reflector in both sensors and shows the radar − LiDAR
  difference in the objects table (`REF` rows), with a marker in both
  bird's-eye views.
- On your data: radar − LiDAR −1.9° at both 5 and 8 m, the radar's
  mounting angle.
- Narrow the range filter to the reflector's distance. A person who walks
  in counts as a change too.

**Radar detection:** the threshold is now 16 dB instead of 14 dB: 25–40%
fewer radar tracks with no LiDAR object behind them, with walker detection
unchanged.

**LiDAR detection:** a roof or ceiling over a person was counted as part of
them. Fixed: the walker under the carport is now classed "pedestrian" in
87–99% of sweeps, up from 80%.

**The rest of the radar processing was checked** against the LiDAR and the
camera: range, speed, timing and angle conventions.

## Facts to know when processing the data

**Frames and conventions**

- **Time:** `t_ref` in the `sync/*_timeline.csv` files (CLOCK_MONOTONIC
  seconds, with the timing offsets applied).
- **Space:** the vehicle frame: x forward, y left, z up. Azimuth is positive
  to the left everywhere in the pipeline.
- **LiDAR points** are turned into the vehicle frame right after decoding,
  with the mounting in `calibration/lidar_extrinsic.json`. Each session
  records the transform it used in `sync/alignment.json`.

**Radar**

- Radial velocity v_r > 0 means receding.
- Radial speeds above ±4.7 m/s (about 17 km/h) wrap around: a faster target
  gets a wrong v_r. This is a property of the current waveform.
- Radar points are drawn at z = 0: radar elevation is not used yet.
- Multipath images (at twice a target's range, or behind it) are flagged in
  the detections' `ghost` column.

**LiDAR (RoboSense RS-Fairy)**

- Each sweep is a fixed grid of 138,240 points. Directions with no return
  come back with **zero range**: filter `range > 0` before any statistics or
  learning.
- The wrapper's 6th column is each point's time offset within the sweep.
  Sweeps are split behind the vehicle, so the forward view is never cut.
- Every 10th sweep lacks a 16.5° sector (question 5).

**Camera (ZED 2i)**

- Recorded as lossless H.264 SVO2; images are extracted offline.
- Dropped frames: typically 0.2–0.3%, single frames, listed in
  `capture/timing/camera_gaps.csv` and flagged in `sync/pairs.csv`.

**Synchronisation**

- Timing offsets (applied automatically): radar +12.65 ms, camera
  +23.34 ms, relative to the LiDAR.
- `sync/pairs.csv`: one row per radar frame, with the nearest camera frame
  and LiDAR sweep and their time errors; rows in a gap are flagged.
- `make_report.py` aligns a session, writes `pairs.csv` and the HTML report.

**Detections (`run_perception.py --session <s>`)**, in `sync/`:

- `perception_radar.npz`: per frame, detections (multipath flagged),
  objects and tracks; stationary reflectors in `static_det`.
- `perception_lidar.npz`: per sweep, objects with boxes and size classes,
  and tracks. The moving/static flag on LiDAR tracks assumes a stationary
  rig.
- `perception.json`: scores against the LiDAR walker track.
- Column layouts are in the script's docstring; method and scores in
  `perception/README.md`.

**Mounting**

All sensors must stay fixed between a calibration set and the recordings it
belongs to. The LiDAR was once found turned 57° after being reconnected.
