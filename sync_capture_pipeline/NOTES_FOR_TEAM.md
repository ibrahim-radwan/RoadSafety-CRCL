# Notes for the research team

The current state of the system, and what to know when recording and
processing. Updated 2026-11-17. Earlier versions of this file are in git
history.

## What to do now

1. **Pull the latest `main`** on the rig and on the processing machines.
   On the rig, then restart the web page's service
   (`sudo systemctl restart capture-web`), or the iPad page stays on the old
   version. Then re-run, on every session you have:
   `python make_report.py --session <s>` and
   `python run_perception.py --session <s>`.
   The radar and LiDAR outputs have changed (see "New since your
   questions").
2. **Connect the Altos V4 radar to the rig** as a fourth sensor: follow
   "Connecting the Altos to the rig" in the next section, step by step.
3. **Mark the LiDAR's housing and its collar** so that any turn shows. The
   housing can turn in its collar.
4. **Record a calibration set on the wheeler, as mounted.** Record the empty
   scene first, as a session named `background`. Then record the reflector
   positions in the same folder, without moving the vehicle. If anything is
   remounted later, record a new set. Mount the Altos before recording the
   set, so it is in the set too.
5. **Record a 10–15 minute pilot, with all four sensors, and share it with
   us before collecting at scale.** It should contain:
   - the calibration set;
   - about a minute of someone walking towards and away from the stopped
     wheeler (a timing check; for the Altos it also sets its time offset);
   - a ride at mixed speeds;
   - a few stops.
6. **We check the pilot and send feedback or corrections:** data health,
   sync and calibration on the vehicle, detection on real roads, and the
   Altos's time offset, which we add to the offsets file. Then
   you record more and share it, and we keep feeding back.

Meanwhile, please try the live view's calibration capture on the rig. So
far it is tested only on playback.

## The Altos V4 radar (fourth sensor)

An imaging radar recorded alongside the TI radar, LiDAR and camera. It
sends a **processed point cloud**: its own detections, each with range,
Doppler, azimuth, elevation and SNR. There is no raw ADC data.

**Connection and power**

- **Power:** the **15 V / 4 A** adaptor; the 1.5 A one is too weak. It
  boots in 10–20 s.
- **Network:** its harness's media converter, then a second USB-Ethernet
  adapter on the Jetson. It keeps its factory addresses: radar
  192.168.3.10, Jetson 192.168.3.1. These don't overlap the TI radar's
  (192.168.33.x), the LiDAR's (192.168.34.x) or the Wi-Fi.
- **Its web page** (on a machine on that network):
  http://192.168.3.10:3000.

**Connecting the Altos to the rig (once)**

On the rig's Jetson, in a terminal, from the repository folder:

1. **Update and restart the page.** Run `git pull`, then
   `sudo systemctl restart capture-web`. Reload the iPad page: the sensor
   checkboxes now include **Altos radar**. If the Jetson's Wi-Fi is the
   hotspot, connect it to the internet for the pull first
   (`docs/knowledge/02_hardware_and_network.md`, "Wi-Fi for the iPad").
2. **Plug it in.**
   - The Altos's connector goes to its harness.
   - The harness's RJ45 goes to a second USB-Ethernet adapter, into any
     free USB port on the Jetson.
   - The harness's power jack goes to the **15 V / 4 A** adaptor.
   - Wait 20 s for it to boot.
3. **Find the new adapter's MAC address:** `ip -br link`. It is the
   interface that was not there before, often named `enx…`. The LiDAR's
   adapter is 00:0E:C6:BC:78:54.
4. **Give the Jetson its address on that adapter.** Put the MAC from
   step 3 in place of `<MAC>`:

   ```bash
   sudo nmcli con add type ethernet con-name "altos-usb-eth" ifname "" \
        802-3-ethernet.mac-address <MAC> ipv4.method manual ipv4.addresses 192.168.3.1/24 \
        ipv4.never-default yes ipv6.method ignore
   sudo nmcli con up altos-usb-eth
   ```

   The profile is tied to that adapter, so it comes back by itself after
   every reboot. Nothing about the TI radar, the LiDAR or the Wi-Fi changes.
5. **Check it answers.**
   - `ip -br addr` lists 192.168.3.1 on the new adapter.
   - `curl -s http://192.168.3.10:3000/status` prints `"machine_name":"Altos V4"`.
6. **Check its settings.** `curl -s http://192.168.3.10:3000/install`
   should show `"profile":"150m_1"`, `"fps_limit":null`,
   `"sync_mode":"FreeRunning"` and `threshold_coeff` `0.0`. If not, set
   them:

   ```bash
   python altos_standalone_capture/altos_settings.py --profile 150m_1 --threshold 0 --fps-limit off
   ```

7. **Mount it.**
   - Facing forward, level, rigid, close to the TI radar.
   - Note where it sits relative to the TI radar (offset in cm, any turn):
     it is needed for the calibration.
8. **Check it live.** On the iPad, tick **Altos radar** in *Live view* and
   start it. The Altos panel shows points, about 15 frames/s, and moving
   people in blue or red.
9. **Make a 30 s test recording** with all four sensors ticked.
   - The Altos card should read about **15 frames/s, 0 lost, 0 restarts**.
   - `check_session` should say **PASS**.
   - Then go on with the calibration set and the pilot (steps 4–5 above).

If the session refuses to start with "Altos radar does not answer": check
its power, its connector, and that 192.168.3.1 is on its adapter
(steps 4–5).

**Settings in use:** profile `150m_1`, threshold coefficient `0` (the most
points), no FPS limit (15 frames/s), free running.

- In a room this gives about 170 points per frame, against about 110 at the
  default threshold.
- **Don't change its settings while recording.** Every change restarts it:
  a 1–2 s gap, and its frame numbers start again. The session check
  reports a restart as a failure.

**Recording and the page**

- **Recording:** "Altos radar" is a checkbox, ticked by default. It starts
  and stops with the LiDAR and camera. The session doesn't start if the
  Altos doesn't answer; untick it to record without it.
- **Status card:** frames/s (normally 15), points per frame, lost frames
  and restarts.
- **Live view:** an Altos top view, an Altos row in the reflector table,
  and Altos data in calibration captures (`<set>/<position>/altos/`).

**After recording**

- `check_session.py` checks it.
- `align_session.py` puts it on the common clock and adds it to
  `pairs.csv`.
- The HTML report has an Altos section.
- The replay has an Altos panel (`--sensors` includes `altos`).

**Not calibrated yet**

- **Mounting:** its points are in its **own** frame (x forward, y left, z
  up). The Altos's place in the vehicle frame is part of your extrinsic
  calibration.
- **Time offset:** its processing delay (under 60 ms per its spec) is still
  in its times.
  - We measure it from your pilot's walking minute (the report's Altos
    timing check) and add it to `calibration/stage3_offsets.json` as
    `altos_ms`.
  - After you pull that, re-run `make_report.py` on the sessions recorded
    so far: offsets are applied when a session is aligned, never to the raw
    data.
- **Signs:** the Doppler and elevation signs follow the vendor's documents;
  they're not yet checked against a target.

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
- **"radar"** in sensor lists, flags and file names is the TI radar; the
  Altos is always **"altos"**.

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

**Altos V4**

- Recorded raw to `altos/altos.pcap`. `altos/metadata.json` holds the
  radar's settings at the start of the session.
- `sync/altos_points.npz` (written by `align_session.py`) holds every point:
  - `points`, one row per point: range (m), Doppler (m/s, > 0 = moving
    away), azimuth (rad, + left), elevation (rad, raw sign: + down), SNR
    (linear), then x, y, z (m; x forward, y left, z up; the Altos's own
    frame);
  - `valid`: false for the few points the radar sends with missing (NaN)
    values (well under 1%);
  - `frame`: the point's row in `sync/altos_timeline.csv`;
  - `t_ref`: each frame's time.
- Radial speeds wrap at ±45 m/s.
- 15 frames/s. Indoors, 70–280 points per frame; the maximum is 12,000.

**Camera (ZED 2i)**

- Recorded as lossless H.264 SVO2; images are extracted offline.
- Dropped frames: typically 0.2–0.3%, single frames, listed in
  `capture/timing/camera_gaps.csv` and flagged in `sync/pairs.csv`.

**Synchronisation**

- Timing offsets (applied automatically): radar +12.65 ms, camera
  +23.34 ms, relative to the LiDAR.
- `sync/pairs.csv`: one row per radar frame, with the nearest camera
  frame, LiDAR sweep and Altos frame (`altos_frame`, `t_altos`,
  `altos_error_ms`) and their time errors. Rows in a gap are flagged
  (`camera_gap`, `lidar_gap`, `altos_gap`).
- **Altos times:** each frame's arrival on the Jetson, smoothed. Its
  processing delay is not yet removed (see the Altos section). Paired with
  the radar frames, the Altos is within ±33 ms.
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
