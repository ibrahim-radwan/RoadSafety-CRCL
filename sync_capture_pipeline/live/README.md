# Live view

Watch the three sensors live on the iPad page (`web_control/`) while you
align them: the radar–LiDAR–camera extrinsic calibration. **Nothing is
saved, except a calibration capture when you press *Save background* or
*Save position*** (2026-11-07, below). Added 2026-10-31.

**Opening the page:** live view is part of the iPad page
(`web_control/`). The easiest way to reach it, anywhere, is the Jetson's
own Wi-Fi hotspot: join `JetsonHotSpot` and open
`http://192.168.50.1:8080`. Setup and details:
`docs/knowledge/02_hardware_and_network.md`, "Wi-Fi for the iPad".

## What the page shows

| Panel | Content | Rate | Size |
|---|---|---|---|
| Camera | ZED 2i left image, 640 × 360 (live: HD720 at 15 fps, no depth) | 5 /s | ~28 kB |
| LiDAR | The replay's top view (`replay/sources.py` `render_lidar`): forward up, vehicle-left on the left, coloured by height above our local ground. Optional overlay of the radar detections | 4 /s | ~16 kB |
| Radar | Range–azimuth fan with our detections (`perception/radar.py`): the **stationary** map (zero Doppler, no clutter removal; shows a corner reflector) or the **moving-target** map | 4 /s | ~28 kB |

That is about 0.3 MB/s (~2.6 Mbit/s) over the Wi-Fi, measured on
`freewalk_ref_02`. The LiDAR and radar views share one extent
(forward 0..R, lateral ±0.75 R), so radar marks drawn over the LiDAR image
are in the same frame. The LiDAR points are turned into the vehicle (radar)
frame with the current mounting (below). What remains between the two is
the residual of that extrinsic.

**The strongest reflector** (the table above the panels), inside the
range window you set on the page:

- **Radar:** the strongest stationary detection, by power. Multipath ghosts
  are left out.
- **LiDAR:** the brightest spot. Returns at or above the sweep's 99.9th
  intensity percentile (at least 100 of 255) are binned in 0.5 m cells, and
  the cell with the highest summed intensity wins. Corner reflectors are by
  far the brightest LiDAR returns.
- **Radar − LiDAR:** the range and azimuth difference. If the two points
  are more than 1.5 m apart, the row says they are probably not the same
  object.

**Take background** (2026-11-06; buttons above the table):

1. Start live view without the reflector and with nobody in the scene,
   rig still.
2. Press *Take background*.
   - The radar worker keeps the median stationary map of 16 frames.
   - The LiDAR worker keeps the voxels (0.15 m, dilated by one) occupied in
     20 sweeps.
3. Place the reflector. The table then shows the **largest change from the
   background**:
   - **Radar:** local maxima of map − background, ≥ 10 dB above it and
     ≥ 6 dB above their range row's median (so a deep null of the background
     is not a change), largest rise first.
   - **LiDAR:** returns in voxels the background did not have, pooled over
     the last 5 sweeps, in 0.3 m cells of at least 3 returns, most returns
     first.
   - Both are marked on the views.
4. *Clear background* returns to the strongest return.

- **Take background is memory only** (*Save background* below keeps one):
  - The server forwards one-shot `take:<n>` / `clear:<n>` tokens and keeps
    none, so a new live view starts without a background.
  - Nothing is written. `tests/test_live_view.py` checks that no file of
    the played session changes, with a background taken and cleared.
  - Recorded sessions and the replay are unaffected.
- **Verified on the team's calibration_dataset** (background from
  `background`):
  - LiDAR: the reflector at 3 and 5 m is picked within 0.1 m of its true
    position. At 8 m it has too few returns.
  - Radar: right on boresight; off-axis limited by the azimuth calibration.

**Calibration capture** (2026-11-07; the team's request: save
synchronised radar and LiDAR frames for the empty background and each
reflector position, reflector points only):

1. Set name (default `calib_<date>`). With the reflector and people out of
   the scene, rig still: **Save background**.
2. Place the reflector, enter a position name (rig's view: left = vehicle
   left), **Save position**. Repeat.

- **Each press saves ~2 s from every running sensor** into
  `<sessions-root>/calibration/<set>/<name>/` (`capture.json`, written by
  the server; ~85 MB per position):

  | Folder | Files |
  |---|---|
  | `radar/` | `frames.npy` (16 raw frames, int16 as the `.bin` files), `t.npy` (CLOCK_MONOTONIC at receipt), `meta.json`. A background also has `background_band.npy` (the mean zero-Doppler channels, complex) and `background_ref.npy` (its own frame-to-frame residual); these are per virtual channel, so a background saved before the measured array layout (2026-11-07: 32 channels instead of 40) is refused with a message: save it again. A position also has `reflector.json` (its `az_deg` refined coherently across both radar chips, `az_deg_incoherent` the map's peak) and `reflector_points.npz`. |
  | `lidar/` | `sweeps.npz` (20 sweeps in the LiDAR's own frame, with counts and t), `meta.json` (the extrinsic used). A background also has `background_voxels.npy`. A position also has `reflector.json` (vehicle frame, plus its own-frame position) and `reflector_points.npz`. |
  | `camera/` | `frame_0..2.jpg` (full-size left, 0.6 s apart), `meta.json` (rectified left intrinsics) |

- **Synchronised:** one press starts all workers together, and every
  frame carries its time. For a still reflector, exact pairing does not
  matter.
- **The reflector** (`perception/core/reflector.py`) is what changed from
  the set's saved background (else from a *Take background* in memory),
  inside the reflector search range:
  - **LiDAR:** new voxels; the cell group present in ≥ 30% of the sweeps
    with the most returns (people move, the reflector does not); its points
    within 0.45 m; height window −0.9..2.5 m (the 2026-11 floor is 1.1 m
    below the LiDAR).
  - **Radar:** each frame's zero-Doppler channels minus the background's
    mean, **coherently** (complex), then beamformed. Walls and their
    sidelobes cancel. The largest rise ≥ 10 dB above the background's own
    residual, interpolated.
- **Reflector only** (tick): the LiDAR view draws only new returns, and the
  radar's stationary map draws the coherent change.
- The page lists the set's positions: LiDAR x / y / z and own-frame
  azimuth, radar range / azimuth, and the difference. When the server sends
  the set (`calib_set`), the workers load its saved background at once.
- **Process a set:** `python calib_reflector.py --set <set folder>`. It
  gives the camera-referenced LiDAR yaw, the per-frame reflector points of
  both sensors, and a figure.
- **Tested** by playing the team's `background` and `5m_center` sessions
  through live view (`tests/test_live_view.py` captures into a temporary
  folder, and checks the played session is unchanged). **Not yet with the
  live sensors.**

**The LiDAR is drawn in the vehicle (radar) frame** with the newest mounting
in `calibration/lidar_extrinsic.json` (with `--from-session`, that session's
mounting). The page names it under the buttons. Update the file when the
rig's mounting changes.

Azimuth is positive to the **left**, as everywhere in the pipeline
(`perception/core/conventions.py`). LiDAR-over-camera needs the team's
camera calibration and is not done yet.

## Live view and recording never run together

- **Both start buttons go through one lock** in `web_control/server.py`.
  - A live view is refused while a recording runs, including one started
    from a terminal: the server looks for `capture_session.py` and the
    recorder processes.
  - A recording is refused while any live-view worker is still running,
    including while one is stopping.
- **Live view switches itself off after 10 minutes.** The server stops it
  at 600 s. Each worker also has its own deadline (615 s), in case the
  server does not.
- **The workers cannot outlive the server.** They are its children, linked
  by pipes. When the server stops or crashes, their stdin closes and they
  stop. Tested: SIGKILL of the server stopped all three workers within 2 s.
  At start-up the server also ends any leftover workers.
- **What the server-side lock does not cover:** `capture_session.py` started
  from a terminal while live view is on. The two radar receivers would then
  share the DCA1000's data port. Don't do that; the page is the way to run
  both.

## How it works

```
web_control/server.py ── liveview.LiveView ──┬─ live/radar_live.py   (conda env, GPU)
   /api/live/start|stop|control|state|log    ├─ live/lidar_live.py   (conda env, GPU)
   /api/live/frame/<sensor>?after=<seq>      └─ live/camera_live.py  (conda env, pyzed)
```

- **One process per sensor,** as for recording. `pyzed.grab()` holds the GIL,
  and the radar's receive thread must not wait on anything.
- **Worker → server** (`live/common.py`): stdout carries length-prefixed
  messages, each a JSON header (numbers, detections, extent) plus a JPEG.
  stdin carries control lines from the server (`{"range_m": 20}`,
  `{"radar_map": "static"}`). All printing, including from C++ (rs_driver,
  the ZED SDK), goes to stderr and appears in the page's "Live view log".
- **The server keeps only the newest message** per sensor, in memory.
- **The page pulls images.** Each panel asks for the next image only after
  the previous one has arrived, so a slow Wi-Fi lowers the rate instead of
  building a backlog. The numbers come from `/api/live/state` four times a
  second, and the page draws them on canvases.

### The recorders are not modified

- **Radar** (`radar_live.py`):
  - It imports `dca_capture_radar_sync.py`'s `send_recv`, constants and
    `Receiver`, and arms the DCA1000 with the same command sequence, waits
    and 25 µs packet delay as its `main()`. **Keep the two sequences in
    step.**
  - The receiver's packet loop is unchanged. Only its disk writer is replaced,
    by a one-frame slot that the newest frame overwrites. The receive thread
    never waits on processing; frames in between are dropped.
  - STOP_RECORD is sent on every exit path.
  - Processing: `RadarDetector(...)` with the measured array layout
    (`calibration/radar_array.json`, 2026-11-07; the log line says so)
    `.process(raw, moving_only=True, static=True)`, about 8–9 ms of GPU per
    frame, 30–60 ms with the rendering.
- **LiDAR** (`lidar_live.py`): our rs_driver wrapper on the live sockets,
  with the recorder's `rs_driver/config/config.yaml` (`split_angle: 180`).
  Then `LidarDetector` for the ground height, and `render_lidar`.
- **Camera** (`camera_live.py`): opens the ZED without `enable_recording`.

`dca_capture_radar_sync.py`, `recorder.py`, `record_lidar_raw.py` and
`capture_session.py` are unchanged. The `tool_sha256` hashes in
`session.json` stay the same.

## Without the sensors: a recorded session

```bash
cd sync_capture_pipeline
python web_control/server.py --port 8099 --sessions-root /tmp/sessions --camera-stage none \
    --live-from-session /media/.../freewalk_ref_02
```

The workers then play the session's radar frames, `lidar.pcap` and
`camera.svo2` in real time, looping. The page says so under the panels.
This is how live view was developed and tested on `ibrahim-desktop`, which
has no sensors. A single worker can be run the same way:
`python live/radar_live.py --from-session <session>`. Its messages are
binary on stdout.

**The LiDAR's live socket path can be tested without the sensor too.**
Send a session's `lidar.pcap` UDP payloads to 127.0.0.1 at their recorded
pace, and run `live/lidar_live.py --config <copy of config.yaml with
lidar_ip and host_ip 127.0.0.1>`. Verified 2026-11-02: about 3.5
images/s, a clean stop, and "waiting for data…" when no packets arrive.

Tests: `tests/test_live_view.py`:
- the lockout both ways (stand-ins, no data needed);
- JSON without NaN, and server failures answered in JSON;
- on a reference session: images and numbers from all three workers, the
  time limit, clean exits, no file in the session changed, and a worker
  stopping when its server goes away.

## On the rig

**Working (team's report, 2026-11-05):** over the Jetson's Wi-Fi hotspot,
the team opened the page, ran live view, and recorded afterwards.

Live view was built on `ibrahim-desktop`, which has no sensors. These were
the checks for the rig. The report covers 1–4 and a recording after live
view; worth confirming in numbers when convenient:

1. Live DCA1000 control in live view, including STOP_RECORD at the end.
2. The live LiDAR sockets.
3. Opening the live ZED at HD720/15.
4. The page on the iPad over the rig's Wi-Fi.
5. A normal recording straight after a live view, which must PASS
   `check_session`.
6. The radar's packet rate in live view. The receiver shares its process
   with the GPU processing, so the "pkt/s" on the radar panel should stay
   near ~17,300 at 8 fps. Lost packets are zero-filled and frame boundaries
   stay correct, so live view tolerates some loss; recording never shares
   that process.
