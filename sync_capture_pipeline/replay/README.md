# Session replay

Replays a three-sensor session in real time, with every sensor on one clock.
It can show all three sensors or any one of them.

| Panel | What it shows | Source |
|---|---|---|
| Camera | ZED left image, 960 px wide | SVO2, decoded on the GPU by the ZED SDK |
| LiDAR | One of three views (below), with the radar FOV (±85°) dashed | `lidar.pcap`, decoded by rs_driver |
| Radar BEV | Range-azimuth power, on the **same metric grid** as the LiDAR BEV | radar `.bin`, processed on the GPU (PyTorch) |
| Radar range-Doppler | Range vs radial velocity | same |

Each panel title shows the session time of the data on screen. For the radar
frame on screen, the status line shows the paired camera frame and LiDAR sweep
from `sync/pairs.csv`, with their time errors and any flags (`camera_gap`,
`lidar_gap`, …).

## LiDAR views

Choose a view from the **LiDAR view** row under the controls, or with
`--lidar-view`, `--lidar-range`, `--lidar-color` and `--lidar-near`. Changes
apply at once, even while paused.

| View | Best for | Notes |
|---|---|---|
| **Top - forward** (default) | Road sessions; direct comparison with the radar BEV | 0 to R m ahead, R/2 either side |
| **Top - 360°** | Seeing the whole surroundings, including behind | ±R m around the LiDAR |
| **Panorama** | Recognising the scene; reads like a photo | Azimuth × elevation, coloured by distance (red = near) |

- **Range:** 5, 10, 20, 40 or 80 m. Use 5–10 m indoors and 40 m on the road.
  Mouse wheel and drag zoom further.
- **Colour:**
  - **height** (top views): the highest return in each cell;
  - **intensity**: reflectivity;
  - **density**: point count.
- **Hide nearer than** (default **0.5 m**) hides returns from the LiDAR's own
  mount and the vehicle body. On the desk bench this was **69% of all
  points**, which is why the earlier view looked almost empty. Set it to 0 to
  see everything; nothing is discarded, only hidden.

## Run

Run these from `sync_capture_pipeline/` on the Jetson desktop. The replay
needs `sync/` from `align_session.py`, and uses its offsets.

```bash
PY=/home/jetson/miniconda3/envs/rel_1_py_3_8/bin/python
$PY align_session.py --session sessions/run05            # once per session
$PY replay/replay_session.py --session sessions/run05
$PY replay/replay_session.py --session sessions/run05 --sensors lidar          # one sensor
$PY replay/replay_session.py --session sessions/run05 --sensors camera,radar --start 30 --speed 0.5
```

| Control | Action |
|---|---|
| **Space** / Play button | Play or pause |
| **← / →** | Previous or next radar frame (all sensors follow) |
| **[ / ]** | Slower or faster (0.25×, 0.5×, 1×, 2×) |
| Slider | Seek |
| **m** | Radar: moving targets only (static scene removed) or everything |
| **s** | Save a screenshot to `<session>/sync/replay_<time>.png` |

Before a sensor's first frame, for example before the radar starts (the radar
always starts last), that sensor's panel reads *no data at this time*.

## Measured on this Jetson

Tested on a 48 s three-sensor session, playing all sensors at 1×:

| Sensor | Updates/s | Lag behind the clock |
|---|---|---|
| Camera | 30.0 | 16–42 ms (one frame is 33 ms) |
| Radar | 3.9 at 4 fps | about 110 ms (one frame is 250 ms); **5–7 ms GPU per frame** |
| LiDAR | 9.7–10.7 | about 70 ms (one sweep is 100 ms) |

Seeking (backwards and forwards), 2× speed, pause, and paused seeks all keep
every sensor within one frame period of the clock.

## How it works (`sources.py`)

**Shared clock.** `ReplayClock` holds the session time on `CLOCK_MONOTONIC`,
in shared memory, so a source in another process follows the same clock.

**Camera.** Runs in its own process. The ZED SDK's `grab()` holds the GIL for
a whole frame decode, which in a thread starved the GUI and the other sensors.
Frames come back through shared memory. Playback decodes forward. It seeks
only on real jumps, because seeking an H.264 stream re-decodes from a
keyframe.

**Radar.** The repo's config, channel geometry and antenna calibration
(`cascaded_radar_capture_pipeline/processing/radar_config.py`), with all FFTs
in PyTorch on the GPU.

**LiDAR.**

- rs_driver can only play a PCAP from the start. A feeder thread writes
  packets into a named pipe in step with the clock, so pause, speed and seek
  work.
- The feeder sends the last device-info packet first, and starts 0.3 s before
  a seek point so a complete sweep is ready.
- Sweeps are split behind the vehicle (`split_angle: 180`).
- A packet index is cached in `sync/lidar_pcap_index.npz` on first use.

## Overlays and navigation (2026-10-11)

- **One range for both bird's-eye views.** The LiDAR *range* selector sets the
  same window on the LiDAR and radar views. The radar BEV is computed out to
  `--bev-range` (default 80 m), and the range–Doppler range axis follows the
  same selector.
- **Colour scales relative to the noise floor.** The radar BEV runs from
  median + 3 dB of that frame, and range–Doppler likewise. An empty scene stays
  dark instead of auto-scaling noise to full colour.
- **LiDAR target overlay.** It shows:
  - a white circle on both bird's-eye views;
  - a dashed line at its range on the range–Doppler map;
  - its range and angle in the info line.

  It comes from `sync/timing_lidar_track.npz`, which `make_report.py` writes,
  so run the report once per session. It is a single-target track: with several
  people it follows one of them.
- **Event strip** above the slider:
  - camera gaps (aqua) and LiDAR gaps (orange) from `pairs.csv`;
  - a time cursor; **click to jump**;
  - a note when radar times are nominal (DCA backlog).

## 3D LiDAR view (2026-10-11)

The **3D LiDAR** button, or `--lidar-3d` at start, opens a second window that
follows the replay. It shows the same sweep as the 2D LiDAR panel, as a point
cloud.

- **Controls:** left drag to orbit, wheel to zoom, middle or ctrl drag to pan.
  *Reset view* returns to the default.
- **Default camera:** behind and above the sensors, looking forward, so
  vehicle-left is on the left as in the other views.
- **Colour:** by height, intensity or range.
- **Drawn as well:**
  - a 1 m ground grid; the ground level is estimated from the first sweep,
    since the mount height is not recorded;
  - the radar's ±60° field of view, in blue;
  - the LiDAR target, in white.
- Only valid returns are shown (zero-range "no return" points are removed),
  and the *hide nearer than* filter applies.
- The *range* selector also sets the 3D view's extent.
- A screenshot (`s` key) also saves `<name>_3d.png` when the window is open.

**OpenGL with conda:** `conda activate` sets `QT_XCB_GL_INTEGRATION=none`
(the `qt-main` activation script), which disables Qt's OpenGL. The replay
clears that value for its own process, so no extra step is needed (fixed
2026-10-14; before that the 3D window failed with "neither GLX nor EGL are
enabled").

**Requires PyOpenGL.** Version 3.1.10, pure Python with no dependencies, was
installed in `rel_1_py_3_8` on 2026-10-11. The only change to the environment
was that one package; `pip check` is clean and every project component and the
report tests were re-verified. To remove it: `pip uninstall PyOpenGL`. Without
it, the button just prints that the 3D view is unavailable.

## Radar point cloud (2026-10-13)

*Radar points* (main window), or `--radar-points {off,raw,filtered}` at start.
The points are drawn on the radar bird's-eye view and in the 3D window,
together with the LiDAR cloud.

- **Raw** (the team's definition: before any filtering). Every range–azimuth
  cell of the current frame, computed live on the GPU: no CFAR, no clustering.
  - Static-clutter removal follows the *moving only* tick box.
  - A *raw threshold* (dB above the frame's median) is a display setting only.
  - z = 0, since there is no elevation in this path.
  - Shown as cyan dots on the bird's-eye view, and coloured by power in 3D.
- **Filtered** (after CFAR and all the filters of the team's pipeline). The
  output of `processing/radar_processing.process_frame`, **unchanged**.
  - It takes ~1 s per frame on the CPU, too slow to run live, so it is
    computed once per session:
    `radar_cloud.py --session <s>` or `make_report.py --session <s> --radar-cloud`.
    That is about 1.1× the recording time on 10 cores; 10.8 min for the
    10-min `freewalk_ref_02`.
  - Saved as `sync/radar_cloud_filtered.npz`, with the team's original values
    plus the points converted to the pipeline frame (x forward, y left,
    mirror-corrected).
  - Coloured by radial velocity: blue −, grey 0, red +.
  - *z = 0* ignores the pipeline's elevation, which is unverified.
  - Typically 1–9 detections per frame, tuned for a single subject.
- The radar is assumed co-located with the LiDAR; there is no extrinsic
  calibration yet.

## Limitations

- **Radar azimuth:** the processing's angle sign was mirrored. On freewalk_ref_02 a
  walker 3 m to the right (camera and LiDAR agree) appeared on the left. The BEV
  now flips it (`RadarProcessor.MIRROR_AZIMUTH`).
- **Radar angular focusing is weak** with the current, uncalibrated channel
  processing. A single person appears as an arc across much of the field of view,
  but the arc's peak sits at the LiDAR target. Range is sharp. The team's antenna
  calibration should address it.
- **Camera decode limit:** lossless H.264 decodes at only ~40–50 frames/s on the
  Orin. While the radar and LiDAR also run, the camera panel can drop to ~15–25
  updates/s. Sync is unaffected: it shows the latest decoded frame for the
  current time, and the panel title gives its timestamp. Use 0.5× for every
  camera frame.
- **Radar data recorded before 2026-10-10 is degraded:** about 50% stale
  buffer words per chirp (a firmware defect, not compression). This affects
  `stage3_01` and `ipad_test_01`. Sessions from `r075m_v2` / `r075m_v2_8fps`
  are clean. Replay picks each session's `.cfg` from
  `session.json["radar_firmware"]`; older sessions are assumed r075m.
  **Before 2026-10-10, replay decoded Q-first, which mirrored the radar range
  spectrum;** it now decodes I first.
- A display is needed. It runs on the Jetson desktop, not over the iPad web page.
