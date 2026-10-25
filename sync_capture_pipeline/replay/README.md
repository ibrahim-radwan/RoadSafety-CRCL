# Session replay

Replays a three-sensor session in real time, with every sensor on one clock,
and runs our GPU detection and tracking live on the radar and the LiDAR
(`../perception/`). It can show all three sensors or any one of them.

One window, six panels (2026-10-24; the 3D view used to be a separate window):

| Panel | What it shows | Source |
|---|---|---|
| Camera | ZED left image, 960 px wide | SVO2, decoded on the GPU by the ZED SDK |
| LiDAR | One of three views (below), filling the panel; object boxes coloured by size class, track IDs `L<n>` | `lidar.pcap`, decoded by rs_driver; objects: `perception/lidar.py` (GPU) |
| LiDAR 3D | The same sweep as an orbitable point cloud, 3D boxes, radar points | same |
| Radar BEV | Range-azimuth power on the **same forward window** as the LiDAR; detections; tracks `R<n>` | radar `.bin`, `perception/radar.py` (GPU) |
| Range-Doppler | Range vs radial velocity (+ = receding), detections | same |
| Objects | Confirmed tracks of both sensors: range, azimuth, speed, radial velocity | the trackers |

- **Maximise a panel:** double-click its title, its ⤢ button, or keys **1–6**
  (in the order above). **Esc** or **0** restores. `--maximise <panel>` at
  start.
- **Resize** with the splitters between panels; the bird's-eye views re-fit.
- Each panel title shows the session time of the data on screen. For the radar
  frame on screen, the status line shows the paired camera frame and LiDAR
  sweep from `sync/pairs.csv`, with their time errors and any flags
  (`camera_gap`, `lidar_gap`, …).

## Run

Run these from `sync_capture_pipeline/` on the Jetson desktop, with the
environment **activated** (that is how users run it; see the OpenGL note
below). The replay needs `sync/` from `align_session.py`, and uses its
offsets.

```bash
conda activate rel_1_py_3_8
python align_session.py --session sessions/run05            # once per session
python replay/replay_session.py --session sessions/run05
python replay/replay_session.py --session sessions/run05 --sensors lidar          # one sensor
python replay/replay_session.py --session sessions/run05 --sensors camera,radar --start 30 --speed 0.5
python replay/replay_session.py --session sessions/run05 --maximise lidar3d
```

| Control | Action |
|---|---|
| **Space** / Play button | Play or pause |
| **← / →** | Previous or next radar frame (all sensors follow) |
| **[ / ]** | Slower or faster (0.25×, 0.5×, 1×, 2×) |
| Slider, event strip | Seek (click the event strip to jump) |
| **m** | Radar: moving targets only (static scene removed) or everything |
| **1–6**, **Esc** | Maximise a panel, restore |
| **s** | Save a screenshot to `<session>/sync/replay_<time>.png` (the 3D panel included) |

Before a sensor's first frame, for example before the radar starts (the radar
always starts last), that sensor's panel reads *no data at this time*.

## LiDAR views

Choose a view in the **LiDAR** group under the controls, or with
`--lidar-view`, `--lidar-range`, `--lidar-color` and `--lidar-near`. Changes
apply at once, even while paused.

| View | Best for | Notes |
|---|---|---|
| **Top - forward** (default) | Road sessions; direct comparison with the radar BEV | 0 to R m ahead, as wide as the panel |
| **Top - 360°** | Seeing the whole surroundings, including behind | ±R m around the LiDAR, as wide as the panel |
| **Panorama** | Recognising the scene; reads like a photo | Azimuth × elevation, coloured by distance (red = near) |

- **The top views fill their panel** (2026-10-24). The forward extent is the
  selected range; the lateral extent follows the panel's shape, and the image
  is rendered at about one cell per screen pixel. Before, the image covered
  only a square R × R window in the middle of a wider panel.
- **Range:** 5, 10, 20 (default), 40 or 80 m. Use 5–10 m indoors, 20 m for
  walking scenes and 40 m on the road. Mouse wheel and drag zoom further.
- **Colour:**
  - **height** (top views): the highest return in each cell, from the ground
    (measured by the detector) to 3 m above it;
  - **intensity**: reflectivity;
  - **density**: point count.
- **Hide <** (default **0.5 m**) hides returns from the LiDAR's own mount and
  the vehicle body. On the desk bench this was **69% of all points**. Set it
  to 0 to see everything; nothing is discarded, only hidden.

## Detections and tracks (2026-10-24)

The processing library (`../perception/`), on the GPU, live during replay.
See `../perception/README.md` and `../run_perception.py` for the details and
the measured scores.

**LiDAR** (`perception/lidar.py`, ~20 ms per sweep on its own, ~35–45 ms
while all sensors replay):

- The ground is local, not one plane, so slopes, kerbs and driveways stay
  ground.
- Overhead structures with free space below (carport roofs, canopies, signs,
  branches) are not obstacles.
- Obstacles are grouped into objects with oriented boxes, coloured by a
  **size heuristic**: pedestrian, two-wheeler, vehicle, other, or structure.
  This is a heuristic, not a learned classifier. A wall segment with a bike
  against it can be boxed as a "vehicle". Use the track's speed to tell
  moving road users from static structure.
- Tracks: `L<n>`.

**Radar** (`perception/radar.py`, ~16 ms per frame on its own, ~30–50 ms
while all sensors replay). Per frame:

1. CFAR on the range-Doppler map after coherent beamforming.
2. One azimuth per detection.
3. TDM-MIMO Doppler compensation.
4. The in-situ phase correction (`calibration/radar_rx_phase.json`;
   `--no-rx-phase` turns it off).
5. Multipath ghosts flagged.
6. Doppler-aware tracking: `R<n>`.

The BEV shows detections as dots coloured by radial velocity (blue
approaching, red receding). Multipath ghosts are hollow grey dots.

**Tracks** appear in the table and as labels. A track is confirmed after 3
hits in 5 updates, and shown while it keeps being updated. A radar track must
also keep its Doppler consistent with its motion. A seek starts the trackers
over.

**Detections group:** the boxes, the track IDs and the LiDAR target can each
be switched off. `--no-detect` turns off the LiDAR detector.

## Radar points

Choose in the **Radar** group, or `--radar-points {detections,raw,off}`
at start. The points are drawn on the radar bird's-eye view and in the 3D
panel. *z = 0* (on by default) draws them in the sensor plane, because the
radar elevation is not verified (see the radar topic page).

- **Detections** (default): ours, above.
- **Raw** (the team's definition: before any filtering). Every range–azimuth
  cell of the current frame above a *raw* threshold (dB above the frame's
  median, a display setting only). No CFAR, no clustering.
- The former **filtered** option (the team's `process_frame`, pre-computed by
  `radar_cloud.py`) was retired on 2026-10-25, when that chain was merged into
  `perception/`. Its general stages live on in the detections.
- The radar is assumed co-located with the LiDAR; there is no extrinsic
  calibration yet. The radar–LiDAR yaw differs between mountings by a few
  degrees (about −5° to +3° on the reference sessions); that is the team's
  extrinsic calibration.

## Overlays and navigation

- **One forward range for both bird's-eye views.** The LiDAR *range*
  selector sets the same forward window on the LiDAR and radar views. The
  radar BEV is computed out to `--bev-range` (default 80 m), and the
  range–Doppler range axis follows the same selector.
- **Radar colour scales.** The radar BEV runs from 3 dB above the frame's
  median, but never more than 30 dB below its peak, so a near target's
  sidelobes do not flood the map. Range–Doppler starts at median + 3 dB.
- **LiDAR target overlay.** It shows:
  - a white circle on both bird's-eye views;
  - a white marker in 3D;
  - a dashed line at its range on the range–Doppler map;
  - its range and angle in the info line.

  It comes from `sync/timing_lidar_track.npz`, which `make_report.py` writes,
  so run the report once per session. It is a single-target reference track:
  with several people it follows one of them.
- **Event strip** above the slider:
  - camera gaps (aqua) and LiDAR gaps (orange) from `pairs.csv`;
  - a time cursor; **click to jump**;
  - a note when radar times are nominal (DCA backlog).

## 3D LiDAR panel

- **Controls:** left drag to orbit, wheel to zoom, middle or ctrl drag to pan.
  *Reset view* returns to the default.
- **Default camera:** behind and above the sensors, looking forward, so
  vehicle-left is on the left as in the other views.
- **Colour:**
  - **height**;
  - **objects**: each object's points in its class colour, with the ground
    dimmed;
  - **intensity**;
  - **range**.
- **Drawn as well:**
  - a 1 m ground grid at the ground height the detector measures;
  - the radar's ±60° field of view, in blue;
  - the LiDAR target, in white;
  - 3D object boxes and track IDs within 1.3 × the selected range;
  - the radar points.
- Only valid returns are shown (zero-range "no return" points are removed),
  and the *hide <* filter applies.

**OpenGL with conda:** `conda activate` sets `QT_XCB_GL_INTEGRATION=none`
(the `qt-main` activation script), which disables Qt's OpenGL. The replay
clears that value for its own process, so no extra step is needed (fixed
2026-10-14). Without PyOpenGL (3.1.10 in `rel_1_py_3_8`), the 3D panel reads
*3D view unavailable* and the rest works.

## Measured on this Jetson

### 2026-10-24, `ibrahim-desktop`

Conditions: `freewalk_ref_02` read from an external USB disk (NTFS through
FUSE), with all sensors, detection on, at 1×.

| Sensor | Updates/s | GPU time per product |
|---|---|---|
| Camera | 29.5–29.9 | (ZED SDK decode) |
| Radar | 7.9–8.0 at 8 fps | 30–50 ms (16 ms alone) |
| LiDAR | 9.9–10.0 | 35–47 ms (20 ms alone) |

- **The radar used to reach only 3.0 updates/s on this disk.** Cold reads of
  a 3 MB frame take 78 ms median (up to 390 ms) on NTFS through FUSE.
- The radar now reads 12 frames ahead in a separate thread. Sessions on ext4
  (the SD card, internal storage) read much faster.
- Do not use scikit-learn in the per-frame path. Its per-call threadpool
  introspection held the GIL long enough to starve the GUI (see
  `docs/knowledge/08_pitfalls_and_lessons.md`).

### 2026-10-11, the original board

Conditions: a 48 s three-sensor session, all sensors, 1×, before detection.

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

**Radar.**

- `perception/radar.RadarDetector` builds the products and detections in one
  pass on the GPU (PyTorch). It uses the repo's config, channel geometry and
  antenna calibration (`perception/core/radar_config.py`).
- A prefetch thread reads frames ahead of the clock.
- `RadarProcessor` (the older, products-only processor) is kept unchanged for
  the session report.

**LiDAR.**

- rs_driver can only play a PCAP from the start. A feeder thread writes
  packets into a named pipe in step with the clock, so pause, speed and seek
  work.
- The feeder sends the last device-info packet first, and starts 0.3 s before
  a seek point so a complete sweep is ready.
- Sweeps are split behind the vehicle (`split_angle: 180`).
- A packet index is cached in `sync/lidar_pcap_index.npz` on first use
  (`lidar_pcap_index()`, shared with `run_perception.py`).
- Each sweep is detected once. View changes re-render without re-detecting.
  Per-point labels are computed only while the 3D panel shows *colour:
  objects*.

## Limitations

- **Radar azimuth convention (settled 2026-10-24):** the processing's
  azimuth is positive to the right (the team's convention), and the pipeline
  frame is y left. The conversion is a sign flip. It is not a mirror error in
  the team's code (see `docs/knowledge/03_radar_firmware_and_dca.md`).
- **Radar angular focusing is still weak.** The in-situ phase correction
  halves the azimuth spread against the LiDAR (5.95° → 2.43° on
  freewalk_ref_02), but sidelobes stay only ~5–10 dB below the peak. A
  person still shows a faint arc. The team's recalibration should fix it.
- **Radar–LiDAR yaw:** a few degrees, different per mounting; there is no
  extrinsic calibration yet.
- **LiDAR size classes are heuristics** (see above).
- **Camera decode limit:** lossless H.264 decodes at only ~40–50 frames/s on
  the Orin. Sync is unaffected: the camera panel shows the latest decoded
  frame for the current time, and its title gives the timestamp. Use 0.5× for
  every camera frame.
- **No camera overlays:** there is no camera–LiDAR/radar extrinsic
  calibration yet.
- **Radar data recorded before 2026-10-10 is degraded:** about 50% stale
  buffer words per chirp (a firmware defect, not compression). This affects
  `stage3_01` and `ipad_test_01`. Sessions from `r075m_v2` / `r075m_v2_8fps`
  are clean. Replay picks each session's `.cfg` from
  `session.json["radar_firmware"]`; older sessions are assumed r075m.
- A display is needed. It runs on the Jetson desktop, not over the iPad web page.
