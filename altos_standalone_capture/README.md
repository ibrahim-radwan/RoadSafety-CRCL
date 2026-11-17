# Altos V4 radar: standalone capture

An **Altos V4** 4D imaging radar: the project's **fourth sensor**, recorded
alongside the TI radar, LiDAR and camera (decided 2026-11-17).

The sensor:
- 16 TX × 16 RX;
- 150° horizontal and 30° vertical field of view;
- up to 15 frames/s;
- outputs a **processed point cloud**, not raw ADC data.

This folder holds its recorder and packet format, and the tools for
working with it on its own.

**In the capture pipeline:**
- `sync_capture_pipeline/capture_session.py` runs `record_altos_raw.py` into
  `<session>/altos/`.
- `check_session.py`, `align_session.py`, the session report and the live
  view all include it.
- See `docs/knowledge/07_synchronisation_and_stage3.md` for its capture
  times.

Our unit: serial V4-260804-0146, firmware 8e75cfeb.

**Use the V4 materials, not the V2 ones.** The V2 driver
(`Altos-Radar/altosRadarRosV2`) has a different packet format and does not
decode this sensor. The V4 materials are:
- the [V4 user guide](https://forum.altosradar.com/t/altos-v4-and-rcu-rf6-user-guide/72);
- the V4/RCU Interface Control Document, linked from that guide;
- `altos-pc-v2.h` and the Wireshark dissector `altos.lua` in
  [Altos-Radar/altosRadarParser](https://github.com/Altos-Radar/altosRadarParser).

## Hardware and network

- **Power:** 9–36 V. The guide asks for a supply of more than 25 W; the spec
  sheet says the radar draws 18 W. We use the 15 V / 4 A adaptor; 1.5 A is not
  enough. The barrel plug is 5.5 × 2.5 mm. Boot takes 10–20 s.
- **Ethernet:** 100BASE-T1, through the harness's media converter to RJ45.
- **Addresses:** radar 192.168.3.10/24, host 192.168.3.1/24.
- **Web pages:** configuration at `http://192.168.3.10:3000`; firmware update
  at port 8080.
- **Data:** UDP multicast to 224.1.2.4:4040.

On `ibrahim-desktop` the radar shares `eth0` with the DCA1000 profile. Add
the address for a session, without changing the saved profiles:

```bash
sudo ip addr add 192.168.3.1/24 dev eth0     # undo: sudo ip addr del 192.168.3.1/24 dev eth0
curl -s http://192.168.3.10:3000/status      # model, serial, firmware
curl -s http://192.168.3.10:3000/install     # profile, FPS limit, threshold, sync mode
```

### Radar settings in use (web page, "Installation Configuration")

| Setting | Value | Why |
|---|---|---|
| Profile | `150m_1` | The finer-resolution variant of the 150 m profile; enough range for two-wheeler scenes. |
| FPS limit | off | The radar runs at its maximum, 15 fps. Setting 30 is reset to 15 (checked 2026-11-15). |
| Threshold coefficient | 0 | The most detections the firmware allows (default 250,000). It is not a plain SNR cut (see "Measured"), so filtering by SNR afterwards only approximates a higher threshold. |
| Sync | Free running | No PTP yet: use host receipt time (see below). |
| Flip points below ground, install height | (stored, no effect) | The web page sends these only for another device type, not for the V4. The flip is done by the vendor's driver on the host (z → −z − 2h for points below the floor); we can apply it offline. |

## Changing settings

```bash
python altos_settings.py                                  # show status and settings
python altos_settings.py --profile 150m_0                 # change
python altos_settings.py --threshold 250000 --fps-limit 10
python altos_settings.py --threshold default --fps-limit off
python altos_settings.py --profile 150m_1 --dry-run       # print the request only
python altos_settings.py --restore settings_backups/<file>.json
```

The script sends the same request as the web page's Apply button, read from
the page's code: POST `/install`. Settings not given on the command line
are kept.

Before each change it:
- saves the current settings to `settings_backups/` (git-ignored), so
  `--restore` can put them back.

After the change it:
1. waits for the radar's restart;
2. waits for fresh frames;
3. reads the settings back and confirms them.

It never touches the network (`/eth`) or time sync (`/time`) settings.

**Every change restarts the radar.** Checked 2026-11-15, for threshold and
profile changes:
- The radar answers `202 Accepted` after 1–6 s.
- The stream stops for about 1.2–1.5 s, once 7 s.
- Frame ids then start again from 0.

The recorder, checker and viewer count this as a radar restart. Don't
change settings while recording.

## Validation in a small room

`analyse_altos.py` measures recordings against the room and targets, measured
with a tape. Describe them in a scene file: copy `scene_example.yaml`. It
holds:
- the radar height;
- the distance from the radar to each wall and to the ceiling;
- each target's position and radius.

The radar must be level and point along the room's long side.

```bash
python record_altos_raw.py --duration 30 --label "plate 2m ahead" --scene my_scene.yaml
python analyse_altos.py dataset/altos_session_<stamp> --plot          # one recording
python analyse_altos.py dataset/altos_session_* --plot --csv sweep.csv  # compare several
```

Per recording it reports:
- frame rate, points per frame, SNR spread;
- **every point classed** as:
  - target;
  - surface (on a wall, the floor or the ceiling);
  - interior (furniture, people);
  - beyond walls (reflection ghosts);
  - below floor (floor bounces);
- **how much of each surface is seen**: 25 cm cells in the field of view,
  seen in ≥5% of frames;
- **per target:**
  - detection rate;
  - range, azimuth and elevation against the tape: error and frame-to-frame
    spread;
  - range and azimuth peaks around it, for the resolution tests;
- **for moving points:** Doppler against the rate of change of range. A
  slope of about +1 means positive Doppler = moving away. Also the azimuth
  at start and end (+ = left).

Across recordings it adds:
- a comparison table;
- a threshold check against the threshold-0 recording of the same profile
  and FPS.

`--plot` writes a top and side view of each recording, with points coloured
by class.

Checked 2026-11-15 on a synthetic recording with known answers:
- target errors 0, detection 100%;
- Doppler = 0.99 × range rate;
- left and right drawn correctly.

Two limits:
- A single surface seen over a range of angles can give two range peaks a
  few cm apart.
- The ceiling is only in view beyond about 6 m from a radar 1.6 m below
  it, so in a small room it reads 0/0.

**Tests** (radar fixed, room measured; for the sweep, change settings with
`altos_settings.py` between runs):

| # | Test | Length |
|---|---|---|
| 1 | Empty room | 60 s |
| 2 | Target at measured positions: 1, 2, 3 m ahead; ±30° to the side; floor level and raised | ~20 s each |
| 3 | Walk towards and away 2–3 times, then across | 30 s |
| 4 | Plate in front of a wall, moved closer: 50, 30, 20, 10 cm (range resolution) | ~15 s each |
| 5 | Two targets at the same distance, moved together (angular resolution) | ~15 s each |
| 6 | Sweep: tests 1 and 2 (one position) per setting: profile 150m_0 / 150m_1 / 320m_1; threshold 0 / 25,000 / 250,000 / 2,500,000; FPS limit off / 10 | 30 s each |

For the targets:
- **Best:** a corner reflector, three square sheets of foil-covered card
  (15–20 cm) taped into a box corner. It returns energy towards the radar
  from any direction.
- **Fallback:** a flat metal plate works, but only when turned to face the
  radar squarely.

## Recording and checking

Run with the project environment. On `ibrahim-desktop`, unset `PYTHONPATH`
first (ROS sets it):

```bash
cd altos_standalone_capture
python record_altos_raw.py --duration 30          # Ctrl+C to stop early
python check_altos_session.py dataset/altos_session_<stamp> [--export]
```

The recorder uses only the standard library. The checker needs numpy.

A session folder (`dataset/` is git-ignored) contains:

| File | Content |
|---|---|
| `altos.pcap` | Every packet as received, stamped with host CLOCK_MONOTONIC at receipt. Opens in Wireshark (use `altos.lua` to decode it). |
| `frames.csv` | One row per frame: id, first and last packet receipt time, radar time, points, packets, complete. |
| `stats.csv` | Per second: packets, frames, points, lost and incomplete frames, queue depth. |
| `clock_map.csv` | CLOCK_MONOTONIC and CLOCK_REALTIME pairs, once per second. |
| `metadata.json` | Addresses, the radar's own settings at the start (`/status`, `/install`, `/eth`, `/time`), and totals. |
| `points.npz` | Written by `--export`: points (range, doppler, azimuth, elevation, snr, x, y, z), a `valid` mask, and per-frame times. |

What the checker reports:
- frame-id gaps (whole frames lost);
- incomplete frames;
- frame spacing on both clocks;
- the radar clock fitted against host time;
- point statistics.

It exits 0 when no frames were lost or incomplete. The last frame, cut off
by the stop, is reported but does not fail the check.

## 3D viewer

```bash
python view_altos.py                                          # live from the radar
python view_altos.py --session dataset/altos_session_<stamp>  # replay a recording
python view_altos.py --colour snr --trail 5 --camera top      # starting options
```

`view_altos.py` uses PyQt5 and pyqtgraph's OpenGL view, the same stack as the
replay's 3D LiDAR panel, and needs a display. It runs with the environment
activated or not.

`conda activate` sets `QT_XCB_GL_INTEGRATION=none`, which switches off Qt's
OpenGL ("Cannot create platform OpenGL context"). The viewer clears that
setting for its own process, as the replay does.

**Live.** Joins the multicast stream and shows each frame as it arrives,
with the frame rate and lost or incomplete frames. It can run while
`record_altos_raw.py` records. Checked 2026-11-15: both received every
frame.

**Replay.** Plays `altos.pcap` at the recorded pace. Controls:
- play/pause, a time slider, and speeds 0.25×–4×;
- keys: space plays or pauses; ← and → step one frame.

**Scene.** The axes are vehicle-style: x forward, y left, z up, with the
origin at the radar. The view draws:
- a ground grid, at the radar's configured install height below it
  (`--height` overrides);
- range rings every 10 m;
- the 150° × 30° field of view, in radar blue.

Mouse: drag orbits, the wheel zooms, ctrl+drag pans. The **Behind**, **Top**
and **Side** buttons reset the camera.

**Colours.**
- doppler: blue approaching, grey still, red receding;
- SNR in dB;
- height above the grid;
- range.

**Filters and display.**
- Filters: minimum SNR (dB), minimum |speed| (shows only moving things),
  maximum range.
- **Trail** overlays the last N frames, older ones fainter; radar clouds are
  sparse, so 3–5 frames help.
- Points with NaN fields are never drawn.

**Signs not yet checked:** that negative Doppler means approaching (taken from
the vendor driver), and that z is up (taken from the vendor guide). Walking
towards the radar should show blue.

`--screenshot out.png --screenshot-after 6` saves the window and exits, for
checks without a person at the screen.

## Data format

Packet layout is in `altos_v4.py`. Each packet has a 32-byte header (magic
"Alto", radar time, frame_id, length, offset), then up to 72 points of 20
bytes: range, doppler, azimuth, elevation and SNR as float32.

- **Multi-packet frames:** a frame of more than 72 points comes in several
  packets, which are put back together by `offset`.
- **Axes:**
  - raw: x forward, y **left**, z **down**;
  - positive azimuth is left, as in our vehicle frame;
  - `frames_to_arrays()` flips the sign of z so that z is up.

## Measured on our unit (2026-11-15, indoors, ibrahim-desktop)

| Item | Result |
|---|---|
| Frame rate | 15.0 frames/s; no lost or incomplete frames over 30 s |
| Points per frame | 72–284 indoors (frames over 72 points come as 2–4 packets) |
| Packet spread within a frame | about 0.5 ms from first to last packet |
| Azimuth spread of points | −72° to +73° |
| Elevation spread of points | −15° to +15° |
| NaN points | Some points come with fields set to `0xFFFFFFFF` (NaN): 0.2% in one recording, 3 points in another, always among a frame's last ~16 points. Kept as sent, marked invalid when decoded. |

**Settings and points per frame**, same room, nothing moving, 2026-11-15:

| Profile | Threshold | Points per frame |
|---|---|---|
| 150m_1 | 250,000 (default) | ~110–113 |
| 150m_1 | 0 | ~168–175 |
| 150m_0 | 0 | ~122 |

**The threshold is not a plain SNR cut** (15 s at each threshold):
- At 250,000 the radar keeps 110 points per frame.
- Cutting the threshold-0 recording to the same count needs an SNR floor
  of 21.1 dB.
- Yet the 250,000 run keeps some points down to 2.9 dB (its lowest 1%).

The threshold is probably relative to the local noise, like a CFAR
detector.

**Radar clock:** free running, it reads March 2025. Its timestamps are
exactly 66.7 ms apart, but the real frame rate wanders against host time:
−505 ppm in one recording, +408 ppm in the next. Use host receipt time, as
for the other sensors. The radar's timestamps are only worth using with PTP.

## Not yet verified

These are the small-room tests above:
- **Axis signs** of azimuth and elevation, and the **Doppler sign**.
- **Range and angle accuracy.**
- **Resolution.**
- **The best setting combination.**
- **Recording on the rig**, next to the other sensors (CPU, network ports).
- **PTP and the second-offset trigger.**
