# Data Collection Guide

Recording radar, camera and LiDAR on the two-wheeler, controlled from an iPad.
Written 2026-10-11 for the current setup. The technical background is in
`SYNC_DESIGN_3_SENSORS.md`.

| Sensor | What it records | Rate |
|---|---|---|
| Cascade radar (AWR2243 ×2 + DCA1000), firmware `r075m_v2_8fps` | raw ADC frames, 2.62 – 75 m | 8 frames/s |
| ZED 2i stereo camera | lossless video (SVO2) + IMU | 30 frames/s |
| RoboSense RS-Fairy LiDAR | raw packets (pcap) | 10 sweeps/s |

Nothing is paired or processed during the ride: every sensor records raw
data with timestamps, and the pairing is done afterwards (see **After the
ride**).

---

## 1. Before the first ride of the day

### Power — check this first

The radar board needs a **full 12 V supply**. On a weak supply (such as a
partly discharged power bank) the radar accepts every command and reports
success, but sends **no data**. No error appears anywhere except the radar
packet rate.

### Hardware

| Device | Connection |
|---|---|
| DCA1000 (radar capture board) | Ethernet → Jetson onboard port |
| LiDAR | Ethernet → USB-Ethernet adapter → Jetson |
| ZED 2i camera | USB → Jetson |
| SD card (1 TB) | Jetson SD slot. **Insert before powering on, never remove while recording** |
| iPad | Same Wi-Fi as the Jetson |

- **Mounting:** all three sensors must face forward and be mounted rigidly.
  The LiDAR was once found turned 57° after being reconnected, which ruins
  its data for pairing. If any sensor was moved or remounted, tell the team
  responsible for calibration.
- **Power-on order:** radar 12 V + DCA1000 → LiDAR → Jetson. The radar starts
  streaming by itself at power-on. No laptop is needed.
- Nothing needs to be typed on the Jetson. Network settings, the SD card
  mount, the web page and the system settings all start automatically.

### Open the web page

On the iPad: **http://jetson-desktop.local:8080**

If it does not load, check that the iPad is on the same Wi-Fi as the Jetson,
then wait a minute after the Jetson boots.

---

## 2. Recording a session

### Fill in the form

| Field | What to enter |
|---|---|
| **Session name** | Pre-filled as `run_<date>_<time>`. Change it if you like. It must be new: letters, digits, `-` and `_` only. |
| **Note** | What is being recorded: route, weather, traffic, anything unusual. **Always fill this in**; you will not remember later. |
| **Duration (s)** | `0` = record until you press **Stop**. Any other value = stops by itself after that many seconds (Stop still works earlier). |
| **Countdown (s)** | Delay before recording starts, to get ready or get on the bike. |
| **Sensors** | Leave all three ticked. |
| **Stage 3 calibration guide** | **Leave OFF.** Only for timing calibration (section 5). |

Press **Start recording**. The tap also enables sound on the iPad.

### What you see and hear

1. **Countdown** (if set), then **"Starting sensors…"** for about 15 s. The
   LiDAR and camera start first; the radar starts last.
2. **One long beep = recording.** The banner shows elapsed time (and the set
   duration).
3. **Live status cards**, updated every second:

   | Card | Normal |
   |---|---|
   | Radar | about **17,300 packets/s**. Below 5,000 it turns red (see section 4). |
   | Camera | about **30 fps**, few or no gaps |
   | LiDAR | packets flowing, **0 lost** |
   | Board (Jetson) | temperature OK (warns at 80 °C, refuses to start at 90 °C) |

4. **Three short beeps = recording ended**: the duration was reached, you
   pressed Stop, or a sensor failed. The page shows "Recording finished".
5. Wait for **finalizing**. The camera verifies its file, which takes up to
   about half the recording time, and is then moved onto the SD card. The
   session check then runs by itself.

### Check the result before the next session

The session appears under **Recorded sessions** with **PASS** or **FAIL**.
Tap it to read the full check.

- **PASS:** the data is complete. Go on.
- **FAIL:** read why before recording more, or you risk a series of
  unusable sessions.
  - **Camera dropped slightly above 1%** (for example 1.05%) is a borderline
    fail. The data is usable, but note it.
  - **Radar or LiDAR loss, or missing sensors:** fix the cause (section 4)
    before continuing.

A plain-language **HINT** appears on the page for the common failures.

---

## 3. Storage

| | Capacity |
|---|---|
| SD card (radar + LiDAR + camera) | 930 GB free ≈ **8 hours** of recording (about 112 GB per hour) |
| Jetson internal disk (camera during recording) | holds about **4 hours** of camera; emptied after each session |

- The camera records to the Jetson's internal disk during a session and is
  moved onto the SD card when the session ends. This avoids camera frame
  drops; it is automatic.
- The launcher refuses to start if the SD card is missing or there is not
  enough space for the set duration.
- **The SD card is ext4 (Linux).** Windows cannot read it directly. Copy
  sessions off through the Jetson (section 6).

---

## 4. If something goes wrong

| Symptom | Cause / action |
|---|---|
| **Radar 0 packets/s**, or HINT "DCA1000 not responding" | Check the radar's **12 V supply** and the DCA1000 Ethernet cable. Power-cycle the radar and DCA1000. |
| **Radar below 5,000 packets/s** (card red; e.g. ~900 or ~3,000/s), or the session **stops by itself** with HINT "radar data rate too low (DCA1000 throttled)" | Rare DCA1000 start-up fault: it sends too slowly, buffers the radar data, then stops capturing. The launcher stops the session automatically after 5 s of low rate. **Power-cycle the radar + DCA1000** (12 V off, wait 10 s, on) and start a new session. |
| HINT "no LiDAR data" | LiDAR power and the USB-Ethernet adapter. |
| HINT "camera not detected" | Unplug and replug the ZED USB cable, start again. |
| HINT "SD card not found" | Insert the card fully. If it was removed while the Jetson ran, reinserting is enough; no commands are needed. |
| HINT "storage full" | Copy finished sessions off, then delete them. |
| HINT "board too hot" | Shade the Jetson, improve airflow, wait. |
| Page does not load | iPad Wi-Fi; wait for the Jetson to finish booting. |
| Page reloaded or iPad locked during a recording | Harmless. The recording continues on the Jetson; reopen the page and it shows the running session. |

---

## 5. Timing calibration (Stage 3): already done

The time offsets between the sensors were measured on 2026-10-11 and are
applied automatically when sessions are processed:

| | Offset (added to its timestamps) |
|---|---|
| Radar | **+12.65 ± 1.45 ms** |
| Camera | **+23.34 ± 3.28 ms** |
| LiDAR | reference (0) |

They are stored in `sync_capture_pipeline/calibration/stage3_offsets.json`.
**They stay valid as long as these do not change:**

- the radar firmware (`r075m_v2_8fps`);
- the DCA1000 settings;
- the camera and LiDAR settings.

If the radar firmware is changed (for example to another range variant),
repeat Stage 3:

1. Record **3 sessions of 90 s**, in daylight, with **"Stage 3 calibration
   guide" ticked** (it clears itself after each Start).
2. Follow the routine. There is a double beep at each phase change:
   1. **Walk:** hold the metal plate square to the radar at radar height,
      straight ahead. Walk **briskly** between **4 m and 14 m**, turning
      sharply at each end.
   2. **Side-steps:** at about 8 m, step **2–3 m left, then 2–3 m right**,
      past the centre each time. Keep moving.
   3. **Flips:** at about 8 m, flip the plate face-on and edge-on about every
      second.
3. Keep everyone else out of view.
4. Run
   `python measure_sync_offset.py --session <s1> <s2> <s3> --out calibration/stage3_offsets.json`.
   Accept the result when the three sessions agree. Slow or narrow walking
   gives wrong offsets.

**Spatial calibration** (where the sensors sit relative to each other) and
**radar antenna calibration** are done by the end-user team.

---

## 6. After the ride

### Pair the sensors (on the Jetson)

```bash
cd ~/dev/CascadedRadarRoadSafetyOnTwoWheelers/sync_capture_pipeline
PY=~/miniconda3/envs/rel_1_py_3_8/bin/python
$PY align_session.py --session /mnt/sdcard/sessions/<name>
```

This writes `<session>/sync/`:

- `pairs.csv`: one row per radar frame, with the matching camera frame and
  LiDAR sweep;
- per-sensor timelines;
- `alignment.json`.

The Stage 3 offsets are applied automatically. The output shows
`offsets: … (Stage 3)`.

### Session report (HTML)

```bash
$PY make_report.py --session /mnt/sdcard/sessions/<name>
```

This writes `<session>/session_report.html`: one self-contained page, with
figures embedded, that opens offline anywhere and travels with the session.
It aligns the session first if needed. It takes about 1/3 of the recording
time (13 min for a 10-min session); `--reuse` re-renders in seconds from the
cached data.

The report is **descriptive**, with no pass/fail. It contains:

- overview and factual observations;
- timeline and data rates;
- synchronisation: intervals, pairing errors, and a velocity-based
  radar−LiDAR timing check;
- radar: range–time, micro-Doppler, raw ADC statistics, channel levels;
- LiDAR: bird's-eye view, occupancy by direction, statistics;
- camera: thumbnails, exposure, drops, IMU;
- paired cross-sensor snapshots.

For the radar and LiDAR detections and tracks of a session (for the research
team, and to score them against the LiDAR walker), run
`$PY sync_capture_pipeline/run_perception.py --session <s>`, which takes about
4 minutes for 10 minutes of data. The replay computes them live.

Tests: `$PY -m pytest sync_capture_pipeline/tests`.

To look at a session:
`$PY replay/replay_session.py --session /mnt/sdcard/sessions/<name>`.
Use the Jetson desktop; see `replay/README.md`.

### Copy data off

The SD card is ext4, so copy through the Jetson:

- **To an external drive plugged into the Jetson:**
  `rsync -a --info=progress2 /mnt/sdcard/sessions/<name> /media/jetson/<drive>/`
- **Over the network:** `rsync -a /mnt/sdcard/sessions/<name> user@<lab-pc>:/data/`

Delete sessions from the card only after the copy has been checked.

**Removing the SD card:** power the Jetson off first, or run
`sudo umount /mnt/sdcard`.

### What a session contains

```
<name>/
  session.json        settings, clocks, radar firmware, thermal, notes
  session.log         full recording log
  check_session.txt   PASS/FAIL report
  radar/              frame_*.bin (3,145,728 B each), frames.csv
  camera/             camera.svo2, capture/frames.csv, IMU, calibration
  lidar/              lidar.pcap, clock_map.csv, stats.csv, metadata.json
  sync/               (after align_session) pairs.csv, timelines, alignment.json
```

Radar frame format: int16 little-endian, **I first**,
`[loop 64][tx 6][dev 2][rx 4][sample 256][I,Q]`. See
`cascaded_radar_capture_pipeline/firmware/FIRMWARE_STATUS.md`.

**Keep** every session with a PASS, its note and `session.json`. Sessions
that fail the check should be fixed or deleted, not annotated.

---

## 7. Changing the radar firmware

Normally not needed. If the team wants a different range variant:

1. Follow `cascaded_radar_capture_pipeline/firmware/FIRMWARE_STATUS.md`. Use
   `flashing/boot/flash_variant.sh <variant>`, which records what was
   flashed; sessions pick this up automatically.
2. Run the 30 s acceptance check described in `FIRMWARE_STATUS.md`.
3. Repeat Stage 3 (section 5).

Only use `_v2` images. The older images produce about 50% corrupted radar
data.
