# Three-sensor sync capture: design

Cascaded radar (DCA1000) + ZED 2i camera + RoboSense RS-Fairy LiDAR on the Jetson AGX Orin.

**Status:** proposal for review. Nothing below has been implemented yet.

---

## 1. Goal

For every radar frame, find the camera image and the LiDAR data that saw
the same moment. Report how far apart in time each pair is.

At 15 m/s, **10 ms of timing error is 15 cm of position error**. So the
target is to know each sensor's *capture* time to within a few
milliseconds, on one shared clock.

## 2. Principles

1. **Capture time, never save time.** File-write time includes queueing
   and storage stalls. eMMC stalls of up to about 200 ms were measured on
   this Jetson (2026-10-08). Host arrival time is better, but still
   includes each sensor's transport delay. Where a sensor reports its own
   capture time, use that.
2. **One reference clock: the Jetson's `CLOCK_MONOTONIC`.** All three
   sensors share one host. Monotonic time never jumps, unlike the wall
   clock, which NTP can step.
3. **Record raw, align offline.**
   - At capture time: each recorder logs the sensor's own timestamp *and*
     the host arrival time.
   - Afterwards: converting, de-jittering and pairing happen in a separate
     tool that can be re-run and audited.
   - The existing pipeline already separates these steps (Stage 3/4). This
     design keeps that separation.
4. **Measure the remaining fixed offsets physically.** A constant latency
   cannot be seen in timestamps alone (see `measure_sync_offset.py`).
   Stage 3 extends from two sensors to three.
5. **Radar frames are the anchor.** It is a radar dataset: each radar frame
   gets one camera image and one LiDAR sector. Each pair records its time
   error.

## 3. Sensor timing model

All times below are converted to `CLOCK_MONOTONIC`.

### 3.1 Radar (DCA1000, target 8 fps)

| Item | Value |
|---|---|
| Frame period | **Set by the firmware image, not by the `.cfg`.** The `.cfg` is never sent to the device. The radar free-runs on its own crystal from firmware load, and streams indefinitely until the session is stopped. **In flash since 2026-10-10: `r075m_v2_8fps`, 125 ms = 8 fps** (measured 125.002 ms). Every `_v2` variant also has a 250 ms build. The design does not assume a value: the period is measured from the data. |
| Chirp timing (`r075m`) | 6 chirps (TDM over 6 TX) × 64 loops × (4 µs idle + 31 µs ramp) = 384 chirps, **13.44 ms active per frame**. Each chirp samples 256 points at 10 Msps (25.6 µs). |
| Frame size | 3,145,728 B (256 samples × 64 loops × 48 virtual channels × 4 B), which is 2,160.5 packets of 1,456 B ADC data |
| Transmit burst | **set by the DCA1000 packet delay, not by the radar:** 2,160.5 packets × (delay + ~12 µs). At **25 µs** (receiver default since 2026-10-10) that is **~80 ms** per frame, then idle. At the old 50 µs it was 134.4 ms, which caps the stream at 7.44 fps (see below). An earlier "54 ms" figure was wrong. |
| Data rate | 4 fps: about 12.6 MB/s, about 45 GB/hour. **8 fps: about 25 MB/s, about 90 GB/hour, about 17,300 pkt/s** |
| Timestamp in data | none. Packets carry only a sequence number and a byte count. |

The radar packets carry no timestamp. Capture time is estimated in three
steps:

1. **Log the raw times (as today).** Record `t_burst`, the arrival time of
   the first packet of each transmit burst, and `t_last`.
2. **De-jitter offline.** Fit a straight line `t_k = t0 + k*T` through the
   `t_burst` values, using the frame index `k`. Use a robust fit, with
   outliers rejected. The radar's crystal makes `T` extremely stable, so
   the fit removes network and scheduling jitter from every frame. Frame
   indices come from byte-count slicing, so lost packets do not shift `k`.
3. **Correct the fixed latency.** Measure it in Stage 3 and subtract it.
   The *reference instant* of a radar frame is the middle of its chirp
   sequence: chirp start + 6.72 ms for `r075m`. The DCA forwards data as it
   arrives over LVDS, so the burst starts shortly after chirping begins.
   `t_burst` therefore sits a small, fixed distance from the reference
   instant, and Stage 3 measures that distance. The 13.44 ms chirp window is
   short compared with the frame period, so a radar frame is close to an
   instantaneous snapshot.

**Found and fixed on hardware (2026-10-09): `t_burst` was one frame late.**
Each burst's first packet starts slightly *before* a frame boundary
(16–1,440 B, because a frame is 2,160.5 packets). That packet carries the
tail of frame n−1 and the head of frame n. So the packet that *completes*
frame n is the first packet of burst n+1. The original receiver stamped a
frame with the burst current at completion, which made `t_burst` 250 ms
late at 4 fps. `check_session.py` and `measure_sync_offset.py` both pair
on it, so any offset measured with the old tool silently absorbed this
error. The receiver now stamps each frame with the burst that carried its
first byte. A 3-sensor session confirmed `t_burst` = the frame's own burst
start, 0.00 ms difference on all 79 frames.

**Radar data validity and the firmware fix (2026-10-10).** All images
booted from QSPI before 2026-10-10 (r017m … r168m, r075m) stream about **50%
stale words per chirp**. The AM273x DSS L3 RAM holding the CSI-2 ping/pong
buffers was never memory-initialised at QSPI boot; CCS loading had hidden
this, because its GEL script initialises that RAM. The stale words are
identical in every frame, in 32-byte blocks, and change on every power-up.
They are **not compression**: no compression exists in this firmware, and
`compressionCfg` in the old `.cfg` files was never sent anywhere.

- **The fix:** the `_v2` images run DSS L3 MEMINIT at boot. They also use
  HPF 0/0, which moves the near floor to 2.62 m on r075m (it was 10.5 m).
- **Confirmed on hardware:** 0.00% stale words at 4 and 8 fps, also after a
  power cycle (sessions `radar_fw_v2_check_02_powercycle` and
  `radar_fw_v2_8fps_check_02`).
- **Records:** the Windows agent's report is in the handover folder;
  `firmware_map.json` marks each variant's `data_valid`.
- **Decode:** I first. The corrected `.cfg` files have `adcbufCfg -1 0 0 1 1`;
  the old ones selected Q-first in `radar_config.py`, which mirrored the
  spectrum. `check_radar_frames.py` runs the stale-word test.

**DCA1000 packet delay (2026-10-10).** The `CONFIG_PACKET_DATA` delay is in
8 ns ticks. The old value, 6250 (50 µs), made a frame take 134.4 ms on the
wire. At 8 fps the DCA then streamed continuously and **dropped LVDS data
internally**, which the sequence-number and byte-count checks cannot see.

- **The fix:** `dca_capture_radar_sync.py --packet-delay-us`, default 25.
- **Warning:** the receiver warns when there are fewer transmit bursts than
  half the frames (no idle gap means saturation).
- **Windows path:** the project's Windows capture config
  (`AM273X_Capture_fixed.json`) still has 50 µs, so 8 fps on Windows would
  lose data.

**Which firmware recorded a session:** `~/.radar_flashed` (written by
`flashing/boot/flash_variant.sh`) is copied into
`session.json["radar_firmware"]`. `radar_firmware.py` maps a session to its
variant and `.cfg`. Stage 3 and replay use it; sessions without the field
are assumed r075m.

**Validation (implemented):** check that each frame boundary falls at a
burst boundary. The receiver slices frames from a continuous byte stream. If
`START_RECORD` ever caught the radar mid-burst, every later `.bin` would
straddle two radar frames. The burst structure makes this checkable.

### 3.2 Camera (ZED 2i, 30 fps)

| Item | Value |
|---|---|
| SDK image timestamp | Capture time estimated by the SDK, on `SYSTEM_CLOCK`, the **wall clock**. It is consistently about 50 ms before host receipt (measured). |
| Logged today | `timestamp_ns` (SDK, wall clock) and `host_receipt_ns` (`time.time_ns()`, wall clock). **Nothing is logged on the monotonic clock.** |

Capture time is computed per frame:

1. **Recorder change (small).** Next to `host_receipt_ns`, also log
   `host_receipt_mono_ns = time.monotonic_ns()`, taken back-to-back.
2. **Offline conversion.**
   `t_capture_mono = timestamp_ns − (host_receipt_ns − host_receipt_mono_ns)`.
   The wall-to-monotonic offset is sampled at every frame, so gradual NTP
   adjustments are absorbed. A step in that offset means the wall clock
   jumped; flag it.
3. **Exposure and rolling shutter.** These add a few milliseconds of fixed
   bias. Stage 3 absorbs it.

Pairing quantisation is ±16.7 ms (nearest frame at 30 fps). Dropped frames
are already logged in `capture/timing/camera_gaps.csv`.

### 3.3 LiDAR (RS-Fairy, 10 Hz)

| Item | Value |
|---|---|
| Packet timestamp | Microseconds, on the LiDAR's own clock, in every MSOP and IMU packet |
| Logged today | Every packet in `lidar.pcap`, with host arrival time on `CLOCK_MONOTONIC` |
| Clock relation | The LiDAR clock runs about +15.7 ppm vs host monotonic (about 57 ms per hour). Arrival jitter p99 is about 1 ms (measured). |
| Sweep | 100 ms per revolution, 138,240 points |

Capture time is computed in two steps:

1. **Clock fit (offline, per session).** Fit `t_mono = a*t_lidar + b`
   over all MSOP packets. Use the lower envelope of the arrival times:
   delay only ever adds, so the earliest arrivals are the true relation.
   This gives every packet, and therefore every point, a monotonic time.
2. **Field-of-view time, not frame time.** A sweep spans 100 ms. The part
   of the scene that matters is inside the radar's field of view.
   - **Radar FOV:** azimuth ±85°, elevation ±5°, range 10.5–75 m
     (`r075m`).
   - **Per-point times:** decode with per-point timestamps (rs_driver
     `PointXYZIRT` instead of `PointXYZI`).
   - **Point selection:** rotate the points into the radar frame using the
     mounting rotation, and keep those inside the radar FOV.
   - **Times reported per sweep:**
     - the **boresight time**: the time of points within a few degrees of
       straight ahead. Pairing uses this.
     - the **FOV time span**: the first to last point inside the FOV.
       With ±85° azimuth, this can approach half a revolution (about
       50 ms). It shows how much the scan itself is spread in time.
   - **Why it matters:** using the frame's first-point time could be up to
     100 ms wrong for the region of interest.

   **Sweep boundary (found 2026-10-09):** rs_driver splits sweeps at
   `split_angle` 0°, which is *straight ahead*. Each sweep therefore held
   the right half of the forward FOV at its start and the left half 100 ms
   later at its end, and the boresight time mixed the two.
   `split_angle: 180` puts the boundary behind. The forward ±5° is then
   scanned at 48–52 ms of the sweep in one pass, and the ±85° FOV spans
   47.5 ms. The LiDAR config sets it, and `align_session.py` applies it when
   decoding, so sessions recorded earlier benefit too (the PCAP is raw).

   **Mounting:** the LiDAR faces straight ahead, the same as the radar. That
   is enough to start. The rotation is a config value with that default. A
   one-time check confirms which LiDAR axis is "forward": put an object
   straight ahead and read its position in a short recording. Precise
   extrinsics come from calibration later. Selecting points by geometry
   works for any mounting orientation, including a dome LiDAR mounted on
   its side, so no assumption about the spin axis is built in.

Later option: the LiDAR supports gPTP (802.1AS), which could make the
clock fit unnecessary. The USB adapter has no hardware timestamping, so the
gain is limited. This is not needed for a first version.

## 4. Pairing (offline)

For each radar frame `k` with reference time `t_k`:

- **Camera:** the nearest frame by capture time. Error at most about
  ±16.7 ms, plus any dropped frames, which are reported.
- **LiDAR:** the sweep whose boresight time is nearest to `t_k`. LiDAR at
  10 Hz vs radar at 8 Hz gives an error of up to ±50 ms that cycles every
  0.5 s (4 radar frames vs 5 sweeps). At 4 fps the cycle is 1 s. This is
  reported per pair.
  - **Decided:** keep the error, report it, and let users filter on it.
  - **Later, if needed:** motion-compensate the LiDAR points to `t_k`,
    using per-point times and IMU ego-motion.
- **Output:** `pairs.csv`, with one row per radar frame. It holds the
  radar file, camera frame index and timestamp, LiDAR sweep index and
  sector time, and each signed time error. The camera images and LiDAR
  clouds are extracted offline from SVO2 and PCAP for the pairs that are
  kept.

## 5. Session layout

Sessions are recorded to the SD card: `/mnt/sdcard/sessions/<run>/` (see
section 9). The camera is staged on the eMMC during recording and moved into
`<run>/camera/` when the session ends.

```
sessions/<run>/
  session.json      launcher manifest: start/stop on monotonic + wall clock,
                    tool versions, config snapshots, exit codes, notes
  session.log       full launcher + recorder output
  radar/            frame_*.bin, frames.csv            (as today)
  camera/           camera.svo2, capture/frames.csv, capture/imu/, calibration/,
                    metadata.json, capture_status.json  (ZED recorder output)
  lidar/            lidar.pcap, stats.csv, clock_map.csv, metadata.json,
                    config_used.yaml                    (record_lidar_raw.py)
  sync/             generated offline: per-sensor timelines on CLOCK_MONOTONIC,
                    fit reports, pairs.csv
```

## 6. Launcher (`capture_session.py`, extended)

**Separate processes.** Three processes, each with its own Python
interpreter lock (GIL), as the current design argues:

- radar: stdlib only
- ZED: conda env `rel_1_py_3_8`
- LiDAR: conda env

**Start order.** The sensors that need coverage start first, and the radar
starts last:

1. LiDAR recorder (instant).
2. ZED recorder (about 6 s to open, warm up and verify).
3. Wait until both report that they are recording (health check).
4. Radar (about 5.5 s of DCA setup, then `START_RECORD`).

**Stop order.** The radar stops first, then a trailing window, then the
ZED and the LiDAR. Every stop is a SIGINT, so each recorder finalises
cleanly. The ZED's post-capture SVO2 verification takes about 0.55× the
recording time, and the launcher must wait for it.

**Live health.** One line per second, teed to `session.log`:

- radar packets/s
- camera FPS and gaps
- LiDAR packets/s and losses
- free disk space

Abort cleanly if any sensor goes silent.

**Manifest.** Clock pairs (monotonic and wall clock) at start and stop,
config snapshots, the git state of each tool, and exit codes. Since
2026-10-10 it also records `radar_firmware` (from `~/.radar_flashed`) and
`camera_staged`.

**Preflight additions (2026-10-10):**

- shows the flashed radar image (variant, period, near floor, md5) and warns
  if it is unknown or marked invalid;
- refuses to record under `/mnt/sdcard` unless the card is mounted;
- `--camera-stage DIR`: the ZED records to DIR (eMMC) and is moved into the
  session afterwards.

## 7. Required changes per component

| Component | Change | Size |
|---|---|---|
| ZED `recorder.py` | Log `host_receipt_mono_ns` per frame and per IMU sample. Accept an explicit output directory (`--session-dir`), so the launcher controls the path. | small |
| `record_lidar_raw.py` | Accept `--session-dir` (exact path instead of a generated name). Otherwise ready. | small |
| `dca_capture_radar_sync.py` | Move `.bin` writing off the receive thread, into a queue and writer thread. At 25 MB/s, a storage stall currently blocks `recvfrom()`. Add a frame/burst alignment check. Timing columns unchanged. | medium |
| `capture_session.py` | Three sensors, start/stop order, health monitor, manifest (section 6). | medium |
| `check_session.py` | Per-sensor integrity: radar sizes and alignment, ZED `capture_status.json`, LiDAR `metadata.json` / pcap counter check. **Done 2026-10-09:** stdlib-only, separates failures from warnings, plus a coverage check on `CLOCK_MONOTONIC`. Pairing has moved to `align_session.py`. Fault injection (truncated/missing radar frame, unfinalized camera, truncated PCAP) catches all of them. The radar + D455 version is in `legacy/`. | medium |
| new `align_session.py` | Section 3 conversions and fits, section 4 pairing, `sync/` outputs, and a fit-quality report. **Done 2026-10-09.** On a 20 s bench session: radar fit residual p99 0.30 ms; camera errors within ±16.4 ms; LiDAR 376/376 sweeps decoded, clock-fit p99 0.73 ms; 0 flagged pairs. LiDAR decode runs at 3× real time (rs_driver drops data above about 4×). | large |
| `measure_sync_offset.py` | Three sensors: radar-camera and radar-LiDAR offsets from one event (section 8). | medium |
| `DATA_COLLECTION_GUIDE.md` | New hardware, three-sensor procedure, storage budget. | small |

## 8. Stage 3: measuring the fixed offsets with three sensors

Use one target that all three sensors see strongly: a **corner reflector
on a pole, moved sharply** (swung, or dropped through the beam). Record
30 s of repeated, varied motion, as the guide does today.

**Motion signals:**

- radar: frame-to-frame raw difference (as today)
- camera: frame-to-frame pixel difference in the target region
- LiDAR: point-count / range change in the target's sector

**Result:**

- radar−camera offset and radar−LiDAR offset, each from cross-correlation
- repeated 2–3 times, results agreeing within a few ms
- stored per campaign, applied by `align_session.py`

Re-measure only when the frame rate, cabling, mounting or firmware changes.

**Implemented 2026-10-09 (`measure_sync_offset.py`).** The LiDAR is the
reference, because of its per-point times and tight clock fit. The
calibration target is a flat metal plate held square to the radar. The
original window was 11.5–14 m because r075m's HPF suppressed returns inside
10.5 m. **With r075m_v2(_8fps) the floor is 2.62 m, so walk about 4–14 m.**
`--range-min` now defaults to the session firmware's floor.

- **Radar ↔ LiDAR, from the target's range trajectory.** Radar: range FFT
  per frame, static background removed, I/Q order detected from the data.
  LiDAR: median range of the target's points, at their per-point times.
  The fit allows a constant range bias.
- **Camera ↔ LiDAR, from lateral motion.** Camera: SVO2 replayed, moving
  pixels against a median background, horizontal centroid converted to an
  angle with the intrinsics. LiDAR: the target's azimuth. The fit allows
  scale and bias.
- **Search and cost:** a 2 ms grid over ±1 s, using a trimmed-RMS cost so
  clutter locks do not dominate.
- **Uncertainty:** a bootstrap gives each session's uncertainty, and an
  estimate is rejected above 20 ms (radar) or 10 ms (camera), or if the
  radar-LiDAR range bias exceeds 1 m.
- **Combining sessions:** inverse-variance weighted mean.
- **Cross-check:** plate flips (radar amplitude vs camera motion energy).
- **Precision, validated on synthetic tracks with known offsets** (4 fps
  radar, 8 cm range noise, 5% clutter): radar **±6–9 ms per session** (true
  value within 0.4–1.7 σ), camera ±1–2 ms. Three sessions bring the radar to
  about ±4–5 ms. 8 fps radar and brisker walking both improve it.
- **Not yet validated on a real calibration recording.** The tool refuses
  to estimate when the target is absent (checked on a bench session).
- **First outdoor attempt (`stage3_01`, night of 2026-10-09):** no offsets.
  - The radar data had the stale-word defect.
  - The camera could not see the target at night.
  - The LiDAR tracked static scenery until voxel background removal was
    added; it now gives ground truth for 198 of 239 frames
    (`sync/lidar_target_truth.csv`).
- **RESULT, 2026-10-11 (3 daylight sessions `/mnt/sdcard/sessions/stage3_01..03`,
  90 s each, r075m_v2_8fps):**
  - **radar +12.65 ± 1.45 ms** (sessions 11.3 / 14.4 / 12.4);
  - **camera +23.34 ± 3.28 ms** (21.7 / 30.6 / 22.5);
  - LiDAR is the reference. Offsets are added to each sensor's time.
  - The values are stored in `calibration/stage3_offsets.json` (with `valid_for`).
    `align_session.py` applies them by default and warns on a firmware
    mismatch.
  - **Lessons:**
    - The LiDAR range must track the **plate surface** (10th percentile of
      the target's point ranges, now the default). The median of body+plate
      gave a walking-direction-dependent bias that looked like an 11 ms
      offset spread.
    - The radar extraction removes static clutter per frame (mean over the 64
      loops).
    - The camera is searched only in the LiDAR's azimuth band, largest blob.
    - Slow, narrow walks gave wrong offsets (+73/+108 ms). The routine needs
      a brisk walk from 4 to 14 m and side-steps of 2–3 m each way (the web
      guide scales the phases to the duration; 90 s used).
    - The LiDAR was found rotated 57° once; check it faces forward before
      calibration.
- **Plan (as executed):** a daylight session with `r075m_v2_8fps`, the final storage
  setup and the frozen configuration. Use 3 sessions; accept when they
  agree and radar σ < ~5 ms, camera σ < ~3 ms.
  - Night camera offsets are not used: long exposures shift camera timing
    by up to ~15 ms.
  - The offsets hold while the timing chain is unchanged: firmware period,
    25 µs DCA delay, camera mode, LiDAR config.

## 9. Storage and system prerequisites

| Item | Value |
|---|---|
| Data rate (measured at 8 fps) | radar 90.6 + ZED 3.9 + LiDAR 17.5 = **about 112 GB per hour**, about 31 MB/s |
| Storage (decided 2026-10-10) | **The team's 1 TB SanDisk SD card** (U3/V30/A2), **ext4**. Label `sensor_data`, UUID-based `/etc/fstab` entry, mounted at **`/mnt/sdcard`**, options `noatime,data=writeback,nofail,x-systemd.automount`. Re-inserting it needs no commands; never remove it while recording. Holds about 8 h. |
| Camera on the eMMC | Any disk shared with the radar's 25 MB/s stream stalls the ZED's writes, causing frame drops. Measured: all on SD 5.44% → SD with writeback tuning 2.36% → **camera staged on eMMC 0.18% (PASS)**. The test sessions were deleted 2026-10-11 after the results were recorded here. The eMMC holds about 4 h of camera. NVMe remains an option if longer sessions are needed. |
| Writeback | `vm.dirty_background_bytes=64 MB`, `vm.dirty_bytes=1 GB` (smooth writeback; persist in `/etc/sysctl.d/90-sensor-capture.conf`) |
| `net.core.rmem_max` | 32 MB, persistent (radar and LiDAR sockets) |
| `jetson_clocks` | **Not used in the field** (heat). MAXN with DVFS is enough. Board temperature is logged and checked by the launcher. |
| Network | radar 192.168.33.30 ("Wired connection 2"), LiDAR 192.168.34.30 ("lidar-usb-eth"). **Both profiles are bound by MAC address, not interface name**, because `eth0`/`eth1` swap depending on whether the USB adapter is present at boot. "Wired connection 1" (DHCP) has autoconnect off. |

## 10. Decisions and open questions

**Resolved 2026-10-09:**

1. **LiDAR mounting:** faces straight ahead, the same as the radar. That
   is enough to start. It is a config default, confirmed once with an
   object straight ahead, and refined by calibration later (section 3.3).
2. **Radar chirp timing:** taken from
   `cascaded_radar_capture_pipeline/firmware/r075m/cascade_r075m.cfg`.
   The active time is 13.44 ms per frame (section 3.1).
3. **Radar session length:** the radar streams indefinitely until the
   session is stopped (firmware build `build_r075m_inf`).
4. **LiDAR timing error in pairs.csv:** keep it, and report it per pair
   (section 4).

**Open:**

5. ~~Frame rate vs firmware.~~ **Resolved 2026-10-10:** `r075m_v2_8fps`
   is flashed and runs at 8.000 fps. All variants are rebuilt as `_v2`
   (4 fps) and `_v2_8fps`; only the r075m pair is proven on hardware so far.
6. **Antenna calibration (owned by the end-user team, decided 2026-10-11).**
   The team will do all sensor calibration (radar antenna and the
   three-sensor extrinsics) for annotation and research. It does not affect
   recording or Stage 3. Notes for them: calibrate with the data's firmware
   (r075m_v2_8fps); keep the mounts rigid; re-record the extrinsic
   calibration if a sensor moves. Original note: antenna calibration must be
   redone with HPF 0/0 images before angle
   estimates are trusted: r017m_calib_v2 for 77 GHz, a corner reflector with
   a 76 GHz v2 image for the others. The TX/RX → azimuth table is
   unverified. Stage 3 (range only) is not affected.
7. **Radar point cloud in the replay. DONE 2026-10-13** (see
   `replay/README.md`).
   - Team definitions: raw = before any filtering (all range–azimuth cells,
     live on GPU, z = 0); filtered = after CFAR and the existing pipeline
     filters (their `process_frame`, unchanged, pre-computed by
     `radar_cloud.py` into `sync/radar_cloud_filtered.npz`, `.npz` only).
   - Checked on stage3_02: range error −0.07 m; lateral 0.54 m with mirror
     correction vs 0.76 m without.
   - Original request notes:
   - The team's `process_frame` already produces CFAR detections as
     `xyz_cloud`, but it runs at ~1 s/frame on the CPU (no CuPy).
   - Options:
     - (A) pre-compute it per session;
     - (B) GPU CFAR + angle in the replay's RadarProcessor, real time.
   - Open questions to the team: export or view only; what "raw" vs
     "filtered" means; whether their `process_frame` sign/coordinate
     conventions should change. See `NOTES_FOR_TEAM.md`.

## 11. Robustness and validation added 2026-10-11

### 11.1 DCA1000 throttle / backlog

Seen in `freewalk_ref_01` (kept as the fault test case):
- The DCA1000 sent a steady ~2,970 pkt/s, about 17% of the expected rate,
  with no idle gaps.
- Its internal buffer filled after ~102 s; it then stopped capturing and
  drained the buffer at full speed.
- The radar data itself was intact. The cause inside the DCA is unknown; a
  power cycle of radar + DCA1000 clears it.

Handling added:

- **Launcher watchdog** (`capture_session.py`): stops the session when 5
  consecutive seconds are below 50% of the expected radar packet rate
  (expected rate from the firmware's frame period). The stop reason is
  "radar rate low … power-cycle the radar + DCA1000", and the web page shows
  a matching HINT. It would have stopped `freewalk_ref_01` at 7 s.
- **`check_session.py`** fails a session whose radar transmit bursts are
  fewer than half its frames ("DCA1000 streamed with NO idle gap"), instead
  of crashing.
- **`align_session.py`** falls back to **nominal radar times** (first frame
  + k × frame period) when arrival times are not a line, and records
  `radar.timing_source` in `alignment.json`. On `freewalk_ref_01`, frames
  arrived up to ~85 s late.

### 11.2 Timing consistency on ordinary sessions

The question: does the radar offset still hold after a power cycle? A
first plate check on the free-walk session suggested −23 ms.

- **Range-based checks are geometry-biased on free walking.**
  - Per 30 s window, the Stage 3 sessions themselves give about +7 ± 10 ms.
  - The free walk gives −7 to −18 ms depending on which LiDAR surface is
    used, because each sensor sees a different part of the body.
- **The velocity-based check is geometry-free:** radar Doppler at the LiDAR
  target range against the LiDAR range rate (scale 0.93–0.99).
  - Stage 3 baseline: +7 ± 5 ms.
  - Free walk: −4 ± 8 ms.
  - Plate segment: +0.3 ± 10 ms.
- **Conclusion:** no significant shift; the Stage 3 offsets stand.
- Tools: `timing_consistency.py` (range, per window),
  `analysis_velocity_timing_experiment.py` (velocity). The session report
  runs the velocity check on every session.

### 11.3 Session report (`make_report.py`)

- **Output:** one self-contained HTML per session
  (`<session>/session_report.html`, ~3.5 MB). It is descriptive, with no
  pass/fail verdict (requested by the team).
- **Process:** aligns the session first if needed, then reads each sensor
  once:
  - LiDAR first; its target track drives the radar Doppler-at-target and the
    snapshot moments;
  - then the radar, on the GPU;
  - then the camera.
- **Contents:**
  - overview and factual observations;
  - timeline;
  - synchronisation: intervals, pairing errors, velocity timing check;
  - radar: range–time, micro-Doppler, raw ADC statistics, channel levels;
  - LiDAR: bird's-eye view, occupancy by direction, statistics;
  - camera: thumbnails, exposure, drops, IMU;
  - paired snapshots.
- **Cost:** about 1/3 of the recording time (13 min for 10 min). The data is
  cached in `sync/report_data.pkl` and `--reuse` re-renders in seconds. It
  also writes `sync/timing_lidar_track.npz`, which the replay uses for its
  target overlay.
- **Tests:** `tests/test_session_report.py`, 96 passing on 5 sessions:
  - builds on every session, including the DCA fault;
  - every figure is a valid PNG;
  - the page is self-contained;
  - its numbers match `check_session.txt` and `alignment.json`.
- Package: `report/` (`collect.py`, `figures.py`, `render.py`). The palette
  is shared with the replay: radar #2a78d6, LiDAR #eb6834, camera #1baf7a.

### 11.4 Reference sessions (SD card)

| Session | Use |
|---|---|
| `freewalk_ref_02` | **Reference**: 10 min, scripted segments (empty scene, left/right check, free walk, close and far range, standing, two people, plate check) |
| `freewalk_ref_01` | DCA1000 throttle fault (first 103 s of radar valid, nominal times) |
| `stage3_01..03` | Stage 3 calibration (the offsets in `calibration/stage3_offsets.json`) |

### 11.5 Replay additions

- One range selector for both bird's-eye views.
- Noise-floor-relative radar colour scales.
- LiDAR target overlay.
- Clickable event strip.
- Radar azimuth conversion: the team's azimuth is right-positive; ours is
  left-positive. This was first described as a "mirror correction"; it is a
  convention, settled 2026-10-24.
- 3D LiDAR window (PyOpenGL 3.1.10 added to `rel_1_py_3_8`; it changed
  nothing else). Since 2026-10-24 it is a panel of the main window.

### 11.6 Detection and tracking (2026-10-24)

`perception/` holds our GPU radar and LiDAR detectors and a shared tracker,
live in the replay and offline in `run_perception.py`. Both use the aligned
times (`t_ref`), so radar and LiDAR objects can be compared directly. The
team's chain is unchanged. See `perception/README.md`.

See `replay/README.md`.
