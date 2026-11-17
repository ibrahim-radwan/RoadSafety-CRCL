# Synchronisation and Stage 3

The full design is in `sync_capture_pipeline/SYNC_DESIGN_3_SENSORS.md`. The
essentials:

## Principles

- **One reference clock:** CLOCK_MONOTONIC on the Jetson. Everything is
  recorded raw with receipt times and aligned **offline** (`align_session.py`).
- **Radar frames are the pairing anchor.** `sync/pairs.csv` has one row per
  radar frame, with its nearest camera frame and LiDAR sweep, the time error of
  each, and flags (`camera_gap`, `lidar_gap`).
- **Per-sensor time models:**
  - radar: a line fitted through the transmit-burst times (lower envelope;
    nominal-period fallback on a DCA backlog);
  - camera: per-frame conversion via `host_receipt_mono_ns`;
  - LiDAR: packet-clock fit plus per-point times, with the boresight points
    (radar field of view) as the sweep's time. The boresight is the radar's,
    in the vehicle frame (the session's LiDAR extrinsic, since 2026-11-06).
    On the 2026-11-03 mounting (LiDAR yawed 40°), the old LiDAR-forward cone
    was scanned ~11 ms away from the radar boresight. The Stage 3 offsets
    were measured on the old mounting with identity, so they are unaffected.
- **Stage 3 offsets** are measured once per setup and added to each sensor's
  time.

## Stage 3 result (2026-10-11, 3 daylight sessions, r075m_v2_8fps)

| | Offset (added) | Per session |
|---|---|---|
| Radar | **+12.65 ± 1.45 ms** | 11.3 / 14.4 / 12.4 |
| Camera | **+23.34 ± 3.28 ms** | 21.7 / 30.6 / 22.5 |
| LiDAR | reference | |

- **Stored in** `sync_capture_pipeline/calibration/stage3_offsets.json`, with
  `valid_for`.
- **Applied by default** in `align_session.py`, which warns on a firmware
  mismatch.
- **Valid while** these stay unchanged: firmware and frame period, the 25 µs
  DCA delay, camera mode, LiDAR config.

## Method (`measure_sync_offset.py`)

- **Target:** a flat metal plate, held square to the radar.
- **Routine** (the web page's opt-in Stage 3 guide):
  1. brisk walk 4 ↔ 14 m;
  2. side-steps 2–3 m each way at ~8 m;
  3. flips.
- **Radar ↔ LiDAR:** range trajectories.
  - Radar: moving-only range profile (mean over the 64 loops removed per frame).
  - LiDAR: target cluster after voxel background removal, at the **10th range
    percentile**, i.e. the plate surface.
- **Camera ↔ LiDAR:** lateral angle. The largest moving blob, searched only
  inside the LiDAR target's azimuth band.
- **Uncertainty:** bootstrap; sessions are combined by inverse variance.

## Lessons (each cost a session)

- **Slow or narrow walks** gave wrong offsets (+73 and +108 ms). Walk briskly,
  use the full range, and make the side-steps wide.
- **LiDAR median range** (body + plate) gave a walking-direction-dependent bias
  that looked like an 11 ms spread. Use the plate surface (10th percentile).
- **Night:** the camera can't see the target. Daylight only.

## Checking timing on ordinary sessions

- **Range-based checks are geometry-biased** on free walking: ±10 ms of
  systematic error.
- **The velocity method is geometry-free:** radar Doppler at the LiDAR target's
  range against the LiDAR range rate.
  - On `freewalk_ref_02` it gave −4 ± 8 ms, so there is **no shift after a
    power cycle**.
  - The session report runs it for every session.
- **Residual lag, radar range vs the LiDAR target (2026-11-07):** within
  ±0.03 s on stage3_01–03, freewalk_ref_02 and run_20261014_1326;
  +0.11 s on `freewalk_ref_01`, whose radar times are the nominal-period
  fallback after a DCA1000 backlog.
- **`timing_consistency.py` wrote the shared target cache
  (`sync/timing_lidar_track.npz`) in the LiDAR's own frame** until
  2026-11-07 (no extrinsic yaw); the replay, `run_perception` and
  `baseline` read that cache in the vehicle frame. Fixed (it now uses the
  session's extrinsic yaw, like `report/collect.py`). The existing caches
  were checked: all in the vehicle frame (the 2026-11-03 ones were written
  by the report).

## The Altos V4 radar (fourth sensor, 2026-11-17)

- **Capture time = host arrival of each frame's first packet, de-jittered.**
  The radar's own timestamps can't be used without PTP. Its date is wrong
  (2025-03), and the stamps are exactly 66.7 ms apart while the real frame
  rate wanders: fitted against host time, −505, +408 and −104 ppm in three
  recordings.
- **How `align_session.py` fits it:**
  - Split at radar restarts and pauses over 1 s.
  - Fit arrival time against frame number. Frame ids are used, so lost
    frames keep their slot. The intercept sits on the lower envelope, as for
    the other sensors.
  - Add a rolling ±5 s lower envelope of the residuals, for the wander.
  - Checked on synthetic arrivals: 2 ms of jitter is cut to < 2 ms of error
    (`tests/test_altos.py`).
- **The radar's processing delay is still in its times.** The spec says
  < 60 ms. The fix is an `altos_ms` key in `calibration/stage3_offsets.json`,
  which `align_session` applies like the others. Until it exists,
  `alignment.json` says `"altos_offset_calibrated": false` and the report
  says so.
- **Measuring `altos_ms`:** the report runs the velocity check for the Altos
  too:
  - its moving points' median Doppler near the LiDAR target (±0.6 m,
    ±20°: the mounting isn't calibrated);
  - fitted against the LiDAR range rate, per 30 s window;
  - the weighted residual is what to add to `altos_ms`.
  
  It needs a session with someone walking towards and away from the rig.
  Not yet run: no 4-sensor session exists.
- **Pairing:** `pairs.csv` gives each TI radar frame the nearest Altos
  frame (`altos_frame`, `t_altos`, `altos_error_ms`). The error stays within
  ±33.3 ms (half an Altos frame), plus any gap. Flags: `outside_altos`,
  `altos_gap`, `altos_incomplete`.
