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
    (radar field of view) as the sweep's time.
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
