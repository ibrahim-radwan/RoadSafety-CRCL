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
- **Its processing delay is removed by `altos_ms` = −64 ms** (measured
  2026-11-27, in `calibration/stage3_offsets.json`). Its times are frame
  arrivals, which are 64 ms late; the spec says < 60 ms for processing alone.
  - **Session:** `offset-session` (recorded as `run_20261126_1225`), 2.2 min,
    someone walking towards and away from the rig between 2.6 and 11.9 m.
  - **Method:** a whole-session range lag against the LiDAR target track,
    r_altos(t) = r_lidar(t + τ) + c. The Altos range is the median of its
    moving points within 1 m and 25° of the LiDAR target. Each straight
    leg informs τ, because the range shift flips sign between approaching
    and receding legs, and c absorbs the constant range difference
    (+0.20 m: body centre vs the LiDAR's nearest surface, plus the mounting).
  - **Result:** τ = −64 ms, block-bootstrap sd 3.3 ms; −56 to −66 ms across
    gates and the two halves; residual scatter 5 cm. With the offset applied:
    −4 ± 3 ms.
  - **Control:** the TI radar's own detections by the same method gave
    +2 ± 13 ms with its offset applied, so the method and the LiDAR reference
    are unbiased.
  - **The 30 s Doppler windows failed on this walk** (σ 45–100 ms). They
    also gave a spurious +178 ms over the whole session. With steady legs at
    ±1.3 m/s, a Doppler fit learns timing only at the turns, and the turns
    are where the tracks are worst (the LiDAR loses the walker below 3.8 m;
    the moving-point filter drops a standing person).
  - **The report** now gives this range-based residual too (`range_timing`
    in `report/collect.py`); it should stay near 0.
  - **Valid for** profile 150m_1, threshold 0, firmware 8e75cfeb. Re-measure
    if those change, and confirm on a second walking session when there is
    one.
- **Pairing:** `pairs.csv` gives each TI radar frame the nearest Altos
  frame (`altos_frame`, `t_altos`, `altos_error_ms`). The error stays within
  ±33.3 ms (half an Altos frame), plus any gap. Flags: `outside_altos`,
  `altos_gap`, `altos_incomplete`.
