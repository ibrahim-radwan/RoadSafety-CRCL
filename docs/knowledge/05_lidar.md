# LiDAR (RoboSense RS-Fairy)

## Capture

- `lidar_standalone_capture/record_lidar_raw.py` records raw UDP packets to
  `lidar.pcap`, stamped with nanosecond CLOCK_MONOTONIC times at receipt. It
  also writes `stats.csv`, `clock_map.csv` (monotonic/realtime pairs) and
  `metadata.json`.
  - **Why raw:** decoded points would be ~27.6 MB/s, against ~4.7 MB/s raw.
- `check_lidar_session.py` checks a session at packet level, and by decoding
  it.

## Packet facts (verified on hardware)

- **MSOP:** 1248 B at ~3,585/s.
  - 16-bit counter at bytes 12–15 (wraps at 65535).
  - LiDAR timestamp at byte 20: 6 B seconds + 4 B microseconds, big-endian.
- **IMU:** 51 B at 200/s. **DIFOP:** ~1/s.
- **Clock:** the LiDAR clock drifts ~15–70 ppm against the host. Alignment fits
  it per session (lower envelope).
- **Fixed 138,240-point grid per sweep.** No-return directions come back as
  **zero-range points**; filter `range > 0`. About 92k returns per sweep are
  valid. About 44% of valid returns are within 1 m on the bench rig (mount and
  surroundings).
- Every 10th sweep has 132,288 points with no loss; that's firmware behaviour.

## Driver and Python wrapper

- **Driver:** `lidar_standalone_capture/rs_driver/` is RoboSense's open-source
  rs_driver (upstream commit 897b14d), with our changes. See its `NOTICE.md`.
- **Our wrapper:** `rs_driver/python_wrapper_2/` (pybind11).
  `robosense_lidar.cpp` and `lidar.py` changes:
  - double-precision timestamps (were truncated to whole seconds);
  - `stop()` releases the GIL and calls `driver.stop()` (deadlock fix);
  - `start()` releases the GIL (needed for the FIFO-fed replay);
  - pcap playback: `pcap_path`, `pcap_rate`, `input_finished()`;
  - PointXYZIRT with a **6th column = per-point time offset** from the cloud
    timestamp;
  - optional `split_angle`.
- **Config:** `rs_driver/config/config.yaml`:
  - lidar 192.168.34.200 → host 192.168.34.30;
  - **`split_angle: 180`**, so sweeps split behind and the forward field of
    view is never cut in two;
  - `ts_first_point: true`.
- **Decoding speed:** decode pcaps at **≤ 3×** real time. Faster than 4×, the
  driver drops data.
- **Build:** see `docs/NEW_BOARD_SETUP.md`. The end user's original prebuilt
  `.so` is kept as `.so.bak`.

## Mounting

- **Facing forward, like the radar.** Alignment assumes rpy = 0 for its
  field-of-view selection.
- **It was once found turned 57°** after the sensors were reconnected; Stage 3
  found it. Check orientation before calibration or collection, for example in
  the replay: a person straight ahead should be at ~0° azimuth.

## Object detection (ours, 2026-10-24)

`sync_capture_pipeline/perception/lidar.py`, live in the replay and offline in
`run_perception.py`. See `perception/README.md`.

- **Local ground, not one plane.** A single RANSAC plane failed on
  `freewalk_ref_02`: a carport with a driveway sloping ~13° down. The plane
  tilted 9°, and the roof plus the sloped ground merged into one 13 × 8 m
  "object".
  - Now: the lowest return per 1 m cell, with a lifted-cell rule (> 0.35 m
    above its 3×3 minimum).
- **Overhead structures removed:** a 0.2 m column is an obstacle only if
  something in it reaches down to within 1 m of the ground. This drops roofs,
  canopies and branches.
- **Density-based clustering (2026-10-26).** Objects grow through dense
  cells only; a sparse cell cannot connect two objects.
  - Before, single stray points bridged objects. In a cluttered office
    (`run_20261014_1325`/`1326`), a chain of one-point cells at a depth edge
    (mixed pixels) merged the walker into a 7 m object with the walls and
    desks. The walker was found in only 27–47% of sweeps there, against
    84–94% now.
  - The core threshold scales with range⁻², because point density falls with
    distance.
- On `freewalk_ref_02`:
  - the walker is detected in 99% of sweeps and tracked in 99%;
  - it is classed `pedestrian` in 85.5% (`other` in most of the rest);
  - about 37 objects per sweep (walls, bins, bikes, the parked car);
  - ~23 ms per sweep on the GPU.
- Size classes are heuristics. Walls with things against them can be boxed as
  "vehicle".
