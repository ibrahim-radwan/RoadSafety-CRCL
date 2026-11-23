# Camera (Stereolabs ZED 2i)

- **Recorder:** `camera_standalone_capture/ZED_2i_Sensor/recorder.py`.
  - Records `camera.svo2` (H264_LOSSLESS via NVENC), IMU CSVs and the
    calibration.
  - Logs `host_receipt_mono_ns` (CLOCK_MONOTONIC) for every frame and IMU row.
  - The SDK's SYSTEM_CLOCK timestamps are converted per frame.
  - Takes `--session-dir`.
- **SDK:** ZED SDK **4.1.4**, with the matching `pyzed` for Python 3.8. Use the
  same SDK version to record and to extract.
- **`config.yaml`** (approved 2026-10-08):
  - `auto_export: false`: **no PNG export at capture**, extraction is offline;
  - `sensors.poll_rate_hz 200`;
  - magnetometer, barometer and temperature off.
- **Verification (2026-11-23):** at stop the recorder checks only the SVO2's
  frame count (`recording.verify_after_stop: false`). Decoding every frame
  took ~0.6× the recording time and held up the next session; it is now
  done offline by **`verify_svo.py --session <session>`**, which writes
  `camera/svo_verification.json`. The full `check_session.py` requires it;
  the session notes (`--quick`) only say it is still to run. It also works on
  a camera that stopped early (e.g. a full disk).
- **Data rate:** 2–4 GB/h indoors, 16.6 GB/h on the road (lossless video
  grows with the scene). See [04](04_storage_and_recording.md).
- **Frame drops:** typically 0.1–0.5%, mostly single frames, listed in
  `capture/timing/camera_gaps.csv`; the same indoors and on the road
  (2026-11-23, 11 sessions). The session notes summarise them (count,
  longest gap and when, per minute).
  - **Cause:** write stalls on any disk shared with the radar, hence the eMMC
    staging (see [04](04_storage_and_recording.md)).
  - **Ruled out:** CPU load, USB-hub sharing with the LiDAR, and the keyboard
    hub.
- **Decode speed:** lossless H.264 decodes at only ~40–50 frames/s on the Orin.
  - The replay camera panel can drop to 15–25 updates/s while radar and LiDAR
    also run (sync is unaffected).
  - Offline extraction is slower than real time under load.
- **Timing:** the camera offset is +23.34 ± 3.28 ms against the LiDAR
  (Stage 3). Night sessions use long exposures that can shift camera timing by
  up to ~15 ms; Stage 3 was done in daylight.
- **`pyzed.grab()` holds the Python GIL.** In the replay the camera runs in its
  own process for that reason.
- **pyzed wheel:** `pyzed-4.1-cp38-cp38-linux_aarch64.whl` is **not in git**.
  It is installed with the SDK's `get_python_api.py` (see
  `docs/NEW_BOARD_SETUP.md`).
