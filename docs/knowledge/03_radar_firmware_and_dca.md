# Radar firmware and the DCA1000

The details are in `cascaded_radar_capture_pipeline/firmware/FIRMWARE_STATUS.md`.
This page is the background an agent needs before touching radar code.

## Key facts

- **The `.cfg` file is never sent to the device.** The firmware in QSPI holds
  the waveform; the `.cfg` only describes it for processing. A wrong `.cfg` is
  silent, so every tool resolves it through `sync_capture_pipeline/radar_firmware.py`:
  - `~/.radar_flashed` records what is in flash; it is written by
    `flashing/boot/flash_variant.sh`;
  - the launcher copies that record into `session.json["radar_firmware"]`;
  - the `.cfg` path comes from `firmware_map.json`.
- **Frame:** 3,145,728 B = 64 loops × 6 TX (TDM) × 2 devices × 4 RX × 256
  samples × int16 (I, Q), little-endian, **I first**. Chirp order:
  chirp = loop × 6 + tx.
- **In flash:** `r075m_v2_8fps`. Range resolution 0.293 m, window 2.62–75 m,
  125 ms frames, v_max ±4.7 m/s (faster targets wrap).

## The v2 fix (2026-10-10)

- **The bug:** the old images left the AM273x CSI-2 ping/pong buffers in
  DSS-L3 RAM uninitialised at QSPI boot, so about 50% of every chirp streamed
  power-up garbage (32-byte blocks, identical in every frame).
  - It was misdiagnosed for a while as "compression"; the firmware has no
    compression.
  - CCS loading hid the bug, because its GEL script initialises that RAM.
- **The fix:** `SOC_rcmStartMemInitDSSL3`/`Wait` at boot. Built by an agent on
  the Windows build machine.
- **Also changed:** HPF 0/0 for every variant, which moves the r075m near floor
  from 10.5 m to 2.62 m.
- **Variants:** 20 in total. Only the r075m pair is hardware-proven.
- **Acceptance check after flashing:**
  1. 30 s radar-only capture;
  2. `check_radar_frames.py`: 0.00% "identical in every frame";
  3. power-cycle and repeat once.

## DCA1000

- **Packet delay:** the `CONFIG_PACKET_DATA` delay is in 8 ns ticks. The old
  6250 (50 µs) makes a frame take 134.4 ms on the wire, which caps the stream
  at 7.44 fps and silently drops LVDS data at 8 fps.
  - The receiver default is **25 µs** (~80 ms per frame).
  - The receiver warns if transmit bursts are fewer than half the frames.
- **Throttle fault:** occasionally after power-up the DCA sends ~3,000 pkt/s
  (once ~900/s) instead of ~17,300.
  - It buffers the data and stops capturing after ~100 s.
  - The radar data itself is fine.
  - Cause unknown. **Fix: power-cycle the radar + DCA1000.**
  - The launcher watchdog stops a session after 5 s below 50% of the expected
    rate; `check_session` flags it; `align_session` falls back to nominal
    frame times.
- **Frame stamping:** each frame takes the time of the transmit burst that
  carried its **first** byte. A frame is 2,160.5 packets, so a burst starts
  just before a frame boundary. An earlier receiver stamped frames one burst
  late.

## Processing caveats (see `sync_capture_pipeline/NOTES_FOR_TEAM.md`)

- **The "azimuth mirror" is a frame convention, not an error** (settled
  2026-10-24).
  - The team's processing azimuth is positive to the **right**; their README
    says so ("x = lateral, positive = right of boresight").
  - The pipeline frame is y **left**, so world azimuth = − processing
    azimuth.
  - Verified against the LiDAR walker on `freewalk_ref_02`: 90% sign agreement
    beyond ±8°.
  - The earlier "mirrored" finding came from the replay reading the
    processing azimuth as left-positive.
  - The conversion lives in one place, `perception/core/conventions.py`,
    which the detector, the replay and the report use.
- **Doppler sign:** positive = receding. 99% sign agreement with the LiDAR
  range rate (corr +0.83 on freewalk_ref_02, +0.98 on stage3).
- **Weak angular focusing: phase calibration errors, measured in situ**
  (`perception/estimate_rx_phase.py`).
  - The residual phase per virtual channel against the LiDAR azimuth is stable.
  - **TX3 and TX4 (the slave AWR2243) sit ~100–125° off TX0–TX2 (the
    master):** an inter-device TX phase error in the cascade calibration.
  - There is also a smaller per-RX pattern.
  - The correction measured on the plate sessions (`stage3_02` + `03`) is
    stored in `calibration/radar_rx_phase.json`. On `freewalk_ref_02`, which
    was not used to measure it, it cuts the azimuth spread from 5.95° to
    2.43°.
  - Even so, sidelobes stay only ~5–10 dB below the peak. A proper
    calibration (a reflector over a wide angle span) belongs to the team; the
    in-situ correction is provisional.
- **The radar–LiDAR yaw differs between mountings:**
  - stage3 (2026-10-11 morning): about +3°;
  - freewalk_ref_02 (afternoon): about −0.4°;
  - run_20261014_1326: about +2° (with the per-channel correction).
  
  This is extrinsic calibration (the team's). The phase correction removes
  its beam-steering part so as not to absorb it.
- **TDM-MIMO Doppler phase:** a moving target's phase advances between the 6
  TX slots of a loop. `perception/radar.py` compensates it per Doppler bin. The
  old chain did not.
- **Stationary reflectors (2026-10-26, experimental):** `perception/radar.py`
  `process(..., static=True)` detects at zero Doppler, without clutter
  removal.
  - Within 6–10 m, 82–100% of these detections have a LiDAR object at the
    same place.
  - Farther out, many are images: azimuth sidelobes of a strong reflector at
    the same range (in `freewalk_ref_02`: 12.0 m at 0°, 12°, 18° and 32° in
    almost every frame), and multipath at 2–3× range. In the 5 m-deep office,
    detections appear at 10–20 m.
  - Beyond ~60 m the static map shows artifacts even indoors, so the static
    path stops at 60 m.
  - Better antenna calibration (lower sidelobes) is the real fix.
- **Multipath ghosts:** a walker produces images at 2× range with 1.9×
  radial velocity (radar → person → rig → person → radar), and images behind
  the walker at the same radial velocity (via walls). `perception/radar.py`
  flags both (`ghost` column).
- **Elevation is unreliable:**
  - It comes from a single 8-channel row on TX5, which is on the slave device
    (see the TX phase error above).
  - The walker reads ~+13° on freewalk_ref_02.
  - Not corrected: that needs a vertical reference.
- **Old `.cfg` files implied Q-first.** Corrected: `adcbufCfg -1 0 0 1 1`.
- **Processing is `sync_capture_pipeline/perception/`** (PyTorch, GPU).
  The old radar chain (`process_frame`) was merged into it and deleted on
  2026-10-25. `perception/radar_chain.py` reproduces the old front end on
  either of its CFARs. Its GPU path asked for CASO-CFAR on range-Doppler;
  its CPU fallback silently ran OS. Details in
  [01](01_status_and_roadmap.md).

## Flashing (from the Jetson)

1. `conda activate rel_1_py_3_8`. Plain `python` lacks pyserial.
2. `flashing/boot/flash_variant.sh <variant> /dev/ttyACM0`.
   - It checks the md5 against the map.
   - It writes only the app slot (0xA0000), never the bootloader.
   - It records `~/.radar_flashed`.
3. `uart_uniflash.py` can exit 0 on error, so **read its output**.
4. Set the boot switch: UART to flash, QSPI to run. Then power-cycle.
