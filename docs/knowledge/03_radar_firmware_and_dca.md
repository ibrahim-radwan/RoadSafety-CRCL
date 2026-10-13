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

- **Azimuth sign is mirrored** in the processing (radar_config's TX/RX table).
  - Replay and report correct it: `RadarProcessor.MIRROR_AZIMUTH`.
  - The team's `process_frame` is unchanged.
- **Weak angular focusing:** a person appears as an arc. The antenna
  calibration was probably measured with the old HPF, and recalibration
  belongs to the team.
- **Elevation is unreliable:** it comes from a single 8-channel row.
- **Old `.cfg` files implied Q-first.** Corrected: `adcbufCfg -1 0 0 1 1`.

## Flashing (from the Jetson)

1. `conda activate rel_1_py_3_8`. Plain `python` lacks pyserial.
2. `flashing/boot/flash_variant.sh <variant> /dev/ttyACM0`.
   - It checks the md5 against the map.
   - It writes only the app slot (0xA0000), never the bootloader.
   - It records `~/.radar_flashed`.
3. `uart_uniflash.py` can exit 0 on error, so **read its output**.
4. Set the boot switch: UART to flash, QSPI to run. Then power-cycle.
