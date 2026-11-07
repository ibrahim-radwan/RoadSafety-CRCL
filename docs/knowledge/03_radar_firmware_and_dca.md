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
- **Weak angular focusing (history, team layout): phase errors measured in
  situ** (`perception/estimate_rx_phase.py`). Superseded on 2026-11-07 by
  the measured array layout (see "Off-axis azimuth" below): these "phase
  errors" were mostly the table's wrong positions.
  - The residual phase per virtual channel against the LiDAR azimuth is stable.
  - **TX3 and TX4 (the slave AWR2243) sit ~100–125° off TX0–TX2 (the
    master).** This was read as an inter-device TX phase error in the
    cascade calibration; it was the table's wrong TX positions.
  - There is also a smaller per-RX pattern.
  - The correction measured on the plate sessions (`stage3_02` + `03`) is
    stored in `calibration/radar_rx_phase.json`. On `freewalk_ref_02`, which
    was not used to measure it, it cuts the azimuth spread from 5.95° to
    2.43°.
  - Even so, sidelobes stayed only ~5–10 dB below the peak: the patch
    only held near boresight.
- **The radar–LiDAR yaw differs between mountings:**
  - stage3 (2026-10-11 morning): about +3°;
  - freewalk_ref_02 (afternoon): about −0.4°;
  - run_20261014_1326: about +2° (with the per-channel correction);
    with the measured layout the radar−LiDAR offsets are −0.1°
    (stage3_01), −3.0° (freewalk_ref_02), +4.7° (run_20261014_1326).
  - **run_20261014_1325 is a set-up session:** the LiDAR is turned ~20°
    against the radar and the camera. Checked on the camera images
    (2026-11-07): where the camera was mounted (the last ~10 s), the
    radar's azimuth falls on the walker and the LiDAR's ~20° right of
    them. Earlier in the session the camera lies on a desk and is
    handled. Its LiDAR-referenced radar scores are not meaningful
    (`baseline.py` notes it).
  - **2026-11-03 rig (team's corner-reflector sessions): the LiDAR is
    yawed +37.2° from the camera** (camera-referenced, 2026-11-07; 39.8°
    against the nominal positions). The radar boresight reads 2.7° right
    of the camera. See `NOTES_FOR_TEAM.md`.
- **Off-axis azimuth: our array table was wrong; fixed with a measured
  layout (2026-11-07).** Not the antenna calibration.
  - **What was seen** (`calibration_dataset` 2026-11-03, camera reference, 9
    positions at 2–5 m): with the team table, the `.cfg` calibration and the
    in-situ phase, a constant −2.7°, 7 of 9 positions within ~1°, but the
    peak 15° off at +9° (another lobe). `.cfg` alone: 8° mean error, up to
    15°.
  - **Why it was not the calibration** (October plate sessions, LiDAR
    reference, raw channels):
    - the `.cfg` `antennaCalibParams` barely matter (coherence 0.85 with
      the phases the data needs, 0.82 with none; the team saw no difference
      from an intrinsic calibration either);
    - the needed correction depended on the target's angle (left vs right
      0.57–0.60, same-side control 0.99), which no per-channel correction
      can fix.
  - **The layout, measured** (`perception/estimate_array_layout.py`): each
    channel's phase against sin(azimuth) gives its position; both chips
    agree, on the plate sessions and `freewalk_ref_02`:

    | chirp | team table (λ/2) | measured, within a chip |
    |---|---|---|
    | 0, 1, 2 | 0, 4, 7 | 0, −2, −4 |
    | 3, 4 | 11, 15 | **elevated** (phase follows the target's height) |
    | 5 | elevation TX | azimuth, +3 |
    | RX 0–3 / 4–7 | 0–3 / 19–22 | the same |
    | time slot | chirp index | the same (TDM compensation right) |

    So each chip's virtual channels form one contiguous group of 11
    positions, and the two groups are 19 apart (a gap of 8).
  - **The two chips' phase:** a constant plus a ramp in range, +0.356 rad/m
    and −11° on `r075m_v2_8fps` (the slave samples ~0.4 µs late; fitted on
    `stage3_02` + `03`). Its frame-to-frame scatter on walkers shrinks with
    range: an extended target's speckle over the 19-position baseline, not
    the hardware.
  - **Processing** (`perception/radar.py`, `layout="measured"`, the default;
    `calibration/radar_array.json`):
    - no `.cfg` calibration and no in-situ phase (both made it worse);
    - the slave's phase removed after the range FFT;
    - each chip's group beamformed (Hann over 11), the two powers added:
      the map's sidelobes are ~20 dB down (an FFT across the gap gives
      fringes every ~6°, only 1–3 dB down);
    - each detection's angle refined coherently across both chips: the
      fringe nearest the incoherent peak, if it beats the others within the
      main lobe by 1 dB (`refine_gate_db`); `refine_azimuth` does the same
      for one static reflector (calibration tools);
    - elevation is 0 for now: the elevated TX positions are not measured.
  - **Result** (nothing fitted on the test sessions):

    | test | in use before | measured layout |
    |---|---|---|
    | 2026-11-03 corner reflector − camera, 9 positions 2–5 m (`calib_reflector.py`) | −2.7°, outliers +9.8°, −16.3°, +3.5° | −2.4° ± 0.37°, none > 1° off |
    | `stage3_01` vs LiDAR: spread / > 5° off | 1.5° / 20% | 1.1° / 4% |
    | `freewalk_ref_02` | 2.7° / 36% | 1.8° / 12% |
    | `freewalk_ref_01` | 9.3° / 75% | 2.8° / 29% |
    | `run_20261014_1326` | 7.4° / 57% | 1.8° / 16% |

    Baseline re-made with the measured layout on 2026-11-07
    (`perception/baseline/scores.json`, `golden_ours.npz`). Against the
    old baseline: radar detected (within 10° of the LiDAR target) +0.08 to
    +0.32 on every session except `run_20261014_1325` (LiDAR turned, see
    above); azimuth spread 2.4° → 2.0° (freewalk_ref_02), 11.7° → 2.6°
    (freewalk_ref_01), 8.1° → 1.6° (run_20261014_1326).

    At 8 m on 2026-11-03 the camera detection fails; the radar there
    matches the nominal geometry within ~2° after the same offset.
  - **The constant offset** (radar vs camera −2.4°, vs LiDAR per mounting)
    is the mounting yaw: extrinsic, the team's.
  - **`layout="team"`** keeps the old table, the `.cfg` calibration and the
    in-situ phase (`radar_rx_phase.json`) for comparisons and for
    `radar_chain.py` (the old chain's reproduction). The replay has
    `--radar-layout team`.
  - **Open:** the elevated TX positions (elevation); the inter-chip phase for
    the other firmware variants (unmeasured: those combine the chips
    incoherently only); the board's drawing, as an independent check.
  - **Near field is not a factor.** The virtual aperture is 5.7 cm; peaks
    move < 0.3° at 2–5 m.
- **Coherent background subtraction works across sessions** (zero-Doppler
  channels, complex, before beamforming;
  `RadarDetector.static_band` / `static_map_of`). On 2026-11-03 the static
  scene left 6–15% after subtracting the `background` session's mean, so a
  reflector stands alone without the walls' sidelobes. The live view's
  calibration capture and `calib_reflector.py` use it.
  
  This is extrinsic calibration (the team's). The phase correction removes
  its beam-steering part so as not to absorb it.
- **TDM-MIMO Doppler phase:** a moving target's phase advances between the 6
  TX slots of a loop. `perception/radar.py` compensates it per Doppler bin. The
  old chain did not.
- **Stationary reflectors (2026-10-26, experimental):** `perception/radar.py`
  `process(..., static=True)` detects at zero Doppler, without clutter
  removal.
  - One frame's detections are single cells. A reflector near the
    thresholds blinks: on AR1 it is in 14% of frames, in 37 separate runs.
    `detect_static_map` runs the same detector on any map. The replay's
    `S1..S3` rows use the mean power of the last 2 s of maps, interpolated
    between cells (2026-11-07).
  - Within 6–10 m, 82–100% of these detections have a LiDAR object at the
    same place.
  - Farther out, many are images: azimuth sidelobes of a strong reflector at
    the same range (in `freewalk_ref_02`: 12.0 m at 0°, 12°, 18° and 32° in
    almost every frame), and multipath at 2–3× range. In the 5 m-deep office,
    detections appear at 10–20 m.
  - Beyond ~60 m the static map shows artifacts even indoors, so the static
    path stops at 60 m.
  - These numbers are from the team table. The measured layout's map
    sidelobes are ~20 dB down ("Off-axis azimuth" above), so the sidelobe
    images should drop; on one stage3_01 frame the static detections went
    from 33 to 11. Not yet re-scored against the LiDAR.
- **Multipath ghosts:** a walker produces images at 2× range with 1.9×
  radial velocity (radar → person → rig → person → radar), and images behind
  the walker at the same radial velocity (via walls). `perception/radar.py`
  flags both (`ghost` column).
- **Elevation is not available (2026-11-07):**
  - The team table took chirp 5 as the elevation TX. It is an azimuth TX;
    chirps 3 and 4 are the elevated ones. That explains the walker's
    ~+13° on freewalk_ref_02.
  - With the measured layout el = 0 (z = 0) until the elevated TX
    positions are measured (a reflector at known heights, e.g. the LiDAR's
    z of the 2026-11-03 reflector).
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
