# Task: investigate the cascade radar's raw data stream and produce a corrected firmware

You are working on the **Windows machine that builds the radar firmware**. It has
the TI SDK, the cascade firmware source (including `cascade_config.h`), and the
project's original code base and documentation. A Jetson AGX Orin in the field
records the radar together with a camera and a LiDAR. The Jetson team found
problems in the radar data and needs you to find the root cause in the firmware
or capture configuration, and to build a corrected image.

**Do not flash anything without the operator's explicit go-ahead.** Report your
findings first.

---

## 1. System as it runs today

| Item | Value |
|---|---|
| Radar | 2-chip cascade (AWR2243 ×2 with AM273x MSS). 6 TX (TDM MIMO) × 8 RX = 48 virtual channels |
| Firmware | Variant **`r075m`**: `am273x_cascade_r075m.appimage`, md5 `f0e6a73d4c7d35d92e2128305f0b1867`, build dir `build_r075m_inf` |
| How firmware runs | Flashed permanently to QSPI (app at `0xA0000`). It streams at power-on, with infinite frames. The `.cfg` file is **not** sent to the device; the firmware carries its own parameters. |
| Source of truth for parameters | `cascade_config.h` (per `firmware_map.json`: "registers transcribed from cascade_config.h") |
| Capture path | Radar → DCA1000 → Ethernet → Jetson. The Jetson configures the DCA over UDP (`192.168.33.180:4096`, data on port 4098). |
| DCA1000 config sent by the Jetson | `CONFIG_FPGA` = bytes `01 01 01 02 03 1E`; `CONFIG_PACKET_DATA` = packet size 1470, delay 6250. Each data packet is a 4 B sequence number + a 6 B byte count + **1456 B payload**. |
| Expected frame | **3,145,728 B** = 256 samples × 64 loops × 48 virtual channels × 4 B (int16 I + Q). That is 2160.5 packets per frame. |
| Frame period | 250 ms (4 fps) per `firmware_map.json`; the measured rate is exactly 4.00 fps. **Target: 8 fps (125 ms).** |

`cascade_r075m.cfg` (copied with the data) records these values. **Some may not
match what the firmware really does.** Verify every one against `cascade_config.h`:

```
dfeDataOutputMode 1
channelCfg 15 7 1 15 7
adcCfg 2 1
adcbufCfg -1 0 1 0 1
profileCfg 0 76 4 3 31 0 0 19.9878 0 256 10000 2 2 42
dataPathConfig 0 1 0 2 64 64 64 0
hsiClockConfig 9 0
hsiLaneConfig 15 1 0 2 0 4 0 5 0 3 0 0
dataFormatConfig 0 1
frameCfg 0 5 64 0 256 50 1 0 2        <- the "50" does not match the 250 ms period in firmware_map.json
compressionCfg -1 1 0 0.5 8           <- compression ENABLED, method 0 (EGE), ratio 0.5, 8 range bins per block
```

The project's own config parser (`cascaded_radar_capture_pipeline/processing/radar_config.py`)
already prints this warning for this cfg:

> WARNING: compression ON — ADC data reaching LVDS is compressed. This was
> confirmed to cause amplitude non-stationarity. Disable with: `compressionCfg -1 0 0 0.5 8`

### How the Jetson decodes a frame (assumptions to confirm or correct)

- `int16` stream, little-endian.
- Per frame, in order: `[chirp 0..383][device 0..1][rx 0..3][sample 0..255][I,Q]`.
- TDM: `chirp = loop * 6 + tx`.
- **I/Q order is ambiguous in the code base:**
  - `capture/config.py` hard-codes I-first.
  - `processing/radar_config.py` derives Q-first from `adcbufCfg` (SampleSwap = 1).
- In the outdoor data, the energy split between the two halves of the range
  spectrum favours I-first (ratio 3.97). This is not proof.

---

## 2. What the Jetson team observed (evidence)

The receiver is not the cause. It placed every payload by the DCA's byte count;
loss was 0.00% by both sequence number and byte count; and every frame boundary
aligned with the DCA's transmit bursts.

### Finding A: about half of every frame is constant data (main issue)

Session `stage3_01`: outdoors, at night, 239 frames, one person walking with a
metal plate at 12–14 m. Indoor session `ipad_test_01`: radar facing a wall, 120
frames. Both show the same structure:

- **53.5%** of all int16 values are **bit-identical in every one of the 239
  frames** (50.6% indoors).
- The constant values come in **runs of 16 int16 (32 bytes)**, and up to 128.
  They are spread across all RX channels and sample positions; no single
  channel is dead.
- Constant values are large: mean |value| about **16,000 counts**. Varying
  values are small: about **127 counts** outdoors, about 19 indoors.
- Example (indoor frame 0, chirp 0, RX 0, I values):
  `-20, -31, -28, -30, -19, -19, -15, -42, -23058, -7266, 30315, -15364`.
  The first 8 are small and noisy; then large values begin.
- **Consequence:**
  - A standard range FFT over all samples shows **no change between frames at
    any range**: 0.0 dB at every range bin across 60 s, with a person walking
    at 12–14 m.
  - With the constant values zeroed, only faint, intermittent traces appear at
    12–14 m.
  - Broadband vertical streaks appear at about 7 s, 31.5 s and 40 s.
  - The radar is effectively unusable for detection and for time calibration.

### Finding B: an episode with ~9.5× smaller frames

Earlier the same evening, a session recorded at a steady **~908 packets/s**
(1.32 MB/s), against the normal ~8,650 packets/s (12.6 MB/s):

- Still 4 bursts per second, but each burst was about **330 KB** instead of
  3.1 MB.
- No packet loss.
- Every frame was misaligned against the 3,145,728 B frame size.

It went away after the operator intervened.
**Operator: fill in what you did → `<OPERATOR NOTE: e.g. power-cycled the radar / changed the 12 V supply / re-flashed>`.**
Explain what in the firmware or hardware can produce this.

### Finding C: other configuration observations

1. **HPF floor:** the high-pass corners suppress returns closer than about
   **10.5 m** on `r075m` (`hpf_floor_m` in `firmware_map.json`). For a
   two-wheeler safety dataset, near targets matter.
   - For comparison: `r050m` is 7 m and `r017m` is 1.17 m.
   - Review whether this trade-off is intended.
2. **Frame rate:** the goal is 8 fps (125 ms). Every variant in
   `firmware_map.json` is 250 ms, except `r017m_calib` at 1000 ms.
3. **Bursts:** each 54 ms transmit burst starts a fraction of a packet *before*
   a frame boundary, because a frame is 2160.5 packets. The Jetson handles
   this, so it is fine. Mentioned only so you are not surprised by it.

---

## 3. Hypotheses: confirm or rule out each, with evidence

1. **Compression is enabled in the firmware** (`compressionCfg … 1 … 0.5 8`).
   - What exactly is compressed: raw ADC samples, or range-FFT output (note
     "8 range bins per block")?
   - What does the LVDS/HSI stream then contain: compressed blocks plus padding
     or fill?
   - Does that explain the constant 32-byte blocks and the ~50% split?
2. **Lane or format mismatch between radar and DCA1000.**
   - Radar side: `hsiLaneConfig`, `hsiClockConfig`, `dataFormatConfig`, CBUFF
     setup.
   - DCA side: the Jetson's `CONFIG_FPGA` bytes `01 01 01 02 03 1E` (logging
     mode, LVDS lane mode, transfer mode, capture mode, data format, timer).
   - Compare with what the Windows tools (mmWave Studio / DCA1000 CLI / the
     original project) use for the same board. A 2-lane/4-lane or
     interleave mismatch can produce interleaved invalid blocks.
3. **Frame layout or I/Q order** differ from the Jetson's assumptions (section 1).
4. **The `.cfg` does not match the firmware** (for example the `frameCfg` "50").
   The Jetson labels range and Doppler axes from the `.cfg`, so any mismatch
   corrupts the processing silently.
5. **Cause of the 908 pkt/s episode** (Finding B).
6. **Broadband streaks** at a few instants: interference, cascade
   synchronisation (LO/frame sync between the two devices), or something else.

---

## 4. Deliverables

1. **Report** (markdown), with `file:line` references into the SDK and
   firmware source:
   - what `r075m` actually streams: data type, compression, per-frame byte
     layout, I/Q order, channel and chirp order, bytes per frame, frame period;
   - the verdict on each hypothesis in section 3;
   - the root cause of the constant blocks, **proven by decoding the provided
     frames**.
2. **Decode check against ground truth.** Decode `stage3_01` with your
   understanding of the format and produce a range-time map. A correct decode
   shows a target following `sync/lidar_target_truth.csv`:
   - **0–30 s:** range moves back and forth between **12.3 and 14.0 m** (4 round
     trips);
   - **30–50 s:** range steady at about 12.5 m;
   - **50–60 s:** standing at 12.4 m.

   That file holds the LiDAR-measured target range and azimuth for each radar
   frame. Expect at most a few cm of range error from the not-yet-calibrated
   time offset.
3. **A corrected firmware build, not flashed:**
   - **compression disabled**, raw ADC out;
   - otherwise the same chirp parameters as `r075m`;
   - optionally a second build at **125 ms (8 fps)**.

   Before building, ask the operator about the HPF floor (Finding C1).
   For each build:
   - update the `.cfg` header and values so they match `cascade_config.h`
     exactly;
   - update `firmware_map.json` (all fields, and the md5 of the `.appimage` and
     `.rprc`; `.xer5f` is non-deterministic, so do not hash it);
   - add a `compression` field to every variant.
4. **Data-format spec for the Jetson**, one page. Include:
   - bytes per frame;
   - exact layout;
   - I/Q order;
   - the virtual-channel order (TX/RX → azimuth position);
   - expected packets/s and MB/s at 4 and 8 fps;
   - any DCA1000 `CONFIG_FPGA` / `CONFIG_PACKET_DATA` changes the Jetson must make.
5. **Flash procedure** for the operator, for after they approve.

---

## 5. Data and files provided (copied from the Jetson)

| Path | Contents |
|---|---|
| `sessions/stage3_01/radar/frame_*.bin` | 239 raw frames, 3,145,728 B each (718 MB). Outdoor, night, person walking with a metal plate. |
| `sessions/stage3_01/radar/frames.csv` | Per frame: timing, `lost_bytes` (all 0), `burst_offset_bytes` |
| `sessions/stage3_01/sync/lidar_target_truth.csv` | **Ground truth:** LiDAR target range and azimuth per radar frame (198 of 239 known) |
| `sessions/stage3_01/session.log`, `session.json` | Full capture log, including DCA command replies and per-second packet counts |
| `sessions/ipad_test_01/radar/` | 120 indoor frames, radar facing a wall (361 MB). Same constant-block structure. |
| `cascaded_radar_capture_pipeline/firmware/firmware_map.json`, `r075m/cascade_r075m.cfg` | As on the Jetson |
| `cascaded_radar_capture_pipeline/processing/radar_config.py`, `radar_reader.py`, `capture/config.py` | The Jetson's decoding assumptions |
| `sync_capture_pipeline/dca_capture_radar_sync.py` | The Jetson's DCA1000 control and receiver |

### Reproduce the constant-block finding (Python + NumPy)

```python
import numpy as np, glob
files = sorted(glob.glob("sessions/stage3_01/radar/frame_*.bin"))
F = np.stack([np.fromfile(f, dtype=np.int16) for f in files])        # [239, 1572864]
const = (F == F[0]).all(axis=0)
print(f"identical in every frame: {100 * const.mean():.1f}%")           # expect ~53.5
edges = np.flatnonzero(np.diff(np.r_[0, const.astype(np.int8), 0]))
runs = edges[1::2] - edges[0::2]
print("constant run lengths (int16): min", runs.min(), "median", int(np.median(runs)), "max", runs.max())
print("first constant runs (start, length):", list(zip(edges[0::2][:6], runs[:6])))
print(f"|value| constant {np.abs(F[0][const]).mean():.0f} vs varying {np.abs(F[:, ~const]).mean():.0f}")

# Range-time on the varying samples only (Jetson layout assumption, I-first)
X = F.astype(np.float32); X[:, const] = 0
X = X.reshape(len(F), 384, 8, 256, 2)
iq = X[..., 0] + 1j * X[..., 1]
iq -= iq.mean(axis=3, keepdims=True)
prof = (np.abs(np.fft.fft(iq * np.hanning(256), axis=3)) ** 2).mean(axis=(1, 2))   # [frames, 256]
fg = 10 * np.log10(np.clip(prof / np.median(prof, axis=0), 1e-3, None))
# range bin k = k * 0.2929 m; compare fg over time with lidar_target_truth.csv
```

---

## 6. How the fix will be verified on the Jetson

After flashing, the Jetson team will record a short session and check:

| Check | Expected after the fix |
|---|---|
| Packets/s | ~8,650 at 4 fps (~17,300 at 8 fps), steady from the first second |
| Frame size and alignment | Matches your spec; every burst aligned |
| Constant-across-frames fraction | About 0%, not ~50% |
| Range-time map | A walking person appears clearly and follows the LiDAR range |

Then the outdoor time calibration will be repeated in daylight.
