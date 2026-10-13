# Follow-up: the DSS L3 fix works on hardware. Rebuild every range variant with it

You are the agent on the **Windows firmware-build machine**. You (or an earlier
session) wrote `investigation/REPORT.md`, `JETSON_DATA_FORMAT.md` and
`FLASH_PROCEDURE.md` in this handover folder, and built `r075m_v2` and
`r075m_v2_8fps`. Read those three files first. This brief:

1. reports the hardware results from the Jetson;
2. corrects one wrong statement in your data-format spec;
3. asks for the same fix in every other range variant.

**Do not flash anything.** Build, verify, document, and hand back.

---

## 1. Hardware results (Jetson, 2026-10-10)

All sessions were checked with your stale-word test; the Jetson repo now has a
copy, `sync_capture_pipeline/check_radar_frames.py`.

| Session | Image | Result |
|---|---|---|
| radar_fw_v2_check_01 (10 s) | r075m_v2 | 0.00% identical in every frame (r075m: 53.5%). 8,642 pkt/s, 0 loss. Returns visible at 1.5–5 m. |
| radar_fw_v2_check_02_powercycle (60 s, after a power cycle of radar + DCA) | r075m_v2 | **0.00%**. A person walking 2 ↔ 18 m gives a clean continuous range track, down to ~2 m. No streaks. |
| radar_fw_v2_8fps_check_02 (30 s) | r075m_v2_8fps | **0.00%**. 8.00 fps, 125.0 ms frames, 0 loss, 240 of 240 bursts aligned. Peak sample 2,155 (no clipping). |
| Three-sensor test (20 s, radar + ZED 2i + LiDAR) | r075m_v2_8fps | Session check PASS; alignment fits a radar period of 125.002 ms |

**Verdict: the DSS L3 MEMINIT fix is confirmed and holds across power cycles.**
Update `REPORT.md` §8 ("hardware confirmation outstanding") accordingly.

## 2. Correction: the DCA1000 packet delay limits the frame rate

`JETSON_DATA_FORMAT.md` says to keep `CONFIG_PACKET_DATA` = 1470 / 6250 ("no
change needed") and describes a ~54 ms burst per frame. **Both are wrong.**

- The delay field is in 8 ns FPGA ticks: 6250 = **50 µs** after every packet.
  With ~12 µs to send each 1470 B packet, a packet takes ~62 µs, so a
  2,160.5-packet frame takes **134.4 ms**. That caps the stream at **7.44 fps**.
- On the first 8 fps run with 6250, the Jetson received exactly 134.4 ms per
  frame, back to back, with no idle gap. The DCA streamed continuously and
  dropped LVDS data internally (7.43 fps received). The loss was **invisible**
  to the sequence-number and byte-count checks, because it happens before the
  DCA counts bytes.
- At 4 fps the 134 ms transfer fits inside 250 ms, so nothing showed.
- **Fix applied on the Jetson:** delay **25 µs (3125)**, about 80 ms per frame.
  Result: 8.00 fps, clean.

**Actions:**

1. Correct `JETSON_DATA_FORMAT.md`: the rates table, the burst duration, and
   the "DCA1000 settings" section.
2. Check the project's Windows capture config (`cf_cascade.json`
   `packetDelay_us`, and any mmWave Studio / DCA1000 CLI setting). If it is
   50 µs, **any 8 fps capture made on Windows will also lose data silently.**
   Report what it is set to; recommend 25 µs.
3. Add a note to `firmware_map.json` `conventions`: frame time on the wire at
   the DCA delay must be shorter than `frame_periodicity_ms`.

## 3. Task: rebuild every range variant with the fix

The operator decided the following:

- **Fix:** DSS L3 MEMINIT (`SOC_rcmStartMemInitDSSL3/WaitMemInitDSSL3(ALL)`),
  exactly as in `r075m_v2`.
- **HPF: 0/0 (175 kHz / 350 kHz) for ALL variants**, as validated on
  r075m_v2.
- **Frame rates: two images per variant**, 250 ms (4 fps) and 125 ms (8 fps).
  Exception: `r017m_calib` gets one image only and keeps its 1000 ms period.
- **Everything else unchanged** from each variant's current `cascade_config.h`
  block: profile, slope, samples, loops, chirps, gain.

| Base | New variants |
|---|---|
| r017m | `r017m_v2`, `r017m_v2_8fps` |
| r017m_calib | `r017m_calib_v2` (1000 ms only) |
| r050m | `r050m_v2`, `r050m_v2_8fps` |
| r100m | `r100m_v2`, `r100m_v2_8fps` |
| r142m | `r142m_v2`, `r142m_v2_8fps` |
| r168m | `r168m_v2`, `r168m_v2_8fps` |
| r075m | already done; **do not change** `r075m_v2` / `r075m_v2_8fps`. Rebuild both once as a determinism check: the `.rprc` and `.appimage` md5 must stay `c17eaa9b…` / `b4ac5b97…` and `e9a61839…` / `8f5eb30a…`. |

**Things to check and report per variant:**

- **Feasibility at 125 ms.** Active chirping is 384 × (idle + ramp). From the
  map: r017m about 25 ms, r142m about 23 ms, r017m_calib about 63 ms, others
  about 13.4 ms. Add ~80 ms of DCA transfer at 25 µs. Confirm each fits in the
  frame period with margin. Also check the per-chirp CBUFF timing (8192 B per
  chirp at 4 lanes × 600 Mbps is ~27 µs) against each variant's chirp time.
  r075m has a 35 µs chirp; the others are longer, so this should be easier.
- **New near floor and HPF1 knee** at HPF 0/0. Use HPF1 = 175/235/350/700 kHz
  and HPF2 = 350/700/1400/2800 kHz for registers 0–3. Fill `hpf_floor_m`,
  `hpf1_floor_m` and `usable_window_m`.
- **Clipping risk.** With the lower HPF corners, strong near returns pass. On
  r075m_v2 the peak was ~2,155 of 32,767, but the short-range variants
  (r017m) may see more. Note the expected headroom.
- **Antenna calibration.** HPF corners change the RX phase and amplitude
  response. Say whether the `antennaCalibParams` in each `.cfg` were measured
  with HPF 2/2. If so, flag that calibration must be repeated with the 0/0
  images (the `r017m_calib_v2` image exists for this). Also, the TX/RX → azimuth
  table is still unverified (your §4).
- **Disassembly diff vs the base image.** As you did for r075m → v2: the only
  functional changes should be MEMINIT, the HPF fields, and (for `_8fps`) the
  frame-period immediate.

## 4. Deliverables: where they go and how they are laid out

### 4.1 Output location: a NEW folder inside the handover root

The handover root is the folder containing this brief:
`Jetson_Radar_Sync_Capture_Handover_Firmware_Investigation/`.
Put **everything you produce** in one new folder inside it:

```
Jetson_Radar_Sync_Capture_Handover_Firmware_Investigation/
├── jetson_firmware_current/       <- EXACT copy of the Jetson's firmware folder today (+ MANIFEST.txt md5s).
│                                     START FROM THIS. DO NOT MODIFY it - it is the baseline the Jetson diffs against
├── firmware/                      <- earlier r075m_v2 delivery (superseded by jetson_firmware_current/); do not modify
├── investigation/                 <- existing; update REPORT.md / JETSON_DATA_FORMAT.md / FLASH_PROCEDURE.md here
├── jetson_updates_20261010/       <- read-only context from the Jetson
├── ...
└── rebuild_all_variants/          <- NEW: all deliverables of this task
    ├── firmware/                  <- COMPLETE drop-in replacement for the Jetson's
    │   │                             cascaded_radar_capture_pipeline/firmware/  (see 4.2)
    │   ├── firmware_map.json
    │   ├── flash_provision.cfg            (unchanged from jetson_firmware_current/)
    │   ├── fill_firmware_map.py           (unchanged; already has the HPF1 fix)
    │   ├── make_flash_cfgs.py             (unchanged unless you fix it; say so)
    │   ├── r017m/ ... r168m/              (old variants: image unchanged, .cfg text corrected)
    │   ├── r075m_v2/  r075m_v2_8fps/      (unchanged)
    │   ├── r017m_v2/  r017m_v2_8fps/  r017m_calib_v2/
    │   ├── r050m_v2/  r050m_v2_8fps/
    │   ├── r100m_v2/  r100m_v2_8fps/
    │   ├── r142m_v2/  r142m_v2_8fps/
    │   ├── r168m_v2/  r168m_v2_8fps/
    │   └── _build_record/<every variant>/
    ├── source/                    <- cascade_config.h, main_mss.c, makefile, build_one_inf.bat as built
    ├── REBUILD_SUMMARY.md
    └── MANIFEST.txt               <- md5 of every file under firmware/, one per line: "<md5>  <relative path>"
```

### 4.2 `rebuild_all_variants/firmware/` must mirror the Jetson tree exactly

The Jetson will copy this folder over
`/home/jetson/dev/CascadedRadarRoadSafetyOnTwoWheelers/cascaded_radar_capture_pipeline/firmware/`
after a diff. So it must be **complete**, including the unchanged variants and
helper scripts, not only the new files. A variant missing from it would look
deleted. Per variant directory, the file names are fixed:

```
<v>/am273x_cascade_<v>.appimage
<v>/cascade_<v>.cfg
<v>/flash_<v>.cfg
_build_record/<v>/<v>.rprc
_build_record/<v>/<v>.xer5f
_build_record/<v>/am273x_cascade_mss.map
```

Directory and variant names: lower case, exactly as in the table in §3.

**Procedure:**

1. Copy `jetson_firmware_current/` to `rebuild_all_variants/firmware/`.
   Leave out `MANIFEST.txt`; you write a new one at the end.
2. **Add** the new variant directories and build records.
3. **Edit only:**
   - `firmware_map.json` (add the entries, extend `conventions`);
   - the old variants' `cascade_<v>.cfg` text corrections.
4. Leave every old `.appimage`, `.rprc`, `.xer5f`, `.map` and flash config
   byte-identical.

**Flash configs** for the new variants: generate them with `make_flash_cfgs.py`,
or write them in the exact format of the existing ones (absolute Jetson paths,
one `--file` at `0xA0000`). Those paths do not exist on Windows. So
`fill_firmware_map.py --root .` run on Windows may report the `flash` column
as failing for **every** variant, old ones included. That is expected. The
Jetson re-runs the check after the copy, where the paths resolve. All other
columns (`derived`, `appimage`, `cfg`, `rprc`) must be `ok` for every
variant, with the `.rprc` fingerprint verified. Paste the output into
`REBUILD_SUMMARY.md`.

**`MANIFEST.txt`:** md5 of every file under `rebuild_all_variants/firmware/`.
The Jetson diffs it against `jetson_firmware_current/MANIFEST.txt`. The only
differences it should find are:
- the new variant files;
- `firmware_map.json`;
- the corrected old `.cfg` files.

List these expected differences in `REBUILD_SUMMARY.md`.

### 4.3 Contents per new variant `<v>`

- `firmware/<v>/am273x_cascade_<v>.appimage`
- `firmware/<v>/cascade_<v>.cfg`: generated from `cascade_config.h` like the
  r075m_v2 one. Requirements:
  - `adcbufCfg -1 0 0 1 1` (I first);
  - `compressionCfg -1 0 …`;
  - `frameCfg` period matching the image;
  - correct `profileCfg` HPF fields;
  - a header with variant, md5, period, window and HPF.
- `firmware/<v>/flash_<v>.cfg`:
  - **absolute Jetson paths**:
    `--flash-writer=/home/jetson/dev/CascadedRadarRoadSafetyOnTwoWheelers/flashing/boot/sbl_prebuilt/am273x-evm/sbl_uart_uniflash.release.tiimage`
    and `--file=/home/jetson/dev/CascadedRadarRoadSafetyOnTwoWheelers/cascaded_radar_capture_pipeline/firmware/<v>/am273x_cascade_<v>.appimage --operation=flash --flash-offset=0xA0000`;
  - **never** a write to 0x0.
- `firmware/_build_record/<v>/`: `<v>.rprc`, `<v>.xer5f`,
  `am273x_cascade_mss.map`.
- `firmware/firmware_map.json`:
  - one entry per new variant, with **all** the fields the r075m_v2 entry has.
    The Jetson reads `appimage`, `appimage_md5`, `rprc_md5`, `processing_cfg`,
    `flash_cfg`, `slope_mhz_us`, `max_range_m`, `hpf_floor_m`,
    `frame_periodicity_ms`, `fps`, `compression`, `dss_l3_meminit` and
    `data_valid`.
  - Keep the old variants: they mark data recorded with them as not valid, and
    sessions need them for attribution.
- **Corrected `.cfg` files for the old images** (r017m … r168m): the same text
  corrections you made to `r075m/cascade_r075m.cfg` (`adcbufCfg`, `frameCfg`,
  `compressionCfg`, `dataPathConfig`, header). Old recordings are decoded with
  them, and the old Q-first `adcbufCfg` mirrors the spectrum. Images unchanged.
- **Updated docs:**
  - `REPORT.md` §8 (fix confirmed) and a new section for this rebuild;
  - `JETSON_DATA_FORMAT.md` (packet delay; rates per variant at 4/8 fps);
  - `FLASH_PROCEDURE.md`: on the Jetson, flash with
    `flashing/boot/flash_variant.sh <variant> <port>`, run inside the conda
    env `rel_1_py_3_8` (plain `python` has no pyserial). It checks the md5
    against the map and writes `~/.radar_flashed`, which the Jetson capture
    pipeline reads.
- A short `REBUILD_SUMMARY.md`: a table of variant / period / window / HPF /
  md5s / feasibility / notes.

## 5. Jetson-side changes you should know about

- `sync_capture_pipeline/radar_firmware.py` resolves each session's firmware.
  It uses `~/.radar_flashed`, copied into `session.json["radar_firmware"]`, and
  `firmware_map.json` `processing_cfg`. A missing or misnamed field breaks
  decoding.
- `cascaded_radar_capture_pipeline/firmware/fill_firmware_map.py` had the HPF1
  table wrong (700 kHz for register 2). It is fixed on the Jetson:
  `{0: 175e3, 1: 235e3, 2: 350e3, 3: 700e3}`. After the transfer, the Jetson
  runs `fill_firmware_map.py --root .` to verify every hash and every flash
  config.
- `processing/radar_config.py`: the compression warning is now informational.
  Its I/Q order follows `adcbufCfg`, so the corrected `.cfg` files decode
  correctly.
- `dca_capture_radar_sync.py`: `--packet-delay-us` (default 25), and a warning
  when the DCA streams with no idle gap between frames.

## 6. When the Jetson gets the build back

For each variant the team will use:

1. Flash with `flash_variant.sh`.
2. Record a 30 s radar-only session and check: about 8,640 pkt/s at 4 fps or
   17,280 at 8 fps, 0 loss, every burst separate and aligned, **0.00%**
   identical-in-every-frame.
3. Check the range-time map: a walking person tracked inside the variant's
   window, including near the new floor.
4. Power-cycle and repeat once.
