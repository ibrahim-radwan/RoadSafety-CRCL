# Radar firmware status (updated 2026-10-10)

> **Licence conditions for these images:** they contain TI components,
> including the AWR2243 firmware patch under the TI Text File License. They
> may be redistributed **only unmodified, for use with TI devices, with no
> reverse engineering**, and only together with the licence texts. See
> [../../THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) §2.

## Which images to use

| Status | Variants |
|---|---|
| **Use, proven on hardware** | `r075m_v2` (250 ms), `r075m_v2_8fps` (125 ms) |
| Use after a 30 s acceptance check | `r017m_v2`, `r050m_v2`, `r100m_v2`, `r142m_v2`, `r168m_v2`, their `_8fps` builds, and `r017m_calib_v2` (1000 ms) |
| **Do not use: data ~50% stale when booted from QSPI** | `r017m`, `r017m_calib`, `r050m`, `r075m`, `r100m`, `r142m`, `r168m` (kept so old recordings can be identified and decoded) |

`firmware_map.json` is the source of truth. Every variant has `data_valid`,
`dss_l3_meminit`, `compression` (always "off"), `frame_periodicity_ms`,
`hpf_floor_m` and md5s.

## What changed in `_v2`

- **DSS L3 MEMINIT at boot.**
  - The old images left the AM273x CSI-2 ping/pong buffers in uninitialised
    DSS L3 RAM.
  - About half of every chirp streamed the RAM's power-up garbage: 32-byte
    blocks, the same in every frame, different after each power-up.
  - CCS loading hid this, because its GEL script initialises that RAM.
  - It is **not compression**: the cascade firmware never compresses, and
    `compressionCfg` in the old `.cfg` files was never sent to the device.
- **HPF 0/0** (175 / 350 kHz) for all variants, which lowers the near floor.
  r075m goes from 10.5 m to 2.62 m. Per-variant windows are in the map.
- **`_8fps` builds:** 125 ms frame period. Everything else is identical,
  checked by disassembly diff.

Confirmed on hardware (r075m_v2 / _8fps): 0.00% stale words, including after
a power cycle. Full investigation and rebuild reports are on the handover
drive: `Jetson_Radar_Sync_Capture_Handover_Firmware_Investigation/investigation/`
and `rebuild_all_variants/REBUILD_SUMMARY.md`.

## Decoding

- int16 LE, **I first**, `[loop 64][tx 6][dev 2][rx 4][sample 256][I,Q]`,
  3,145,728 B per frame.
- The `.cfg` files were corrected 2026-10-10 (`adcbufCfg -1 0 0 1 1`,
  `frameCfg` period, `compressionCfg … 0`). The old ones made
  `radar_config.py` decode Q-first, which mirrors the range spectrum.
- The `.cfg` is **never sent to the device**. It only describes the image
  for processing.

## Flashing (Jetson)

```bash
conda activate rel_1_py_3_8          # plain `python` lacks pyserial
cd ~/dev/CascadedRadarRoadSafetyOnTwoWheelers/flashing/boot
./flash_variant.sh <variant> /dev/ttyACM0
```

`flash_variant.sh`:

- checks the image md5 against the map;
- checks that the flash config writes only this image at `0xA0000`;
- writes **`~/.radar_flashed`**, which the capture pipeline copies into every
  `session.json`.

`uart_uniflash.py` can report an error and still exit successfully, so
**read its output**. If it failed, the record is wrong.

After a flash or any manual change, verify the tree with
`python3 fill_firmware_map.py --root .`. It must show ok in every column for
all variants.

## Acceptance check after flashing a variant

1. Record 30 s radar-only:
   `python3 ../../sync_capture_pipeline/dca_capture_radar_sync.py --secs 30 --out <dir>/radar`.
2. Check the result:
   - about 8,640 pkt/s at 4 fps, or 17,280 at 8 fps;
   - 0 loss;
   - every burst separate and aligned.
3. Run `python ../../sync_capture_pipeline/check_radar_frames.py <dir> --png rt.png`:
   **0.00%** identical in every frame, and no clipping.
4. Power-cycle and repeat once.

**DCA1000:** keep the packet delay at 25 µs (the receiver's default). The old
50 µs caps the stream at 7.44 fps and loses data silently at 8 fps.

## DCA1000 throttle fault (seen 2026-10-11)

- **What happens:** occasionally after power-up, the DCA1000 sends only
  ~3,000 packets/s (or ~900/s) instead of ~17,300 at 8 fps.
  - It buffers the radar data, and after ~100 s it stops capturing.
  - The radar itself is fine.
  - The cause is unknown. It has been seen with both r075m (~900/s) and
    r075m_v2_8fps (~3,000/s).
- **Known fix:** power-cycle the radar + DCA1000.
- **Handling:**
  - the capture launcher stops such a session automatically after 5 s;
  - `check_session` flags it;
  - `align_session` falls back to nominal frame times.
- Example: `/mnt/sdcard/sessions/freewalk_ref_01`.

## Open

- **Antenna calibration** must be redone with the HPF 0/0 images:
  `r017m_calib_v2` for 77 GHz, a corner reflector with a 76 GHz v2 image for
  the rest. The TX/RX → azimuth table is unverified.
- r050m_v2 has the least clipping headroom (~17 dB). Watch the peak sample
  on close targets.
- The Windows capture config `AM273X_Capture_fixed.json` still uses
  `packetDelay_us` 50.
- Backup of the folder before the all-variant rebuild:
  `../firmware_backup_20261010_before_v2_all/`.
